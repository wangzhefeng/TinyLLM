from __future__ import annotations

import contextlib
from collections.abc import Callable

import torch


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        grad_accum_steps: int = 1,
        grad_clip: float = 1.0,
        use_amp: bool = False,
        use_ddp: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.grad_clip = grad_clip or 0.0
        self.use_ddp = use_ddp
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp)
        self.optimizer.zero_grad()

    def _amp_context(self):
        if not self.use_amp:
            return contextlib.nullcontext()
        return torch.amp.autocast(device_type=self.device.type)

    def train_step(self, batch, loss_fn: Callable, batch_idx: int, num_batches: int):
        is_step = (batch_idx + 1) % self.grad_accum_steps == 0 or batch_idx + 1 == num_batches
        sync = (
            self.model.no_sync()
            if self.use_ddp and not is_step and hasattr(self.model, "no_sync")
            else contextlib.nullcontext()
        )
        with sync, self._amp_context():
            result = loss_fn(batch)
            loss = result[0] if isinstance(result, tuple) else result
            scaled_loss = loss / self.grad_accum_steps
        if self.use_amp:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        if is_step:
            if self.use_amp:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            self.optimizer.zero_grad()
        return result, is_step

    @torch.inference_mode()
    def evaluate(self, data_loader, loss_fn: Callable, num_batches: int | None = None) -> float:
        was_training = self.model.training
        self.model.eval()
        total = 0.0
        count = 0
        limit = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
        for batch in data_loader:
            if count >= limit:
                break
            result = loss_fn(batch)
            loss = result[0] if isinstance(result, tuple) else result
            total += float(loss.item())
            count += 1
        if was_training:
            self.model.train()
        if count == 0:
            return float("nan")
        return total / count

    def state_dict(self):
        return {"scaler": self.scaler.state_dict() if self.use_amp else None}

    def load_state_dict(self, state):
        if self.use_amp and state and state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])
