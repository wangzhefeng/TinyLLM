# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-23
# * Version     : 1.0.012322
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import contextlib
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler

# data
from pre_training.data_provider.data_loader import create_dataloader
# tokenizer
from layers.tokenizers.tokenization import (
    choose_tokenizer,
    text_to_token_ids, 
    token_ids_to_text
)
# model
from pre_training.experiments.exp_basic import Exp_Basic
# training
from utils.optimizer import select_optimizer
from utils.criterion import select_criterion
from utils.calc_loss import calc_loss_batch, calc_loss_loader
from utils.learning_rate import adjust_learning_rate, warmup_cosine_lr, set_optimizer_lr
from utils.early_stopping import EarlyStopping
from tools.checkpoint import CheckpointManager, load_checkpoint
from layers.tokenizers.extension import pad_vocab_size
from utils.plot_losses import plot_losses_llm, plot_losses
from layers.inference import generate
# utils
from utils.model_memory import model_memory_size
from utils.timestamp_utils import from_unix_time
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Model_Pretrain(Exp_Basic):

    def __init__(self, args, global_rank, world_size, local_rank):
        logger.info(f"{41 * '-'}")
        logger.info("Initializing Experiment...")
        logger.info(f"{41 * '-'}")
        super(Model_Pretrain, self).__init__(args, local_rank)
        self.global_rank = global_rank
        self.world_size = world_size
        self.local_rank = local_rank
    
    def _get_tokenizer(self):
        """
        get tokenizer
        """
        tokenizer = choose_tokenizer(tokenizer_model = self.args.tokenizer_model)
        
        return tokenizer
    
    def _get_data(self):
        """
        get dataset and dataloader
        """
        train_data, train_loader = create_dataloader(
            data_source=self.args.data_source,
            url=self.args.data_url,
            data_path=self.args.data_path,
            data_file=self.args.data_file,
            flag="train",
            train_ratio=self.args.train_ratio,
            tokenizer=self.tokenizer,
            batch_size=self.args.batch_size,
            max_len=self.args.context_length,
            stride=self.args.context_length,
            num_workers=self.args.num_workers,
            device=self.device,
            global_rank=self.global_rank,
            world_size=self.world_size,
        )
        valid_data, valid_loader = create_dataloader(
            data_source=self.args.data_source,
            url=self.args.data_url,
            data_path=self.args.data_path,
            data_file=self.args.data_file,
            flag="valid",
            train_ratio=self.args.train_ratio,
            tokenizer=self.tokenizer,
            batch_size=self.args.batch_size,
            max_len=self.args.context_length,
            stride=self.args.context_length,
            num_workers=self.args.num_workers,
            device=self.device,
            global_rank=self.global_rank,
            world_size=self.world_size,
        )
        
        return train_loader, valid_loader 

    @staticmethod
    def _unwrap_model(model):
        """
        剥掉 DP/DDP 的 module 与 torch.compile 的 _orig_mod 包装，返回原始模型。
        """
        while True:
            if hasattr(model, "_orig_mod"):
                model = model._orig_mod
            elif hasattr(model, "module"):
                model = model.module
            else:
                return model

    def _build_model(self):
        """
        模型构建
        """
        # 词表对齐 padding（ch05/10）：提升 matmul 效率，0/1 表示不 padding
        pad_multiple = getattr(self.args, "pad_vocab_to_multiple_of", 0) or 0
        if pad_multiple > 1:
            padded_vocab_size = pad_vocab_size(self.args.vocab_size, pad_multiple)
            if padded_vocab_size != self.args.vocab_size:
                logger.info(
                    f"Pad vocab size: {self.args.vocab_size} -> {padded_vocab_size} "
                    f"(multiple of {pad_multiple})"
                )
                self.args.vocab_size = padded_vocab_size
        # 模型实例化
        logger.info(f"Initializing model {self.args.model_name}...")
        model = self.model_dict[self.args.model_name].Model(self.args)
        # 单机多卡训练
        if self.args.use_gpu and self.args.use_dp:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        elif self.args.use_gpu and self.args.use_ddp:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank])
        # 打印模型参数量
        model_memory_size(model, input_dtype=self.args.dtype, verbose=True)
        
        return model

    def _get_model_path(self, setting):
        """
        模型保存路径
        """
        # 模型保存路径
        model_path = Path(self.args.checkpoints).joinpath(setting)
        model_path.mkdir(parents=True, exist_ok=True)
        # 最优模型保存路径
        model_checkpoint_path = f"{model_path}/checkpoint.pth"
        
        return model_checkpoint_path

    def _get_results_path(self, setting):
        """
        结果保存路径
        """
        results_path = Path(self.args.test_results).joinpath(setting)
        results_path.mkdir(parents=True, exist_ok=True)
        
        return results_path 

    def train(self, training_iter: int, setting: str, eval_freq: int=10, eval_iter: int=1):
        logger.info(f"{43 * '-'}")
        logger.info(f"Model start training...")
        logger.info(f"{43 * '-'}")
        # ------------------------------
        # data
        # ------------------------------
        train_loader, valid_loader = self._get_data()
        logger.info(f"Train and valid dataloader has builded...")
        # train steps
        train_steps = len(train_loader)
        logger.info(f"Train total steps: {train_steps}")
        # ------------------------------
        # model and results path
        # ------------------------------
        # checkpoint path
        model_checkpoint_path = self._get_model_path(setting)
        logger.info(f"Train checkpoint will be saved in path: {model_checkpoint_path}")
        # test results path
        results_path = self._get_results_path(setting)
        logger.info(f"Train results will be saved in path: {results_path}") 
        # ------------------------------
        # train optimizer
        # ------------------------------
        optimizer = select_optimizer(
            self.model, 
            learning_rate=self.args.learning_rate, 
            weight_decay=self.args.weight_decay,
        )
        logger.info(f"Train optimizer has builded...")
        # ------------------------------
        # 
        # ------------------------------
        # criterion
        criterion = select_criterion()
        logger.info(f"Train criterion has builded...")
        # ------------------------------
        # early stopping
        # ------------------------------
        early_stopping = EarlyStopping(
            patience=self.args.patience, 
            verbose=True,
            delta=0,
            use_ddp=self.args.use_ddp, 
            gpu=self.global_rank,
        )
        logger.info(f"Train early stopping instance has builded, patience: {self.args.patience}")
        # ------------------------------
        # auto mix precision
        # ------------------------------
        if self.args.use_amp:
            scaler = torch.amp.GradScaler(device = self.device)
            logger.info(f"Train auto mix precision instance has builded...") 
        # ------------------------------
        # learning rate schedule（附录 D：step 级 warmup + cosine）
        # ------------------------------
        lr_schedule = getattr(self.args, "lr_schedule", "warmup_cosine")
        total_training_steps = train_steps * self.args.train_epochs
        warmup_steps = int(getattr(self.args, "warmup_ratio", 0.2) * total_training_steps)
        peak_lr = self.args.learning_rate
        current_lr = peak_lr
        # 梯度累积与梯度裁剪
        grad_accum_steps = max(1, getattr(self.args, "grad_accum_steps", 1))
        grad_clip = getattr(self.args, "grad_clip", 1.0) or 0.0
        if grad_accum_steps > 1:
            logger.info(f"Gradient accumulation enabled: {grad_accum_steps} micro-batches per optimizer step "
                        f"(effective batch size = {self.args.batch_size * grad_accum_steps * self.world_size})")
        logger.info(f"LR schedule: {lr_schedule}, total steps: {total_training_steps}, "
                    f"warmup steps: {warmup_steps}, peak lr: {peak_lr}, grad clip: {grad_clip}")
        # ------------------------------
        # resume from checkpoint
        # ------------------------------
        ckpt_manager = None
        save_every_steps = getattr(self.args, "save_every_steps", 0) or 0
        if save_every_steps > 0 or getattr(self.args, "resume", ""):
            ckpt_manager = CheckpointManager(
                Path(model_checkpoint_path).parent / "steps",
                keep_last=getattr(self.args, "keep_last_checkpoints", 3),
            )
        start_epoch, resume_batch = 0, 0
        resume_arg = getattr(self.args, "resume", "") or ""
        if resume_arg:
            resume_path = ckpt_manager.latest() if resume_arg == "latest" else Path(resume_arg)
            if resume_path is None or not Path(resume_path).exists():
                raise FileNotFoundError(f"resume checkpoint not found: {resume_arg}")
            raw_model = self._unwrap_model(self.model)
            payload = load_checkpoint(
                resume_path, model=raw_model, optimizer=optimizer,
                map_location=self.device, strict=True,
            )
            extra = payload.get("extra", {})
            start_epoch = extra.get("epoch", 0)
            resume_batch = extra.get("batch", 0)
            if self.args.use_amp and extra.get("scaler"):
                scaler.load_state_dict(extra["scaler"])
            logger.info(f"Resumed from {resume_path}: epoch {start_epoch}, batch {resume_batch}, "
                        f"global step {extra.get('global_step', -1)}")
        # ------------------------------
        # training data collactor
        # ------------------------------
        # initialize train iter global steps
        global_step = -1
        # initialize list to track train iter losses
        train_losses, valid_losses = [], []
        # initialize tokens seen
        track_tokens_seen, total_tokens_seen, last_tokens = [], 0, 0
        # Variables for cumulative average tokens/sec
        cumulative_tokens, cumulative_time = 0.0, 0.0
        if resume_arg:
            global_step = extra.get("global_step", -1)
            total_tokens_seen = extra.get("total_tokens_seen", 0)
            last_tokens = total_tokens_seen
        # ------------------------------
        # CUDA-specific timing setup
        # ------------------------------
        use_cuda = self.device.type == "cuda"
        if self.args.use_gpu and use_cuda:
            t_start = torch.cuda.Event(enable_timing=True)
            t_end = torch.cuda.Event(enable_timing=True)
            # Ensure all prior CUDA operations are done
            torch.cuda.synchronize()
            # Start the timer for the first interval
            t_start.record()
        else:
            # Start the timer for the first interval
            t0 = time.time()
        
        # training start time
        train_start_time = time.time()
        logger.info(f"Train start time: {from_unix_time(train_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        # main training loop
        for epoch in range(start_epoch, self.args.train_epochs):
            # epoch 模型训练开始时间
            epoch_start_time = time.time()
            logger.info(f"\t\tEpoch {epoch + 1}: start time: {from_unix_time(epoch_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
            
            # initialize epcoh iter count
            iter_count = 0

            # set epoch for DistributedSampler so each process gets a unique shuffle order
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            
            # training mode
            self.model.train()
            # ------------------------------
            # model training
            # ------------------------------
            for batch, (input_batch, target_batch) in enumerate(train_loader):
                # resume: skip batches already finished in the start epoch
                if epoch == start_epoch and batch < resume_batch:
                    continue
                # update train iter global step
                global_step += 1
                # update epoch iter count
                iter_count += 1

                # learning rate: step-level linear warmup + cosine decay (appendix D)
                if lr_schedule == "warmup_cosine":
                    current_lr = warmup_cosine_lr(
                        global_step=global_step,
                        total_steps=total_training_steps,
                        warmup_steps=warmup_steps,
                        peak_lr=peak_lr,
                        initial_lr=self.args.initial_lr,
                        min_lr=self.args.min_lr,
                    )
                    set_optimizer_lr(optimizer, current_lr)

                # gradient accumulation: only step on the last micro-batch
                is_step_batch = ((batch + 1) % grad_accum_steps == 0) or (batch + 1 == train_steps)
                # DDP: skip gradient all-reduce on non-step micro-batches
                if self.args.use_ddp and not is_step_batch and hasattr(self.model, "no_sync"):
                    sync_context = self.model.no_sync()
                else:
                    sync_context = contextlib.nullcontext()

                with sync_context:
                    # forward
                    # ------------------------
                    # calculate train loss
                    if self.args.use_amp:
                        with torch.amp.autocast(device_type=self.args.gpu_type):
                            loss = calc_loss_batch(
                                task_name=self.args.task_name,
                                input_batch=input_batch,
                                target_batch=target_batch,
                                model=self.model,
                                criterion=criterion,
                                device=self.device,
                            )
                    else:
                        loss = calc_loss_batch(
                            task_name=self.args.task_name,
                            input_batch=input_batch,
                            target_batch=target_batch,
                            model=self.model,
                            criterion=criterion,
                            device=self.device,
                        )
                    # normalize loss over accumulation steps
                    loss = loss / grad_accum_steps

                    # backward
                    # ------------------------
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                # optimizer step (with gradient clipping, appendix D)
                # ------------------------
                if is_step_batch:
                    if self.args.use_amp:
                        if grad_clip > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                        optimizer.step()
                    # reset loss gradients for the next accumulation window
                    optimizer.zero_grad()

                # periodic resume checkpoint (rank 0 only)
                # ------------------------
                if (
                    ckpt_manager is not None and save_every_steps > 0
                    and is_step_batch and global_step % save_every_steps == 0
                    and self.global_rank == 0
                ):
                    ckpt_path = ckpt_manager.save(
                        step=global_step,
                        model=self._unwrap_model(self.model),
                        optimizer=optimizer,
                        extra={
                            "epoch": epoch,
                            "batch": batch + 1,
                            "global_step": global_step,
                            "total_tokens_seen": total_tokens_seen + input_batch.numel(),
                            "scaler": scaler.state_dict() if self.args.use_amp else None,
                        },
                    )
                    logger.info(f"\t\tCheckpoint saved: {ckpt_path}")

                # update tokens seen
                # ------------------------
                total_tokens_seen += input_batch.numel()

                # optional evaluation step
                # ------------------------
                if global_step % eval_freq == 0:
                    if self.args.use_ddp:
                        # End timing for the current interval
                        if self.args.use_gpu and use_cuda:
                            t_end.record()
                            # Wait for all CUDA ops to complete.
                            torch.cuda.synchronize()
                            # Convert ms to seconds
                            elapsed = t_start.elapsed_time(t_end) / 1000
                            # Reset timer for the next interval
                            t_start.record()
                        else:
                            elapsed = time.time() - t0
                            # Reset timer for the next interval
                            t0 = time.time()
                        
                        # Calculate local tokens processed during this interval
                        local_interval = total_tokens_seen - last_tokens
                        last_tokens = total_tokens_seen

                        # Aggregate the tokens processed over all devices
                        local_tensor = torch.tensor([local_interval], device=self.device, dtype=torch.float)
                        global_tensor = local_tensor.clone()
                        torch.distributed.all_reduce(global_tensor, op=torch.distributed.ReduceOp.SUM)
                        global_interval = global_tensor.item()

                        # Global tokens per second for this interval
                        global_tps = global_interval / elapsed if elapsed > 0 else 0

                        # Update cumulative tokens (local) and aggregate globally
                        cumulative_tokens += local_interval
                        local_cum_tensor = torch.tensor([cumulative_tokens], device=self.device, dtype=torch.float)
                        global_cum_tensor = local_cum_tensor.clone()
                        torch.distributed.all_reduce(global_cum_tensor, op=torch.distributed.ReduceOp.SUM)
                        global_cumulative_tokens = global_cum_tensor.item()
                        cumulative_time += elapsed
                        global_avg_tps = global_cumulative_tokens / cumulative_time if cumulative_time > 0 else 0
                    
                    # evaluate model
                    train_loss, valid_loss = self.valid(criterion, train_loader, valid_loader, eval_iter)
                    # collect losses and tokens seen
                    train_losses.append(train_loss)
                    valid_losses.append(valid_loss)
                    track_tokens_seen.append(total_tokens_seen)
                    # calculate training left time
                    if self.global_rank == 0:
                        speed = (time.time() - train_start_time) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - batch)
                        logger.info(f'\t\tEpoch {epoch + 1}: Batch {batch + 1} (Step {global_step:06d}): Train loss: {train_loss:.3f}, Val loss {valid_loss:.3f}, LR {current_lr:.2e} | Speed: {speed:.4f}s/batch; left time: {left_time:.2f}seconds.')
                        if self.args.use_ddp:
                            # Only print logs once per GPU (choosing the rank 0 GPU)
                            logger.info(f'\t\tEpoch {epoch + 1}, Batch {batch + 1} (Step {global_step:06d}):, Step tok/sec: {round(global_tps)}, Global avg tok/sec: {round(global_avg_tps)}')
                
                    # init epoch iter count
                    iter_count = 0
                    # init train start time
                    train_start_time = time.time()
            # ------------------------------
            # 早停机制、模型保存
            # ------------------------------
            early_stopping(epoch=epoch + 1, val_loss=valid_loss, model=self.model,
                           optimizer=optimizer, model_path=model_checkpoint_path)
            if early_stopping.early_stop:
                logger.info(f"\t\tEpoch {epoch + 1}: Early stopping...")
                break
            # 学习率调整（epoch 级策略；warmup_cosine 已在 step 级调整）
            if lr_schedule == "epoch":
                adjust_learning_rate(optimizer=optimizer, epoch=epoch + 1, args=self.args)
            # ------------------------------
            # model inference
            # ------------------------------
            if self.global_rank == 0 and epoch % 5 == 0:
                # print a sample text after each epoch
                self.inference(epoch=epoch + 1, setting=setting, load=False)
                # Memory stats
                if torch.cuda.is_available():
                    current_device = torch.cuda.current_device()
                    allocated = torch.cuda.memory_allocated(current_device) / 1024 ** 3
                    reserved = torch.cuda.memory_reserved(current_device) / 1024 ** 3
                    logger.info(f"Allocated memory: {allocated:4f}GB, Reserved memory: {reserved:4f}GB")
            
            # calculate one epoch training used time
            logger.info(f"\t\tEpoch {epoch + 1}: cost time: {(time.time() - epoch_start_time):.2f}seconds")
        # ------------------------------
        # 模型训练结果保存、模型加载
        # ------------------------------
        logger.info(f"{43 * '-'}")
        logger.info(f"Training Finished!")
        logger.info(f"{43 * '-'}")
        # calculate all epoch training used time
        logger.info(f"Training Iter {training_iter + 1} cost time: {((time.time() - train_start_time) / 60):.2f}mins")
        
        # plot loss
        logger.info("Plot and save train/valid losses...")
        plot_losses_llm(self.args.train_epochs, track_tokens_seen, train_losses, valid_losses, "loss_llm", results_path)
        
        # model load
        logger.info("Loading best model...")
        self._unwrap_model(self.model).load_state_dict(
            torch.load(model_checkpoint_path, map_location=self.device, weights_only=True)["model"]
        )
        # final resume checkpoint
        if ckpt_manager is not None and save_every_steps > 0 and self.global_rank == 0:
            ckpt_path = ckpt_manager.save(
                step=global_step,
                model=self._unwrap_model(self.model),
                optimizer=optimizer,
                extra={
                    "epoch": self.args.train_epochs,
                    "batch": 0,
                    "global_step": global_step,
                    "total_tokens_seen": total_tokens_seen,
                    "scaler": scaler.state_dict() if self.args.use_amp else None,
                },
            )
            logger.info(f"Final checkpoint saved: {ckpt_path}")
        
        # return model and train results
        logger.info("Return training results...")
        return self.model

    def valid(self, criterion, train_loader, valid_loader, eval_iter):
        """
        model evaluation
        """
        logger.info(f"\t\tModel start validating...")
        # deactivate dropout
        self.model.eval()
        # model evaluation
        with torch.no_grad():
            train_loss = calc_loss_loader(
                task_name=self.args.task_name,
                data_loader=train_loader,
                model=self.model, 
                criterion=criterion,
                device=self.device,
                num_batches=eval_iter
            )
            valid_loss = calc_loss_loader(
                task_name=self.args.task_name,
                data_loader=valid_loader,
                model=self.model, 
                criterion=criterion,
                device=self.device,
                num_batches=eval_iter
            )
        # training mode
        self.model.train()
        
        return train_loss, valid_loss

    def inference(self, epoch, setting: str, load: bool=False, 
                  start_context: str="Every effort moves you", eos_id: int=None,
                  temperature: float=0.0, top_k: int=1):
        """
        model inference
        """
        logger.info(f"\t\tModel start inference...")
        # load model
        if load:
            model_checkpoint_path = self._get_model_path(setting)
            logger.info(f"Pretrained model has loaded from: {model_checkpoint_path}")
            self._unwrap_model(self.model).load_state_dict(
                torch.load(model_checkpoint_path, map_location=self.device, weights_only=True)["model"]
            )
        # inference mode
        self.model.eval()
        # eos: 优先取参数，其次 args，默认 GPT-2 的 50256
        if eos_id is None:
            eos_id = getattr(self.args, "eos_token_id", 50256)
        # context size: 优先 args（RoPE 模型没有 pos_embed）
        context_length = getattr(self.args, "context_length", None)
        if context_length is None:
            raw_model = self._unwrap_model(self.model)
            context_length = raw_model.pos_embed.weight.shape[0]
        # start context tokenization
        start_context_encoded = text_to_token_ids(start_context, self.tokenizer).to(self.device)
        # generate text
        with torch.no_grad():
            completion_id = generate(
                model = self.model, 
                token_ids = start_context_encoded,
                max_new_tokens = self.args.max_new_tokens,
                context_length = context_length,
                temperature = temperature,
                top_k = top_k,
                eos_token_id = eos_id,
                use_cache = self.args.use_cache,
            )
            # 词表 padding 后可能采到未训练的 padding id，解码前过滤掉
            n_vocab = getattr(self.tokenizer, "n_vocab", None)
            if n_vocab is not None:
                completion_id = completion_id[completion_id < n_vocab]
            completion = token_ids_to_text(completion_id, self.tokenizer).replace("\n", " ")
            logger.info(f"\t\tEpoch {epoch}: Model inference [start context]: {start_context}, [completion]: {completion}")
        # train mode
        self.model.train()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
