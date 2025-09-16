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
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler

# data
from data_provider.data_loader import create_dataloader
# tokenizer
from layers.tokenizers.tokenization import (
    choose_tokenizer,
    text_to_token_ids, 
    token_ids_to_text
)
# model
from exp.exp_basic import Exp_Basic
# training
from utils.llm.optimizer import select_optimizer
from utils.llm.criterion import select_criterion
from utils.llm.calc_loss import calc_loss_batch, calc_loss_loader
from utils.llm.train_funcs import adjust_learning_rate, EarlyStopping
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

    def _build_model(self):
        """
        模型构建
        """
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
        # TODO learning rate
        # ------------------------------
        """
        track_lrs = []
        # 从优化器中获取最大学习率
        peak_lr = 0.001  # peak_lr = optimizer.param_groups[0]["lr"]
        # 计算训练过程中总的迭代次数
        total_training_steps = len(train_loader) * self.args.train_epochs
        # warmup steps
        warmup_steps = int(0.2 * total_training_steps) 
        # 计算 warmup 阶段的迭代次数
        lr_increment = (peak_lr - self.args.initial_lr) / warmup_steps
        """
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
        for epoch in range(self.args.train_epochs):            
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
                # update train iter global step
                global_step += 1
                # update epoch iter count
                iter_count += 1 
                
                # reset loss gradients from previous batch iteration
                optimizer.zero_grad()
                
                """
                # TODO learning rate warmup
                # ------------------------
                # 根据当前阶段（预热或余弦衰减）调整学习率
                if global_step < warmup_steps:
                    # 线性预热
                    lr = self.args.initial_lr + global_step * lr_increment
                else:
                    # 预热后余弦衰减
                    progress = (global_step - warmup_steps) / (total_training_steps - warmup_steps)
                    lr = self.args.min_lr + (peak_lr - self.args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                # 将计算出的学习率应用到优化器中
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                # 记录当前学习率
                track_lrs.append(lr)
                """
                
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

                # backward
                # ------------------------
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    # TODO 在预热阶段后应用梯度裁剪，防止梯度爆炸
                    # if global_step > warmup_steps:
                    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # calculate loss gradient
                    loss.backward()
                    # TODO 在预热阶段后应用梯度裁剪，防止梯度爆炸
                    # if global_step > warmup_steps:
                    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)
                    # update model weights using loss gradients
                    optimizer.step()
                
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
                        logger.info(f'\t\tEpoch {epoch + 1}: Batch {batch + 1} (Step {global_step:06d}): Train loss: {train_loss:.3f}, Val loss {valid_loss:.3f} | Speed: {speed:.4f}s/batch; left time: {left_time:.2f}seconds.')
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
            early_stopping(epoch=epoch + 1, val_loss=valid_loss, model=self.model, model_path=model_checkpoint_path)
            if early_stopping.early_stop:
                logger.info(f"\t\tEpoch {epoch + 1}: Early stopping...")
                break
            # 学习率调整
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
        self.model.load_state_dict(torch.load(model_checkpoint_path, map_location=self.device, weights_only=True)["model"])
        
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
                  start_context: str="Every effort moves you", eos_id: int=50256,
                  temperature: float=0.0, top_k: float=1.0):
        """
        model inference
        """
        logger.info(f"\t\tModel start inference...")
        # load model
        if load:
            logger.info(f"Pretrained model has loaded from: {model_checkpoint_path}")
            model_checkpoint_path = self._get_model_path(setting)
            self.model.load_state_dict(torch.load(model_checkpoint_path, map_location=self.device, weights_only=True)["model"])
        # inference mode
        self.model.eval()
        # context size
        context_length = self.model.module.pos_embed.weight.shape[0] \
            if isinstance(self.model, nn.parallel.DistributedDataParallel) \
            else self.model.pos_embed.weight.shape[0]
        # start context tokenization
        start_context_encoded = text_to_token_ids(start_context, self.tokenizer).to(self.device)
        # generate text
        with torch.no_grad():
            completion_id = generate(
                model = self.model, 
                token_idx = start_context_encoded,
                max_new_tokens = self.args.max_new_tokens,
                context_length = context_length,
                temperature = temperature,
                top_k = top_k,
                eos_id = eos_id,
                use_cache = self.args.use_cache,
            )
            completion = token_ids_to_text(completion_id, self.tokenizer).replace("\n", " ")
            logger.info(f"\t\tEpoch {epoch}: Model inference [start context]: {start_context}, [completion]: {completion}")
        # train mode
        self.model.train()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
