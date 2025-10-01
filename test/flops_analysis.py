# -*- coding: utf-8 -*-

# ***************************************************
# * File        : flops_analysis.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-30
# * Version     : 1.0.083021
# * Description : description
# * Link        : https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/02_performance-analysis/flops-analysis.ipynb
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import time
import warnings
warnings.filterwarnings("ignore")

from thop import profile
import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def flops_analysis_fixed_batch_size(cfg, model, batch_size=2, device=None):
    """
    FLOPS Analysis
        - FLOPs (Floating Point Operations Per Second) measure the computational complexity of neural network models by counting the number of floating-point operations executed
        - High FLOPs indicate more intensive computation and energy consumption
    """
    # input tensor
    input_tensor = torch.randint(0, cfg.vocab_size, size=(batch_size, 1024)).to(device)
    # model
    model = model.bfloat16()
    model.to(device)
    # MACS = multiply-accumulate operations. MACS are typically counted as two FLOPS (one multiply and one accumulate)
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops = 2 * macs
    logger.info(f"{flops:.1e} FLOPS")

    del model
    torch.cuda.empty_cache()



# Theoretical max flops per second provided by the GPU manufacturer
flops_per_second = {
    # https://www.techpowerup.com/gpu-specs/h100-pcie-80-gb.c3899
    "H100": {
        torch.float32: 51.22e12,  # 51.22 TFLOPs for FP32 on NVIDIA H100
        torch.float16: 204.9e12,  # 204.9 TFLOPs for FP16 on NVIDIA H100
        torch.bfloat16: 204.9e12
    },
    # https://www.techpowerup.com/gpu-specs/l4.c4091
    "L4": {
        torch.float32: 30.29e12,  # 30.29 TFLOPs for FP32 on NVIDIA L4
        torch.float16: 30.29e12,  # 30.29 TFLOPs for FP16 on NVIDIA L4
        torch.bfloat16: 30.29e12
    },
    # https://www.techpowerup.com/gpu-specs/tesla-t4.c3316
    "T4": {
        torch.float32: 8.1e12,  # 8.1 TFLOPs for FP32 on NVIDIA T4
        torch.float16: 65.13e12,  # 65.13 TFLOPs for FP16 on NVIDIA T4
        torch.bfloat16: 65.13e12
    },
    # https://www.techpowerup.com/gpu-specs/a10g.c3798
    "A10G": {
        torch.float32: 31.52e12,  # 31.52 TFLOPs for FP32 on NVIDIA A10G
        torch.float16: 31.52e12,  # 31.52 TFLOPs for FP16 on NVIDIA A10G
        torch.bfloat16: 31.52e12
    },
    # https://www.techpowerup.com/gpu-specs/a100-pcie-40-gb.c3623
    "A100": {
        torch.float32: 19.49e12,  # 19.49 TFLOPs for FP32 on NVIDIA A100
        torch.float16: 77.97e12,  # 77.97 TFLOPs for FP16 on NVIDIA A100
        torch.bfloat16: 77.97e12
    },
    # https://www.techpowerup.com/gpu-specs/geforce-rtx-3080.c3621
    "RTX_3080": {
        torch.float32: 29.77e12,  # 29.77 TFLOPs for FP32 on NVIDIA RTX 3080
        torch.float16: 29.77e12,  # 29.77 TFLOPs for FP16 on NVIDIA RTX 3080
        torch.bfloat16: 29.77e12
    },
    # https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622
    "RTX_3090": {
        torch.float32: 35.58e12,  # 35.58 TFLOPs for FP32 on NVIDIA RTX 3090
        torch.float16: 35.58e12,  # 35.58 TFLOPs for FP16 on NVIDIA RTX 3090
        torch.bfloat16: 35.58e12
    }
}


def get_gpu_model(flops_per_second_dict):
    device_name = torch.cuda.get_device_name(0)
    for model in flops_per_second_dict.keys():
        if model in device_name:
            return model
    return "Unknown"  # Default if no matching model is found


def flops_analysis_automatic_batch_size(cfg, model_cls, device, forward_only=True):
    # batch size
    min_batch_size = 1
    max_batch_size = None
    max_possible_batch_size = 4096
    # gpu model
    gpu_model = get_gpu_model(flops_per_second)
    if gpu_model != "Unknown":
        while min_batch_size <= max_possible_batch_size:
            batch_size = (min_batch_size + max_possible_batch_size) // 2
            try:
                # input tensor
                input_tensor = torch.randint(0, cfg.vocab_size, (batch_size, cfg.context_length), device=device)
                # model
                model = model_cls(cfg).bfloat16().to(device)
                
                # Forward and backward pass
                if not forward_only:
                    model.train()

                    # Start timing
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    output = model(input_tensor)
                    loss = output.sum()  # compute a dummy loss
                    loss.backward()
                    
                    # End timing
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    total_time_seconds = end_time - start_time
                
                # MACS = multiply-accumulate operations. MACS are typically counted as two FLOPS (one multiply and one accumulate)
                # Calculate FLOPs for forward pass
                macs, params = profile(model, inputs=(input_tensor,), verbose=False)
                flops_forward = 2 * macs  # Assuming one MAC equals two FLOPs
                logger.info(f"\tBatch size {batch_size}: {flops_forward:.1e} FLOPS")

                if not forward_only:
                    # Estimate FLOPs for backward pass (typically 2x forward FLOPs)
                    flops_backward = 2 * flops_forward

                    # Total FLOPs for forward + backward passes
                    total_flops = flops_forward + flops_backward  # Or total_flops = flops_forward * 3

                    data_type = next(model.parameters()).dtype
                    max_flops_per_second = flops_per_second[gpu_model].get(data_type, 0)

                    # Compute tokens per second
                    tokens_processed = batch_size * cfg.context_length
                    tokens_per_second = tokens_processed / total_time_seconds

                    # Compute FLOPs per token
                    flops_per_token = total_flops / tokens_processed

                    # Compute theoretical max tokens per second
                    if flops_per_token > 0:
                        theoretical_max_tokens_per_second = max_flops_per_second / flops_per_token
                    else:
                        theoretical_max_tokens_per_second = 0  # Avoid division by zero

                    # Compute MFU
                    if theoretical_max_tokens_per_second > 0:
                        mfu = tokens_per_second / theoretical_max_tokens_per_second
                    else:
                        mfu = 0  # Avoid division by zero

                    logger.info(f"\tBatch size {batch_size}: Tokens/sec: {tokens_per_second:.2f}, MFU: {mfu:.4f}")

                # If successful, try a larger batch size
                min_batch_size = batch_size + 1
                max_batch_size = batch_size
                # Clean up
                if forward_only:
                    del model, input_tensor
                else:
                    del model, input_tensor, output, loss
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Try smaller batch size
                    max_possible_batch_size = batch_size - 1
                    # Clean up
                    try:
                        del model, input_tensor
                        torch.cuda.empty_cache()
                    except NameError:
                        pass
                else:
                    raise e
    else:
        print("Unknown GPU model. Please update the flops_per_second dictionary with your GPU information.")




# 测试代码 main 函数
def main():
    from models.gpt2_124M import Model
    from models.gpt2_model_cfg.model_cfgs import (
        device, 
        GPT2_124M_CONFIG, 
        GPT2_MEDIUM_CONFIG, 
        GPT2_LARGE_CONFIG, 
        GPT2_XL_CONFIG,
    )
    
    # params
    model_configs = {
        "gpt-small (124M)": GPT2_124M_CONFIG,
        "gpt-medium (355M)": GPT2_MEDIUM_CONFIG,
        "gpt-large (774M)": GPT2_LARGE_CONFIG,
        "gpt-xl (1558M)": GPT2_XL_CONFIG,
    }

    # ------------------------------
    # simple benchmark with fixed batch size(forward pass only)
    # ------------------------------ 
    # analysis
    for size, cfg in model_configs.items():
        flops_analysis_fixed_batch_size(cfg, model=Model(cfg), batch_size=2, device=device)
    
    # ------------------------------
    # simple benchmark with automatic batch size finding(forward pass only)
    # ------------------------------
    # analysis
    for size, cfg in model_configs.items():
        logger.info(f"Processing {size}...")
        flops_analysis_automatic_batch_size(
            cfg=cfg,
            model=Model, 
            device=device,
            forward_only=True,
        )
    
    # ------------------------------
    # Benchmark with automatic batch size finding and Model FLOP Utilization (MFU)(forward and forward and backward pass)
    # ------------------------------
    # analysis
    for size, cfg in model_configs.items():
        logger.info(f"Processing {size}...")
        flops_analysis_automatic_batch_size(
            cfg=cfg,
            model=Model, 
            device=device,
            forward_only=False,
        )

if __name__ == "__main__":
    main()
