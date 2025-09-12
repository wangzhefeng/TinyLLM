#!/bin/bash

# *********************************************
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2021.01.01
# * Version     : 1.0.0
# * Description : description
# * Link        : link
# **********************************************

# ------------------------------------------------------------
# nproc_per_node: 控制启动多少个进程，每个进程负责一张卡, nproc_per_node 应 ≤ CUDA_VISIBLE_DEVICES 中的设备数量
# CUDA_VISIBLE_DEVICES: CUDA_VISIBLE_DEVICES 是一个底层系统环境变量，用于限定当前进程可以使用的 GPU 物理设备，
#                       并允许自定义设备编号映射。在 PyTorch 分布式训练中，它和 LOCAL_RANK + --nproc_per_node 配合，
#                       实现精确的 GPU 资源隔离与分配。
# LOCAL_RANK: 当前进程在本节点内的编号（0 ~ nproc_per_node-1），用于映射到 CUDA_VISIBLE_DEVICES 中的设备, 
#             每个进程使用 LOCAL_RANK 作为索引访问 CUDA_VISIBLE_DEVICES 列表中的设备
# ------------------------------------------------------------

# 设置使用的 GPU 物理编号（例如只用第 2、3、6、7 张卡）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 设置日志名称
export LOG_NAME=exp_test

# tokenizer 模型名称
tk_model_name=exp_test
# LM 模型名称
lm_model_name=exp_test


# python -u ./exp/exp_test.py
python ./exp/exp_test.py
