# -*- coding: utf-8 -*-

# ***************************************************
# * File        : moe.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-17
# * Version     : 0.1.021721
# * Description : https://zhuanlan.zhihu.com/p/701777558
# *               https://oilbeater.com/en/2025/03/29/deepseek-moe/
# *               https://oilbeater.com/2025/03/29/deepseek-moe/?continueFlag=f8a3608be78ba94b8ef26443ea262ec6
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class _Expert(nn.Module):
    
    def __init__(self, cfgs):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfgs.embed_dim, cfgs.d_ff),
            nn.ReLU(),
            nn.Linear(cfgs.d_ff, cfgs.embed_dim),
            nn.Dropout(cfgs.dropout),
        )

    def forward(self, x):
        out = self.layers(x)

        return out


class _NoisyTopkRouter(nn.Module):
    
    def __init__(self, cfgs):
        super().__init__()

        self.top_k = cfgs.top_k
        self.topk_route_linear = nn.Linear(cfgs.embed_dim, cfgs.num_experts)
        # add noise
        self.noise_linear = nn.Linear(cfgs.embed_dim, cfgs.num_experts)
    
    def forward(self, mh_output):
        """
        假设：
        top_k = 2
        mh_output.shape = [batch_size, tokens, embed_dim] = [2, 4, 32]
        linear_layer.shape = [batch_size, tokens, num_experts] = [2, 4, 4]
        """
        # logits
        logits = self.topk_route_linear(mh_output)
        # nose logits
        noise_logits = self.noise_linear(mh_output)
        # add scaled gaussian noise to logits（Noisy Top-k Gating 用标准正态噪声）
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        # 获取前 k 大的值和索引
        top_k_logits, indices = torch.topk(noisy_logits, self.top_k, dim=-1)
        # 创建一个形状和 logits 相同的 '-inf' 矩阵, shape=[2, 4, 4]
        zeros = torch.full_like(noisy_logits, float('-inf'))
        # 按照索引和值填充上述 zeros 矩阵
        sparse_logits = torch.scatter(zeros, -1, indices, top_k_logits)
        # 对 sparse_logits 进行 softmax 操作，未被填充的位置为 0
        router_output = F.softmax(sparse_logits, dim=-1)

        return router_output, indices


class SparseMoE(nn.Module):
    
    def __init__(self, cfgs):
        super().__init__()
        
        self.router = _NoisyTopkRouter(cfgs)
        self.experts = nn.ModuleList([
            _Expert(cfgs)
            for _ in range(cfgs.num_experts)
        ])
        self.top_k = cfgs.top_k
        # 尊重调用方配置；未提供时才使用默认值 1.0（不要回写覆盖 cfgs）。
        # 注意 DotDict 缺键抛 KeyError，不能直接用 getattr 默认值。
        try:
            self.capacity_factor = cfgs["capacity_factor"] if isinstance(cfgs, dict) else cfgs.capacity_factor
        except (AttributeError, KeyError, TypeError):
            self.capacity_factor = 1.0
        self.num_experts = cfgs.num_experts
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # step1: 输入进入 router 得到两个输出
        gating_output, indices = self.router(x)
        # step2: 初始化全零矩阵，后续叠加为最终结果
        final_output = torch.zeros_like(x)
        # step3: 展平，即把 batch 拼接到一起，这里对输入 x 和 router 后的结果都进行了展平
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))
        # 每个 batch 的 token 个数
        tokens_per_batch = batch_size * seq_len * self.top_k
        # 定义专家容量
        expert_capacity = int((tokens_per_batch / self.num_experts) * self.capacity_factor)
        updates = torch.zeros_like(flat_x)
        # step4: 以每个专家为单位进行操作，即把当前专家处理的所有 tokens 都进行加权
        for i, expert in enumerate(self.experts):
            # step4.1: 对当前的专家(比如专家 0)，查看其对所有 tokens 中哪些在前 topk
            expert_mask = (indices == i).any(dim=-1)
            # step4.2: 展平操作
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)
            # 进行容量判断
            limited_indices = selected_indices[:expert_capacity] \
                if selected_indices.numel() > expert_capacity \
                else selected_indices
            if limited_indices.numel() > 0:
                # step4.3: 得到该专家对哪几个 token 起作用后，选取 token 的维度表示
                expert_input = flat_x[limited_indices]
                # step4.4: 将 token 输入 expert 得到输出
                expert_output = expert(expert_input)
                # step4.5: 计算当前专家对于有作用的 token 的权重分数
                gating_scores = flat_gating_output[limited_indices, i].unsqueeze(1)
                # step4.6: 将 expert 输出乘上权重分数 
                weighted_output = expert_output * gating_scores
                updates.index_add_(0, limited_indices, weighted_output)
                # step 4.7: 循环进行结果叠加
                # final_output[expert_mask] += weighted_output.squeeze(1)

        # Reshape updates to match the original dimensions of x
        final_output += updates.view(batch_size, seq_len, -1)

        return final_output


class MoEFeedForward(nn.Module):
    """Qwen3 式 MoE FFN，可扩展 DeepSeekMoE 两个关键设计（2026-10-20）：

    1. **shared expert**：``num_shared_experts >= 1`` 时无条件激活的共享专家，
       输出与 routed experts 相加（DeepSeekMoE §2.1，保证公共知识总被计算）。
    2. **负载均衡 aux loss**：Switch Transformer 式
       ``L_aux = N * Σ_i f_i · P_i``（f_i=专家 i 分到的 token 比例，
       P_i=平均路由概率）；路由完全均衡时最小。
       ``return_aux_loss=True`` 时返回 (out, aux_loss)；
       默认 False 保持既有调用兼容（aux loss 挂在 ``self.last_aux_loss``）。
    """

    def __init__(self, cfg):
        super().__init__()

        self.num_experts_per_tok = cfg.num_experts_per_tok
        self.num_experts = cfg.num_experts
        self.embed_dim = cfg.embed_dim
        self.gate = nn.Linear(cfg.embed_dim, cfg.num_experts, bias=False, dtype=cfg.dtype)

        self.fc1 = nn.ModuleList([
            nn.Linear(cfg.embed_dim, cfg.moe_intermediate_size, bias=False, dtype=cfg.dtype)
            for _ in range(cfg.num_experts)
        ])
        self.fc2 = nn.ModuleList([
            nn.Linear(cfg.embed_dim, cfg.moe_intermediate_size, bias=False, dtype=cfg.dtype)
            for _ in range(cfg.num_experts)
        ])
        self.fc3 = nn.ModuleList([
            nn.Linear(cfg.moe_intermediate_size, cfg.embed_dim, bias=False, dtype=cfg.dtype)
            for _ in range(cfg.num_experts)
        ])

        # DeepSeekMoE：可选 shared expert（总是激活，与 routed 输出相加）
        try:
            num_shared = cfg["num_shared_experts"] if isinstance(cfg, dict) \
                else getattr(cfg, "num_shared_experts", 0)
        except (KeyError, AttributeError, TypeError):
            # DotDict 缺键抛 KeyError，getattr 对 SimpleNamespace 缺属性抛 AttributeError
            num_shared = 0
        self.num_shared_experts = int(num_shared or 0)
        if self.num_shared_experts > 0:
            shared_dim = cfg.moe_intermediate_size
            self.shared_fc1 = nn.ModuleList([
                nn.Linear(cfg.embed_dim, shared_dim, bias=False, dtype=cfg.dtype)
                for _ in range(self.num_shared_experts)
            ])
            self.shared_fc2 = nn.ModuleList([
                nn.Linear(cfg.embed_dim, shared_dim, bias=False, dtype=cfg.dtype)
                for _ in range(self.num_shared_experts)
            ])
            self.shared_fc3 = nn.ModuleList([
                nn.Linear(shared_dim, cfg.embed_dim, bias=False, dtype=cfg.dtype)
                for _ in range(self.num_shared_experts)
            ])
        self.last_aux_loss = None

    @staticmethod
    def load_balancing_loss(topk_probs, topk_indices, num_experts):
        """Switch 式 aux loss：N · Σ_i f_i · P_i。

        Args:
            topk_probs: (N_tokens, top_k) 每 token 的路由概率
            topk_indices: (N_tokens, top_k) 每 token 选中的专家
        Returns:
            标量 loss；路由完全均衡时 ≈ 1.0，越偏斜越大
        """
        n_tokens = topk_probs.shape[0]
        # f_i：专家 i 被选中的 token 比例
        one_hot = torch.zeros(n_tokens, num_experts, device=topk_probs.device)
        one_hot.scatter_(1, topk_indices, 1.0)
        f = one_hot.mean(dim=0)                            # (N,)
        # P_i：专家 i 的平均路由概率（softmax 全量分布的期望近似）
        # topk 概率未归一到全 vocab，这里用 one-hot 加权 topk_probs 的均值
        prob_mass = torch.zeros(num_experts, device=topk_probs.device)
        prob_mass.index_add_(0, topk_indices.reshape(-1), topk_probs.reshape(-1))
        p = prob_mass / n_tokens
        return num_experts * torch.sum(f * p)

    def forward(self, x, return_aux_loss=False):
        scores = self.gate(x)  # (b, seq_len, num_experts)
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)

        # 负载均衡 aux loss（训练时加到总 loss：total + aux_weight * aux）
        aux_loss = self.load_balancing_loss(
            topk_probs.reshape(-1, self.num_experts_per_tok),
            topk_indices.reshape(-1, self.num_experts_per_tok),
            self.num_experts,
        )

        batch, seq_len, _ = x.shape
        x_flat = x.reshape(batch * seq_len, -1)
        out_flat = torch.zeros(batch * seq_len, self.embed_dim, device=x.device, dtype=x.dtype)

        topk_indices_flat = topk_indices.reshape(-1, self.num_experts_per_tok)
        topk_probs_flat = topk_probs.reshape(-1, self.num_experts_per_tok)

        unique_experts = torch.unique(topk_indices_flat)

        for expert_id_tensor in unique_experts:
            expert_id = int(expert_id_tensor.item())
            mask = topk_indices_flat == expert_id
            if not mask.any():
                continue

            token_mask = mask.any(dim=-1)
            selected_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)
            if selected_idx.numel() == 0:
                continue

            expert_input = x_flat.index_select(0, selected_idx)
            hidden = torch.nn.functional.silu(self.fc1[expert_id](expert_input)) * self.fc2[expert_id](expert_input)
            expert_out = self.fc3[expert_id](hidden)

            mask_selected = mask[selected_idx]
            slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)
            selected_probs = torch.gather(topk_probs_flat.index_select(0, selected_idx), dim=-1, index=slot_indices).squeeze(-1)

            out_flat.index_add_(0, selected_idx, expert_out * selected_probs.unsqueeze(-1))

        out = out_flat.reshape(batch, seq_len, self.embed_dim)

        # shared expert：无条件激活，输出直接相加（不经过路由权重）
        for s in range(self.num_shared_experts):
            hidden = torch.nn.functional.silu(self.shared_fc1[s](x)) * self.shared_fc2[s](x)
            out = out + self.shared_fc3[s](hidden)

        self.last_aux_loss = aux_loss
        if return_aux_loss:
            return out, aux_loss
        return out




# 测试代码 main 函数
def main():
    from utils.args_tools import DotDict
    from utils.log_util import logger

    # random seed
    torch.manual_seed(2025)
    
    # input tensor
    input = torch.randn(2, 3, 6)
    logger.info(f"input: \n{input}")

    # config
    cfgs = {
        "embed_dim": 6,
        "d_ff": 24,
        "dropout": 0.1,
        "top_k": 2,
        "num_experts": 4,
        "capacity_factor": 1.0
    }
    cfgs = DotDict(cfgs)

    # layer
    sparse_moe = SparseMoE(cfgs)
    output = sparse_moe(input)
    logger.info(f"output: \n{output}")

if __name__ == "__main__":
    main()
