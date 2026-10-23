# -*- coding: utf-8 -*-

# ***************************************************
# * File        : dataset.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-20
# * Version     : 1.0.082002
# * Description : 分布式示例共用的 toy 分类数据集。特征为随机正态向量，
# *               标签由一个固定线性规则 + 少量噪声生成（二分类，可学习），
# *               通过 seed 保证所有 rank 看到完全一致的数据。
# *               提供普通版与 DistributedSampler 版 dataloader 工厂。
# * Link        : https://pytorch.org/docs/stable/data.html
# * Requirement : torch >= 2.5
# ***************************************************

__all__ = [
    "ToyDataset",
    "make_toy_dataset",
    "make_toy_dataloaders",
    "make_dataloader",
]

# python libraries
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# global variable
DEFAULT_NUM_FEATURES = 20
DEFAULT_NUM_CLASSES = 2


class ToyDataset(Dataset):
    """
    (features, label) 对的内存数据集。

    features: FloatTensor[num_samples, num_features]
    labels:   LongTensor[num_samples]，取值 [0, num_classes)
    """

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def make_toy_dataset(
    num_samples: int = 2048,
    num_features: int = DEFAULT_NUM_FEATURES,
    num_classes: int = DEFAULT_NUM_CLASSES,
    seed: int = 0,
    noise: float = 0.1,
    rule_seed: int = 12345,
) -> ToyDataset:
    """
    生成确定的 toy 分类数据集：label = argmax 规则由固定的投影向量给出。
    投影向量只由 rule_seed 决定（与数据 seed 无关），因此 train/test 切分
    共享同一分类规则；数据 seed 只控制样本本身，保证所有 rank 数据一致。
    """
    gen = torch.Generator().manual_seed(seed)
    rule_gen = torch.Generator().manual_seed(rule_seed)
    features = torch.randn(num_samples, num_features, generator=gen)
    weight = torch.randn(num_features, num_classes, generator=rule_gen)
    logits = features @ weight + noise * torch.randn(num_samples, num_classes, generator=gen)
    labels = logits.argmax(dim=1)
    return ToyDataset(features, labels)


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    distributed: bool = False,
    shuffle: bool = True,
    seed: int = 0,
) -> DataLoader:
    """
    dataloader 工厂。distributed=True 时使用 DistributedSampler 切分数据
    （要求进程组已初始化），每个 epoch 前需调用 loader.sampler.set_epoch(epoch)。
    """
    if distributed:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # 由 DistributedSampler 负责打乱
            sampler=DistributedSampler(dataset, shuffle=shuffle, seed=seed),
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def make_toy_dataloaders(
    batch_size: int = 32,
    num_samples: int = 2048,
    num_features: int = DEFAULT_NUM_FEATURES,
    num_classes: int = DEFAULT_NUM_CLASSES,
    distributed: bool = False,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    一步构造 (train_loader, test_loader)。test_loader 始终为非分布式、
    不打乱；每个 rank 持有完整测试集，可用于各 rank 独立评估。
    """
    train_ds = make_toy_dataset(num_samples, num_features, num_classes, seed=seed)
    test_ds = make_toy_dataset(num_samples // 4, num_features, num_classes, seed=seed + 1)
    train_loader = make_dataloader(train_ds, batch_size, distributed=distributed, shuffle=True, seed=seed)
    test_loader = make_dataloader(test_ds, batch_size, distributed=False, shuffle=False)
    return train_loader, test_loader




# 测试代码 main 函数
def main():
    train_loader, test_loader = make_toy_dataloaders(batch_size=32, num_samples=64)
    x, y = next(iter(train_loader))
    print(f"batch features: {tuple(x.shape)}, labels: {tuple(y.shape)}")
    print(f"train batches: {len(train_loader)}, test batches: {len(test_loader)}")

if __name__ == "__main__":
    main()
