import os

from omegaconf import DictConfig
import hydra
import torch
from torch.utils.data import random_split
from torch.distributed import init_process_group, destroy_process_group

from char_dataset import CharDataset, DataConfig
from model import GPT, GPTConfig, OptimizerConfig, create_optimizer
from trainer import Trainer, TrainerConfig


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def get_train_objs(gpt_cfg: GPTConfig, opt_cfg: OptimizerConfig, data_cfg: DataConfig):
    # dataset
    dataset = CharDataset(data_cfg)
    # dataset split
    train_len = int(len(dataset) * data_cfg.train_split)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])
    # dataset parameters
    gpt_cfg.vocab_size = dataset.vocab_size
    gpt_cfg.block_size = dataset.block_size
    # model
    model = GPT(gpt_cfg)
    # optimizer
    optimizer = create_optimizer(model, opt_cfg)
    
    return model, optimizer, train_set, test_set


@hydra.main(version_base=None, config_path=".", config_name="gpt2_train_cfg")
def main(cfg: DictConfig):
    ddp_setup()
    
    # configs
    gpt_cfg = GPTConfig(**cfg['gpt_config'])
    opt_cfg = OptimizerConfig(**cfg['optimizer_config'])
    data_cfg = DataConfig(**cfg['data_config'])
    trainer_cfg = TrainerConfig(**cfg['trainer_config'])

    # train objects
    model, optimizer, train_data, test_data = get_train_objs(gpt_cfg, opt_cfg, data_cfg)

    # training
    trainer = Trainer(trainer_cfg, model, optimizer, train_data, test_data)
    trainer.train()

    destroy_process_group()

if __name__ == "__main__":
    main()
