# -*- coding: utf-8 -*-

# ***************************************************
# * File        : text_classification.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-01
# * Version     : 1.0.050121
# * Description : description
# * Link        : https://blog.csdn.net/HUSTHY/article/details/115174978
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
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class TextRNN(nn.Module):
    pass


def load_data(args):
    # train data
    train_data = None
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    # dev data
    dev_data = None
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=True)

    return train_loader, dev_loader


def load_model():
    teacher_model = torch.load("saved_models/TextBert_model.bin")
    student_model = TextRNN()

    return teacher_model, student_model


def train(teacher_model, student_model, train_loader, dev_loader, args):
    # model
    teacher_model.to('cuda')
    student_model.to('cuda')
 
    # teacher 网络参数不更新
    for name, params in teacher_model.named_parameters():
        params.requires_grad = False
 
    # 初始学习率，student 网络参数梯度更新
    optimizer_params = {'lr': 1e-3, 'eps': 1e-8}
    optimizer = optim.AdamW(student_model.parameters(), **optimizer_params)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',  # mode max表示当监控量停止上升时，学习率将减小；min表示当监控量停止下降时，学习率将减小；这里监控的是dev_acc因此应该用max
        factor=0.5, 
        min_lr=1e-6, 
        patience=2, 
        verbose=True,
        eps=1e-8
    )
    # teacher 网络输出和 student 网络输出进行损失计算
    # soft_criterion = nn.KLDivLoss()
    soft_criterion = nn.MSELoss()
 
    # student 网络和 label 进行损失计算
    hard_criterion = nn.CrossEntropyLoss()
    
    # alpha(0,1)之间——两个loss的权重系数
    alpha = args.alpha
 
    #T_softmax()的超参[1,10,20]等等值可以多测试几个
    T = 10
 
    early_stop_step = 50000
    last_improve = 0 #记录上次提升的step
    flag = False  # 记录是否很久没有效果提升
    dev_best_acc = 0
    dev_loss = float(50)
    dev_acc = 0
    correct = 0
    total = 0
    global_step = 0
    epochs = args.epochs
 
    for epoch in range(args.epochs):
        for step,batch in enumerate(tqdm(train_loader,desc='Train iteration:')):
            global_step += 1
            optimizer.zero_grad()
            batch = tuple(t.to('cuda') for t in batch)
            input_ids = batch[0]
            input_mask = batch[1]
            label = batch[2]
 
            student_model.train()
            stu_output = student_model(input_ids)
            tea_output = teacher_model(input_ids,input_mask).detach()
 
            #soft_loss————studetn和teach之间做loss，使用的是散度loss
            soft_loss = soft_criterion(F.log_softmax(stu_output/T,dim=1),F.softmax(tea_output/T,dim=1))*T*T
 
            # #soft_loss————studetn和teach之间做loss，使用的是logits的Mse损失
            # soft_loss = soft_criterion(stu_output,tea_output)
 
            #hard_loss————studetn和label之间的loss，交叉熵
            hard_loss = hard_criterion(stu_output,label)
 
            loss = soft_loss*alpha + hard_loss*(1-alpha)
 
            loss.backward()
            optimizer.step()
            total += label.size(0)
            _,predict = torch.max(stu_output,1)
            correct += (predict==label).sum().item()
            train_acc = correct / total
            if (step+1)%1000 == 0:
                print('Train Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,loss:{:.6f}'.format(epoch,epochs,step,len(train_loader),train_acc*100,loss.item()))
            if (step+1)%(len(train_loader)/2)==0:
                dev_acc,dev_loss = dev(student_model, dev_loader)
                dev_loss = dev_loss.item()
                if dev_best_acc < dev_acc:
                    dev_best_acc = dev_acc
                    path = 'savedmodel/TextRnn_distillation_model_mse.bin'
                    torch.save(student_model,path)
                    last_improve = global_step
                print("DEV Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,dev_acc{:.6f} %,best_dev_acc{:.6f} %,train_loss:{:.6f},dev_loss:{:.6f}".format(epoch, epochs, step, len(train_loader), train_acc * 100, dev_acc * 100,dev_best_acc*100,loss.item(),dev_loss))
            if global_step-last_improve >= early_stop_step:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
            writer.add_scalar('textBert_distillation_bilstm/train_loss', loss.item(), global_step=global_step)
            writer.add_scalar('textBert_distillation_bilstm/dev_loss', dev_loss, global_step=global_step)
            writer.add_scalar('textBert_distillation_bilstm/train_acc', train_acc, global_step=global_step)
            writer.add_scalar('textBert_distillation_bilstm/dev_acc', dev_acc, global_step=global_step)
        scheduler.step(dev_best_acc)
        if flag:
            break
    writer.close()



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
