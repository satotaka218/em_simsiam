# -*- coding: utf-8 -*-
#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# tensor : C, W, H
# mini-batch : B, C, W, H

import argparse
import builtins
from email.policy import default
import math
import os
from statistics import variance
import sys
import random
import shutil
import time
import warnings
from cv2 import transform
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
import torchvision.datasets as datasets
import torchvision.models as models

import torchvision
from torch.utils.tensorboard import SummaryWriter

import simsiam.loader
from simsiam.loader import Gaussian_Noise
import simsiam.builder
from simsiam.validation import KNNValidation

from utils.utils import ProgressMeter, AverageMeter, adjust_learning_rate
from utils.dataset_Simsiam import SimSiamDataset
from utils.dataset_Simsiam import Random_Rotation


from tqdm.contrib import tenumerate
import matplotlib.pyplot as plt

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


# ------------------------------------------------ #
#   setting dict
# ------------------------------------------------ #

normalization_parameter_dict = {'ImageNet': [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]], 
                                'CIFAR': [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]}

# ==================================================================================================================
#
#   Main method
#
# ==================================================================================================================

def main() :

    # ------------------------------------------------ #
    #   setting
    # ------------------------------------------------ #

    epochs = 800 #epoch数ここ変える！！！
    micro_batch_size = 1024
    accum_steps = 1
    batch_size = micro_batch_size
    # default_lr = 0.1
    # init_lr = default_lr * batch_size/256
    # init_lr = 0.03
    init_lr = 0.03 * (batch_size * accum_steps) / 256
    weight_decay = 1e-4
    momentum = 0.9
    basemodel = 'resnet18'
    dim = 2048
    pred_dim = 512
    num_workers = 8
    print_freq = 50
    train_warmup_epochs = 10
    save_checkpoint_freq = 100
    # device = 0 if torch.cuda.is_available() else 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    dataset_name = "stl10" #デーセセットの指定！！
    
    dataset_label = "STL10" if dataset_name == "stl10" else "CIFAR10"
    # dataset_name = "cifar10"
    normalization_parameter = normalization_parameter_dict['ImageNet'] if dataset_name == "stl10" else normalization_parameter_dict['CIFAR']
    # torch.cuda.set_device(device)
    fp_16 = True
    scalar = torch.cuda.amp.GradScaler() # setting for FP16
    # ------------------------------------------------ #
    #   choose training method (interactive)
    # ------------------------------------------------ #
    # NOTE: outputs are separated per run to allow parallel executions.
    while True:
        _in = input("Select model [phinet/simsiam] (default: phinet): ").strip().lower()
        if _in == "":
            method = "phinet"
            model_name = "PhiNet"
            break
        if _in in ("phinet", "phi", "p", "PhiNet", "Phinet"):
            method = "phinet"
            model_name = "PhiNet"
            break
        if _in in ("simsiam", "sim", "SimSiam", "Simsiam", "s"):
            method = "simsiam"
            model_name = "SimSiam"
            break
        print("Invalid input. Please type 'phinet' or 'simsiam'.")

    seed = 345 if method == "phinet" else 234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True

    lambda_sim2 = 1.0      # Sim-1+Sim-2
    sim2_loss_type = "mse" # "mse" or "cos"
    ema_beta = 0.99  # X-PhiNet: encoder_long EMA係数

    # ------------------------------------------------ #
    #   output directories (separated per run)
    # ------------------------------------------------ #
    os.makedirs("data", exist_ok=True)
    run_dir = os.path.join("result_figure", "{}_{}".format(model_name, epochs))
    log_dir  = os.path.join("log", "{}_{}".format(model_name, epochs))
    ckpt_dir = os.path.join("checkpoint", "{}_{}".format(model_name, epochs))

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    effective_batch_size = micro_batch_size * accum_steps
    print("[INFO] method =", method, "| run_dir =", run_dir)
    print("[INFO] dataset =", dataset_name,
          "| pretrain_split = unlabeled" if dataset_name == "stl10" else "| pretrain_split = train",
          "| knn_memory_split = train",
          "| knn_eval_split = test")
    print("[INFO] micro_batch_size =", micro_batch_size, "| accum_steps =", accum_steps,
          "| effective_batch_size =", effective_batch_size)


    # ------------------------------------------------ #
    #   define model, criterion, optimizer
    # ------------------------------------------------ #

    if method == "phinet":
        model = simsiam.builder.PhiNet(models.__dict__[basemodel], dim, pred_dim)
    # elif method == "xphinet":
    #     model = simsiam.builder.XPhiNet(models.__dict__[basemodel], dim, pred_dim, ema_beta=ema_beta)
    else:
        model = simsiam.builder.SimSiam(models.__dict__[basemodel], dim, pred_dim)

    criterion = nn.CosineSimilarity(dim = 1)
    optimizer = torch.optim.SGD(model.parameters(), lr = init_lr, momentum=momentum, weight_decay=weight_decay)

    # model.cuda(device=device)
    # criterion.cuda(device = device)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    criterion = criterion.to(device)

    # ------------------------------------------------ #
    #   data loading code
    # ------------------------------------------------ #

    # input_size = 96 if dataset_name == "stl10" else 32
    input_size = 32
    '''data augmentation method'''
    augumentation = [
        transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([Gaussian_Noise(sigma=700)], 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean = normalization_parameter[0],
                             std = normalization_parameter[1])
    ]


    '''training dataset'''
    if method in ("phinet", "xphinet"):
        strong_transform = transforms.Compose(augumentation)

        third_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=normalization_parameter[0],
                std=normalization_parameter[1]
            ),
        ])

        if dataset_name == "stl10":
            train_set = datasets.STL10(
                root="./data",
                split="unlabeled",
                download=True,
                transform=simsiam.loader.ThreeCropsTransform(
                    strong_transform,
                    third_transform=third_transform,
                ),
            )
        else:
            train_set = datasets.CIFAR10(
                root="./data",
                train=True,
                download=True,
                transform=simsiam.loader.ThreeCropsTransform(
                    strong_transform,
                    third_transform=third_transform
                )
            )
    else:
        if dataset_name == "stl10":
            train_set = datasets.STL10(
                root="./data",
                split="unlabeled",
                download=True,
                transform=simsiam.loader.TwoCropsTransform(transforms.Compose(augumentation)),
            )
        else:
            train_set = datasets.CIFAR10(
                root="./data",
                train=True,
                download=True,
                transform=simsiam.loader.TwoCropsTransform(transforms.Compose(augumentation))
            )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # ------------------------------------------------ #
    #   trainig code
    # ------------------------------------------------ #

    loss_total_per_epoch = []
    loss_sim1_per_epoch = []
    loss_sim2_per_epoch = []
    kNN_acc_log = []
    writer = SummaryWriter(log_dir=log_dir)
    # --- kNN評価に使うencoderをモードで切り替える ---
    _m = model.module if hasattr(model, "module") else model  # DP/DDP対策

    if method == "xphinet":
        knn_encoder = _m.encoder_long   # X-PhiNet: stable encoderで評価
    else:
        knn_encoder = _m.encoder        # SimSiam / PhiNet: online encoderで評価

    validation = KNNValidation(
        data_root="./data",
        batch_size=128,
        num_workers=2,
        model=knn_encoder,
        K=200,
        num_classes=10,
        dataset_name=dataset_name,
        input_size=input_size,
        normalization_parameter=normalization_parameter,
    )

    for _, epoch in tenumerate(range(epochs)) :
        current_lr = adjust_learning_rate(optimizer, init_lr, epoch, epochs, train_warmup_epochs)
        if fp_16 :
            loss_total, loss_sim1, loss_sim2 = train(
                train_loader=train_loader, device=device, model=model, criterion=criterion,
                optimizer=optimizer, epoch=epoch,
                method=method, lambda_sim2=lambda_sim2, sim2_loss_type=sim2_loss_type,
                ema_beta=ema_beta,
                scalar=scalar, print_freq=print_freq, accum_steps=accum_steps, debug_first_batch=(epoch == 0)
            )
        else :
            loss_total, loss_sim1, loss_sim2 = train(
                train_loader=train_loader, device=device, model=model, criterion=criterion,
                optimizer=optimizer, epoch=epoch,
                method=method, lambda_sim2=lambda_sim2, sim2_loss_type=sim2_loss_type,
                ema_beta=ema_beta,
                scalar=None, print_freq=print_freq, accum_steps=accum_steps, debug_first_batch=(epoch == 0)
            )
        
        loss_total_per_epoch.append(loss_total)
        loss_sim1_per_epoch.append(loss_sim1)
        loss_sim2_per_epoch.append(loss_sim2)

        eval_warmup_epochs = 50
        eval_every = 5
        run_eval = (epoch < eval_warmup_epochs) or ((epoch + 1) % eval_every == 0) or (epoch + 1 == epochs)
        if run_eval:
            print("Validating...")
            val_top1_acc = validation.eval()
            print(f"[VAL] epoch={epoch} kNN_top1={val_top1_acc*100:.2f}%")
            kNN_acc_log.append(val_top1_acc * 100)
            tensorbord_visualizer(writer,
                          loss_total, loss_sim1, loss_sim2,
                          current_lr, val_top1_acc * 100, epoch)
        else:
            kNN_acc_log.append(np.nan)
            tensorbord_visualizer(writer,
                          loss_total, loss_sim1, loss_sim2,
                          current_lr, None, epoch)
        # --- checkpoint save policy ---
        # extra_ckpt_epochs = {1, 2, 5, 10} if method == "xphinet" else set()
        extra_ckpt_epochs = set()

        save_now = (
            ((epoch + 1) % save_checkpoint_freq == 0)
            or ((epoch + 1) == epochs and epochs < save_checkpoint_freq)
            or ((epoch + 1) in extra_ckpt_epochs)   # debug for EMA check
        )

        if save_now:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint_{:04d}.pth.tar".format(epoch + 1))
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": basemodel,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                filename=ckpt_path,
            )

    '''visualize traing process'''
    fig = plt.figure()
    xs = [i for i in range(epochs)]
    plt.plot(xs, loss_total_per_epoch, label="loss_total")
    plt.plot(xs, loss_sim1_per_epoch, label="loss_sim1")
    plt.plot(xs, loss_sim2_per_epoch, label="loss_sim2")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(run_dir, f"{dataset_label}_resnet18_loss.png"))

    fig = plt.figure()
    plt.plot([i for i in range(epochs)], kNN_acc_log, label="kNN acc")
    plt.xlabel('Epochs')
    plt.ylabel('k-nearest neigihbor accuracy')
    plt.yticks(np.arange(0, 100, 10))
    plt.ylim((0, 100))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(run_dir, f"{dataset_label}_resnet18_kNN.png"))




# ==================================================================================================================
#
#   Method for learning rate schedular
#
# ==================================================================================================================

def adjust_learning_rate(optimizer, init_lr, epoch, epochs, warmup_epochs):
    """Decay the learning rate based on schedule"""
    if epoch < warmup_epochs:
        cur_lr = init_lr * float(epoch + 1) / float(warmup_epochs)
    else:
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
    
    return cur_lr 

# ==================================================================================================================
#
#   Training Method
#
# ==================================================================================================================

def train(train_loader, device, model, criterion, optimizer, epoch,
          method="simsiam", lambda_sim2=0.0, sim2_loss_type="mse",
          ema_beta=0.99,scalar=None, print_freq=50, accum_steps=1, debug_first_batch=False):
    """ visualize progress """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_total = AverageMeter('Loss', ':.4f')
    losses_sim1 = AverageMeter('Sim1', ':.4f')
    losses_sim2 = AverageMeter('Sim2', ':.4f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_total, losses_sim1, losses_sim2],
        prefix="Epoch: [{}]".format(epoch)
    )
    model.train()

    end = time.time()
    num_batches = len(train_loader)
    optimizer.zero_grad()
    for i, (images, _) in tenumerate(train_loader):
        if debug_first_batch and i == 0:
            print("[DEBUG] len(images) =", len(images))
            for k in range(len(images)):
                print(f"[DEBUG] images[{k}] shape =", tuple(images[k].shape), "dtype =", images[k].dtype, "device =", images[k].device)

        data_time.update(time.time() - end)

        images[0] = images[0].cuda(device, non_blocking=True)
        images[1] = images[1].cuda(device, non_blocking=True)

        if method in ("phinet","xphinet"):
            images[2] = images[2].cuda(device, non_blocking=True)

        # compute output and loss
        if scalar != None :
            '''FP16'''
            with torch.cuda.amp.autocast():
                if method in ("phinet","xphinet"):
                    h1, h2, z1, z2, y1, y2, z0 = model(x1=images[0], x2=images[1], x0=images[2])
                    if epoch == 0 and i == 0:
                        print("[DEBUG] method=phinet or xphinet",
                              "x1", tuple(images[0].shape),
                              "x2", tuple(images[1].shape),
                              "x0", tuple(images[2].shape))
                        diff = (images[0].float() - images[2].float()).abs().mean().item()
                        print("[DEBUG] mean|x1-x0| =", diff)
                        print("[DEBUG] h1", tuple(h1.shape),
                              "z2", tuple(z2.shape),
                              "y1", tuple(y1.shape),
                              "z0", tuple(z0.shape))
                        print("[DEBUG] dtype:", "images[0] =", images[0].dtype, "h1 =", h1.dtype, "z2 =", z2.dtype)

                    # Sim-1（図(b)のSim-1 + SG-1）
                    loss_sim1 = -(criterion(h1, z2).mean() + criterion(h2, z1).mean()) * 0.5

                    # Step3: Sim-2 は計算しても lambda_sim2=0.0 なので損失に寄与しない
                    if sim2_loss_type == "mse":
                        loss_sim2 = 0.5 * (F.mse_loss(y1, z0) + F.mse_loss(y2, z0))
                    else:
                        loss_sim2 = -(criterion(y1, z0).mean() + criterion(y2, z0).mean()) * 0.5

                    loss = loss_sim1 + lambda_sim2 * loss_sim2
                else:
                    p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
                    loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

            bs = images[0].size(0)
            losses_total.update(loss.item(), bs)

            # sim1/sim2はテンソルなので item() にする
            # SimSiam分岐では loss_sim1/loss_sim2 が無い可能性があるので安全に分ける
            if method in ("phinet","xphinet"):
                losses_sim1.update(loss_sim1.item(), bs)
                losses_sim2.update(loss_sim2.item(), bs)
            else:
                # SimSiamはSim2=0扱い
                losses_sim1.update(loss.item(), bs)
                losses_sim2.update(0.0, bs)

            scaled_loss = loss / accum_steps
            scalar.scale(scaled_loss).backward()
            if ((i + 1) % accum_steps == 0) or (i + 1 == num_batches):
                scalar.step(optimizer)
                scalar.update()
                optimizer.zero_grad()
                m = model.module if hasattr(model, "module") else model
                if hasattr(m, "momentum_update"):
                    m.momentum_update(ema_beta)
        else :
            '''Not FP16'''
            if method in ("phinet","xphinet"):
                h1, h2, z1, z2, y1, y2, z0 = model(x1=images[0], x2=images[1], x0=images[2])
                if epoch == 0 and i == 0:
                    print("[DEBUG] method=phinet or xphinet",
                          "x1", tuple(images[0].shape),
                          "x2", tuple(images[1].shape),
                          "x0", tuple(images[2].shape))
                    diff = (images[0].float() - images[2].float()).abs().mean().item()
                    print("[DEBUG] mean|x1-x0| =", diff)
                    print("[DEBUG] h1", tuple(h1.shape),
                          "z2", tuple(z2.shape),
                          "y1", tuple(y1.shape),
                          "z0", tuple(z0.shape))

                loss_sim1 = -(criterion(h1, z2).mean() + criterion(h2, z1).mean()) * 0.5

                if sim2_loss_type == "mse":
                    loss_sim2 = 0.5 * (F.mse_loss(y1, z0) + F.mse_loss(y2, z0))
                else:
                    loss_sim2 = -(criterion(y1, z0).mean() + criterion(y2, z0).mean()) * 0.5

                loss = loss_sim1 + lambda_sim2 * loss_sim2
            else:
                p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

            bs = images[0].size(0)
            losses_total.update(loss.item(), bs)

            # sim1/sim2はテンソルなので item() にする
            # SimSiam分岐では loss_sim1/loss_sim2 が無い可能性があるので安全に分ける
            if method in ("phinet","xphinet"):
                losses_sim1.update(loss_sim1.item(), bs)
                losses_sim2.update(loss_sim2.item(), bs)
            else:
                # SimSiamはSim2=0扱い
                losses_sim1.update(loss.item(), bs)
                losses_sim2.update(0.0, bs)

            scaled_loss = loss / accum_steps
            scaled_loss.backward()
            if ((i + 1) % accum_steps == 0) or (i + 1 == num_batches):
                optimizer.step()
                optimizer.zero_grad()
                m = model.module if hasattr(model, "module") else model
                if hasattr(m, "momentum_update"):
                    m.momentum_update(ema_beta)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 :
            progress.display(i)

    return losses_total.avg, losses_sim1.avg, losses_sim2.avg

# ==================================================================================================================
#
#   Visualize Live Loss
#
# ==================================================================================================================

def tensorbord_visualizer(writer: SummaryWriter,
                          loss_total, loss_sim1, loss_sim2,
                          learning_rate, knn_acc, epoch):
    # losses
    writer.add_scalar('loss/total', loss_total, epoch)
    writer.add_scalar('loss/sim1', loss_sim1, epoch)
    writer.add_scalar('loss/sim2', loss_sim2, epoch)

    # others
    writer.add_scalar('learning_rate', learning_rate, epoch)
    if knn_acc is not None:
        writer.add_scalar('kNN_accuracy', knn_acc, epoch)


# ==================================================================================================================
#
#   Save method
#
# ==================================================================================================================

def save_checkpoint(state, is_best, filename='./checkpoint/checkpoint.pth.tar'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename), 'model_best.pth.tar'))

if __name__ == "__main__" :
    main()
