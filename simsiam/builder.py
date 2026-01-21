# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import copy
from typing import Optional


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True) # num_classesは全結合層の出力の次元

        '''Modifying resnet to the same network as the paper.'''
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.maxpool = nn.Identity()

        '''build a 3-layer projector'''
        # prev_dim = self.encoder.fc.weight.shape[1]
        # self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
        #                                 nn.BatchNorm1d(prev_dim),
        #                                 nn.ReLU(inplace=True), # first layer
        #                                 nn.Linear(prev_dim, prev_dim, bias=False),
        #                                 nn.BatchNorm1d(prev_dim),
        #                                 nn.ReLU(inplace=True), # second layer
        #                                 self.encoder.fc,
        #                                 nn.BatchNorm1d(dim, affine=False)) # output layer
        # self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        '''build a 3-layer projector (paper default)'''
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # second layer
            self.encoder.fc,
            nn.BatchNorm1d(dim, affine=False),  # output layer
        )


        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer

                                        nn.Linear(pred_dim, dim)) # output layer
        

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()
    
class PhiNet(nn.Module):
    """
    PhiNet: SimSiam を拡張し predictor を2段(h,g)にしたモデル。
    図(b) PhiNet architecture に対応。

    - encoder f:   self.encoder （SimSiamと同じ構造: backbone + projector）
    - predictor h: self.predictor_h （SimSiamのpredictor相当）
    - predictor g: self.predictor_g （新規）
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, g_pred_dim=None):
        super().__init__()

        # ---- encoder f（SimSiamと同等の作り）----
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # CIFAR向け調整（builder.pyのSimSiamと同じ方針）
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.maxpool = nn.Identity()

        # projector（SimSiamと同様の3層MLP+BN）
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            self.encoder.fc,
            nn.BatchNorm1d(dim, affine=False)
        )

        # ---- predictor h（SimSiamと同じ2層MLP）----
        self.predictor_h = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )

        # ---- predictor g（新規: hの出力→y）----
        if g_pred_dim is None:
            g_pred_dim = pred_dim
        self.predictor_g = nn.Sequential(
            nn.Linear(dim, g_pred_dim, bias=False),
            nn.BatchNorm1d(g_pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(g_pred_dim, dim),
            nn.Tanh(),
        )

    def forward(self, x1, x2, x0):
        """
        入力:
          x1, x2: 図(b)の x^(1), x^(2)
          x0:     図(b)右枝の入力（image x 由来の第3view）

        出力:
          h1, h2: predictor h の出力（図(b)の h^(1), h^(2)）
          z1_sg, z2_sg: SG-1 用の stop-grad 表現（図(b)の z^(1), z^(2)）
          y1, y2: predictor g の出力（図(b)の y^(1), y^(2)）
          z0_sg: SG-2 用の stop-grad 表現（図(b)の z）
        """
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        # z0 is only used as a stop-grad target, so avoid storing its graph.
        with torch.no_grad():
            z0 = self.encoder(x0)

        h1 = self.predictor_h(z1)
        h2 = self.predictor_h(z2)

        y1 = self.predictor_g(h1)
        y2 = self.predictor_g(h2)

        return h1, h2, z1.detach(), z2.detach(), y1, y2, z0.detach()

class XPhiNet(nn.Module):
    """X-PhiNet: PhiNet + stable encoder f_long (EMA).

    3-view 入力のうち z0 を encoder_long (f_long) から算出し，
    encoder_long は encoder の EMA で更新する．
    """

    def __init__(
        self,
        base_encoder,
        dim: int = 2048,
        pred_dim: int = 512,
        g_pred_dim: Optional[int] = None,
        ema_beta: float = 0.99,
    ):
        super().__init__()

        # ---- online encoder f（PhiNetと同等）----
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.maxpool = nn.Identity()

        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            self.encoder.fc,
            nn.BatchNorm1d(dim, affine=False),
        )

        # ---- predictor h ----
        self.predictor_h = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim),
        )

        # ---- predictor g ----
        if g_pred_dim is None:
            g_pred_dim = pred_dim
        self.predictor_g = nn.Sequential(
            nn.Linear(dim, g_pred_dim, bias=False),
            nn.BatchNorm1d(g_pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(g_pred_dim, dim),
        )

        # ---- stable encoder f_long ----
        self.encoder_long = copy.deepcopy(self.encoder)
        for p in self.encoder_long.parameters():
            p.requires_grad = False

        self.ema_beta = float(ema_beta)

    @torch.no_grad()
    def momentum_update(self, beta: Optional[float] = None) -> None:
        """Update encoder_long with EMA of encoder parameters."""
        if beta is None:
            beta = self.ema_beta
        beta = float(beta)
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_long.parameters()):
            param_k.data.mul_(beta).add_(param_q.data, alpha=1.0 - beta)

    def train(self, mode: bool = True):
        super().train(mode)
        # keep stable encoder always in eval to avoid BN running stats drifting
        self.encoder_long.eval()
        return self
    
    def forward(self, x1, x2, x0):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        
        with torch.no_grad():
            was_training = self.encoder_long.training
            self.encoder_long.eval()
            z0 = self.encoder_long(x0)
            if was_training:
                self.encoder_long.train()

        h1 = self.predictor_h(z1)
        h2 = self.predictor_h(z2)

        y1 = self.predictor_g(h1)
        y2 = self.predictor_g(h2)

        return h1, h2, z1.detach(), z2.detach(), y1, y2, z0.detach()
