import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from torchvision.utils import make_grid, save_image
from spikingjelly.activation_based import base, neuron, functional, surrogate, layer
from spikingjelly.datasets.n_mnist import NMNIST
import torchvision.transforms as transforms
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
import sys
from SpikeSlicer_utils.neuron import IFNode, LIFNode


def reset_net(net):
    functional.reset_net(net)
    net.I = []


class BatchNorm2d(nn.BatchNorm2d, base.StepModule):
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        """
        * :ref:`API in English <BatchNorm2d-en>`

        .. _BatchNorm2d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.BatchNorm2d`

        * :ref:`中文 API <BatchNorm2d-cn>`

        .. _BatchNorm2d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.BatchNorm2d` for other parameters' API
        """
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: torch.Tensor):
        if self.step_mode == 's':
            return super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 5:
                raise ValueError(f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
            T, N, C, H, W = x.shape
            y = []
            for t in range(T):
                y.append(super(nn.BatchNorm2d, self).forward(x[t]))
            y = torch.stack(y, dim=0)
            return y

class MembraneLoss:
    def __init__(self, mse=torch.nn.MSELoss(), v_decay=1, i_decay=1, alpha=0.4, alpha_lr=1e-1):
        """
        :param mse: loss function
        :param v_decay: coefficient of v
        :param i_decay: coefficient of I
        :param alpha: weight of upper bound
        """
        self.mse = mse
        self.v_decay = v_decay
        self.i_decay = i_decay
        self.alpha = torch.tensor(alpha, requires_grad=True)
        self.alpha_optim = torch.optim.SGD([self.alpha], lr=alpha_lr)

    def __call__(self, mem_seq, I, batch_idx, spike_idx, max_idx, Vth):
        """
        :param mem_seq: membrane potential sequence (with gradient)
        :param I: current sequence (with gradient)
        :param batch_idx: global index of batch
        :param spike_idx: global index of spike
        :param Vth: threshold of membrane potential
        """
        ## monotonic-assuming loss
        spike_mem = mem_seq[spike_idx][batch_idx]
        target = (Vth * (spike_idx + 1) / (max_idx + 1)).to(spike_mem.device)
        mono_loss = self.mse(spike_mem, target)

        ## membrane loss
        if max_idx > spike_idx:
            pre_mem_v = mem_seq[spike_idx][batch_idx]
            added_I = 0
            for i in range(spike_idx + 1, max_idx + 1):
                pre_mem_v = pre_mem_v * self.v_decay + self.i_decay * I[i, batch_idx].clamp(0)
                added_I = added_I + I[i, batch_idx].clamp(0).detach()
            mem_v = pre_mem_v
        else:
            mem_v = mem_seq[max_idx][batch_idx]
            added_I = 1
        up_bound_target = (torch.tensor(Vth) * self.v_decay + self.i_decay * I[max_idx, batch_idx].detach().clamp(0)).clamp(min=Vth)
        low_bound_target = torch.tensor(Vth)
        target = self.alpha * up_bound_target + (1 - self.alpha) * low_bound_target
        mem_loss = self.mse(mem_v, target)
        if added_I == 0:
            mem_loss = mem_loss + mono_loss
        ## negative I loss
        neg_I = I.clamp(max=0)
        I_loss = self.mse(neg_I, torch.zeros_like(neg_I))
        return mem_loss, I_loss


class Weighted_MembraneLoss:
    def __init__(self, mse=torch.nn.MSELoss(), tau=1, alpha=0.4, alpha_lr=1e-1):
        """
        :param mse: loss function
        :param tau: time decreasing constant of the neuron model
        :param alpha: weight of upper bound
        """
        self.mse = mse
        self.tau = tau
        self.alpha = torch.tensor(alpha, requires_grad=True)
        self.alpha_optim = torch.optim.SGD([self.alpha], lr=alpha_lr)

    def __call__(self, mem_seq, I, batch_idx, spike_idx, weights, Vth):
        """
        :param mem_seq: membrane potential sequence (with gradient)
        :param I: current sequence (with gradient)
        :param batch_idx: global index of batch
        :param spike_idx: global index of spike
        :param weights: weight for each extend step
        :param Vth: threshold of membrane potential
        """
        total_loss = 0
        total_I_loss = 0
        for extend_idx in range(weights.shape[0]):
            weight = weights[extend_idx]
            ## monotonic-assuming loss
            spike_mem = mem_seq[spike_idx][batch_idx]
            target = (Vth * (spike_idx + 1) / (extend_idx + 1)).to(spike_mem.device)
            mono_loss = self.mse(spike_mem, target)

            ## membrane loss
            if extend_idx > spike_idx:
                pre_mem_v = mem_seq[spike_idx][batch_idx]
                added_I = 0
                for i in range(spike_idx + 1, extend_idx + 1):
                    pre_mem_v = pre_mem_v * self.tau + I[i, batch_idx].clamp(0)
                    added_I = added_I + I[i, batch_idx].clamp(0).detach()
                mem_v = pre_mem_v
            else:
                mem_v = mem_seq[extend_idx][batch_idx]
                added_I = 1
            up_bound_target = torch.tensor(Vth) * self.tau + I[extend_idx, batch_idx].detach().clamp(0)
            low_bound_target = torch.tensor(Vth)
            target = self.alpha * up_bound_target + (1 - self.alpha) * low_bound_target
            mem_loss = self.mse(mem_v, target)
            if added_I == 0:
                mem_loss = mem_loss + mono_loss
            ## negative I loss
            neg_I = I.clamp(max=0)
            I_loss = self.mse(neg_I, torch.zeros_like(neg_I))
            total_loss = total_loss + mem_loss * weight
            total_I_loss = total_I_loss + I_loss * weight
        return total_loss, total_I_loss

class SpikeSlicerNet_B_LIF(nn.Module):
    def __init__(self, resolution, output_num=1):
        super(SpikeSlicerNet_B_LIF, self).__init__()
        self.I = []
        H, W = resolution
        flatten_size = 64 * int(H / 8) * int(W / 8)
        self.encoder = nn.Sequential(
            layer.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            layer.GroupNorm(16, 16),
            neuron.LIFNode(tau=10.),
            layer.AvgPool2d(kernel_size=2),  ## N/2, N/2

            layer.Conv2d(16, 32, kernel_size=3, padding=1),
            layer.GroupNorm(32, 32),
            neuron.LIFNode(tau=10.),
            layer.AvgPool2d(kernel_size=2),  ## N/4, N/4

            layer.Conv2d(32, 64, kernel_size=3, padding=1),
            layer.GroupNorm(64, 64),
            layer.AvgPool2d(kernel_size=2),  ## N/8, N/8
            neuron.LIFNode(tau=10.),

            layer.Flatten(),
            layer.Linear(flatten_size, 512, bias=False),
            neuron.LIFNode(tau=10.)
        )
        self.linear = layer.Linear(512, output_num, bias=False)
        # self.node = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), v_threshold=0.5)
        self.node = LIFNode(tau=10.)

    def forward(self, x):
        feature_map = self.encoder(x)
        I = self.linear(feature_map)
        if self.node.step_mode == 'm':
            self.I = I
        else:
            self.I.append(I)
        x = self.node(I)
        return x

    def adjust_batch(self, idx):
        for single_layer in self.encoder:
            if isinstance(single_layer, neuron.LIFNode) or isinstance(single_layer, neuron.IFNode) or isinstance(
                    single_layer, neuron.ParametricLIFNode):
                if not isinstance(single_layer.v, float):
                    single_layer.v = single_layer.v[idx]

        if isinstance(self.node, LIFNode) or isinstance(self.node, IFNode) or isinstance(self.node,
                                                                                         neuron.ParametricLIFNode):
            if not isinstance(self.node.v, float):
                self.node.v = self.node.v[idx]



class SpikeSlicerNet_S_IF(nn.Module):
    def __init__(self, resolution, output_num=1, pool_size=4):
        super(SpikeSlicerNet_S_IF, self).__init__()
        self.I = []
        H, W = resolution
        pool_H = round(H / W * pool_size)
        pool_W = pool_size
        # flatten_size = 64 * int(H / 8) * int(W / 8)
        flatten_size = 64 * pool_H * pool_W
        self.encoder = nn.Sequential(
            layer.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            layer.GroupNorm(16, 16),
            neuron.IFNode(),
            layer.AvgPool2d(kernel_size=2),  ## N/2, N/2

            layer.Conv2d(16, 32, kernel_size=3, padding=1),
            layer.GroupNorm(32, 32),
            neuron.IFNode(),
            layer.AvgPool2d(kernel_size=2),  ## N/4, N/4

            layer.Conv2d(32, 64, kernel_size=3, padding=1),
            layer.GroupNorm(64, 64),
            # layer.AvgPool2d(kernel_size=2),  ## N/8, N/8
            neuron.IFNode(),
            # layer.AvgPool2d(kernel_size=2),  ## N/8, N/8
            layer.AdaptiveAvgPool2d((pool_H, pool_W)),
        )
            
        
        self.linear = nn.Sequential(
            layer.Flatten(),
            layer.Linear(flatten_size, 512, bias=False),
            neuron.IFNode(),
            layer.Linear(512, output_num, bias=False))
        # self.node = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), v_threshold=0.5)
        self.node = IFNode()

    def forward(self, x):
        feature_map = self.encoder(x)
        I = self.linear(feature_map)
        if self.node.step_mode == 'm':
            self.I = I
        else:
            self.I.append(I)
        x = self.node(I)
        return x

    def adjust_batch(self, idx):
        for single_layer in self.encoder:
            if isinstance(single_layer, neuron.LIFNode) or isinstance(single_layer, neuron.IFNode) or isinstance(
                    single_layer, neuron.ParametricLIFNode):
                if not isinstance(single_layer.v, float):
                    single_layer.v = single_layer.v[idx]

        if isinstance(self.node, LIFNode) or isinstance(self.node, IFNode) or isinstance(self.node,
                                                                                         neuron.ParametricLIFNode):
            if not isinstance(self.node.v, float):
                self.node.v = self.node.v[idx]

    # def reset(self):
    #     self.I = []

class SpikeSlicerNet_B_IF(nn.Module):
    def __init__(self, resolution, output_num=1, pool_size=4):
        super(SpikeSlicerNet_B_IF, self).__init__()
        self.I = []
        H, W = resolution
        flatten_size = 64 * int(H / 8) * int(W / 8)
        self.encoder = nn.Sequential(
            layer.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            layer.GroupNorm(16, 16),
            neuron.IFNode(),
            layer.AvgPool2d(kernel_size=2),  ## N/2, N/2

            layer.Conv2d(16, 32, kernel_size=3, padding=1),
            layer.GroupNorm(32, 32),
            neuron.IFNode(),
            layer.AvgPool2d(kernel_size=2),  ## N/4, N/4

            layer.Conv2d(32, 64, kernel_size=3, padding=1),
            layer.GroupNorm(64, 64),
            layer.AvgPool2d(kernel_size=2),  ## N/8, N/8
            neuron.IFNode(),
            # layer.AvgPool2d(kernel_size=2),  ## N/8, N/8
            # layer.AdaptiveAvgPool2d((pool_H, pool_W)),
        )
            
        
        self.linear = nn.Sequential(
            layer.Flatten(),
            layer.Linear(flatten_size, 512, bias=False),
            neuron.IFNode(),
            layer.Linear(512, output_num, bias=False))
        # self.node = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), v_threshold=0.5)
        self.node = IFNode()

    def forward(self, x):
        feature_map = self.encoder(x)
        I = self.linear(feature_map)
        if self.node.step_mode == 'm':
            self.I = I
        else:
            self.I.append(I)
        x = self.node(I)
        return x

    def adjust_batch(self, idx):
        for single_layer in self.encoder:
            if isinstance(single_layer, neuron.LIFNode) or isinstance(single_layer, neuron.IFNode) or isinstance(
                    single_layer, neuron.ParametricLIFNode):
                if not isinstance(single_layer.v, float):
                    single_layer.v = single_layer.v[idx]

        if isinstance(self.node, LIFNode) or isinstance(self.node, IFNode) or isinstance(self.node,
                                                                                         neuron.ParametricLIFNode):
            if not isinstance(self.node.v, float):
                self.node.v = self.node.v[idx]

