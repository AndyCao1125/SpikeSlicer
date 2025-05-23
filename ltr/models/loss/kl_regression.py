import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class KLRegression(nn.Module):
    """KL-divergence loss for probabilistic regression.
    It is computed using Monte Carlo (MC) samples from an arbitrary distribution."""

    def __init__(self, eps=0.0, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, scores, sample_density, gt_density, mc_dim=-1):
        """Args:
            scores: predicted score values
            sample_density: probability density of the sample distribution
            gt_density: probability density of the ground truth distribution
            mc_dim: dimension of the MC samples"""

        exp_val = scores - torch.log(sample_density + self.eps)

        L = torch.logsumexp(exp_val, dim=mc_dim) - math.log(scores.shape[mc_dim]) - \
            torch.mean(scores * (gt_density / (sample_density + self.eps)), dim=mc_dim)
        if self.reduction == 'none':
            return L.view(-1, 3).mean(1)
        else:
            return L.mean()


class MLRegression(nn.Module):
    """Maximum likelihood loss for probabilistic regression.
    It is computed using Monte Carlo (MC) samples from an arbitrary distribution."""

    def __init__(self, eps=0.0):
        super().__init__()
        self.eps = eps

    def forward(self, scores, sample_density, gt_density=None, mc_dim=-1):
        """Args:
            scores: predicted score values. First sample must be ground-truth
            sample_density: probability density of the sample distribution
            gt_density: not used
            mc_dim: dimension of the MC samples. Only mc_dim=1 supported"""

        assert mc_dim == 1
        assert (sample_density[:,0,...] == -1).all()

        exp_val = scores[:, 1:, ...] - torch.log(sample_density[:, 1:, ...] + self.eps)

        L = torch.logsumexp(exp_val, dim=mc_dim) - math.log(scores.shape[mc_dim] - 1) - scores[:, 0, ...]
        loss = L.mean()
        return loss


class KLRegressionGrid(nn.Module):
    """KL-divergence loss for probabilistic regression.
    It is computed using the grid integration strategy."""

    def forward(self, scores, gt_density, grid_dim=-1, grid_scale=1.0, reduction='mean'):
        """Args:
            scores: predicted score values
            gt_density: probability density of the ground truth distribution
            grid_dim: dimension(s) of the grid
            grid_scale: area of one grid cell"""
        if reduction == 'none':
            score_corr = grid_scale * torch.sum(scores * gt_density.unsqueeze(1), dim=grid_dim)
        else:
            score_corr = grid_scale * torch.sum(scores * gt_density, dim=grid_dim)

        L = torch.logsumexp(scores, dim=grid_dim) + math.log(grid_scale) - score_corr

        if reduction == 'none':
            return L.mean(1)
        else:
            return L.mean()
