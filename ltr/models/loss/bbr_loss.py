import torch
import torch.nn as nn


class GIoULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, weights=None):
        if pred.dim() == 4:
            pred = pred.unsqueeze(0)

        pred = pred.permute(0, 1, 3, 4, 2)
        raw_shape = pred.shape[:-1]
        pred = pred.reshape(-1, 4) # nf x ns x x 4 x h x w
        target = target.permute(0, 1, 3, 4, 2)
        target = target.reshape(-1, 4) #nf x ns x 4 x h x w

        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(
            pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_union = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect + 1e-7
        ious = (area_intersect) / (area_union)
        gious = ious - (ac_union - area_union) / ac_union

        losses = 1 - gious

        if weights is not None and weights.sum() > 0:
            if len(weights.shape) == 4:
                weights = weights.unsqueeze(2)
            weights = weights.permute(0, 1, 3, 4, 2).reshape(-1) # nf x ns x x 1 x h x w
            if self.reduction == 'mean':
                loss_mean = losses[weights>0].mean()
            elif self.reduction == 'none':
                losses[weights <= 0] = 0
                loss_mean = losses.view(raw_shape).mean((0, 2, 3))
            ious = ious[weights>0]
        else:
            if self.reduction == 'mean':
                loss_mean = losses.mean()
            elif self.reduction == 'none':
                loss_mean = losses.view(raw_shape).mean((0, 2, 3))

        return loss_mean, ious
