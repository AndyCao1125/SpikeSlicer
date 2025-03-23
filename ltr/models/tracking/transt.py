import torch.nn as nn
from ltr import model_constructor
import numpy as np
import torch
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2,
                       accuracy)

from ltr.models.backbone.transt_backbone import build_backbone
from ltr.models.loss.matcher import build_matcher
from ltr.models.neck.featurefusion_network import build_featurefusion_network


class TransT(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self, backbone, featurefusion_network, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See transt_backbone.py
            featurefusion_network: torch module of the featurefusion_network architecture, a variant of transformer.
                                   See featurefusion_network.py
            num_classes: number of object classes, always 1 for single object tracking
        """
        super().__init__()
        self.featurefusion_network = featurefusion_network
        hidden_dim = featurefusion_network.d_model
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, search, template):
        """ The forward expects a NestedTensor, which consists of:
               - search.tensors: batched images, of shape [batch_size x 3 x H_search x W_search]
               - search.mask: a binary mask of shape [batch_size x H_search x W_search], containing 1 on padded pixels
               - template.tensors: batched images, of shape [batch_size x 3 x H_template x W_template]
               - template.mask: a binary mask of shape [batch_size x H_template x W_template], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits for all feature vectors.
                                Shape= [batch_size x num_vectors x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all feature vectors, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image.

        """
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        if not isinstance(template, NestedTensor):
            template = nested_tensor_from_tensor(template)
        feature_search, pos_search = self.backbone(search)
        feature_template, pos_template = self.backbone(template)
        src_search, mask_search= feature_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search), mask_search, pos_template[-1], pos_search[-1])

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def track(self, search):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor_2(search)
        features_search, pos_search = self.backbone(search)
        feature_template = self.zf
        pos_template = self.pos_template
        src_search, mask_search= features_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search), mask_search, pos_template[-1], pos_search[-1])

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def template(self, z):
        if not isinstance(z, NestedTensor):
            z = nested_tensor_from_tensor_2(z)
        zf, pos_template = self.backbone(z)
        self.zf = zf
        self.pos_template = pos_template

class SetCriterion(nn.Module):
    """ This class computes the loss for TransT.
    The process happens in two steps:
        1) we compute assignment between ground truth box and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, reduction='elementwise_mean'):
        """ Create the criterion.

        Parameters:
            num_classes: number of object categories, always be 1 for single object tracking.
            matcher: module able to compute a matching between target and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.reduction = reduction
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction=self.reduction)
        if self.reduction == 'none':
            loss_ce = loss_ce.mean(1).cpu() ## [batch,]

        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum(1)

        losses = {}
        if self.reduction == 'elementwise_mean':
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        elif self.reduction == 'none':
            batch_loss_bbox = self.recover_batch_loss(loss_bbox, indices)
            losses['loss_bbox'] = batch_loss_bbox
        giou, iou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        giou = torch.diag(giou)
        iou = torch.diag(iou)
        loss_giou = 1 - giou
        iou = iou

        if self.reduction == 'elementwise_mean':
            losses['loss_giou'] = loss_giou.sum() / num_boxes
            losses['iou'] = iou.sum() / num_boxes

        elif self.reduction == 'none':
            batch_iou = self.recover_batch_loss(iou, indices)
            batch_giou_loss = self.recover_batch_loss(loss_giou, indices)
            losses['loss_giou'] = batch_giou_loss
            losses['iou'] = batch_iou 
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def recover_batch_loss(self, iou, indices):
        per_batch_iou_losses = []
        indices_length = list(map(lambda x: len(x[0]), indices))
        current_idx = 0
        for length in indices_length:
            # Get the corresponding iou losses for the current batch
            if length == 0:
                batch_avg_iou_loss = torch.tensor(1, dtype=torch.float64).to(iou.device)
            else:
                batch_iou_losses = iou[current_idx:current_idx + length]
                current_idx += length

                # Compute the average IoU loss for the current batch
                batch_avg_iou_loss = batch_iou_losses.mean()

            per_batch_iou_losses.append(batch_avg_iou_loss)

        return torch.tensor(per_batch_iou_losses)


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the target
        indices = self.matcher(outputs_without_aux, targets)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_pos = sum(len(t[0]) for t in indices)

        num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_boxes_pos = torch.clamp(num_boxes_pos, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes_pos))

        return losses




class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@model_constructor
def transt_resnet50(settings):
    num_classes = 1
    backbone_net = build_backbone(settings, backbone_pretrained=True)
    featurefusion_network = build_featurefusion_network(settings)
    model = TransT(
        backbone_net,
        featurefusion_network,
        num_classes=num_classes
    )
    device = torch.device(settings.device)
    model.to(device)
    # input_tensor = torch.randn(1, 2, 260, 340).cuda()
    # energy_calculator = Energy(model)
    # energy_calculator.register_hooks()
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(trainable_params/1000000.0)
    # from thop import profile
    # flops, params = profile(model, inputs=(input_tensor, input_tensor))
    # print(f"FLOPs: {flops/1000000000.0} G")
    # print(f"Params: {params/1000000.0} M")
    # from fvcore.nn import FlopCountAnalysis
    # flop_count = FlopCountAnalysis(model, (input_tensor, input_tensor))
    # print(f"Total FLOPs: {flop_count.total()}")
    # output = model(input_tensor, input_tensor)
    # print(f"Total FLOPs: {energy_calculator.flops_meter.sum}")
    # print(f"Total FLOPs: {energy_calculator.flops.item()}")
    return model

def transt_loss(settings):
    num_classes = 1
    matcher = build_matcher()
    weight_dict = {'loss_ce': 8.334, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    losses = ['labels', 'boxes']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.0625, losses=losses)
    device = torch.device(settings.device)
    criterion.to(device)
    return criterion

def spikeslicer_transt_loss(settings):
    num_classes = 1
    matcher = build_matcher()
    # weight_dict = {'loss_ce': 8.334, 'loss_bbox': 5}
    weight_dict = {'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    losses = ['labels', 'boxes']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.0625, losses=losses, reduction='none')
    device = torch.device(settings.device)
    criterion.to(device)
    return criterion


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    # def reduce_update(self, tensor, num=1):
    #     link.allreduce(tensor)
    #     self.update(tensor.item(), num=num)

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


class Energy:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.energy = torch.tensor(0.0, device=device)
        self.energy_meter = AverageMeter()
        self.flops = torch.tensor(0.0, device=device)
        self.flops_meter = AverageMeter()
        self.samples_processed = 0
        self.sparsity_meters = {}
        self.first_conv_encountered = False  # 标记是否遇到第一个Conv2d层
        self.current_threshold = None

    def calculate_flops(self, layer, inputs, outputs):
        if isinstance(layer, nn.Conv2d):
            in_channels = layer.in_channels
            out_channels = layer.out_channels
            kernel_size = torch.prod(torch.tensor(layer.kernel_size)).item()
            output_size = torch.prod(torch.tensor(outputs.size()[-2:])).item()
            flops = in_channels * out_channels * kernel_size * output_size
            # print(f'output size:{outputs.size()}')
            # print(in_channels,out_channels,kernel_size,output_size,outputs.size())
        elif isinstance(layer, nn.Conv1d):
            in_channels = layer.in_channels
            out_channels = layer.out_channels
            kernel_size = torch.prod(torch.tensor(layer.kernel_size)).item()
            output_size = torch.prod(torch.tensor(outputs.size()[-1:])).item()
            flops = in_channels * out_channels * kernel_size * output_size
            # print(f'output size:{outputs.size()}')
            # print(in_channels,out_channels,kernel_size,output_size,outputs.size())
        elif isinstance(layer, nn.Linear):
            in_features = layer.in_features
            out_features = layer.out_features
            flops = in_features * out_features
        else:
            flops = 0
        self.flops += flops
        return flops

    def update_sparsity(self, layer_name, density):
        if layer_name not in self.sparsity_meters:
            self.sparsity_meters[layer_name] = AverageMeter()
        self.sparsity_meters[layer_name].update(density)

    def get_artificial_energy(self, layer, inputs, outputs):
        FLOPs = self.calculate_flops(layer, inputs, outputs)
        energy = 4.6 * FLOPs  # 第一层Conv使用的能量计算公式
        self.energy += energy / (10 ** 9)
        self.energy_meter.update(energy / (10 ** 9))
        self.flops_meter.update(FLOPs / (10 ** 9) )

    def get_spike_energy(self, layer, inputs, outputs):
        FLOPs = self.calculate_flops(layer, inputs, outputs)
        density = torch.count_nonzero(inputs[0]).item() / inputs[0].numel()
        # print(inputs[0].shape)
        # print(density)
        # density = torch.sum(inputs[0]).item() / inputs[0].numel()
        # print(density)
        energy = 0.9 * FLOPs * density * inputs[0].shape[0]  # 考虑稀疏性的能量计算
        self.energy += energy / (10 ** 9)
        self.energy_meter.update(energy / (10 ** 9))
        self.flops_meter.update(FLOPs / (10 ** 9))
        layer_name = layer.__class__.__name__
        self.update_sparsity(layer_name, density)


    def register_hooks(self):
        for name, module in self.model.named_modules():
            # print(name, module)
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                # print(self.first_conv_encountered)
                if not self.first_conv_encountered:
                    self.first_conv_encountered = True
                    print(f"{name} registered for artificial energy calculation!")
                    module.register_forward_hook(self.get_artificial_energy)

                else:
                    module.register_forward_hook(self.get_spike_energy)
                    print(f"{name} registered for spike energy calculation!")

    def print_sparsity(self):
        # 在验证结束后打印每一层的平均稀疏度
        for name, meter in self.sparsity_meters.items():
            print(f"Layer {name}: Average Sparsity = {meter.avg}")