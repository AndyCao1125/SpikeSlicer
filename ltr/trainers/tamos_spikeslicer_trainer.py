import copy
import os
from collections import OrderedDict
from ltr.trainers import BaseTrainer
from ltr.admin.stats import AverageMeter, StatValue
from ltr.admin.tensorboard import TensorboardWriter, wandb_write
import torch
import time
from tqdm import tqdm
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from SpikeSlicer_utils.SpikeSliceNet import reset_net
import numpy as np
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pytracking import TensorDict
from ltr.data.loader import ltr_collate_stack1, ltr_collate

class TAMOS_SpikeSlicerTrainer(BaseTrainer):
    def __init__(self, actor, snn_net, snn_loaders, ann_loaders, snn_optimizer, snn_scheduler, ann_optimizer, ann_scheduler, settings, processing, mem_loss):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, snn_loaders, snn_optimizer, settings, snn_scheduler)
        self.snn = snn_net
        self.extend_step = settings.extend_step
        self.raw_extend_width = settings.extend_width
        self.extend_width = settings.extend_width
        self.extend_width_epoch = settings.extend_width_epoch
        self.group_num = settings.group_num
        self._set_default_settings()
        self.processing = processing
        self.mem_loss = mem_loss
        self.split_train_epoch = settings.split_train_epoch
        self.train_model_name = settings.train_model_name
        self.ann_optimizer = ann_optimizer
        self.ann_scheduler = ann_scheduler
        self.ann_loaders = ann_loaders
        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)

    def multi_process(self, data):
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(self.processing, data))
        return ltr_collate_stack1(results)
        
    def divide_data(self, data):
        divided_data = []
        B = data['test_images'].shape[0]
        for b in range(B):
            data_ = {}
            for key in data.keys():
                if isinstance(data[key], np.ndarray):
                    data_[key] = data[key][b:b+1]
                elif 'anno' in key:
                    data_[key] = [data[key][b]]
                elif isinstance(data[key], list):
                    data_[key] = data[key][0]
                else:
                    data_[key] = data[key]
            divided_data.append(TensorDict(data_))
        return divided_data

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    @staticmethod
    def get_united_bbox(boxes, start_idx, end_idx):
        # xywh to xyxy
        new_boxes = copy.deepcopy(boxes[start_idx:end_idx + 1])
        new_boxes[:, 2] = new_boxes[:, 0] + new_boxes[:, 2]
        new_boxes[:, 3] = new_boxes[:, 1] + new_boxes[:, 3]
        xmin, ymin, xmax, ymax = new_boxes[:, 0].min().item(), new_boxes[:, 1].min().item(), new_boxes[:,
                                                                                            2].max().item(), new_boxes[
                                                                                                            :,
                                                                                                            3].max().item()
        return torch.tensor([xmin, ymin, xmax - xmin, ymax - ymin]).to(boxes.device)

    def get_anno(self, refer_anno, start_ids, end_ids):
        anno = []
        if not isinstance(refer_anno, torch.Tensor):
            refer_anno = torch.stack([refer_anno_[0] for refer_anno_ in refer_anno], dim=1)
        B = refer_anno.shape[0]
        # print(f'refer_anno.shape: {refer_anno.shape}, ids.shape: {ids.shape}')
        T = end_ids.shape[0]
        for t in range(T):
            batch_anno = []
            for b in range(B):
                # select the nearest anno as ground truth
                # print(f'ids[t, b]: {ids[t, b]}')
                batch_anno.append(self.get_united_bbox(refer_anno[b], start_ids[t, b], end_ids[t, b]))
            batch_anno = torch.stack(batch_anno, dim=0)
            anno.append(batch_anno)
        anno = torch.stack(anno, dim=0)
        return anno

    def update_width(self):
        self.extend_width = math.ceil(self.raw_extend_width * (1 - (1 + self.epoch) / self.extend_width_epoch))
        if self.extend_width < 1:
            self.extend_width = 1

    def anno_wrap(self, annos):
        return [{0:anno} for anno in annos]
        # return [{0:annos}]

    def anno_to_singledict(self, annos):
        return [{0:torch.stack([anno[0] for anno in annos], dim=1)}]

    def cycle_dataset_snn(self, loader):
        """Do a cycle of training or validation."""
        if 'test_clf' in self.actor.loss_weight.keys():
            self.actor.loss_weight['test_clf'] = 0

        self.processing.train_mode = 'spikeslicer'
        self.actor.objective['giou'].reduction = 'none'
        self.actor.objective['test_clf'].reduction = 'none'

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)
        both_extend_steps = self.extend_step * 2 + 1
        self._init_timing()
        functional.set_step_mode(self.snn, "m")
        for i, data in enumerate(loader, 1):
            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings
            ## change the data from torch.uint8 into torch.float32
            data['test_images'] = data['test_images'].float()
            data['train_images'] = data['train_images'].float()
            B = data['train_images'].shape[0]
            T = data['train_images'].shape[1]
            if self.epoch >= self.split_train_epoch:
                train_vox = data['train_images'].permute(1, 0, 2, 3, 4).contiguous()
                with torch.no_grad():
                    spikes = self.snn(train_vox)
                reset_net(self.snn)
                spike_local_idx = []
                ## spikes with gradient may have no spikes, so we need to find the first spike, it there is no spike, let the last element be the first spike
                for idx in range(B):
                    if spikes[:, idx, 0].nonzero().shape[0] == 0:
                        spike_local_idx.append(torch.tensor([T - 1]))
                    else:
                        spike_local_idx.append(torch.tensor([spikes.detach()[:, idx, 0].nonzero().min()]))
                spike_local_idx = torch.tensor(spike_local_idx)
                train_split_vox = []
                for b in range(B):
                    split_vox = train_vox[:spike_local_idx[b] + 1, b].sum(0)
                    train_split_vox.append(split_vox)
                train_split_vox = torch.stack(train_split_vox, dim=0)
                data['train_images'] = train_split_vox
                data['train_anno'] = self.get_anno(data['train_anno'], torch.zeros_like(spike_local_idx.unsqueeze(0)), spike_local_idx.unsqueeze(0)).squeeze(0)
            else:
                data['train_images'] = data['train_images'][:, :self.group_num].sum(1)
                data['train_anno'] = self.get_anno(data['train_anno'], torch.zeros(1, B).to(torch.int64), torch.ones(1, B).to(torch.int64) * (self.group_num - 1)).squeeze(0)
            # print(f'data["test_images"].shape: {data["test_images"].shape}, {data["test_images"].dtype}, {data["train_images"].dtype}')
            # sys.exit(0)
            # forward pass
            B = data['test_images'].shape[0]
            T = data['test_images'].shape[1]
            vox_events = data['test_images'].permute(1, 0, 2, 3, 4).contiguous()
            # print(f'vox_events.shape: {vox_events.shape}')
            spikes_with_gradient = self.snn(vox_events)
            # print(f'spikes_with_gradient.shape: {spikes_with_gradient.shape}')
            # print([spikes_with_gradient.detach()[:, idx, 0].nonzero() for idx in range(B)])

            spike_local_idx = []
            ## spikes with gradient may have no spikes, so we need to find the first spike, it there is no spike, let the last element be the first spike
            for idx in range(B):
                if spikes_with_gradient.detach()[:, idx, 0].nonzero().shape[0] == 0:
                    spike_local_idx.append(torch.tensor([T - 1]))
                else:
                    spike_local_idx.append(torch.tensor([spikes_with_gradient.detach()[:, idx, 0].nonzero().min()]))
                
            # spike_local_idx = torch.tensorwww([spikes_with_gradient.detach()[:, idx, 0].nonzero().min() for idx in range(B)])
            spike_local_idx = torch.tensor(spike_local_idx)
            

            
            extend_ids = torch.cat([(idx - self.extend_step) * self.extend_width + spike_local_idx.unsqueeze(0) for idx in range(both_extend_steps)]).clamp(min=0, max=T - 1) # (both_extend_steps, B)
            # print(f'spike_local_idx.shape: {spike_local_idx.shape}, extend_ids.shape: {extend_ids.shape}')
            stacked_vox_events = []
            for t in range(extend_ids.shape[0]):
                t_vox_events = []
                for b in range(B):
                    t_vox_events.append(vox_events[:extend_ids[t, b] + 1, b].sum(0))
                t_vox_events = torch.stack(t_vox_events, dim=0)
                stacked_vox_events.append(t_vox_events)
            stacked_vox_events = torch.stack(stacked_vox_events, dim=0) # (both_extend_steps, B, C, H, W)
            test_images = stacked_vox_events

            # stack events belonging to a natural frame into a single frame
            data['train_images'] = data['train_images'].unsqueeze(0).repeat(both_extend_steps, 1, 1, 1, 1).flatten(0, 1).cpu().numpy()
            data['test_images'] = test_images.flatten(0, 1).cpu().numpy()
            data['test_anno'] = self.get_anno(data['test_anno'], torch.zeros_like(extend_ids), extend_ids).flatten(0, 1).cpu() # obtain the middle anno value of each stacked frame
            data['train_anno'] = data['train_anno'].unsqueeze(0).repeat(both_extend_steps, 1, 1).flatten(0, 1).cpu()

            data['train_anno'] = self.anno_wrap(data['train_anno'])
            data['test_anno'] = self.anno_wrap(data['test_anno'])
            if 'settings' in data.keys():
                del data['settings']
            # data = self.divide_data(data)
            # data = self.multi_process(data)
            data = self.processing(data)
            data['train_anno'] = self.anno_to_singledict(data['train_anno'])
            data['test_anno'] = self.anno_to_singledict(data['test_anno'])
            for key in data.keys():
                if isinstance(data[key], np.ndarray):
                    data[key] = torch.from_numpy(data[key])
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].unsqueeze(0)
            
            # data['test_images'] = data['test_images'].unsqueeze(1).repeat(1, 3, 1, 1, 1)
            # data['train_images'] = data['train_images'].unsqueeze(1).repeat(1, 3, 1, 1, 1)
            # data['train_anno'] = data['train_anno'].unsqueeze(1).repeat(1, 3, 1)
            # data['test_anno'] = data['test_anno'].unsqueeze(1).repeat(1, 3, 1)

            # for key in ['test_images', 'train_images', 'test_anno', 'train_anno', 'test_proposals', 'test_label', 'proposal_iou']:
            #     data[key] = data[key].to(self.device)
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(self.device)
                # print(f'{key}:{isinstance(data[key], torch.Tensor)}')
                
                # if isinstance(data[key], np.ndarray):
                #     data[key] = torch.from_numpy(data[key]).to(self.device)

            # for key in data.keys():
            #     print(f'{key}:{data[key].device}' )
            # print(f'device: {self.device}')
            with torch.no_grad():
                tracker_loss, stats = self.actor(data)
            # here we could use various metrics to evaluate the performance of the tracker, like loss, iou, etc.
            tracker_loss = tracker_loss.view(both_extend_steps, B)
            tracker_loss = (tracker_loss.max(dim=0)[0] - tracker_loss) / (tracker_loss.max(dim=0)[0] - tracker_loss.min(dim=0)[0] + 1e-6)
            total_loss = 0
            desired_idx_ls = 0
            mem_loss_ls = 0
            I_loss_ls = 0
            if self.snn.node.step_mode == 's':
                self.snn.I = torch.stack(self.snn.I)
            for b in range(B):
                ## membrane loss
                # local_max_idx = tracker_loss[:, b].argmax().min()
                # global_max_idx = extend_ids[local_max_idx, b]
                # mem_loss_per_img, I_loss_per_img = self.mem_loss(self.snn.node.past_v, self.snn.I, b, spike_local_idx[b], global_max_idx,
                #                             self.snn.node.v_threshold)
                # desired_idx = local_max_idx
                ## weighted membrane loss
                reward = tracker_loss[:, b]
                if reward.argmax().shape != torch.Size([]):
                    max_idx = reward.argmax()[(reward.argmax() - self.extend_step).abs().argmin()]
                else:
                    max_idx = reward.argmax()
                local_max_idx = torch.argmax(reward).min()
                max_idx = extend_ids[local_max_idx, b]
                mem_loss_per_img, I_loss_per_img = self.mem_loss(self.snn.node.past_v, self.snn.I, b,
                                                                 spike_local_idx[b], max_idx, self.snn.node.v_threshold)
                # desired_idx = (reward * torch.arange(both_extend_steps).to(reward.device)).sum() / reward.sum()
                desired_idx = local_max_idx
                # loss_per_img = mem_loss_per_img + I_loss_per_img
                loss_per_img = mem_loss_per_img
                mem_loss_ls = mem_loss_ls + mem_loss_per_img.detach().cpu()
                I_loss_ls = I_loss_ls + I_loss_per_img.detach().cpu()
                total_loss = total_loss + loss_per_img
                desired_idx_ls = desired_idx_ls + desired_idx
            total_loss = total_loss / B
            stats['spike_local_idx'] = spike_local_idx.float().mean().item()
            stats['desired_idx'] = desired_idx_ls.item() / B
            stats['mem_loss'] = mem_loss_ls.item() / B
            stats['I_loss'] = I_loss_ls.item() / B
            stats['alpha'] = self.mem_loss.alpha.item()
            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
            reset_net(self.snn)
            # update statistics
            batch_size = data['test_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

    def cycle_dataset_ann(self, loader):
        if 'test_clf' in self.actor.loss_weight.keys():
            self.actor.loss_weight['test_clf'] = 100
        if 'clf_ce' in self.actor.loss_weight.keys():
            self.actor.loss_weight['clf_ce'] = 0.25
        self.processing.train_mode = 'tamos'
        self.actor.objective['giou'].reduction = 'mean'
        self.actor.objective['test_clf'].reduction = 'mean'

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)
        self._init_timing()
        functional.set_step_mode(self.snn, "m")
        for i, data in enumerate(loader, 1):
            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings
            ## change the data from torch.uint8 into torch.float32
            data['test_images'] = data['test_images'].float()
            data['train_images'] = data['train_images'].float()
            B = data['train_images'].shape[0]
            T = data['train_images'].shape[1]
            if self.epoch >= self.split_train_epoch:
                permute_data = data['train_images'].permute(1, 0, 2, 3, 4).contiguous()
                train_split_vox_ls = []
                start_idx_ls = []
                spike_local_idx_ls = []
                for idx in range(3):
                    start_idx = torch.randint(0, T - 20, (B,))
                    start_idx_ls.append(start_idx)
                    train_vox = []
                    for b in range(B):
                        train_vox_ = permute_data[start_idx[b]:start_idx[b] + 20, b] # (T, C, H, W)
                        train_vox.append(train_vox_)
                    train_vox = torch.stack(train_vox, dim=1) # (T, B, C, H, W)
                    with torch.no_grad():
                        spikes = self.snn(train_vox)
                    reset_net(self.snn)
                    spike_local_idx = []
                    ## spikes with gradient may have no spikes, so we need to find the first spike, it there is no spike, let the last element be the first spike
                    for idx in range(B):
                        if spikes[:, idx, 0].nonzero().shape[0] == 0:
                            spike_local_idx.append(torch.tensor([train_vox.shape[0] - 1]))
                        else:
                            spike_local_idx.append(torch.tensor([spikes.detach()[:, idx, 0].nonzero().min()]))
                    spike_local_idx = torch.tensor(spike_local_idx)
                    train_split_vox = []
                    for b in range(B):
                        split_vox = train_vox[:spike_local_idx[b] + 1, b].sum(0)
                        train_split_vox.append(split_vox)
                    train_split_vox = torch.stack(train_split_vox, dim=0)
                    train_split_vox_ls.append(train_split_vox)
                    spike_local_idx_ls.append(spike_local_idx)
                start_idx_ls = torch.stack(start_idx_ls, dim=0).to(data['train_images'].device)
                spike_local_idx_ls = torch.stack(spike_local_idx_ls, dim=0).to(data['train_images'].device)
                data['train_images'] = torch.stack(train_split_vox_ls, dim=1) #B, 3, C, H, W
                data['train_anno'] = self.get_anno(data['train_anno'], start_idx_ls, start_idx_ls + spike_local_idx_ls).permute(1, 0, 2) # B, 3, 4
            else:
                start_idx_ls = []
                train_vox_ls = []
                for idx in range(3):
                    start_idx = torch.randint(0, T - 20, (B,))
                    start_idx_ls.append(start_idx)
                    train_vox = []
                    for b in range(B):
                        train_vox_ = data['train_images'][b, start_idx[b]:start_idx[b] + self.group_num].sum(1) # (C, H, W)
                        train_vox.append(train_vox_)
                    train_vox = torch.stack(train_vox, dim=0) # (B, C, H, W)
                    train_vox_ls.append(train_vox)
                start_idx_ls = torch.stack(start_idx_ls, dim=0).to(data['train_images'].device)
                data['train_images'] = torch.stack(train_vox_ls, dim=1) #B, 3, C, H, W
                data['train_anno'] = self.get_anno(data['train_anno'], start_idx_ls, start_idx_ls + self.group_num - 1).permute(1, 0, 2) # B, 3, 4
            # forward pass
            B = data['test_images'].shape[0]
            T = data['test_images'].shape[1]

            permute_data = data['test_images'].permute(1, 0, 2, 3, 4).contiguous()
            test_split_vox_ls = []
            start_idx_ls = []
            spike_local_idx_ls = []
            for idx in range(3):
                start_idx = torch.randint(0, T - 20, (B,))
                start_idx_ls.append(start_idx)
                test_vox = []
                for b in range(B):
                    test_vox_ = permute_data[start_idx[b]:start_idx[b] + 20, b] # (T, C, H, W)
                    test_vox.append(test_vox_)
                test_vox = torch.stack(test_vox, dim=1) # (T, B, C, H, W)
                with torch.no_grad():
                    spikes = self.snn(test_vox)
                reset_net(self.snn)
                spike_local_idx = []
                ## spikes with gradient may have no spikes, so we need to find the first spike, it there is no spike, let the last element be the first spike
                for idx in range(B):
                    if spikes[:, idx, 0].nonzero().shape[0] == 0:
                        spike_local_idx.append(torch.tensor([test_vox.shape[0] - 1]))
                    else:
                        spike_local_idx.append(torch.tensor([spikes.detach()[:, idx, 0].nonzero().min()]))
                spike_local_idx = torch.tensor(spike_local_idx)
                test_split_vox = []
                for b in range(B):
                    split_vox = test_vox[:spike_local_idx[b] + 1, b].sum(0)
                    test_split_vox.append(split_vox)
                test_split_vox = torch.stack(test_split_vox, dim=0)
                test_split_vox_ls.append(test_split_vox)
                spike_local_idx_ls.append(spike_local_idx)
            start_idx_ls = torch.stack(start_idx_ls, dim=0).to(data['test_images'].device)
            spike_local_idx_ls = torch.stack(spike_local_idx_ls, dim=0).to(data['test_images'].device)
            data['test_images'] = torch.stack(test_split_vox_ls, dim=1) #B, 3, C, H, W
            data['test_anno'] = self.get_anno(data['test_anno'], start_idx_ls, start_idx_ls + spike_local_idx_ls).permute(1, 0, 2) # B, 3, 4
            


            # stack events belonging to a natural frame into a single frame
            anno_shape = data['train_anno'].shape
            pre_shape = anno_shape[:2]
            data['train_images'] = data['train_images'].flatten(0, 1).cpu().numpy()
            data['test_images'] = data['test_images'].flatten(0, 1).cpu().numpy()
            data['test_anno'] = data['test_anno'].flatten(0, 1).cpu()
            data['train_anno'] = data['train_anno'].flatten(0, 1).cpu()
            data['train_anno'] = self.anno_wrap(data['train_anno'])
            data['test_anno'] = self.anno_wrap(data['test_anno'])
            data = self.processing(data)
            for key in data.keys():
                if isinstance(data[key], np.ndarray):
                    data[key] = torch.from_numpy(data[key])
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].unsqueeze(0)

            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(self.device)
                    # data[key] = data[key].view(*pre_shape, *data[key].shape[1:])
            tracker_loss, stats = self.actor(data)
            loss = tracker_loss
            # here we could use various metrics to evaluate the performance of the tracker, like loss, iou, etc.

            stats['spike_local_idx'] = spike_local_idx_ls.float().mean().item()
            # to be continued

            # backward pass and update weights
            if loader.training:
                self.ann_optimizer.zero_grad()
                loss.backward()
                self.ann_optimizer.step()

            # update statistics
            batch_size = data['test_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                self.cycle_dataset_snn(loader)

        self._stats_new_epoch()
        self.update_alpha()
        wandb_write(self.stats, self.epoch, prefix='snn/')
        self.update_width()

        if self.epoch >= self.split_train_epoch:
            for loader in self.ann_loaders:
                if self.epoch % loader.epoch_interval == 0:
                    self.cycle_dataset_ann(loader)

            self.ann_scheduler.step()
            self._stats_new_epoch()
            wandb_write(self.stats, self.epoch, prefix='ann/')



    def update_alpha(self):
        self.mem_loss.alpha_optim.zero_grad()
        self.mem_loss.alpha.grad = torch.tensor(2 * (self.stats['train']['desired_idx'].history[-1] - self.extend_step)).to(self.mem_loss.alpha.device)
        self.mem_loss.alpha_optim.step()
        self.mem_loss.alpha.data = self.mem_loss.alpha.data.clamp(min=0, max=1)
    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
                    print_str += '%s: %.5f  ,  ' % (name, val.avg)
            print(print_str[:-5])

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                lr_list = self.lr_scheduler.get_lr()
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()
