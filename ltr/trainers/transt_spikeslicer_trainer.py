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
class TransT_SpikeSlicerTrainer(BaseTrainer):
    def __init__(self, actor, snn_net, snn_loaders, ann_loaders, snn_optimizer, snn_scheduler, ann_optimizer, ann_scheduler, settings, processing, mem_loss, quantization_layer):
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
        self.snn_max_epoch = settings.snn_max_epoch
        self.max_epoch = settings.max_epoch
        self.group_num = settings.group_num
        self._set_default_settings()
        self.processing = processing
        self.mem_loss = mem_loss
        self.split_template_epoch = settings.split_template_epoch
        self.ann_optimizer = ann_optimizer
        self.ann_scheduler = ann_scheduler
        self.ann_loaders = ann_loaders
        self.quantization_layer = quantization_layer

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    @staticmethod
    def get_united_bbox(boxes, end_ids):
        # xywh to xyxy
        new_boxes = copy.deepcopy(boxes[:end_ids + 1])
        new_boxes[:, 2] = new_boxes[:, 0] + new_boxes[:, 2]
        new_boxes[:, 3] = new_boxes[:, 1] + new_boxes[:, 3]
        xmin, ymin, xmax, ymax = new_boxes[:, 0].min().item(), new_boxes[:, 1].min().item(), new_boxes[:,
                                                                                             2].max().item(), new_boxes[
                                                                                                              :,
                                                                                                              3].max().item()
        return torch.tensor([xmin, ymin, xmax - xmin, ymax - ymin]).to(boxes.device)

    def get_anno(self, refer_anno, ids):
        anno = []
        B = refer_anno.shape[0]
        # print(f'refer_anno.shape: {refer_anno.shape}, ids.shape: {ids.shape}')
        T = ids.shape[0]
        for t in range(T):
            batch_anno = []
            for b in range(B):
                # select the nearest anno as ground truth
                # print(f'ids[t, b]: {ids[t, b]}')
                batch_anno.append(self.get_united_bbox(refer_anno[b], ids[t, b]))
            batch_anno = torch.stack(batch_anno, dim=0)
            anno.append(batch_anno)
        anno = torch.stack(anno, dim=0)
        return anno

    def update_width(self):
        self.extend_width = math.ceil(self.raw_extend_width * (1 - (1 + self.epoch) / self.extend_width_epoch))
        if self.extend_width < 1:
            self.extend_width = 1

    def events2frames(self, events, start_idx, end_idx):
        """
        :param events: (N, 6) (x, y, t, p, l, b)
        :param start_idx: (b, )
        :param end_idx: (b, )
        :return: event_frames: (b, 2, h, w)
        """
        sliced_events = []
        B = int(events[:, 5].max().item()) + 1
        for b in range(B):
            batch_events = events[events[:, 5] == b]
            idx = (batch_events[:, 4] <= end_idx[b]) & (batch_events[:, 4] >= start_idx[b])
            batch_events = batch_events[idx, :4]
            batch_events = torch.cat([batch_events, torch.ones(batch_events.shape[0], 1).to(batch_events.device) * b], dim=1)
            sliced_events.append(batch_events)
        sliced_events = torch.cat(sliced_events, dim=0)
        frames = self.quantization_layer(sliced_events)
        return frames


    def cycle_dataset_snn(self, loader):
        """Do a cycle of training or validation."""
        self.actor.objective.weight_dict['loss_ce'] = 0
        self.processing.train_mode = 'spikeslicer'
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
            data['search_images'] = data['search_images'].float()
            data['template_images'] = data['template_images'].float()
            B = data['template_images'].shape[0]
            T = data['template_images'].shape[1]
            if self.epoch >= self.split_template_epoch:
                template_vox = data['template_images'].permute(1, 0, 2, 3, 4).contiguous()
                with torch.no_grad():
                    spikes = self.snn(template_vox)
                reset_net(self.snn)
                spike_local_idx = []
                ## spikes with gradient may have no spikes, so we need to find the first spike, it there is no spike, let the last element be the first spike
                for idx in range(B):
                    if spikes[:, idx, 0].nonzero().shape[0] == 0:
                        spike_local_idx.append(torch.tensor([T - 1]))
                    else:
                        spike_local_idx.append(torch.tensor([spikes.detach()[:, idx, 0].nonzero().min()]))
                spike_local_idx = torch.tensor(spike_local_idx)
                template_split_vox = []
                for b in range(B):
                    split_vox = template_vox[:spike_local_idx[b] + 1, b].sum(0)
                    template_split_vox.append(split_vox)
                template_split_vox = torch.stack(template_split_vox, dim=0)
                template_split_vox = self.events2frames(data['template_events'], torch.zeros_like(spike_local_idx), spike_local_idx)
                data['template_images'] = template_split_vox
                data['template_anno'] = self.get_anno(data['template_anno'], spike_local_idx.unsqueeze(0)).squeeze(0)
            else:
                # data['template_images'] = data['template_images'][:, :self.group_num].sum(1)
                data['template_images'] = self.events2frames(data['template_events'], torch.zeros(B), torch.ones(B) * (self.group_num - 1))
                data['template_anno'] = self.get_anno(data['template_anno'], torch.ones(1, B).to(torch.int64) * (self.group_num - 1)).squeeze(0)
            # print(f'data["search_images"].shape: {data["search_images"].shape}, {data["search_images"].dtype}, {data["template_images"].dtype}')
            # sys.exit(0)
            # forward pass
            B = data['search_images'].shape[0]
            T = data['search_images'].shape[1]
            vox_events = data['search_images'].permute(1, 0, 2, 3, 4).contiguous()
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
                
            # spike_local_idx = torch.tensor([spikes_with_gradient.detach()[:, idx, 0].nonzero().min() for idx in range(B)])
            spike_local_idx = torch.tensor(spike_local_idx)
            

            
            extend_ids = torch.cat([(idx - self.extend_step) * self.extend_width + spike_local_idx.unsqueeze(0) for idx in range(both_extend_steps)]).clamp(min=0, max=T - 1) # (both_extend_steps, B)
            # print(f'spike_local_idx.shape: {spike_local_idx.shape}, extend_ids.shape: {extend_ids.shape}')

            stacked_vox_events = []
            ## without event representation
            # for t in range(extend_ids.shape[0]):
            #     t_vox_events = []
            #     for b in range(B):
            #         t_vox_events.append(vox_events[:extend_ids[t, b] + 1, b].sum(0))
            #     t_vox_events = torch.stack(t_vox_events, dim=0)
            #     stacked_vox_events.append(t_vox_events)
            # stacked_vox_events = torch.stack(stacked_vox_events, dim=0) # (both_extend_steps, B, C, H, W)
            ## without event representation

            ## with event representation
            for t in range(extend_ids.shape[0]):
                t_vox_events = self.events2frames(data['search_events'], torch.zeros(B), extend_ids[t])
                stacked_vox_events.append(t_vox_events)
            stacked_vox_events = torch.stack(stacked_vox_events, dim=0)
            ## with event representation

            search_images = stacked_vox_events

            # stack events belonging to a natural frame into a single frame
            data['template_images'] = data['template_images'].unsqueeze(0).repeat(both_extend_steps, 1, 1, 1, 1).flatten(0, 1).permute(0, 2, 3, 1).contiguous().cpu().numpy()
            data['search_images'] = search_images.flatten(0, 1).permute(0, 2, 3, 1).contiguous().cpu().numpy()
            data['search_anno'] = self.get_anno(data['search_anno'], extend_ids).flatten(0, 1).cpu() # obtain the middle anno value of each stacked frame
            data['template_anno'] = data['template_anno'].unsqueeze(0).repeat(both_extend_steps, 1, 1).flatten(0, 1).cpu()

            data = self.processing(data)
            for key in ['search_images', 'template_images', 'search_anno', 'template_anno']:
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

                local_max_idx = torch.argmax(reward).min()
                max_idx = extend_ids[local_max_idx, b]
                mem_loss_per_img, I_loss_per_img = self.mem_loss(self.snn.node.past_v, self.snn.I, b,
                                                                 spike_local_idx[b], max_idx, self.snn.node.v_threshold)
                desired_idx = (reward * torch.arange(both_extend_steps)).sum() / reward.sum()
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
            # to be continued

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
            reset_net(self.snn)
            # update statistics
            batch_size = data['search_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

    def cycle_dataset_ann(self, loader):
        self.actor.objective.weight_dict['loss_ce'] = 8.334
        self.processing.train_mode = 'transt'
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
            data['search_images'] = data['search_images'].float()
            data['template_images'] = data['template_images'].float()
            B = data['template_images'].shape[0]
            T = data['template_images'].shape[1]
            if self.epoch >= self.split_template_epoch:
                template_vox = data['template_images'].permute(1, 0, 2, 3, 4).contiguous()
                with torch.no_grad():
                    spikes = self.snn(template_vox)
                reset_net(self.snn)
                spike_local_idx = []
                ## spikes with gradient may have no spikes, so we need to find the first spike, it there is no spike, let the last element be the first spike
                for idx in range(B):
                    if spikes[:, idx, 0].nonzero().shape[0] == 0:
                        spike_local_idx.append(torch.tensor([T - 1]))
                    else:
                        spike_local_idx.append(torch.tensor([spikes.detach()[:, idx, 0].nonzero().min()]))
                spike_local_idx = torch.tensor(spike_local_idx)
                # template_split_vox = []
                # for b in range(B):
                #     split_vox = template_vox[:spike_local_idx[b] + 1, b].sum(0)
                #     template_split_vox.append(split_vox)
                # template_split_vox = torch.stack(template_split_vox, dim=0)
                template_split_vox = self.events2frames(data['template_events'], torch.zeros_like(spike_local_idx), spike_local_idx)
                data['template_images'] = template_split_vox
                data['template_anno'] = self.get_anno(data['template_anno'], spike_local_idx.unsqueeze(0)).squeeze(0)
            else:
                # data['template_images'] = data['template_images'][:, :self.group_num].sum(1)
                data['template_images'] = self.events2frames(data['template_events'], torch.zeros(B), torch.ones(B) * (self.group_num - 1))
                data['template_anno'] = self.get_anno(data['template_anno'], torch.ones(1, B).to(torch.int64) * (self.group_num - 1)).squeeze(0)
            # forward pass
            B = data['search_images'].shape[0]
            T = data['search_images'].shape[1]
            vox_events = data['search_images'].permute(1, 0, 2, 3, 4).contiguous()
            with torch.no_grad():
                spikes = self.snn(vox_events)
            reset_net(self.snn)
            spike_local_idx = []
            ## spikes with gradient may have no spikes, so we need to find the first spike, it there is no spike, let the last element be the first spike
            for idx in range(B):
                if spikes.detach()[:, idx, 0].nonzero().shape[0] == 0:
                    spike_local_idx.append(torch.tensor([T - 1]))
                else:
                    spike_local_idx.append(torch.tensor([spikes.detach()[:, idx, 0].nonzero().min()]))

            spike_local_idx = torch.tensor(spike_local_idx)


            ## without event representation
            # stacked_vox_events = []
            # for b in range(B):
            #     stacked_vox_events.append(vox_events[:spike_local_idx[b] + 1, b].sum(0))
            # stacked_vox_events = torch.stack(stacked_vox_events, dim=0) # (B, C, H, W)
            ## without event representation

            ## with event representation
            stacked_vox_events = self.events2frames(data['search_events'], torch.zeros(B), spike_local_idx)
            ## with event representation
            search_images = stacked_vox_events
            # stack events belonging to a natural frame into a single frame
            data['template_images'] = data['template_images'].permute(0, 2, 3, 1).contiguous().cpu().numpy()
            data['search_images'] = search_images.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            data['search_anno'] = self.get_anno(data['search_anno'], spike_local_idx.unsqueeze(0)).flatten(0, 1).cpu() # obtain the middle anno value of each stacked frame
            data['template_anno'] = data['template_anno'].cpu()

            data = self.processing(data)
            for key in ['search_images', 'template_images', 'search_anno', 'template_anno']:
                data[key] = data[key].to(self.device)

            tracker_loss, stats = self.actor(data)
            loss = tracker_loss.mean()
            # here we could use various metrics to evaluate the performance of the tracker, like loss, iou, etc.

            stats['spike_local_idx'] = spike_local_idx.float().mean().item()
            # to be continued

            # backward pass and update weights
            if loader.training:
                self.ann_optimizer.zero_grad()
                loss.backward()
                self.ann_optimizer.step()

            # update statistics
            batch_size = data['search_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

    def train_epoch(self):
        """Do one epoch for each loader."""
        if self.epoch < self.snn_max_epoch:
            for loader in self.loaders:
                if self.epoch % loader.epoch_interval == 0:
                    self.cycle_dataset_snn(loader)

            self._stats_new_epoch()
            self.update_alpha()
            wandb_write(self.stats, self.epoch, prefix='snn/')
            self.update_width()

        if self.epoch >= self.split_template_epoch:
            if self.epoch >= self.max_epoch / 2 and len(self.ann_loaders) > 1:
                loader = self.ann_loaders[1]
                if self.epoch % loader.epoch_interval == 0:
                    self.cycle_dataset_ann(loader)
            else:
                loader = self.ann_loaders[0]
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
