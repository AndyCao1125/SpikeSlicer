import copy
import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
from ltr.admin.environment import env_settings
import time
import tonic
from dv import AedatFile
from tqdm import tqdm
events_struct = np.dtype([("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", bool)])
def make_structured_array(*args, dtype=events_struct):
    """
    Make a structured array given a variable number of argument values

    Args:
        *args: Values in the form of nested lists or tuples or numpy arrays. Every except the first argument can be of a primitive data type like int or float
    Returns:
        struct_arr: numpy structured array with the shape of the first argument
    """
    assert not isinstance(args[-1], np.dtype), "The `dtype` must be provided as a keyword argument."
    names = dtype.names
    assert len(args) == len(names)
    struct_arr = np.empty_like(args[0], dtype=dtype)
    for arg,name in zip(args,names):
        struct_arr[name] = arg
    return struct_arr


class FE108(BaseVideoDataset):

    def __init__(self, root, subset='train', groupnum=1, txt_suffix="_all.txt"):
        super().__init__('FE108', root, None)

        # all folders inside the root
        self.subset = subset
        self.sequence_names, self.sequence_list = self._get_sequence_list(subset + txt_suffix)
        self.groupnum = groupnum
        pair_path = os.path.join(root, "pair.txt")
        self.pair = self.get_pair(pair_path)
        self.time_series = self.get_time_series()
        self.seq_per_class = self._build_seq_per_class()
        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def get_name(self):
        return 'toolkit'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_pair(self, pair_path):
        pair = {}
        with open(pair_path, 'r') as f:
            for line in f.readlines():
                file, start_frame = line.split()
                pair[file] = int(start_frame) + 1
        return pair

    ## for image saver
    def get_time_series(self):
        time_series = {}
        for idx in range(len(self.sequence_names)):
            seq_name = self.sequence_names[idx]
            gt_path = os.path.join(self.sequence_list[idx], 'groundtruth_rect.txt')
            file_path = os.path.join(self.sequence_list[idx], 'events.aedat4')
            time_series[seq_name] = self.tag(seq_name, gt_path, file_path)
        return time_series

    def tag(self, eventtype, gtPath, filePath):
        ## 根据pair.txt得到目标数据集的起始自然帧的位置，然后对应到事件数据集的起始时间

        start_frame = self.pair[eventtype]

        time_series = []
        count = 0
        frame_num = len(open(gtPath, 'r').readlines())
        print(f'{eventtype} has {frame_num} frames.')
        # frame_num = 956
        f = AedatFile(filePath)

        for frame in f["frames"]:  ## 定位event数据的timestamp
            count += 1
            if count >= start_frame and count <= start_frame + frame_num:
                time_series.append(frame.timestamp_start_of_frame)
            else:
                continue
        return time_series

    def image_saver(self, seq_id, groupnum, save_suffix="event_frame"):
        f = AedatFile(os.path.join(self.sequence_list[seq_id], 'events.aedat4'))
        save_dir = os.path.join(self.sequence_list[seq_id], save_suffix, 'groupnum_' + str(groupnum))
        if not os.path.exists(os.path.join(self.sequence_list[seq_id], save_suffix)):
            os.mkdir(os.path.join(self.sequence_list[seq_id], save_suffix))
        if not os.path.exists(save_dir):
            os.mkdir(os.path.join(save_dir))
        events = np.hstack([packet for packet in f['events'].numpy()])
        all_ts, all_x, all_y, all_p = events['timestamp'], events['x'], events['y'], events['polarity']
        frame_id = list(range(len(self.time_series[self.sequence_names[seq_id]]) - 1))
        time_series = self.time_series[self.sequence_names[seq_id]]
        for idx in tqdm(frame_id):
            # around 0.5 seconds to get the index
            start = np.where(all_ts > time_series[idx])  # 找到这一个自然帧内的event数据的起止timestamp的索引
            end = np.where(all_ts > time_series[idx + 1])
            xytp = make_structured_array(
                all_x[start[0][0]:end[0][0]],
                all_y[start[0][0]:end[0][0]],
                all_ts[start[0][0]:end[0][0]],
                all_p[start[0][0]:end[0][0]],
                dtype=np.dtype([("x", int), ("y", int), ("t", int), ("p", int)]),
            )
            frame = tonic.transforms.functional.to_frame_numpy(
                events=xytp,
                sensor_size=(346, 260, 2),
                # time_window=30000,
                # event_count=int(count),
                n_time_bins=groupnum,
                # n_event_bins=groupnum,
                # overlap=self.overlap,
                # include_incomplete=self.include_incomplete,
            )
            frame = frame.astype(np.uint8)
            if groupnum == 1:
                np.save(os.path.join(save_dir, f'{idx}.npy'), frame[0])
            else:
                for i in range(groupnum):
                    np.save(os.path.join(save_dir, f'{idx}_{i}.npy'), frame[i])

    def raw_event_saver(self, seq_id, groupnum, save_suffix="raw_event"):
        f = AedatFile(os.path.join(self.sequence_list[seq_id], 'events.aedat4'))
        save_dir = os.path.join(self.sequence_list[seq_id], save_suffix, 'groupnum_' + str(groupnum))
        if not os.path.exists(os.path.join(self.sequence_list[seq_id], save_suffix)):
            os.mkdir(os.path.join(self.sequence_list[seq_id], save_suffix))
        if not os.path.exists(save_dir):
            os.mkdir(os.path.join(save_dir))
        events = np.hstack([packet for packet in f['events'].numpy()])
        all_ts, all_x, all_y, all_p = events['timestamp'], events['x'], events['y'], events['polarity']
        frame_id = list(range(len(self.time_series[self.sequence_names[seq_id]]) - 1))
        time_series = self.time_series[self.sequence_names[seq_id]]
        for idx in tqdm(frame_id):
            # cc_time = time.time()
            ## old version np.where is so slow
            # around 0.5 seconds to get the index
            # start = np.where(all_ts > time_series[idx])  # 找到这一个自然帧内的event数据的起止timestamp的索引
            # end = np.where(all_ts > time_series[idx + 1])
            
            # xytp = make_structured_array(
            #     all_x[start[0][0]:end[0][0]],
            #     all_y[start[0][0]:end[0][0]],
            #     all_ts[start[0][0]:end[0][0]],
            #     all_p[start[0][0]:end[0][0]],
            #     dtype=np.dtype([("x", int), ("y", int), ("t", int), ("p", int)]),
            # )
            start = np.searchsorted(all_ts, time_series[idx], side='right')  # 找到这一个自然帧内的event数据的起止timestamp的索引
            end = np.searchsorted(all_ts, time_series[idx + 1], side='left')
            xytp = make_structured_array(
                all_x[start:end],
                all_y[start:end],
                all_ts[start:end],
                all_p[start:end],
                dtype=np.dtype([("x", int), ("y", int), ("t", int), ("p", int)]),
            )
            # print(start[0][0], end[0][0], start1, end2)
            # print(f'before structure time:{time.time() - cc_time}')
            # print(f'cc_time: {time.time() - cc_time}')
            # sys.exit(0)
            if groupnum == 1:
                np.save(os.path.join(save_dir, f'{idx}.npy'), xytp)
            else:
                # Divide the time range into 'groupnum' intervals and save events for each interval
                start_time = all_ts[start:end].min()
                end_time = all_ts[start:end].max()
                interval = (end_time - start_time) / groupnum
                # print(f'start_time: {start_time}, end_time: {end_time}, interval: {interval}')
                for i in range(groupnum):
                    # cir_time = time.time()
                    interval_start_time = start_time + i * interval
                    interval_end_time = start_time + (i + 1) * interval if i < groupnum - 1 else end_time
                    if i == groupnum -1:
                        interval_data = xytp[(xytp['t'] >= interval_start_time) & (xytp['t'] <= interval_end_time)]
                    else:
                        interval_data = xytp[(xytp['t'] >= interval_start_time) & (xytp['t'] < interval_end_time)]
                    standard_data = np.concatenate([interval_data['x'][:, np.newaxis], interval_data['y'][:, np.newaxis], 
                                        interval_data['t'][:, np.newaxis], interval_data['p'][:, np.newaxis]], axis=1)
                    np.save(os.path.join(save_dir, f'{idx}_{i}.npy'), standard_data)
                    # print(f'cir_time: {time.time() - cir_time}')
                    # np.save(os.path.join(save_dir, f'{idx}_{i}.npy'), interval_data)
            
            ## validate:
            
            # original_data = np.concatenate([xytp['x'][:, np.newaxis], xytp['y'][:, np.newaxis], 
            #                             xytp['t'][:, np.newaxis], xytp['p'][:, np.newaxis]], axis=1)
            # # 对每个分割的文件进行重组
            # recombined_data = []
            
            # for i in range(groupnum):
            #     file_name = os.path.join(save_dir, f'{idx}_{i}.npy')
            #     split_data = np.load(file_name)
            #     print(f'split_data.shape: {split_data.shape}, filename:{file_name}')
            #     recombined_data.append(split_data)
            # recombined_data = np.concatenate(recombined_data, axis=0)

            # print(f'originshape: {original_data.shape}, recombinedshape: {recombined_data.shape}')

            # # print(f'Validate: {all(np.array_equal(recombined_data[field], original_data[field]) for field in original_data.dtype.names)}')
            # print(f'Validate: {all(np.array_equal(recombined_data[:, i], original_data[:, i]) for i in range(4))}')
            
            # if idx == 2:
            #     sys.exit(0)

    def image_loader(self, seq_id, frame_ids):
        event_dir = os.path.join(self.sequence_list[seq_id], "event_frame", 'groupnum_' + str(self.groupnum))
        frames = []
        for idx in frame_ids:
            if self.groupnum > 1:
                frame = np.load(os.path.join(event_dir, f'{idx // self.groupnum}_{idx % self.groupnum}.npy'))
                frames.append(frame)
            else:
                frame = np.load(os.path.join(event_dir, f'{idx}.npy'))
                frames.append(frame)
        # frames = np.concatenate(frames, axis=0)
        return np.stack(frames, 0)
    

    def load_and_split_npy(self, file_path, start_idx):

        groupnum = self.groupnum

        # 加载npy文件
        data = np.load(file_path)
        
        # 获取时间轴上的最小和最大时间戳
        min_time, max_time = data['t'].min(), data['t'].max()

        # 计算时间分割
        time_split = (max_time - min_time) / groupnum

        # 计算开始和结束时间戳
        start_time = min_time + start_idx * time_split
        end_time = min_time + (start_idx + 1) * time_split

        # 提取时间范围内的数据
        extracted_data = data[(data['t'] >= start_time) & (data['t'] < end_time)]
        standard_data = np.concatenate([extracted_data['x'][:, np.newaxis], extracted_data['y'][:, np.newaxis], 
                                        extracted_data['t'][:, np.newaxis], extracted_data['p'][:, np.newaxis]], axis=1)
        return standard_data
    
    def raw_event_loader(self, seq_id, frame_ids):
        ### groupnum 1
        # event_dir = os.path.join(self.sequence_list[seq_id], "raw_event", 'groupnum_' + str(1))
        # events_list = []
        # for idx, groupnum_idx in enumerate(frame_ids):
        #     origin_idx = groupnum_idx // self.groupnum  # 原始的自然帧的索引
        #     local_idx = groupnum_idx % self.groupnum  # 在一个自然帧内的索引
        #     events = self.load_and_split_npy(os.path.join(event_dir, f'{origin_idx}.npy'), local_idx)
        #     events = np.concatenate([events, np.ones((events.shape[0], 1)) * idx], axis=1)
        #     events_list.append(events)
        # events_list = np.concatenate(events_list, axis=0)

        ### groupnum 5
        event_dir = os.path.join(self.sequence_list[seq_id], "raw_event", 'groupnum_' + str(5))
        events_list = []
        for idx, groupnum_idx in enumerate(frame_ids):
            origin_idx = groupnum_idx // self.groupnum  # 原始的自然帧的索引
            local_idx = groupnum_idx % self.groupnum  # 在一个自然帧内的索引
            file_path = os.path.join(event_dir, f'{origin_idx}_{local_idx}.npy')
            events = np.load(file_path)
            if events.shape[0] == 0:
                events = np.array([[0, 0, 1, 0]], dtype=np.float64)
            events = np.concatenate([events, np.ones((events.shape[0], 1)) * idx], axis=1)
            events_list.append(events)
        events_list = np.concatenate(events_list, axis=0)
        return events_list


    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_names[i]
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self, txt_file):
        list_file = os.path.join(self.root, txt_file)
        with open(list_file, 'r') as f:
            seq_names = f.read().strip().split('\n')
        # seq_names = ['airplane_mul','ball333']
        if self.subset == 'val':
            seq_dirs = [os.path.join(self.root, 'train', s) for s in seq_names]
            print(f'-----{self.subset} set has {len(seq_dirs)} sequences.')
        else:
            seq_dirs = [os.path.join(self.root, self.subset, s) for s in seq_names]
            print(f'-----{self.subset} set has {len(seq_dirs)} sequences.')
        return seq_names, seq_dirs

    def _read_bb_anno(self, seq_path, suffix):
        if self.groupnum == 1:
            bb_anno_file = os.path.join(seq_path, "groundtruth_rect" + ".txt")
        else:
            bb_anno_file = os.path.join(seq_path, "groundtruth_rect_{}".format(self.groupnum) + suffix + ".txt")
        gt = np.genfromtxt(bb_anno_file, delimiter=',')
        return torch.from_numpy(gt)

    ## for training
    def get_sequence_info(self, seq_id, suffix="_linear"):
        seq_path = self.sequence_list[seq_id]
        bbox = self._read_bb_anno(seq_path, suffix)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0) & (bbox[:, 2] < 346) & (bbox[:, 3] < 260) & (bbox[:, 0] >= 0) & (bbox[:, 1] >= 0) & (bbox[:, 0] < 346) & (bbox[:, 1] < 260)
        visible = copy.deepcopy(valid)

        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': None}

    def get_class_name(self, seq_id):
        return self.sequence_names[seq_id]

    def get_frames(self, seq_id, frame_ids, anno=None, is_extend=False):

        frame_list = self.image_loader(seq_id, frame_ids)
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        anno_frames = {}
        ## extend the bbox
        if is_extend:
            if frame_ids[-1] < len(anno['visible']) - 1:
                if anno['visible'][frame_ids[-1] + 1]:
                    frame_ids.extend([frame_ids[-1] + 1])
                else:
                    frame_ids.extend([frame_ids[-1]])
            else:
                frame_ids.extend([frame_ids[-1]])
        for key, value in anno.items():
            if value is not None:
                anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
            else:
                anno_frames[key] = None
        w, h = anno_frames['bbox'][0][2], anno_frames['bbox'][0][3]
        anno_frames['bbox'] = torch.stack(anno_frames['bbox'], dim=0)
        assert w > 0 and h > 0, "bbox width and height should be positive, seq_id: {} frame_ids: {}".format(seq_id, frame_ids)
        return frame_list, anno_frames, None
