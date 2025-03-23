import h5py
import numpy as np
import tonic
import torch
import matplotlib.pyplot as plt
from PIL import Image
from dv import AedatFile
import argparse
import os
import tqdm
from typing import List
import time
pair = {}

events_struct = np.dtype([("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", bool)])
def get_start_frame(seq_name):
    return pair[seq_name]
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

def tag(eventtype,gtPath,pairPath,filePath):        
    ## 根据pair.txt得到目标数据集的起始自然帧的位置，然后对应到事件数据集的起始时间

    match_file = pairPath
    with open(match_file, 'r') as f:
        for line in f.readlines():
            file, start_frame = line.split()
            pair[file] = int(start_frame) + 1

    start_frame = get_start_frame(eventtype)

    time_series = []
    count = 0
    frame_num = len(open(gtPath,'r').readlines())
    # print(f'{eventtype} has {frame_num} frames in total.')
    # frame_num = 956
    f = AedatFile(filePath)

    for frame in f["frames"]:    ## 定位event数据的timestamp
        count += 1
        if count >= start_frame and count <= start_frame + frame_num:
            time_series.append(frame.timestamp_start_of_frame)
        else:
            continue
    return time_series


# def generate_grouped_timestamps(original_timestamps: List[int], groupnum: int) -> List[int]:
#     """
#     根据原始时间戳列表和组数，生成平均切割后的新时间戳列表。

#     参数:
#     original_timestamps (List[int]): 原始时间戳列表。
#     groupnum (int): 组数。

#     返回:
#     List[int]: 新的时间戳列表。
#     """
#     # 确保原始时间戳列表至少有两个时间戳以计算持续时间
#     if len(original_timestamps) < 2:
#         raise ValueError("原始时间戳列表需要至少包含两个时间戳。")

#     # 计算总持续时间
#     total_duration = original_timestamps[-1] - original_timestamps[0]

#     # 计算每个组的持续时间
#     group_duration = total_duration / groupnum

#     # 生成新的时间戳列表
#     new_timestamps = [original_timestamps[0] + i * group_duration for i in range(groupnum + 1)]

#     return new_timestamps


def calculate_timestamp(raw_time_series, index, groupnum):
    # 确定原始区间
    original_interval_index = index // groupnum
    # 计算区间时间分割
    time_split = (raw_time_series[original_interval_index + 1] - raw_time_series[original_interval_index]) / groupnum
    # 计算新索引对应的时间戳
    timestamp = raw_time_series[original_interval_index] + time_split * (index % groupnum)
    return timestamp


def extract_event_data(seq_name: str, start_idx: List[int], end_idx: List[int], event_dir, pair_dir,  groupnum=5) -> List[np.ndarray]:
    """
    提取给定时间范围内的事件流数据。

    参数:
    seq_name (str): 序列名称。
    start_idx (List[int]): 起始索引列表。
    end_idx (List[int]): 结束索引列表。
    time_series (List[int]): 时间戳列表。
    event_dir (str): 包含事件数据的目录。

    返回:
    List[np.ndarray]: 包含每个索引范围内事件数据的列表。
    """
    # ## 读取所有序列名字
    # with open(list_dir, 'r') as f:
    #     seq_names = f.read().strip().split('\n')
    #     seq_names = seq_names[0:]
    


    ## 获取事件数据文件路径
    seq_dir = os.path.join(event_dir, seq_name, 'events.aedat4')
    gt_dir = os.path.join(event_dir, seq_name, 'groundtruth_rect.txt')   ## label
    start = time.time()
    raw_time_series=tag(seq_name,gt_dir,pair_dir,seq_dir)
    print("tag", time.time()-start)
    # ## 这一步通过线性插值去生成一个新的时间戳序列，存在小数
    # time_series = generate_grouped_timestamps(raw_time_series, groupnum)

    # 读取事件数据文件
    start = time.time()
    f = AedatFile(seq_dir)
    events = np.hstack([packet for packet in f['events'].numpy()])
    print("AedatFile", time.time()-start)
    # 解析事件数据
    all_ts, all_x, all_y, all_p = events['timestamp'], events['x'], events['y'], events['polarity']

    # print(all_ts[:1000])
    # 存储每个范围内的事件数据
    extracted_data = []
    start_time = time.time()
    for start, end in zip(start_idx, end_idx):
        # # 根据索引找到实际的时间戳
        # start_time = time_series[start]
        # end_time = time_series[end]
        start_timestamp = calculate_timestamp(raw_time_series, start, groupnum)
        end_timestamp = calculate_timestamp(raw_time_series, end, groupnum)

        # 提取时间范围内的事件
        mask = (all_ts >= start_timestamp) & (all_ts < end_timestamp)
        x, y, t, p = all_x[mask], all_y[mask], all_ts[mask], all_p[mask]

        # 使用 make_structured_array 函数创建结构化数组
        xytp = make_structured_array(x, y, t, p, dtype=np.dtype([("x", int), ("y", int), ("t", int), ("p", int)]))

        # 将结构化数组添加到列表中
        extracted_data.append(xytp)
    print("Append data", time.time()-start_time)
    return extracted_data


# if __name__ == '__main__':
#     event_dir = '/home/lsf_node01/dataset/FE108/train'
#     pair_dir = '/home/lsf_node01/dataset/FE108/pair.txt'
#     seq_name = 'ball333'
#     start_idx = [0, 1, 2, 3, 4]
#     end_idx = [1, 2, 3, 4, 5]
#     extracted_data = extract_event_data(seq_name, start_idx, end_idx, event_dir, pair_dir)
#     print(extracted_data)

if __name__ == '__main__':
    event_dir = '/home/lsf_node01/dataset/FE108/train'
    pair_dir = '/home/lsf_node01/dataset/FE108/pair.txt'
    seq_name = 'ball333'
    start_idx = np.array([0, 1, 2, 3, 4])
    end_idx = np.array([1, 2, 3, 4, 5])
    extracted_data = extract_event_data(seq_name, start_idx, end_idx, event_dir, pair_dir, groupnum=5)
    print(extracted_data)
    temp = np.concatenate(extracted_data, axis=0)
    for i in range(100):
        start = time.time()
        extracted_data_ = extract_event_data(seq_name, start_idx, end_idx, event_dir, pair_dir, groupnum=1)
        end = time.time()
        print("total", end - start)
    print(extracted_data_)
    print(temp == extracted_data_[0])
