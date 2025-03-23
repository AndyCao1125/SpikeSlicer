## example: python run_tracker_fe108.py --tracker_name dimp --tracker_param dimp50 --trained_model Dimp50_small_ep0050.pth.tar --eval_save_dir dimp50_spikeslicer --event_representation frame --snn_type small
import os
import sys
import argparse
import torch
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import get_dataset
from pytracking.evaluation.running import run_dataset
from pytracking.evaluation import Tracker
from spikingjelly.activation_based import functional
import numpy as np
from pytracking.utils.load_text import load_text
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import time
from collections import OrderedDict
import importlib
from SpikeSlicer_utils.SpikeSliceNet import SpikeSlicerNet_S_IF, SpikeSlicerNet_B_IF

import warnings
warnings.filterwarnings("ignore")
import random
import copy
from SpikeSlicer_utils.quantization_layer import QuantizationLayer
from SpikeSlicer_utils.SpikeSliceNet import reset_net


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)



def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='eotb', sequence=None, debug=0, threads=0,
                visdom_info=None):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
        visdom_info: Dict optionally containing 'use_visdom', 'server' and 'port' for Visdom visualization.
    """

    visdom_info = {} if visdom_info is None else visdom_info

    dataset = get_dataset(dataset_name)

    # print(f'dataset: {dataset}')
    if sequence == 'train':
        train_list = [f.strip() for f in open('eotb_train_split.txt', 'r').readlines()]
        dataset = [dataset[i] for i in train_list]
    elif sequence == 'val':
        val_list = [f.strip() for f in open('eotb_val_split.txt', 'r').readlines()]
        dataset = [dataset[i] for i in val_list]
    elif sequence == 'mix':
        mix_list = [f.strip() for f in open('eotb_mix_split.txt', 'r').readlines()]
        dataset = [dataset[i] for i in mix_list]
    elif sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, run_id)]

    run_dataset(dataset, trackers, debug, threads, visdom_info=visdom_info)

def save_tracker_output(seq: Sequence, tracker: Tracker, output: dict, save_dir):
    """Saves the output of the tracker."""

    save_results_dir = os.path.join(save_dir)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    
    base_results_path = os.path.join(save_results_dir, seq.name)

    frame_names = [os.path.splitext(os.path.basename(f))[0] for f in seq.frames]

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)
            
            print(f'Result is saved at: {bbox_file}')


        elif key == 'frame_idx':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_frameidx.txt'.format(base_results_path, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = '{}_frameidx.txt'.format(base_results_path)
                save_time(timings_file, data)



def run_tracker_subseq(tracker_name, tracker_param, seq):
    tracker = Tracker(tracker_name, tracker_param)
    output = tracker.run_sequence(seq)
    return output

def track_sub_sequence(tracker, seq, quantization_layer, init_info, snn_net, groupnum, device):
        
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i
        # segmentation[i] is the segmentation mask for frame i (numpy array)

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i
        # segmentation[i] is the multi-label segmentation mask for frame i (numpy array)

        output = {'target_bbox': [],
                  'time': [],
                  'segmentation': [],
                  'frame_idx': []}

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        def _read_image_npy(image_file: str):
            im = np.load(image_file)
            return im.transpose(1, 2, 0)  ## HWC

        def frame_path2events_path(frame_path):
            event_path = frame_path.replace('event_frame', 'raw_event')
            return event_path

        def get_united_bbox(boxes, start_idx, end_idx):
            # xywh to xyxy
            new_boxes = copy.deepcopy(boxes[start_idx:end_idx + 1])
            new_boxes[:, 2] = new_boxes[:, 0] + new_boxes[:, 2]
            new_boxes[:, 3] = new_boxes[:, 1] + new_boxes[:, 3]
            xmin, ymin, xmax, ymax = new_boxes[:, 0].min().item(), new_boxes[:, 1].min().item(), new_boxes[:,
                                                                                                 2].max().item(), new_boxes[
                                                                                                                  :,
                                                                                                                  3].max().item()
            return [xmin, ymin, xmax - xmin, ymax - ymin]

        # Initialize
        images = []
        events = []
        local_idx_perseq = []
        functional.set_step_mode(snn_net, 'm')
        for i in range(4 * groupnum):
            images.append(_read_image_npy(seq.frames[i]))  ## read numpy
            events.append(np.load(frame_path2events_path(seq.frames[i])))  ## lists of np-array raw events (N, 4)
        images = np.stack(images, 0).astype(np.float32)
        torch_image = torch.from_numpy(images.transpose(0, 3, 1, 2)).unsqueeze(1).to(device) # T, 1, C, H, W
        with torch.no_grad():
            spikes = snn_net(torch_image)
            
        if spikes.sum() > 0:
            spike_local_idx = spikes.squeeze().nonzero().min().item()
        else:
            spike_local_idx = 4 * groupnum - 1

        local_idx_perseq.append(spike_local_idx)

        ## reset SNN
        reset_net(snn_net)

        events = torch.from_numpy(np.concatenate(events[:spike_local_idx + 1], axis=0)).to(device)
        events = torch.cat([events, torch.zeros(events.shape[0], 1).to(torch.int64).to(events.device)], dim=1)
        event_representation = quantization_layer(events).squeeze(0).cpu().numpy().transpose(1, 2, 0)
        bbox = get_united_bbox(seq.ground_truth_rect, 0, spike_local_idx)
        init_info['init_bbox'] = bbox

        start_time = time.time()
        out = tracker.initialize(event_representation, init_info)

        if out is None:
            out = {}

        prev_output = OrderedDict(out)

        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time,
                        'segmentation': init_info.get('init_mask'),
                        'frame_idx': spike_local_idx}

        _store_outputs(out, init_default)
        end_idx = spike_local_idx
        start_idx = end_idx + 1

        while start_idx < len(seq.frames):
            
            start_time = time.time()
            images = []
            torch_images = []
            events = []

            ## drop last group
            if start_idx + 4* groupnum > len(seq.frames):
                break


            ## Step I: Construct event cells and feed them into the SNN to obtain the slicing time

            for i in range(4*groupnum):
                frame_path = seq.frames[start_idx + i]
                image = _read_image_npy(frame_path).astype(np.float32)
                events.append(np.load(frame_path2events_path(frame_path)))  ## lists of np-array raw events (N, 4)
                images.append(image)
                torch_image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)
                torch_images.append(torch_image)
            torch_images = torch.stack(torch_images, 0)

            functional.set_step_mode(snn_net, 'm')
            with torch.no_grad():
                spikes = snn_net(torch_images)
            if spikes.sum() > 0:
                spike_local_idx = spikes.squeeze().nonzero().min().item()
            else:
                spike_local_idx = 4*groupnum - 1

            # reset SNN
            reset_net(snn_net)

            ## Step II: Build the event representations from raw events with the output slicing time

            events = torch.from_numpy(np.concatenate(events[:spike_local_idx + 1], axis=0)).to(device)
            events = torch.cat([events, torch.zeros(events.shape[0], 1).to(torch.int64).to(events.device)], dim=1)
            event_representation = quantization_layer(events).squeeze(0).cpu().numpy().transpose(1, 2, 0)
            end_idx = start_idx + spike_local_idx

            local_idx_perseq.append(spike_local_idx)

            ## leverage the middle index to get the frame info (target)
            mid_idx = start_idx + (end_idx - start_idx) // 2
            start_idx = end_idx + 1
            info = seq.frame_info(end_idx)
            info['previous_output'] = prev_output


            ## Setp III: Tracking with the newly constructed event representation
            out = tracker.track(event_representation, info)

            out['frame_idx'] = end_idx
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

            segmentation = out['segmentation'] if 'segmentation' in out else None
            # if self.visdom is not None:
            #     tracker.visdom_draw_tracking(image, out['target_bbox'], segmentation)
            # elif tracker.params.visualization:
            #     self.visualize(image, out['target_bbox'], segmentation)

        for key in ['target_bbox', 'segmentation']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        mean_spkidx_perseq = np.mean(local_idx_perseq)
        print(f'Average spkidx of Seq {seq.name} is {mean_spkidx_perseq}')
        return output, mean_spkidx_perseq



def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name', type=str, default='dimp', help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default='dimp50', help='Name of parameter file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='eotb_test', help='Name of dataset (eotb_origin, eotb_train, eotb_test, eotb_val).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--use_visdom', type=bool, default=True, help='Flag to enable visdom.')
    parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom.')
    parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom.')
    parser.add_argument('--trained_model', type=str, default="Dimp50_ep0050.pth.tar", help='Checkpoint directory for trained tracker model.')
    parser.add_argument('--event_resolution', type=tuple, default=(260, 346), help='Event resolution.')
    parser.add_argument('--groupnum', type=int, default=5, help='Group number of event frames.')
    parser.add_argument('--eval_save_dir', type=str, default="dimp50_pytracking_small_snn_epoch50")
    parser.add_argument('--event_representation', type=str, default="frame", choices=['frame', 'EST', 'timesurface'])
    parser.add_argument('--snn_type', type=str, default="small", choices=['small', 'base'])


    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset = get_dataset(args.dataset_name, groupnum=args.groupnum)
    quantization_layer = QuantizationLayer(args.event_representation, args.event_resolution).to(device)
    print(f'--------Loading {args.event_representation} representation--------')
    print(f'--------Loading SpikeSlicer Model from {args.trained_model}--------')

    total_spkidx = []
    print(f'Evaluating SpikeSlicer on {len(dataset)} Seqences.')
    total_start = time.time()
    for idx, seq in enumerate(dataset):
        print(f'--->SpikeSlicer processing on: Seq-{idx+1}/{len(dataset)}-{seq.name}')
        seq_start = time.time()

        init_info = seq.init_info()

        # load parameter module
        tracker_module = importlib.import_module('pytracking.tracker.{}'.format(args.tracker_name))
        tracker_class = tracker_module.get_tracker_class()

        # load parameter module
        param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(args.tracker_name, args.tracker_param))
        params = param_module.parameters(args.trained_model)
        tracker = tracker_class(params)

        if args.snn_type == 'small':
            snn_net = SpikeSlicerNet_S_IF(args.event_resolution, output_num=1)
        elif args.snn_type == 'base':
            snn_net = SpikeSlicerNet_B_IF(args.event_resolution, output_num=1)
        else:
            raise Exception
        
        snn_net.load_state_dict(torch.load(args.trained_model)['snn'])
        snn_net.to(device)
        snn_net.eval()
        output, mean_spkidx = track_sub_sequence(tracker, seq, quantization_layer, init_info, snn_net, args.groupnum, device)

        total_spkidx.append(mean_spkidx)
        if isinstance(output['time'][0], (dict, OrderedDict)):
            exec_time = sum([sum(times.values()) for times in output['time']])
            num_frames = len(output['time'])
        else:
            exec_time = sum(output['time'])
            num_frames = len(output['time'])

        print('FPS: {}'.format(num_frames / exec_time))

        save_tracker_output(seq, tracker, output, args.eval_save_dir)

        seq_end = time.time()
        print(f'Seq-{idx+1}/{len(dataset)}-{seq.name} processing time: {(seq_end - seq_start)/60.0} min')

    total_end = time.time()
    print(f'Total time: {(total_end - total_start)/60.0} min')
    print(f'Total average spkidx:{np.mean(total_spkidx)}')


if __name__ == '__main__':
    main()
