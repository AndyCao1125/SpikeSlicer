## example: python eval_results.py --tracker_name transt --tracker_param transt50 --eval_results_dir eval_results/transt50 --mode all

import os
import sys
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 8]

sys.path.append('../..')
from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist

from argparse import ArgumentParser
parser = ArgumentParser()

# Add arguments
parser.add_argument('--tracker_name', type=str, default='transt', help='Name of tracking method.')
parser.add_argument('--tracker_param', type=str, default='transt50', help='Name of parameter file.')
parser.add_argument('--eval_results_dir', type=str, default="", help='Path to the directory containing the results or path to zip file.')
parser.add_argument('--mode', type=str, default="all", choices=['hdr', 'll', 'fwb', 'fnb','all','single'], help='extract sequences in specfic mode')
args = parser.parse_args()



trackers = []

## load results after run_tracker_fe1080p.py
eval_results_dir = os.path.join(args.eval_results_dir)   

## loading trackers info
trackers.extend([Tracker(args.tracker_name, args.tracker_param, None, 'E-'+ args.tracker_param)])

## loading test dataset
dataset = get_dataset('eotb_test')

list_hdr = ["giraffe222", "bottle_mul222", "cow_mul222", "whale_mul222","cup222","star_mul222","bike222","elephant222","dove_mul222","airplane_mul222",]
list_ll = ["cup_low", "bike_low", "giraffe_low", "tank_low", "box_low",]
list_fwb = ["tower", "ship", "star", "truck", "dog"]
list_fnb = ["ship_motion", "giraffe_motion", "fighter_mul", "bike333", "dog_motion", "dove_motion", "star_motion", "tank_low"]
list_single = ["airplane_mul222" ]
dataset_mode = ['hdr', 'll', 'fwb', 'fnb','all','single']

## extract sequences in specfic mode
def extract_seq(dataset, mode):
    if mode == 'hdr':
        dataset = [dataset[s] for s in list_hdr]
    elif mode == 'll':
        dataset = [dataset[s] for s in list_ll]
    elif mode == 'fwb':
        dataset = [dataset[s] for s in list_fwb]
    elif mode == 'fnb':
        dataset = [dataset[s] for s in list_fnb]
    elif mode == 'all':
        dataset = dataset
    elif mode == 'single':
        dataset = [dataset[s] for s in list_single]
    print(f'-------Loading {mode} sequences with len {len(dataset)}-------')
    return dataset


dataset = extract_seq(dataset, args.mode)


print_results(trackers, dataset, 'eotb', merge_results=False, plot_types=('success', 'prec', 'norm_prec'), eval_dir = eval_results_dir)