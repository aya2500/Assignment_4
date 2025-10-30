import os
import sys
import argparse
import torch
import importlib

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

# Prefer GPU when available; fall back to CPU otherwise. This only changes device selection.
if torch.cuda.is_available():
    # Use CUDA as the default tensor type so modules created without explicit device
    # will be placed on GPU by default when possible.
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

from lib.test.evaluation.datasets import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker

def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset.
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    # If parameter module exposes multiple checkpoints and no run_id is provided,
    # create one Tracker per checkpoint (run_id is 1-based index).
    try:
        param_module = importlib.import_module(f'lib.test.parameter.{tracker_param}')
        params = param_module.parameters(tracker_param)
    except Exception:
        params = None

    trackers = []
    if params is not None and hasattr(params, 'checkpoints') and run_id is None:
        num_ckpts = len(params.checkpoints)
        for i in range(1, num_ckpts + 1):
            trackers.append(Tracker(tracker_name, tracker_param, dataset_name, i))
    else:
        trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='lasot', help='Name of dataset (otb, nfs, uav, got10k_test, lasot, trackingnet, lasot_extension_subset, tnl2k).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=8)

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus)


if __name__ == '__main__':
    main()