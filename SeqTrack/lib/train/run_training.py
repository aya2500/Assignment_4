import os
import sys
import argparse
import importlib
import cv2 as cv
import torch.backends.cudnn
import torch.distributed as dist

import random
import numpy as np

torch.backends.cudnn.benchmark = False

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import _init_paths
import lib.train.admin.settings as ws_settings


def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(script_name, config_name, cudnn_benchmark=True, local_rank=-1,
                 save_dir=None, base_seed=None, use_lmdb=False,
                 resume=None, phase="phase_1", repo_id=None):  # ✅ added repo_id
    """Run the train script."""
    if save_dir is None:
        print("save_dir dir is not given. Use the default dir instead.")

    cv.setNumThreads(0)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    print(f'script_name: {script_name}.py  config_name: {config_name}.yaml')

    if base_seed is not None:
        if local_rank != -1:
            init_seeds(base_seed + local_rank)
        else:
            init_seeds(base_seed)

    settings = ws_settings.Settings()
    settings.script_name = script_name
    settings.config_name = config_name
    settings.project_path = os.path.join('train', script_name, config_name)
    settings.local_rank = local_rank
    settings.save_dir = os.path.abspath(save_dir)
    settings.use_lmdb = use_lmdb
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    settings.cfg_file = os.path.join(prj_dir, f'experiments/{script_name}/{config_name}.yaml')

    # ---- START: ADDED FOR AUTOMATION ----
    settings.resume_checkpoint = resume
    settings.phase_name = phase
    settings.repo_id = repo_id  # ✅ added to settings
    # ---- END: ADDED FOR AUTOMATION ----

    expr_module = importlib.import_module('lib.train.train_script')
    expr_func = getattr(expr_module, 'run')
    expr_func(settings)


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--script', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--cudnn_benchmark', type=bool, default=True)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)

    # ---- START: ADDED FOR AUTOMATION ----
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--phase', type=str, default="phase_1")
    parser.add_argument('--repo_id', type=str, default=None,
                        help='Hugging Face repo ID, e.g. "ayamohamed2500/seqtrack-checkpoint"')  # ✅ added
    # ---- END: ADDED FOR AUTOMATION ----

    args = parser.parse_args()

    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(0)
    print("local_rank:", args.local_rank)

    if torch.cuda.is_available():
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Training on CPU.")
    print(f"SEED: {args.seed} (our TEAM Number)")

    run_training(args.script, args.config, cudnn_benchmark=args.cudnn_benchmark,
                 local_rank=args.local_rank, save_dir=args.save_dir, base_seed=args.seed,
                 use_lmdb=args.use_lmdb, resume=args.resume, phase=args.phase, repo_id=args.repo_id)  # ✅ pass repo_id


if __name__ == '__main__':
    main()
