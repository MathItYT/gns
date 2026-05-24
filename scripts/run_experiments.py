#!/usr/bin/env python3
"""Run multiple training experiments with identical protocol across architectures

Usage:
  scripts/run_experiments.py --data_path PATH [--seeds 0,1,2] [--architectures gns,sparse_egnn] \
      [--ntraining_steps 1000] [--model_path models/] [--extra "--batch_size=4"]

The script launches `python -m gns.train` (via CLI) for each combination of
architecture and seed, creating a distinct model output directory per run.
"""
import argparse
import subprocess
import shlex
import os
from datetime import datetime

DEFAULT_ARCHS = ["gns", "sparse_egnn"]
DEFAULT_SEEDS = [0, 1, 2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', required=True)
    p.add_argument('--seeds', default=','.join(map(str, DEFAULT_SEEDS)))
    p.add_argument('--architectures', default=','.join(DEFAULT_ARCHS))
    p.add_argument('--ntraining_steps', type=int, default=None)
    p.add_argument('--model_path', default='models/')
    p.add_argument('--extra', default='', help='Extra flags to forward to training (quoted)')
    p.add_argument('--dry_run', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    archs = [a for a in args.architectures.split(',') if a.strip()]

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    for arch in archs:
        for seed in seeds:
            run_id = f"{arch}-seed{seed}-{timestamp}"
            out_dir = os.path.join(args.model_path, run_id)
            os.makedirs(out_dir, exist_ok=True)

            cmd = [
                'python3', '-m', 'gns.train',
                '--mode=train',
                f'--architecture={arch}',
                f'--model_path={out_dir}/',
                f'--data_path={args.data_path}',
                f'--seed={seed}',
            ]
            if args.ntraining_steps is not None:
                cmd.append(f'--ntraining_steps={args.ntraining_steps}')

            if args.extra:
                # split extra flags safely
                extra_flags = shlex.split(args.extra)
                cmd.extend(extra_flags)

            print('\nRunning:', ' '.join(cmd))
            if args.dry_run:
                continue

            # Prepare log files per run
            stdout_path = os.path.join(out_dir, 'train.stdout.log')
            stderr_path = os.path.join(out_dir, 'train.stderr.log')
            with open(stdout_path, 'wb') as out_f, open(stderr_path, 'wb') as err_f:
                # limit OpenMP threads to reduce resource contention
                env = os.environ.copy()
                env.setdefault('OMP_NUM_THREADS', '1')
                env.setdefault('MKL_NUM_THREADS', '1')
                proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, env=env)
                ret = proc.wait()
            if ret != 0:
                print(f"Run logs: stdout={stdout_path}, stderr={stderr_path}")
                raise SystemExit(f"Training process failed with exit code {ret}")

    print('All experiments finished.')


if __name__ == '__main__':
    main()
