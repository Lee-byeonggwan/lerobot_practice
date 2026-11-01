"""Train the original real-data LoRA recipe; requires a local 20-episode recording."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, help='LeRobot dataset repo ID; local files need no upload')
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args()
    info = json.loads((a.dataset_root/'meta/info.json').read_text())
    if info['total_episodes'] != 20 or info['fps'] != 30:
        p.error('This recipe expects 20 episodes at 30 FPS')
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.gpu < 0:
        p.error('gpu must be nonnegative')
    command = [sys.executable, '-m', 'lerobot.scripts.lerobot_train',
        '--policy.path=lerobot/smolvla_base', f'--dataset.repo_id={a.dataset}',
        f'--dataset.root={a.dataset_root.resolve()}', '--batch_size=64', '--steps=20000',
        '--save_freq=5000', '--log_freq=100', '--peft.method_type=LORA',
        '--peft.r=64', '--peft.lora_alpha=64', '--policy.optimizer_lr=1e-3',
        '--policy.scheduler_decay_lr=1e-4', f'--output_dir={a.output.resolve()}',
        '--job_name=smolvla_real_lora_20k', '--policy.device=cuda',
        '--policy.push_to_hub=false', '--wandb.enable=false']
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    return subprocess.run(command, env=env).returncode

if __name__ == '__main__':
    raise SystemExit(main())
