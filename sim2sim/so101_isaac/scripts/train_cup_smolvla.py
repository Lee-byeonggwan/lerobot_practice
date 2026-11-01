"""Launch an isolated, local-only SmolVLA LoRA run after dataset validation."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--steps', type=int, choices=[2000, 20000], default=20000)
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.run = args.run.resolve()
    if args.gpu < 0: parser.error('gpu must be nonnegative')
    validation = json.loads((args.dataset/'conversion_validation.json').read_text())
    assert validation['valid'] and validation['episodes'] == 20 and validation['all_train']
    provenance = json.loads((args.dataset/'conversion_provenance.json').read_text())
    args.run.mkdir(parents=True, exist_ok=False)
    output = args.run/'training'
    rename = {f'observation.images.{name}': f'observation.images.camera{i}'
              for i, name in enumerate(('top', 'wrist', 'belly'), start=1)}
    command = [sys.executable, '-m', 'lerobot.scripts.lerobot_train',
        '--policy.path=lerobot/smolvla_base', f'--dataset.repo_id={provenance["repo_id"]}',
        f'--dataset.root={args.dataset.resolve()}', '--dataset.eval_split=0.0',
        '--dataset.video_backend=pyav', '--batch_size=64', f'--steps={args.steps}',
        '--save_freq=2000', '--log_freq=100', '--num_workers=4', '--prefetch_factor=2',
        '--env_eval_freq=0', '--eval_steps=0', '--seed=20261101',
        '--peft.method_type=LORA', '--peft.r=64', '--peft.lora_alpha=64',
        '--policy.optimizer_lr=0.001', '--policy.scheduler_decay_lr=0.0001',
        '--policy.scheduler_decay_steps=20000', '--policy.scheduler_warmup_steps=1000',
        '--policy.device=cuda', '--policy.push_to_hub=false', '--wandb.enable=false',
        '--rename_map='+json.dumps(rename), f'--output_dir={output.resolve()}',
        '--job_name=smolvla_sim_cup_folded_train20_lora']
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false',
               PYTHONUNBUFFERED='1', TQDM_DISABLE='1')
    config = {'command': command, 'gpu': args.gpu, 'steps': args.steps, 'batch_size': 64,
              'all_20_episodes_train': True, 'task': provenance['task'], 'camera_rename_map': rename,
              'action_units': provenance['action_units'], 'state_units': provenance['state_units'],
              'joint_order': provenance['joint_order'], 'save_every_steps': 2000,
              'dataset_validation': validation, 'automatic_real_robot_execution': False,
              'environment_packages': {name: importlib.metadata.version(name) for name in
                 ('torch', 'lerobot', 'peft', 'transformers', 'h5py', 'av')},
              'environment_note': 'Independent training environment; see envs/requirements-train.txt.',
              'started_utc': datetime.now(timezone.utc).isoformat()}
    (args.run/'run_config.json').write_text(json.dumps(config, indent=2)+'\n')
    with (args.run/'train.log').open('w') as log:
        child = subprocess.Popen(command, cwd=args.run, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        (args.run/'process.json').write_text(json.dumps({'pid': child.pid, 'supervisor_pid': os.getpid(),
                        'status': 'running', 'log': str(args.run/'train.log')}, indent=2)+'\n')
        rc = child.wait()
    checkpoint = output/f'checkpoints/{args.steps:06d}'
    step_path = checkpoint/'training_state/training_step.json'
    final_step = json.loads(step_path.read_text())['step'] if step_path.exists() else None
    completed = rc == 0 and final_step == args.steps and (checkpoint/'pretrained_model/adapter_model.safetensors').exists()
    status = {'status': 'complete' if completed else 'failed', 'returncode': rc,
              'final_step': final_step, 'checkpoint': str(checkpoint),
              'finished_utc': datetime.now(timezone.utc).isoformat(),
              'learned_policy_success_rate_evaluated': False}
    (args.run/'result.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(status), flush=True)
    return 0 if completed else 1


if __name__ == '__main__':
    raise SystemExit(main())
