"""Resume a completed 2,000-step run on two GPUs; never stop other processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='002000 directory including training_state')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--gpus', type=int, nargs=2, default=[0,1])
    a = p.parse_args()
    checkpoint, run = a.checkpoint.resolve(), a.run.resolve()
    if len(set(a.gpus)) != 2 or min(a.gpus) < 0:
        p.error('Specify two different nonnegative GPU indices')
    step = json.loads((checkpoint/'training_state/training_step.json').read_text())
    if step['step'] != 2000 or step['batch_size'] != 64 or step['num_processes'] != 1:
        p.error('Expected a 2,000-step, single-GPU, batch-64 checkpoint')
    for name in ('optimizer_state.safetensors', 'rng_state.safetensors'):
        if not (checkpoint/'training_state'/name).is_file():
            raise FileNotFoundError(name)
    run.mkdir(parents=True, exist_ok=False)
    output = run/'training'
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
               '--nproc-per-node=2', '--module', 'lerobot.scripts.lerobot_train',
               f'--config_path={checkpoint}/pretrained_model/train_config.json', '--resume=true',
               '--steps=20000', '--batch_size=32', '--num_workers=4', f'--output_dir={output}',
               '--job_name=smolvla_sim_cup_folded_train20_lora_dual_gpu']
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,a.gpus)), OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    (run/'run_config.json').write_text(json.dumps(dict(command=command, gpus=a.gpus,
        effective_batch_size=64, resumed_from=str(checkpoint)), indent=2)+'\n')
    with (run/'train.log').open('w') as log:
        result = subprocess.run(command, env=env, cwd=run, stdout=log, stderr=subprocess.STDOUT)
    final = output/'checkpoints/020000'
    step_file = final/'training_state/training_step.json'
    final_step = json.loads(step_file.read_text())['step'] if step_file.exists() else None
    complete = result.returncode == 0 and final_step == 20000 and (final/'pretrained_model/adapter_model.safetensors').is_file()
    (run/'result.json').write_text(json.dumps(dict(status='complete' if complete else 'failed',
        returncode=result.returncode, final_step=final_step, checkpoint=str(final)), indent=2)+'\n')
    return 0 if complete else 1

if __name__ == '__main__':
    raise SystemExit(main())
