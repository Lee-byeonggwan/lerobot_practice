"""Resume a bounded, two-GPU collection of independently verified simulation episodes.

Only this program's child process groups may be terminated on timeout. Existing
datasets, failed attempts, robot ports and calibration files are never modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', type=Path, required=True)
parser.add_argument('--plan', type=Path, required=True)
parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
parser.add_argument('--max-attempts', type=int, default=4)
parser.add_argument('--center-release', action='store_true')
parser.add_argument('--grasp-heights', type=float, nargs='+', default=[.052,.051,.053,.054])
parser.add_argument('--gentle-approach', action='store_true')
parser.add_argument('--grasp-x', type=float, default=.002)
args = parser.parse_args()
assert all(g >= 0 for g in args.gpus) and len(set(args.gpus)) == len(args.gpus)
assert 1 <= args.max_attempts <= 4
assert len(args.grasp_heights)>=args.max_attempts and all(.049<=z<=.055 for z in args.grasp_heights)
dataset = args.dataset.resolve()
plan = json.loads(args.plan.read_text())
assert len(plan) == 20 and [p['episode_index'] for p in plan] == list(range(20))
manifest_path = dataset / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
assert manifest['folded_start'] and manifest['target_successful_episodes'] == 20
lock = threading.Lock()
stop = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: stop.set())
signal.signal(signal.SIGINT, lambda *_: stop.set())
jobs = queue.Queue()
done = {entry['episode_index'] for entry in manifest['episodes']}
for entry in plan:
    if entry['episode_index'] not in done:
        jobs.put(entry)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def child_run(command, log_path, timeout, environment, status_path):
    with log_path.open('w') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
        atomic_json(status_path, {'pid': proc.pid, 'command': command, 'log': str(log_path)})
        started = time.monotonic()
        while proc.poll() is None:
            if stop.is_set() or time.monotonic() - started > timeout:
                # The process was created above in its own session; never target other jobs.
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                return -1
            stop.wait(1)
        return proc.returncode


def worker(gpu):
    environment = os.environ.copy()
    environment.update(OMNI_KIT_ACCEPT_EULA='YES', LEISAAC_TASK_SCOPE='lift_cube',
                       LEISAAC_ASSETS_ROOT=str(ROOT/'assets'), OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    while not stop.is_set():
        try:
            target = jobs.get_nowait()
        except queue.Empty:
            return
        index = target['episode_index']
        for attempt, grasp_z in enumerate(args.grasp_heights[:args.max_attempts]):
            if stop.is_set():
                return
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            folder = dataset/f'episode_{index:03d}_attempt_{attempt:02d}_{stamp}'
            log_path = dataset/(folder.name+'.log')
            command = [str(ROOT/'.venv-sim/bin/python'), str(ROOT/'scripts/scripted_cup_demo.py'),
                       '--device', f'cuda:{gpu}', '--livestream', '0', '--output', str(folder),
                       '--record', '--folded-start', '--hold-seconds', '0', '--episode-index', str(index),
                       '--cube-offset-x', str(target['offset_x_m']),
                       '--cube-offset-y', str(target['offset_y_m']), '--grasp-z', str(grasp_z)]
            if args.center_release:
                command.append('--center-release')
            if args.gentle_approach:
                command.append('--gentle-approach')
            command.extend(['--grasp-x',str(args.grasp_x)])
            with lock:
                print('START', index, 'GPU', gpu, 'attempt', attempt, target, flush=True)
            rc = child_run(command, log_path, 360, environment, dataset/f'worker_gpu{gpu}.json')
            result_file = folder/'result.json'
            result = json.loads(result_file.read_text()) if result_file.exists() else {}
            verified = False
            if rc == 0 and result.get('success'):
                verify_command = [str(ROOT/'.venv-sim/bin/python'),
                                  str(ROOT/'scripts/verify_scripted_episode.py'), str(folder)]
                rc = child_run(verify_command, folder/'verification.log', 180, environment,
                               dataset/f'worker_gpu{gpu}.json')
                verified = rc == 0 and (folder/'validation.json').exists()
            with lock:
                manifest.setdefault('attempts', []).append({
                    'episode_index': index, 'gpu': gpu, 'attempt': attempt,
                    'folder': folder.name, 'grasp_z_m': grasp_z, 'verified': verified,
                    'reason': result.get('reason', f'Process failed; see {log_path.name}')})
                if verified:
                    v = json.loads((folder/'validation.json').read_text())
                    assert v['valid'] and v['folded_start'] and v['episode_index'] == index
                    manifest['episodes'].append({
                        **target, 'file': str(Path(v['hdf5']).relative_to(dataset)),
                        'validation': str((folder/'validation.json').relative_to(dataset)),
                        'review_video': str(Path(v['review_video']).relative_to(dataset)),
                        'frames': v['frames'], 'fps': v['fps'], 'sha256': v['sha256']})
                    manifest['episodes'].sort(key=lambda e: e['episode_index'])
                    manifest['successful_episodes'] = len(manifest['episodes'])
                atomic_json(manifest_path, manifest)
                print('VERIFIED' if verified else 'FAILED', index, 'count',
                      len(manifest['episodes']), 'reason', result.get('reason'), flush=True)
            if verified:
                break
        jobs.task_done()


with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
    futures = [pool.submit(worker, gpu) for gpu in args.gpus]
    for future in futures:
        future.result()
complete = len(manifest['episodes']) == 20
atomic_json(dataset/'batch_result.json', {'complete': complete,
            'successful_episodes': len(manifest['episodes']),
            'missing_indices': sorted(set(range(20))-{e['episode_index'] for e in manifest['episodes']})})
print('BATCH_COMPLETE', complete, 'count', len(manifest['episodes']), flush=True)
raise SystemExit(0 if complete else 1)
