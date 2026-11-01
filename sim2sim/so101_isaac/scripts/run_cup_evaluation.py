"""Supervise simulation evaluation and the isolated policy environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--dataset',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--sim-device',default='cuda:0')
parser.add_argument('--policy-device',default='cuda:1')
parser.add_argument('--episodes',type=int,default=20)
parser.add_argument('--seconds',type=float,default=40)
args=parser.parse_args()
if not 1 <= args.episodes <= 20 or args.seconds <= 0:
    parser.error('episodes must be in [1,20]; seconds must be positive')
out=args.output.resolve()
out.mkdir(parents=True,exist_ok=False)
checkpoint=args.checkpoint.resolve()
(ROOT/'outputs').mkdir(exist_ok=True)
socket=str(out/'policy.sock')
environment=os.environ.copy()
environment.update(OMNI_KIT_ACCEPT_EULA='YES',LEISAAC_TASK_SCOPE='lift_cube',
    LEISAAC_ASSETS_ROOT=str(ROOT/'assets'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
server_cmd=[str(ROOT/'.venv-train/bin/python'),str(ROOT/'scripts/cup_policy_server.py'),
    '--checkpoint',str(checkpoint),'--dataset',str(args.dataset.resolve()),
    '--socket',socket,'--output',str(out),'--device',args.policy_device]
sim_cmd=[str(ROOT/'.venv-sim/bin/python'),str(ROOT/'scripts/evaluate_cup_policy.py'),
    '--output',str(out),'--socket',socket,'--device',args.sim_device,'--livestream','0',
    '--episodes',str(args.episodes),'--seconds',str(args.seconds)]
processes=[]
info={'output':str(out),'checkpoint':str(checkpoint),'server_command':server_cmd,'sim_command':sim_cmd,
      'supervisor_pid':os.getpid(),'status':'starting'}
pointer=ROOT/'outputs/active_cup_evaluation.json'
def status():
    pointer.write_text(json.dumps(info,indent=2)+'\n')
    (out/'process.json').write_text(json.dumps(info,indent=2)+'\n')
try:
    with (out/'policy.log').open('w') as log:
        server=subprocess.Popen(server_cmd,env=environment,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    processes.append(server);info['policy_pid']=server.pid;status()
    started=time.monotonic()
    while not (out/'policy_ready.json').exists():
        if server.poll() is not None: raise RuntimeError(f'Policy exited {server.returncode}')
        if time.monotonic()-started>300: raise TimeoutError('Policy load timeout')
        time.sleep(1)
    with (out/'sim.log').open('w') as log:
        sim=subprocess.Popen(sim_cmd,env=environment,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    processes.append(sim);info.update(sim_pid=sim.pid,status='running');status()
    rc=sim.wait(timeout=10800)
    if rc != 0: raise RuntimeError(f'Simulator exited {rc}; see {out}/sim.log')
    info.update(sim_returncode=rc,status='complete' if rc==0 else 'failed');status()
except BaseException as exc:
    info.update(status='failed',error=repr(exc));status();raise
finally:
    for proc in reversed(processes):
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
