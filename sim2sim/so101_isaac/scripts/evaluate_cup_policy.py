"""20 closed-loop simulation trials. Object states are used for scoring only."""
import argparse
import hashlib
import itertools
import json
from multiprocessing.connection import Client
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ['LEISAAC_LIFTCUBE_SCENE'] = str(ROOT/'assets/scenes/table_with_wood_cube_cup/collection_scene.usda')
import torch
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--socket', required=True)
p.add_argument('--episodes', type=int, default=20)
p.add_argument('--seconds', type=float, default=40.)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if not 1 <= a.episodes <= 20 or a.seconds <= 0:
    p.error('episodes must be in [1, 20] and seconds must be positive')
gpu_index = int(a.device.split(':')[1])
a.headless = True; a.enable_cameras = True; a.multi_gpu = False
a.kit_args = (a.kit_args or '')+f' --/renderer/activeGpu={gpu_index} --/renderer/multiGpu/enabled=false'
torch.cuda.set_device(gpu_index)
launcher = AppLauncher(a)
env = None
connection = None

def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)

try:
    import numpy as np
    from PIL import Image, ImageDraw
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
    from cup_collection_config import make_cfg, fixed_camera, CUBE_REFERENCE_XY
    from so101_scripted_kinematics import JointChain, NAMES

    cfg = make_cfg(a.device)
    cfg.scene.review = fixed_camera('review', (.65,-.95,.55), (.18,-.48,.13))
    cfg.scene.review.spawn.focal_length = 28
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()
    connection = Client(a.socket, family='AF_UNIX', authkey=b'cup-sim-local-evaluation')
    robot, cube, cup = env.scene['robot'], env.scene['cube'], env.scene['cup']
    ids = [robot.joint_names.index(n) for n in NAMES]
    chain = JointChain(ROOT/'assets/robots/so101_follower_purple.usda',
        robot.data.root_pos_w.torch[0].cpu().numpy(), robot.data.root_quat_w.torch[0].cpu().numpy())
    folded = torch.deg2rad(torch.tensor([0.,-95.,70.,85.,0.,10.],device=a.device))
    corners = torch.tensor(list(itertools.product((-.01,.01), repeat=3)),device=a.device)
    dt = float(env.step_dt)
    rng = np.random.default_rng(20261102)
    groups = [(0,0),(.01,0),(-.01,0),(0,.01),(0,-.01)]
    plan = [{'episode_index': i, 'seed': 2026110200+i, 'position_group': i//4,
             'offset_xy_m': (np.array(groups[i//4])+rng.uniform(-.002,.002,2)).tolist()}
            for i in range(a.episodes)]
    report = {'status': 'running', 'requested_episodes': a.episodes, 'episodes': [],
        'successes': 0, 'evaluation_type': 'sim-to-sim learned-policy closed-loop',
        'protocol': 'New jittered positions in the training distribution; fixed cup, folded start; no retries.',
        'horizon_sim_seconds': a.seconds, 'control_hz': 30, 'action_chunk_steps': 50,
        'simulation_pauses_during_inference': True,
        'success_criterion': 'Lift >=4 cm; cube fully inside upright cup, released, stationary for >=1 second at termination.',
        'floor_height_m': .001, 'floor_contact_tolerance_m': .0005,
        'strict_floor_metric_also_recorded': True,
        'source_sha256': {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [Path(__file__), ROOT/'scripts/cup_policy_server.py', ROOT/'scripts/cup_collection_config.py',
             ROOT/'assets/scenes/table_with_wood_cube_cup/collection_scene.usda',
             ROOT/'assets/robots/so101_follower_purple.usda']}}
    save(a.output/'evaluation_plan.json', plan)
    save(a.output/'summary.json', report)
    for trial in plan:
        index = trial['episode_index']
        folder = a.output/f'episode_{index:03d}'; folder.mkdir()
        env.reset(seed=trial['seed'])
        # Check scoring FK against the actual rigid body rather than assuming its frame.
        grip_body = robot.body_names.index('gripper')
        pose = cube.data.default_root_state.torch[:,:7].clone()
        pose[0,:3] = torch.tensor([CUBE_REFERENCE_XY[0]+trial['offset_xy_m'][0],
                                  CUBE_REFERENCE_XY[1]+trial['offset_xy_m'][1],.05191131225],device=a.device)
        cube.write_root_pose_to_sim(pose)
        cube.write_root_velocity_to_sim(torch.zeros((1,6),device=a.device))
        reset_q = robot.data.joint_pos.torch.clone(); reset_q[0,ids] = folded
        robot.write_joint_state_to_sim(reset_q,torch.zeros_like(reset_q))
        for _ in range(30): env.step(folded[None])
        assert float((robot.data.joint_pos.torch[0,ids]-folded).abs().max()) < .15
        fk_body = chain.fk(robot.data.joint_pos.torch[0,ids[:5]].cpu().numpy())[2]
        assert np.linalg.norm(fk_body-robot.data.body_pos_w.torch[0,grip_body].cpu().numpy()) < .002
        initial_cube = cube.data.root_pos_w.torch[0].clone()
        initial_cup = cup.data.root_pos_w.torch[0].clone()
        connection.send({'op': 'reset','seed': trial['seed']}); assert connection.recv()['ok']
        video = folder/'rollout.mp4'
        encoder = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
            '-video_size','1280x1024','-framerate','30','-i','pipe:0','-an','-c:v','libx264',
            '-threads','4','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
        trace = []; actions = []; latencies = []; stable = 0.; strict_stable = 0.; peak_lift = 0.; lifted = False
        limits_clipped = 0; success = False
        print('EPISODE_START', index, trial, flush=True)
        start = time.monotonic()
        try:
            for frame in range(round(a.seconds/dt)):
                state = robot.data.joint_pos.torch[0,ids].cpu().numpy().copy()
                images = {name:env.scene[sensor].data.output['rgb'].torch[0,:,:,:3].cpu().numpy().copy()
                          for name,sensor in [('top','top'),('wrist','wrist'),('belly','front'),('overview','review')]}
                if not len(actions):
                    connection.send({'op':'predict','state':state,
                                     'images':{k:images[k] for k in ('top','wrist','belly')}})
                    response = connection.recv(); actions = list(response['actions'])
                    latencies.append(response['inference_seconds'])
                predicted = actions.pop(0)
                command = predicted.copy()
                # Enforce only the simulator's joint bounds, never use object-state corrections.
                bounds = robot.data.joint_pos_limits.torch[0,ids].cpu().numpy()
                command = np.clip(command,bounds[:,0],bounds[:,1])
                limits_clipped += int(np.any(command != predicted))
                env.step(torch.tensor(command,device=a.device)[None])
                cp,cq = cube.data.root_pos_w.torch[0],cube.data.root_quat_w.torch[0]
                pp,pq = cup.data.root_pos_w.torch[0],cup.data.root_quat_w.torch[0]
                tip = chain.fk(robot.data.joint_pos.torch[0,ids[:5]].cpu().numpy())[0]
                peak_lift = max(peak_lift,float(cp[2]-initial_cube[2]))
                lifted = lifted or (float(cp[2]-initial_cube[2])>.04 and np.linalg.norm(cp.cpu().numpy()-tip)<.04)
                world = quat_apply(cq.expand(8,4),corners)+cp
                local = quat_apply_inverse(pq.expand(8,4),world-pp)
                radius = .024+.010*local[:,2]/.07
                strict_inside = bool(((local[:,2]>.001)&(local[:,2]<.069)&(local[:,:2].norm(dim=-1)<radius-.001)).all())
                # Cup floor is 1 mm thick. Allow 0.5 mm contact penetration tolerance;
                # keep the original wall/rim margins and also report the strict metric.
                inside = bool(((local[:,2]>.0005)&(local[:,2]<.069)&(local[:,:2].norm(dim=-1)<radius-.001)).all())
                upright = float(quat_apply(pq[None],torch.tensor([[0.,0.,1.]],device=a.device))[0,2])>.966
                slow = float(cube.data.root_lin_vel_w.torch.norm())<.02 and float(cube.data.root_ang_vel_w.torch.norm())<.3
                cup_slow = float(cup.data.root_lin_vel_w.torch.norm())<.02 and float(cup.data.root_ang_vel_w.torch.norm())<.3
                released = float(robot.data.joint_pos.torch[0,ids[-1]])>.35 and np.linalg.norm(cp.cpu().numpy()-tip)>.04
                cup_near = float((pp[:2]-initial_cup[:2]).norm())<.025
                good = inside and upright and slow and cup_slow and released and cup_near and lifted
                stable = stable+dt if good else 0.
                strict_stable = strict_stable+dt if good and strict_inside else 0.
                success = stable >= 1.
                row = {'frame':frame,'state':state.tolist(),'predicted_action':predicted.tolist(),
                       'applied_action':command.tolist(),'cube':cp.cpu().tolist(),'cup':pp.cpu().tolist(),
                       'inside':inside,'upright':upright,'released':bool(released),'lifted':bool(lifted),
                       'stable_s':stable,'strict_inside':strict_inside,'strict_stable_s':strict_stable,
                       'cube_bottom_z':float(world[:,2].min()),'cube_quat_xyzw':cq.cpu().tolist(),
                       'cup_quat_xyzw':pq.cpu().tolist(),'local_corners':local.cpu().tolist(),
                       'slow':slow,'cup_slow':cup_slow,'cup_near':cup_near}
                trace.append(row)
                canvas = np.zeros((1024,1280,3),dtype=np.uint8)
                for j,name in enumerate(('overview','top','wrist','belly')):
                    y,x = (j//2)*512,(j%2)*640
                    canvas[y+32:y+512,x:x+640] = images[name]
                img=Image.fromarray(canvas);draw=ImageDraw.Draw(img)
                for j,name in enumerate(('overview','top','wrist','belly')):
                    draw.text(((j%2)*640+8,(j//2)*512+8),f'{name.upper()} | learned policy | trial {index+1}/{a.episodes} | {frame/30:.1f}s',fill='white')
                encoder.stdin.write(img.tobytes())
                if frame==0: img.save(folder/'initial.png')
                if frame%150==0:
                    save(a.output/'live_status.json',{'episode':index+1,'frame':frame,'successes':report['successes'],'latest':row})
                    print('PROGRESS',index,frame,'lift',peak_lift,'inside',inside,flush=True)
                if success: break
            img.save(folder/'final.png')
        finally:
            encoder.stdin.close()
            assert encoder.wait()==0, 'Video encoding failed'
        result = {**trial,'success':bool(success),'frames':len(trace),'simulation_seconds':len(trace)*dt,
            'wall_seconds':time.monotonic()-start,'peak_lift_m':peak_lift,'stable_seconds':stable,
            'strict_floor_success':strict_stable>=1.,
            'clipped_action_frames':limits_clipped,'mean_inference_seconds':float(np.mean(latencies)),
            'video':str(video),'final':trace[-1],
            'reason':'stable released placement in cup' if success else 'time limit without stable released placement'}
        save(folder/'trace.json',trace);save(folder/'result.json',result)
        report['episodes'].append(result);report['successes']+=int(success)
        report['completed_episodes']=len(report['episodes'])
        report['success_rate']=report['successes']/len(report['episodes'])
        save(a.output/'summary.json',report)
        print('EPISODE_RESULT',json.dumps(result),flush=True)
    report['status']='complete';save(a.output/'summary.json',report)
    connection.send({'op':'close'})
except BaseException:
    import traceback
    save(a.output/'error.json',{'traceback':traceback.format_exc()})
    traceback.print_exc()
    raise
finally:
    if connection: connection.close()
    if env: env.close()
    launcher.app.close()
