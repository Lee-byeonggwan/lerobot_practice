"""Bounded contact-based cube-to-cup pilot. No serial port or Leader connection.

State-machine phases follow LeIsaac's approach/grasp/lift/transport/release pattern.
The original six joint-position actions, gravity, collisions and three RGB sensors are retained.
Only initial episode reset writes object poses; manipulation uses joint targets exclusively.
"""
import argparse
import hashlib
import json
import os
import signal
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ['LEISAAC_LIFTCUBE_SCENE']=str(ROOT/'assets/scenes/table_with_wood_cube_cup/collection_scene.usda')
import torch
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--record',action='store_true')
parser.add_argument('--grasp-x',type=float,default=.002)
parser.add_argument('--grasp-z',type=float,default=.054)
parser.add_argument('--close',type=float,default=-.10)
parser.add_argument('--hold-seconds',type=float,default=30)
parser.add_argument('--folded-start',action='store_true')
parser.add_argument('--cube-offset-x',type=float,default=0.)
parser.add_argument('--cube-offset-y',type=float,default=0.)
parser.add_argument('--episode-index',type=int,default=0)
parser.add_argument('--center-release',action='store_true',help='Align held cube center above cup before opening the gripper')
parser.add_argument('--gentle-approach',action='store_true',help='Open wider, settle above cube and reject pre-grasp displacement')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
assert args.device.startswith('cuda:') and int(args.device.split(':')[1]) >= 0
gpu_index=int(args.device.split(':')[1]);torch.cuda.set_device(gpu_index)
args.headless=True;args.hide_ui=False;args.enable_cameras=True;args.multi_gpu=False
args.width,args.height=1280,720
args.kit_args=(args.kit_args or '')+f' --/renderer/activeGpu={gpu_index} --/renderer/multiGpu/enabled=false'
args.output.mkdir(parents=True,exist_ok=False)
launcher=AppLauncher(args)
env=None;stop=False
def stopping(*_):
    global stop
    stop=True
signal.signal(signal.SIGTERM,stopping)
signal.signal(signal.SIGINT,stopping)
def json_default(value):
    if hasattr(value,'item'):
        return value.item()
    if isinstance(value,Path):
        return str(value)
    raise TypeError(f'Unsupported JSON value: {type(value)}')

def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,default=json_default)+'\n')
try:
    import itertools
    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation
    import omni.ui as ui
    import omni.usd
    from pxr import UsdGeom,Gf
    from omni.kit.viewport.utility import get_active_viewport,capture_viewport_to_file
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.datasets.episode_data import EpisodeData
    from isaaclab.utils.datasets.hdf5_dataset_file_handler import HDF5DatasetFileHandler
    from isaaclab.utils.math import quat_apply,quat_apply_inverse
    from cup_collection_config import make_cfg,CAMERAS,CUBE_REFERENCE_XY,INSTRUCTION
    from so101_scripted_kinematics import JointChain,NAMES
    from live_camera_panel import LiveCameraPanel

    cfg=make_cfg(args.device)
    env=ManagerBasedRLEnv(cfg=cfg);env.reset()
    robot,cube,cup=env.scene['robot'],env.scene['cube'],env.scene['cup']
    ids=[robot.joint_names.index(n) for n in NAMES]
    grip_body=robot.body_names.index('gripper')
    pose=cube.data.default_root_state.torch[:,:7].clone()
    pose[0,:3]=torch.tensor([CUBE_REFERENCE_XY[0]+args.cube_offset_x,CUBE_REFERENCE_XY[1]+args.cube_offset_y,.05191131225],device=args.device)
    cube.write_root_pose_to_sim(pose);cube.write_root_velocity_to_sim(torch.zeros((1,6),device=args.device))
    if args.folded_start:
        folded=torch.deg2rad(torch.tensor([0.,-95.,70.,85.,0.,10.],device=args.device))
        reset_q=robot.data.joint_pos.torch.clone();reset_q[0,ids]=folded
        robot.write_joint_state_to_sim(reset_q,torch.zeros_like(reset_q))
    action=robot.data.joint_pos.torch[:,ids].clone()
    if args.folded_start:action=folded[None].clone()
    for _ in range(30):env.step(action)
    chain=JointChain(ROOT/'assets/robots/so101_follower_purple.usda',
        robot.data.root_pos_w.torch[0].cpu().numpy(),robot.data.root_quat_w.torch[0].cpu().numpy(),tip=(args.grasp_x,0,-.095))
    q=robot.data.joint_pos.torch[0,ids].cpu().numpy()
    if args.folded_start:
        assert np.max(np.abs(q-folded.cpu().numpy()))<.15,('Folded pose failed to settle',q)
    _,rot,pred=chain.fk(q[:5])
    real=robot.data.body_pos_w.torch[0,grip_body].cpu().numpy()
    fk_error=float(np.linalg.norm(pred-real))
    print('FK_CHECK',fk_error,pred.tolist(),real.tolist(),flush=True)
    assert fk_error<.002,('FK must match physics before generating actions',fk_error,pred,real)
    stage=omni.usd.get_context().get_stage()
    cam=UsdGeom.Camera.Define(stage,'/ScriptedOverview')
    cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(.65,-.95,.55),Gf.Vec3d(.18,-.48,.13),Gf.Vec3d(0,0,1)).GetInverse())
    cam.CreateFocalLengthAttr(32)
    viewport=get_active_viewport();viewport.set_active_camera('/ScriptedOverview')
    panel=LiveCameraPanel(env);panel.update(force=True)
    window=ui.Window('Automatic Cube-to-Cup Pilot',width=600,height=140)
    with window.frame:
        with ui.VStack():
            ui.Label('SIM ONLY | physical contact grasp | no object teleport during trajectory')
            label=ui.Label('Planning...')
            ui.Button('Stop Pilot',clicked_fn=stopping)
    dt=float(env.step_dt)
    initial_cube=cube.data.root_pos_w.torch[0].cpu().numpy().copy()
    initial_cup=cup.data.root_pos_w.torch[0].cpu().numpy().copy()
    cx,cy=map(float,initial_cube[:2]);px,py=map(float,initial_cup[:2])
    # All lateral transport stays above the 7 cm cup rim (world z approximately .111 m).
    opening=.9 if args.gentle_approach else .65
    waypoints=[('approach',[cx,cy,.12],opening,5.0 if args.folded_start else 3.5),
               ('descend',[cx,cy,args.grasp_z],opening,3.5 if args.gentle_approach else 2.5),
               ('close',[cx,cy,args.grasp_z],args.close,1.5),
               ('lift',[cx,cy,.135],args.close,2.5),
               ('clearance',[cx,cy,.15],args.close,1.5),
               ('transport',[px,py,.15],args.close,4.0),
               ('release',[px,py,.15],opening,1.5),
               ('retract',[cx,cy,.13],opening,3.0),
               ('settle',[cx,cy,.13],opening,2.0)]
    if args.center_release:
        waypoints.insert(6,('align',[px,py,.15],args.close,1.5))
    if args.gentle_approach:
        waypoints.insert(1,('open_clear',[cx,cy,.12],opening,1.0))
    plan=[];seed=q[:5].copy()
    if args.folded_start:
        plan.append({'phase':'folded_wait','tip_target':chain.fk(q[:5])[0].tolist(),
                     'joint_target_rad':q.tolist(),'seconds':1.0})
    for name,pos,grip,duration in waypoints:
        seed=chain.solve(np.array(pos),seed)
        plan.append({'phase':name,'tip_target':pos,'joint_target_rad':[*seed,grip],'seconds':duration})
    source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ('scripts/scripted_cup_demo.py','scripts/so101_scripted_kinematics.py','scripts/cup_collection_config.py',
         'assets/scenes/table_with_wood_cube_cup/collection_scene.usda','assets/robots/so101_follower_purple.usda')}
    write_json(args.output/'plan.json',{'fk_error_m':fk_error,'phases':plan,'settings':vars(args)|{'output':str(args.output)},
        'language_instruction':INSTRUCTION,'action_units':'absolute radians, six joints',
        'state_machine_uses_privileged_object_positions':True,'gravity_and_collision_enabled':True,'source_sha256':source_hashes})
    print('PLAN_READY',json.dumps(plan,default=json_default),flush=True)
    episode=EpisodeData() if args.record else None
    trace=[];bias=np.zeros(5);last_command=action[0].cpu().numpy().copy()
    corners=torch.tensor(list(itertools.product((-.01,.01),repeat=3)),device=args.device)
    stable=0.;peak_lift=0.;pregrasp_drift=0.;lift_verified=False;transport_verified=False;success=False;reason='not completed'
    start=time.monotonic();frame=0;previous_tip=chain.fk(q[:5])[0];previous_grip=q[-1]
    def phase_images(name):
        for camera in CAMERAS:
            rgb=env.scene[camera].data.output['rgb'].torch[0,:,:,:3].cpu().numpy()
            Image.fromarray(rgb).save(args.output/(name+'_'+camera+'.png'))
        capture_viewport_to_file(viewport,str(args.output/(name+'_overview.png')))
    phase_images('initial')
    for entry in plan:
        name=entry['phase'];label.text='Running: '+name
        if name=='align':
            measured_cube=cube.data.root_pos_w.torch[0].cpu().numpy()
            measured_cup=cup.data.root_pos_w.torch[0].cpu().numpy()
            measured_q=robot.data.joint_pos.torch[0,ids[:5]].cpu().numpy()
            measured_tip=chain.fk(measured_q)[0]
            correction=np.clip(measured_cup[:2]-measured_cube[:2],-.015,.015)
            aligned_target=np.r_[measured_tip[:2]+correction,.15]
            aligned_q=chain.solve(aligned_target,measured_q)
            for future in plan:
                if future['phase'] in ('align','release'):
                    future['tip_target']=aligned_target.tolist()
                    future['joint_target_rad'][:5]=aligned_q.tolist()
            write_json(args.output/'release_alignment.json',{'cube_before':measured_cube.tolist(),
                'cup_before':measured_cup.tolist(),'tip_before':measured_tip.tolist(),
                'correction_xy_m':correction.tolist(),'aligned_tip_target':aligned_target.tolist()})
        goal=np.array(entry['tip_target']);steps=round(entry['seconds']/dt)
        desired_q=robot.data.joint_pos.torch[0,ids[:5]].cpu().numpy().copy()
        # Approach starts by joint-space interpolation; remaining phases interpolate tip paths.
        start_q=desired_q.copy();planned_q=np.array(entry['joint_target_rad'][:5]);grip_start=previous_grip
        for step in range(steps):
            if stop or time.monotonic()-start>180:raise RuntimeError('Pilot stopped or exceeded 180 seconds')
            tick=time.monotonic();u=min((step+1)/(steps*.8),1.);alpha=u*u*(3-2*u)
            if name in ('approach','folded_wait'):desired_q=(1-alpha)*start_q+alpha*planned_q
            elif name in ('open_clear','close','release','settle'):desired_q=planned_q.copy()
            else:desired_q=chain.solve((1-alpha)*previous_tip+alpha*goal,desired_q)
            measured=robot.data.joint_pos.torch[0,ids].cpu().numpy()
            # Small integral bias compensates gravity sag without disabling gravity.
            bias=np.clip(bias+.04*(desired_q-measured[:5]),-.15,.15)
            cmd=np.r_[desired_q+bias,grip_start+alpha*(entry['joint_target_rad'][-1]-grip_start)]
            cmd[:5]=np.clip(cmd[:5],chain.limits[:,0],chain.limits[:,1])
            cmd=np.clip(cmd,last_command-.8*dt,last_command+.8*dt)
            action=torch.tensor(cmd,dtype=torch.float32,device=args.device)[None]
            if episode is not None:
                obs={'joint_pos':robot.data.joint_pos.torch[0,ids].cpu().clone(),
                     'joint_vel':robot.data.joint_vel.torch[0,ids].cpu().clone(),
                     **{c:env.scene[c].data.output['rgb'].torch[0,:,:,:3].cpu().clone() for c in CAMERAS}}
                episode.add('obs',obs,clone=False);episode.add('actions',action[0].cpu())
                episode.add('timestamp',torch.tensor(frame*dt,dtype=torch.float64))
                episode.add('frame_index',torch.tensor(frame));episode.add('episode_index',torch.tensor(args.episode_index))
            env.step(action);last_command=cmd
            panel.update()
            cp=cube.data.root_pos_w.torch[0];cq=cube.data.root_quat_w.torch[0]
            if name in ('folded_wait','approach','open_clear','descend'):
                pregrasp_drift=max(pregrasp_drift,float(np.linalg.norm(cp[:2].cpu().numpy()-initial_cube[:2])))
            pp=cup.data.root_pos_w.torch[0];pq=cup.data.root_quat_w.torch[0]
            tip=chain.fk(robot.data.joint_pos.torch[0,ids[:5]].cpu().numpy())[0]
            peak_lift=max(peak_lift,float(cp[2])-float(initial_cube[2]))
            if name=='lift' and float(cp[2])>initial_cube[2]+.04 and np.linalg.norm(cp.cpu().numpy()-tip)<.04:lift_verified=True
            world_corners=quat_apply(cq.expand(8,4),corners)+cp
            cube_bottom=float(world_corners[:,2].min())
            cup_rim=float(pp[2])+.07
            if name=='transport' and lift_verified and float(torch.linalg.vector_norm(cp[:2]-pp[:2]))<.025 and cube_bottom>cup_rim+.005:transport_verified=True
            local=quat_apply_inverse(pq.expand(8,4),world_corners-pp)
            radius=.024+.010*local[:,2]/.07
            inside=bool(((local[:,2]>.001)&(local[:,2]<.069)&(local[:,:2].norm(dim=-1)<radius-.001)).all())
            upright=float(quat_apply(pq[None],torch.tensor([[0.,0.,1.]],device=args.device))[0,2])>.966
            slow=float(cube.data.root_lin_vel_w.torch.norm())<.02 and float(cube.data.root_ang_vel_w.torch.norm())<.3
            cup_slow=float(cup.data.root_lin_vel_w.torch.norm())<.02 and float(cup.data.root_ang_vel_w.torch.norm())<.3
            released=float(robot.data.joint_pos.torch[0,ids[-1]])>.35 and np.linalg.norm(cp.cpu().numpy()-tip)>.04
            cup_near=np.linalg.norm(pp[:2].cpu().numpy()-initial_cup[:2])<.025
            good=inside and upright and slow and cup_slow and released and cup_near and lift_verified and transport_verified
            stable=stable+dt if good else 0.
            success=stable>=1.
            row={'frame':frame,'phase':name,'cube':cp.cpu().tolist(),'cup':pp.cpu().tolist(),'tip':tip.tolist(),
                 'joint_pos':robot.data.joint_pos.torch[0,ids].cpu().tolist(),'action':cmd.tolist(),
                 'inside':inside,'released':released,'lift_verified':lift_verified,'transport_verified':transport_verified,'stable_s':stable,
                 'pregrasp_max_xy_displacement_m':pregrasp_drift,
                 'cube_bottom_world_z':cube_bottom,'cup_rim_world_z':cup_rim,'rim_clearance_m':cube_bottom-cup_rim}
            trace.append(row)
            if episode is not None:
                episode.add('next_joint_pos',robot.data.joint_pos.torch[0,ids].cpu().clone())
                episode.add('cube_pose',torch.cat((cp,cq)).cpu());episode.add('cup_pose',torch.cat((pp,pq)).cpu())
                episode.add('task_success',torch.tensor(success))
            frame+=1
            if step%30==0:
                write_json(args.output/'live_status.json',row)
            time.sleep(max(0,dt-(time.monotonic()-tick)))
        previous_tip=goal;previous_grip=entry['joint_target_rad'][-1]
        phase_images(name)
        print('PHASE_DONE',name,json.dumps(trace[-1],default=json_default),flush=True)
        if args.gentle_approach and name=='descend' and pregrasp_drift>.001:
            reason='cube moved more than 1 mm before gripper closed';break
        if name=='lift' and not lift_verified:
            reason='cube not lifted by contact grasp';break
        if name=='transport' and not transport_verified:
            reason='cube not transported above cup';break
    else:reason='verified lift, transport, release and stable placement' if success else 'final placement checks not satisfied'
    write_json(args.output/'trace.json',trace)
    result={'success':success,'reason':reason,'frames':frame,'peak_cube_lift_m':peak_lift,
            'gentle_approach':args.gentle_approach,'pregrasp_max_xy_displacement_m':pregrasp_drift,
            'lift_verified':lift_verified,'transport_verified':transport_verified,'stable_seconds':stable,
            'joint_order':NAMES,'recorded':args.record,'final':trace[-1]}
    if episode is not None:
        episode.pre_export();episode.success=success;episode.seed=cfg.seed
        path=args.output/('episode_000.hdf5' if success else 'failed_attempt.hdf5')
        h=HDF5DatasetFileHandler();h.create(str(path),env_name='SO101-WoodCube-Cup-Scripted-v1')
        h.add_env_args({'source':'scripted privileged-state expert','split':'train' if success else 'excluded',
                        'task':INSTRUCTION,'joint_order':NAMES,'action_units':'absolute radians',
                        'camera_keys':CAMERAS,'control_dt':dt,'no_object_teleport_during_trajectory':True,
                        'episode_index':args.episode_index,'folded_start':args.folded_start,
                        'gentle_approach':args.gentle_approach,'grasp_tip_offset_x_m':args.grasp_x,
                        'opening_radians':opening,'pregrasp_max_xy_displacement_m':pregrasp_drift,
                        'start_joint_positions_rad':q.tolist()})
        h.add_env_args({'source_sha256':source_hashes,'camera_alignment':'provisional; not calibrated against real',
            'observation_action_timing':'obs[t] is captured before applied action[t]; next_joint_pos[t] after one control step',
            'cube_initial_position_m':initial_cube.tolist(),'cup_initial_position_m':initial_cup.tolist(),
            'cameras':{c:{'position_m':getattr(cfg.scene,c).offset.pos,'orientation_xyzw':getattr(cfg.scene,c).offset.rot,
                         'convention':getattr(cfg.scene,c).offset.convention,'focal_length_mm':getattr(cfg.scene,c).spawn.focal_length,
                         'horizontal_aperture_mm':getattr(cfg.scene,c).spawn.horizontal_aperture} for c in CAMERAS}})
        h.write_episode(episode,demo_id=0);h.flush();h.close()
        import h5py
        with h5py.File(path,'r') as f:
            g=f['data/demo_0'];assert g['actions'].shape==(frame,6)
            for c in CAMERAS:assert g['obs'][c].shape==(frame,480,640,3)
        result['episode_file']=str(path)
    write_json(args.output/'result.json',result)
    print('PILOT_RESULT',json.dumps(result,default=json_default),flush=True)
    label.text=('SUCCESS: ' if success else 'FAILED: ')+reason
    end=time.monotonic()+min(max(args.hold_seconds,0),120)
    while launcher.app.is_running() and not stop and time.monotonic()<end:
        launcher.app.update();time.sleep(.02)
except BaseException as exc:
    import traceback
    traceback.print_exc()
    write_json(args.output/'error.json',{'error':repr(exc),'traceback':traceback.format_exc()})
finally:
    if env is not None:env.close()
    launcher.app.close()
