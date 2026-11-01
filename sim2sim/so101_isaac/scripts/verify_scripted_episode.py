"""Validate a recorded scripted episode and export a three-camera review video."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import h5py
import numpy as np
from PIL import Image,ImageDraw,ImageFont

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('folder',type=Path)
args=parser.parse_args()
folder=args.folder
result=json.loads((folder/'result.json').read_text())
assert result['success'] and result['recorded'],result
source=Path(result['episode_file'])
trace=json.loads((folder/'trace.json').read_text())
with h5py.File(source,'r') as f:
    g=f['data/demo_0'];n=len(g['actions'])
    actions=g['actions'][:];state=g['obs/joint_pos'][:];next_state=g['next_joint_pos'][:]
    assert actions.shape==state.shape==next_state.shape==(n,6)
    assert np.isfinite(actions).all() and np.isfinite(state).all()
    assert np.array_equal(g['frame_index'][:],np.arange(n))
    info=json.loads(f['data'].attrs['env_args'])
    episode_index=info.get('episode_index',0)
    assert np.all(g['episode_index'][:]==episode_index)
    assert np.allclose(g['timestamp'][:],np.arange(n)/30,atol=1e-9)
    assert np.allclose(next_state[:-1],state[1:],atol=1e-6),'Observation/action temporal continuity failed'
    assert np.allclose(actions,np.array([r['action'] for r in trace]),atol=1e-6)
    assert np.max(np.abs(np.diff(actions,axis=0)))<=.8/30+1e-6
    assert np.all(g['task_success'][-30:]),'Final second must remain successful'
    cameras={c:g['obs'][c][:] for c in ('top','wrist','front')}
    for c,v in cameras.items():assert v.shape==(n,480,640,3) and v.dtype==np.uint8,(c,v.shape,v.dtype)
    assert info['source']=='scripted privileged-state expert'
    assert info['no_object_teleport_during_trajectory']
    shapes={c:list(v.shape) for c,v in cameras.items()}
    success_attr=bool(g.attrs['success'])
    assert success_attr
    if info.get('folded_start'):
        assert trace[0]['phase']=='folded_wait'
        assert np.max(np.abs(state[0]-np.deg2rad([0,-95,70,85,0,10])))<.15
    if info.get('gentle_approach'):
        assert result['pregrasp_max_xy_displacement_m']<=.001
    transport=[r for r in trace if r['phase']=='transport']
    assert min(r['rim_clearance_m'] for r in transport)>.005

video=folder/f'episode_{episode_index:03d}_three_cameras.mp4'
font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
font=ImageFont.truetype(str(font_path),20) if font_path.exists() else ImageFont.load_default()
cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1920x512',
     '-framerate','30','-i','pipe:0','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','22',
     '-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
try:
    for i in range(n):
        canvas=np.zeros((512,1920,3),dtype=np.uint8)
        for j,c in enumerate(('top','wrist','front')):canvas[32:,j*640:(j+1)*640]=cameras[c][i]
        image=Image.fromarray(canvas);draw=ImageDraw.Draw(image)
        for j,label in enumerate(('TOP','WRIST','BELLY')):
            draw.text((j*640+8,4),f'{label} | {trace[i]["phase"]} | {i/30:.1f}s',font=font,fill='white')
        process.stdin.write(image.tobytes())
finally:
    process.stdin.close()
assert process.wait()==0,'Video encoding failed'
transport=[r for r in trace if r['phase']=='transport']
report={'valid':True,'episode_index':episode_index,'folded_start':info.get('folded_start',False),
        'gentle_approach':info.get('gentle_approach',False),
        'pregrasp_max_xy_displacement_m':result.get('pregrasp_max_xy_displacement_m'),
        'initial_joint_positions_rad':state[0].tolist(),'frames':n,'fps':30,'simulation_seconds':n/30,
        'observation_shape':list(state.shape),'action_shape':list(actions.shape),'cameras':shapes,
        'pre_action_observation_continuity':True,'action_rate_limit_passed':True,'final_second_success':True,
        'minimum_transport_rim_clearance_m':min(r['rim_clearance_m'] for r in transport),
        'peak_cube_lift_m':result['peak_cube_lift_m'],'stable_seconds':result['stable_seconds'],
        'hdf5':str(source),'bytes':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'review_video':str(video),'note':'RGB/state/action from simulation. Controller used privileged simulator object state; not an evaluated learned policy.'}
(folder/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
