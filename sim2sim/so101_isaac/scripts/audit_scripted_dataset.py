"""Audit the complete folded-start training manifest without modifying episodes."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('dataset',type=Path)
args=parser.parse_args()
root=args.dataset.resolve()
manifest=json.loads((root/'manifest.json').read_text())
entries=manifest['episodes']
assert manifest['split']=='train' and manifest['folded_start']
assert len(entries)==20 and [e['episode_index'] for e in entries]==list(range(20))
assert Counter(e['position_group'] for e in entries)=={0:4,1:4,2:4,3:4,4:4}
rows=[];starts=[];hashes=[]
for entry in entries:
    folder=(root/entry['file']).parent
    validation=json.loads((root/entry['validation']).read_text())
    assert validation['valid'] and validation['folded_start']
    assert validation['episode_index']==entry['episode_index']
    frames=validation['frames'];assert frames in (735,780,795,840) and validation['fps']==30
    if manifest.get('gentle_approach'):
        assert validation['gentle_approach'] and validation['pregrasp_max_xy_displacement_m']<=.001
    assert validation['minimum_transport_rim_clearance_m']>.005
    assert validation['stable_seconds']>=1
    source=root/entry['file']
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest==entry['sha256']==validation['sha256']
    hashes.append(digest)
    with h5py.File(source,'r') as f:
        g=f['data/demo_0'];info=json.loads(f['data'].attrs['env_args'])
        assert bool(g.attrs['success']) and info['folded_start']
        assert info['no_object_teleport_during_trajectory'] and info['split']=='train'
        assert np.all(g['episode_index'][:]==entry['episode_index'])
        assert np.all(g['task_success'][-30:])
        assert g['actions'].shape==g['obs/joint_pos'].shape==(frames,6)
        for camera in ('wrist','front','top'):
            ds=g['obs'][camera]
            assert ds.shape==(frames,480,640,3) and ds.dtype==np.uint8
            assert ds[0].max()>30 and ds[-1].max()>30
        start=g['obs/joint_pos'][0];starts.append(start)
        assert np.max(np.abs(start-np.deg2rad([0,-95,70,85,0,10])))<.15
        cube=np.array(info['cube_initial_position_m']);cup=np.array(info['cup_initial_position_m'])
        requested=np.array([.19+entry['offset_x_m'],-.43+entry['offset_y_m']])
        assert np.linalg.norm(cube[:2]-requested)<.002
        assert np.linalg.norm(cup[:2]-[.25,-.60])<.002
        rows.append({'episode_index':entry['episode_index'],'position_group':entry['position_group'],
            'cube_x_m':float(cube[0]),'cube_y_m':float(cube[1]),'frames':frames,'seconds':frames/30,
            'minimum_rim_clearance_m':validation['minimum_transport_rim_clearance_m'],
            'pregrasp_max_xy_displacement_m':validation.get('pregrasp_max_xy_displacement_m'),
            'stable_seconds':validation['stable_seconds'],'file':entry['file'],
            'video':entry['review_video']})
assert len(set(hashes))==20
assert len({(round(r['cube_x_m'],5),round(r['cube_y_m'],5)) for r in rows})==20
assert max(np.ptp(np.array(starts),axis=0))<.05,'Inconsistent initial folded posture'
with (root/'episodes.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
report={'valid':True,'successful_episodes':20,'all_split':'train','folded_start':True,
    'gentle_approach':manifest.get('gentle_approach',False),
    'maximum_pregrasp_xy_displacement_m':max((r['pregrasp_max_xy_displacement_m'] or 0) for r in rows),
    'frames_per_episode':sorted(set(r['frames'] for r in rows)),
    'total_frames':sum(r['frames'] for r in rows),'fps':30,
    'total_simulation_seconds':sum(r['seconds'] for r in rows),
    'camera_shapes':{k:[480,640,3] for k in ('wrist','front','top')},'state_dim':6,'action_dim':6,
    'action_units':'absolute joint radians','unique_cube_start_positions':20,
    'position_group_counts':dict(Counter(e['position_group'] for e in entries)),
    'minimum_transport_rim_clearance_m':min(r['minimum_rim_clearance_m'] for r in rows),
    'minimum_stable_seconds':min(r['stable_seconds'] for r in rows),
    'max_start_posture_spread_rad':float(np.ptp(np.array(starts),axis=0).max()),
    'dataset_bytes':sum((root/e['file']).stat().st_size for e in entries),
    'excluded_attempts':sum(not a['verified'] for a in manifest.get('attempts',[])),
    'format':'Isaac Lab EpisodeData HDF5; LeRobot conversion pending',
    'limitations':['Scripted expert uses simulator object state; this is not learned-policy evaluation.',
                  'Camera extrinsics/intrinsics are provisional, not calibrated against real.',
                  'Cube positions cover a small local area only.'],
    'episodes':rows}
(root/'dataset_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='episodes'},indent=2))
