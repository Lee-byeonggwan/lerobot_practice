"""Convert only verified manifest episodes, preserving every synchronized frame."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
CAMERAS = {'top': 'top', 'wrist': 'wrist', 'belly': 'front'}
TASK = 'Pick up the wooden cube and place it inside the cup.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    args = parser.parse_args()
    manifest = json.loads((args.source / 'manifest.json').read_text())
    audit = json.loads((args.source / 'dataset_validation.json').read_text())
    assert audit['valid'] and audit['successful_episodes'] == 20
    entries = manifest['episodes']
    assert [e['episode_index'] for e in entries] == list(range(20))
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    features = {
        'observation.state': {'dtype': 'float32', 'shape': (6,), 'names': JOINTS},
        'action': {'dtype': 'float32', 'shape': (6,), 'names': JOINTS},
        **{f'observation.images.{name}': {'dtype': 'video', 'shape': (480, 640, 3),
                                         'names': ['height', 'width', 'channels']} for name in CAMERAS},
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, root=args.output, fps=30, robot_type='so101_follower', features=features,
        streaming_encoding=True, encoder_threads=4, video_backend='pyav',
        rgb_encoder=RGBEncoderConfig(vcodec='h264', crf=18, preset='fast', g=2),
    )
    provenance = {'source': str(args.source.resolve()), 'repo_id': args.repo_id,
                  'task': TASK, 'split': 'train', 'joint_order': JOINTS,
                  'state_units': 'joint radians', 'action_units': 'absolute joint target radians',
                  'camera_mapping': CAMERAS, 'dropped_frames': 0, 'episodes': [],
                  'camera_alignment': 'provisional; not calibrated against real',
                  'real_robot_direct_deployment_ready': False,
                  'controller': 'scripted privileged simulator state expert',
                  'images': 'RGB; H264 CRF18 encoding; no overlays or spatial resizing'}
    for entry in entries:
        source = args.source / entry['file']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry['sha256']
        with h5py.File(source, 'r') as f:
            g = f['data/demo_0']
            assert bool(g.attrs['success'])
            state = g['obs/joint_pos'][:].astype(np.float32)
            action = g['actions'][:].astype(np.float32)
            n = len(action)
            assert n == entry['frames'] == 840
            assert np.isfinite(state).all() and np.isfinite(action).all()
            # Read in blocks to avoid repeatedly decompressing large HDF5 chunks.
            for start in range(0, n, 120):
                stop = min(start + 120, n)
                images = {name: g[f'obs/{src}'][start:stop] for name, src in CAMERAS.items()}
                for i in range(start, stop):
                    dataset.add_frame({'observation.state': state[i], 'action': action[i], 'task': TASK,
                        **{f'observation.images.{name}': value[i-start] for name, value in images.items()}})
            dataset.save_episode()
        provenance['episodes'].append({'episode_index': entry['episode_index'],
                                       'source_file': entry['file'], 'source_sha256': entry['sha256'],
                                       'frames': n})
        (args.output / 'conversion_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        print(f'SAVED {entry["episode_index"]+1}/20 ({n} frames)', flush=True)
    dataset.finalize()
    del dataset
    # Verify all numerical values and representative decoded RGB frames after reopening.
    dataset = LeRobotDataset(args.repo_id, root=args.output, video_backend='pyav')
    assert dataset.num_episodes == 20 and dataset.num_frames == 16800
    checks = []
    offset = 0
    for entry in entries:
        with h5py.File(args.source / entry['file'], 'r') as f:
            g = f['data/demo_0']; n = entry['frames']
            rows = dataset.hf_dataset.select(range(offset, offset+n))[:]
            np.testing.assert_array_equal(np.asarray(rows['observation.state']), g['obs/joint_pos'][:].astype(np.float32))
            np.testing.assert_array_equal(np.asarray(rows['action']), g['actions'][:].astype(np.float32))
            np.testing.assert_array_equal(np.asarray(rows['episode_index']), np.full(n, entry['episode_index']))
            np.testing.assert_array_equal(np.asarray(rows['frame_index']), np.arange(n))
            np.testing.assert_allclose(np.asarray(rows['timestamp']), np.arange(n)/30, atol=2e-6, rtol=0)
            for i in (0, 315, 600, 839):
                decoded = dataset[offset+i]
                assert decoded['task'] == TASK
                for name, src in CAMERAS.items():
                    rgb = decoded[f'observation.images.{name}'].permute(1,2,0).numpy()*255
                    original = g[f'obs/{src}'][i].astype(np.float32)
                    error = float(np.abs(rgb-original).mean())
                    assert error < 5, (entry['episode_index'], i, name, error)
                    checks.append(error)
        offset += n
        print(f'ROUNDTRIP {entry["episode_index"]+1}/20 PASS', flush=True)
    report = {'valid': True, 'episodes': 20, 'frames': 16800, 'fps': 30,
              'all_train': True, 'all_state_action_values_match': True,
              'timestamps_and_episode_boundaries_match': True,
              'decoded_image_checks': len(checks), 'max_decoded_rgb_mae_255': max(checks),
              'features': features, 'task': TASK}
    (args.output / 'conversion_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print('CONVERSION_VALIDATED', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
