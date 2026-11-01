"""Local IPC inference for the trained cup policy; no simulator or robot access."""
import argparse
import json
from multiprocessing.connection import Listener
from pathlib import Path
import time

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--dataset', type=Path, required=True)
p.add_argument('--socket', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--device', default='cuda:1')
a = p.parse_args()
torch.set_num_threads(4)
cfg = PreTrainedConfig.from_pretrained(a.checkpoint)
cfg.pretrained_path = a.checkpoint
cfg.device = a.device
provenance = json.loads((a.dataset/'conversion_provenance.json').read_text())
meta = LeRobotDatasetMetadata(provenance['repo_id'], root=a.dataset)
rename = {f'observation.images.{n}': f'observation.images.camera{i}'
          for i, n in enumerate(('top', 'wrist', 'belly'), 1)}
policy = make_policy(cfg, ds_meta=meta, rename_map=rename).eval()
pre, post = make_pre_post_processors(cfg, pretrained_path=str(a.checkpoint),
    preprocessor_overrides={'device_processor': {'device': a.device}},
    postprocessor_overrides={'device_processor': {'device': 'cpu'}})
# Keep the adapter loaded exactly as trained, with the checkpoint's own processors.
with Listener(a.socket, family='AF_UNIX', authkey=b'cup-sim-local-evaluation') as listener:
    Path(a.socket).chmod(0o600)
    (a.output/'policy_ready.json').write_text(json.dumps({
        'checkpoint': str(a.checkpoint), 'device': a.device,
        'chunk_size': cfg.chunk_size, 'n_action_steps': cfg.n_action_steps,
        'camera_mapping': rename, 'inputs': 'RGB and six measured joint radians only'}, indent=2))
    print('POLICY_READY', flush=True)
    with listener.accept() as conn:
        while True:
            request = conn.recv()
            if request['op'] == 'close':
                break
            if request['op'] == 'reset':
                policy.reset(); pre.reset(); post.reset()
                torch.manual_seed(request['seed'])
                conn.send({'ok': True})
                continue
            tick = time.monotonic()
            observation = {'observation.state': torch.from_numpy(request['state']).float(),
                           'task': 'Pick up the wooden cube and place it inside the cup.'}
            for name, rgb in request['images'].items():
                observation['observation.images.'+name] = torch.from_numpy(rgb).permute(2,0,1).float()/255.
            with torch.inference_mode():
                actions = post(policy.predict_action_chunk(pre(observation))).cpu().numpy()[0]
            assert actions.shape == (50,6) and np.isfinite(actions).all(), actions.shape
            conn.send({'actions': actions, 'inference_seconds': time.monotonic()-tick})
