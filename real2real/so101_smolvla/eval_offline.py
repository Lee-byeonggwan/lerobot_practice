"""Open-loop sanity check: compare predicted action chunks against the recorded demonstrations.

For a set of random frames, the policy sees the recorded cameras/state/task and predicts the next
50 actions; we report the mean absolute error (degrees for arm joints, 0-100 for the gripper)
against what the human actually did. A "hold still" baseline (repeat the current joint state)
is reported for scale. Note: these frames come from the training set, so this checks that the
checkpoint loads and fits the data, not how well it generalizes.
"""

import argparse

import numpy as np
import torch
from torch.utils.data import default_collate

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--dataset-root", default=None, help="Local dataset directory (no Hub upload needed)")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--episodes", type=int, nargs="+", default=None, help="only sample frames from these")
    p.add_argument("--num_frames", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.num_frames < 1 or args.batch_size < 1:
        p.error('num_frames and batch_size must be positive')

    dataset = LeRobotDataset(args.dataset, root=args.dataset_root)
    fps = dataset.meta.fps
    ep_index = np.array(dataset.hf_dataset["episode_index"])
    actions = np.stack(dataset.hf_dataset["action"])
    states = np.stack(dataset.hf_dataset["observation.state"])

    rng = np.random.default_rng(args.seed)
    candidates = np.arange(len(dataset))
    if args.episodes is not None:
        candidates = candidates[np.isin(ep_index, args.episodes)]
    if not len(candidates):
        p.error('No frames match the requested episodes')
    frames = np.sort(rng.choice(candidates, size=min(args.num_frames, len(candidates)), replace=False))

    for ckpt in args.checkpoints:
        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.pretrained_path = ckpt
        cfg.device = args.device
        policy = make_policy(cfg, ds_meta=dataset.meta)
        policy.eval()
        pre, post = make_pre_post_processors(cfg, pretrained_path=ckpt,
            preprocessor_overrides={"device_processor": {"device": args.device}},
            postprocessor_overrides={"device_processor": {"device": args.device}})
        chunk = cfg.chunk_size

        model_err, hold_err = [], []
        for start in range(0, len(frames), args.batch_size):
            idx = frames[start : start + args.batch_size]
            batch = pre(default_collate([dataset[int(i)] for i in idx]))
            with torch.no_grad():
                pred = post(policy.predict_action_chunk(batch)).float().cpu().numpy()
            for b, i in enumerate(idx):
                # Ground-truth future actions, truncated at the end of the episode.
                same_ep = np.nonzero(ep_index[i : i + chunk] == ep_index[i])[0]
                gt = actions[i : i + len(same_ep)]
                model_err.append(np.abs(pred[b, : len(gt)] - gt).mean(0))
                hold_err.append(np.abs(states[i][None] - gt).mean(0))

        model_err, hold_err = np.mean(model_err, 0), np.mean(hold_err, 0)
        print(f"\n== {ckpt}  ({len(frames)} frames, chunk {chunk} = {chunk / fps:.1f}s)")
        print(f"{'joint':<14}{'model MAE':>10}{'hold MAE':>10}")
        for j, name in enumerate(JOINTS):
            print(f"{name:<14}{model_err[j]:>10.2f}{hold_err[j]:>10.2f}")
        print(f"{'mean':<14}{model_err.mean():>10.2f}{hold_err.mean():>10.2f}")

        del policy
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
