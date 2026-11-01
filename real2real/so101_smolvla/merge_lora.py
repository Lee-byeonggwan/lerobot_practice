"""Merge a LoRA checkpoint into smolvla_base and save a standalone policy folder.

The async-inference policy server (and anything else that calls `SmolVLAPolicy.from_pretrained`)
cannot load adapter-only checkpoints, so we bake the adapter into the base weights once.
The merged model is checked against the adapter model on a real dataset frame before saving.
"""

import argparse
import shutil
from pathlib import Path

import torch
from torch.utils.data import default_collate

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--dataset-root", default=None, help="Local dataset directory (no Hub upload needed)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    ckpt, out = Path(args.checkpoint), Path(args.output)

    if out.exists():
        raise FileExistsError(f"Output exists: {out}")
    dataset = LeRobotDataset(args.dataset, root=args.dataset_root)
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    cfg.device = args.device
    peft_policy = make_policy(cfg, ds_meta=dataset.meta)
    peft_policy.eval()
    device_override = {"device_processor": {"device": args.device}}
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=str(ckpt),
        preprocessor_overrides=device_override,
        postprocessor_overrides=device_override,
    )
    if not len(dataset):
        raise ValueError("Dataset has no frames")
    batch = pre(default_collate([dataset[min(100, len(dataset)-1)], dataset[min(3000, len(dataset)-1)]]))

    def predict(policy):
        torch.manual_seed(0)  # same flow-matching noise for both models
        with torch.no_grad():
            return post(policy.predict_action_chunk(batch)).float()

    before = predict(peft_policy)
    merged = peft_policy.merge_and_unload()
    after = predict(merged)
    diff = (before - after).abs().max().item()
    print(f"max |adapter - merged| action difference: {diff:.4f}")
    assert diff < 0.5, "merged model disagrees with the adapter model"

    merged.config.use_peft = False
    merged.config.pretrained_path = None
    merged.save_pretrained(out)
    for f in ckpt.glob("policy_*processor*"):
        shutil.copy(f, out / f.name)
    print(f"saved merged policy to {out}")


if __name__ == "__main__":
    main()
