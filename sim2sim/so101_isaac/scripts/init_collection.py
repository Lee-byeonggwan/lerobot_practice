"""Create an empty collection directory from the committed experiment plan."""
import argparse
import json
from pathlib import Path
import shutil

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    a = p.parse_args()
    a.dataset.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__).resolve().parents[1]/'configs/collection_plan.json',
                 a.dataset/'collection_plan.json')
    manifest = dict(target_successful_episodes=20, successful_episodes=0,
                    folded_start=True, gentle_approach=True, split='train',
                    pregrasp_displacement_limit_m=.001,
                    source='scripted privileged-state expert',
                    camera_alignment='provisional; not calibrated against real',
                    episodes=[], attempts=[])
    (a.dataset/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(a.dataset.resolve())

if __name__ == '__main__':
    main()
