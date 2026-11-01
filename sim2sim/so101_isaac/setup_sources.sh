#!/usr/bin/env bash
# Fetch pinned upstream source; no simulator/robot is started.
set -euo pipefail
cd "$(dirname "$0")"
[[ ! -e IsaacLab && ! -e leisaac ]] || { echo 'Upstream directories already exist; inspect them before retrying.' >&2; exit 1; }
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab
git -C IsaacLab checkout --detach e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4
git clone https://github.com/LightwheelAI/leisaac.git leisaac
git -C leisaac checkout --detach 24d3bcd3f1e4585740fc79921782c41617237812
git -C leisaac apply --check ../patches/leisaac-isaaclab-v3.patch
git -C leisaac apply ../patches/leisaac-isaaclab-v3.patch
