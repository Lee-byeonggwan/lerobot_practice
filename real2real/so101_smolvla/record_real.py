"""Record 20 teleoperated episodes using the same units/cameras as run_real.py."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from settings import load_settings, select_follower

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--episode-seconds', type=float, default=30)
    p.add_argument('--reset-seconds', type=float, default=30)
    a = p.parse_args()
    c = load_settings(required=True)
    from serial.tools import list_ports
    port = select_follower(c, list_ports.comports()).device
    if not c['leader_port'] or c['leader_port'] == port:
        p.error('Set leader_port to the distinct SO-101 leader port in config.local.json')
    if a.root.exists():
        raise FileExistsError(a.root)
    cameras = {key: dict(type='opencv', index_or_path=value, width=640, height=480,
                        fps=30, fourcc='MJPG') for key,value in c['cameras'].items()}
    command = [sys.executable, '-m', 'lerobot.scripts.lerobot_record',
        '--robot.type=so101_follower', f'--robot.port={port}', f'--robot.id={c["robot_id"]}',
        '--robot.use_degrees=true', '--robot.cameras='+json.dumps(cameras),
        '--teleop.type=so101_leader', f'--teleop.port={c["leader_port"]}',
        f'--teleop.id={c["leader_id"]}', '--teleop.use_degrees=true',
        f'--dataset.repo_id={c["dataset_repo_id"]}', f'--dataset.root={a.root.resolve()}',
        f'--dataset.single_task={c["task"]}', '--dataset.num_episodes=20', '--dataset.fps=30',
        f'--dataset.episode_time_s={a.episode_seconds}', f'--dataset.reset_time_s={a.reset_seconds}',
        '--dataset.push_to_hub=false', '--display_data=false']
    return subprocess.run(command).returncode

if __name__ == '__main__':
    raise SystemExit(main())
