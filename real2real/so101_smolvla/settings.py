"""Local, untracked configuration shared by the Windows runner and Linux server."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def load_settings(required=False):
    path = Path(os.environ.get('SO101_CONFIG', Path(__file__).with_name('config.local.json')))
    if not path.is_file():
        if required:
            raise FileNotFoundError('Copy config.example.json to config.local.json and set your robot, cameras and server model path.')
        path = Path(__file__).with_name('config.example.json')
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    if set(config['cameras']) != {'camera1', 'camera2', 'camera3'}:
        raise ValueError('Camera keys must match the trained three-camera model: camera1, camera2, camera3')
    return config

def select_follower(config, ports):
    serial, port = config.get('follower_usb_serial'), config.get('follower_port')
    if not serial and not port:
        raise ValueError('Set follower_usb_serial or follower_port in config.local.json; identify the follower, not the leader.')
    matches = [p for p in ports if (not serial or p.serial_number == serial) and (not port or p.device == port)]
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one follower matching the local configuration, found {len(matches)}')
    return matches[0]
