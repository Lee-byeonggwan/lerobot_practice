"""Real-data SmolVLA runner: stock joint units, no sim mapping or policy clipping.

Start with other robot programs stopped and the arm resting in its demo start
pose. This program restores normal volatile motor limits, enables the robot,
and EXECUTES model actions. S/F/Q or Ctrl+C stops policy commands; support the
arm at the final prompt before torque is released.
"""

import argparse
import functools
import hashlib
import json
import math
import pickle
from queue import Empty, Queue
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
from settings import ROOT, load_settings, select_follower
SETTINGS = load_settings()
CAMERAS = SETTINGS['cameras']
MODEL = SETTINGS['server_model_path']
TASK = SETTINGS['task']
RUNNER_REVISION = 'startup-v4'


def install_setup_read_retries(bus):
    """Use the SDK's bounded communication retries on the preparation bus only."""
    if getattr(bus, '_real_setup_read_retries', False):
        return
    original = bus.read

    @functools.wraps(original)
    def read(*args, **kwargs):
        kwargs.setdefault('num_retry', 2)  # Three total attempts; motor faults still raise.
        return original(*args, **kwargs)

    bus.read = read
    bus._real_setup_read_retries = True


def release_setup_torque(bus, path):
    """Try every joint even if one fails; record exactly what was verified."""
    report = {}
    for n in NAMES:
        try:
            bus.write('Torque_Enable', n, 0, normalize=False, num_retry=2)
            value = bus.read('Torque_Enable', n, normalize=False, num_retry=2)
            report[n] = {'torque': value, 'off_verified': value == 0}
        except Exception as exc:
            report[n] = {'off_verified': False, 'error': repr(exc)}
    path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    if not all(r['off_verified'] for r in report.values()):
        raise RuntimeError('Motor OFF unverified. Support arm and disconnect motor power. '
                           f'Details: {path.name}')


def prepare_bus(bus, saved_calibration, backup_path, confirm_release=None, confirm_pose=None):
    """Restore RAM while OFF; then latch each current pose as a separate phase."""
    install_setup_read_retries(bus)
    def read_all(reg):
        return {n: bus.read(reg, n, normalize=False) for n in NAMES}

    trace_path = backup_path.with_name('startup_trace.jsonl')
    bus._real_target_write_started = False

    def check_torque(stage, latched=()):
        with trace_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'time': time.time(), 'stage': stage, 'event': 'read_begin'})+'\n')
        torque = read_all('Torque_Enable')
        with trace_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'time': time.time(), 'stage': stage,
                                'torque': torque, 'latched': list(latched)})+'\n')
        unexpected = {n: value for n, value in torque.items()
                      if value not in (0, 1) or (value != 0 and n not in latched)}
        if unexpected:
            raise RuntimeError(f'Startup torque changed at {stage}: {unexpected}. '
                               f'No further targets sent. See {trace_path.name}')
        return torque

    actual_calibration = {n: asdict(c) for n, c in bus.read_calibration().items()}
    if actual_calibration != saved_calibration:
        raise RuntimeError('Calibration mismatch: stopped without recalibrating')
    before = {reg: read_all(reg) for reg in (
        'Torque_Enable', 'Operating_Mode', 'Status', 'Max_Torque_Limit', 'Goal_Velocity',
        'Torque_Limit', 'Goal_Position', 'Present_Position', 'Acceleration',
    )}
    if any(before['Operating_Mode'].values()) or any(before['Status'].values()):
        raise RuntimeError('Unexpected motor mode or fault; inspect before running')
    expected_max = {n: 500 if n == 'gripper' else 1000 for n in NAMES}
    if before['Max_Torque_Limit'] != expected_max:
        raise RuntimeError('Stored maximum torque differs from the existing real setup')
    with backup_path.open('x', encoding='utf-8') as f:
        json.dump({'calibration': actual_calibration, 'before': before}, f, indent=2)
    if any(before['Torque_Enable'].values()):
        if confirm_release is None:
            raise RuntimeError('Torque remains ON. Use the supervised startup prompt to release it.')
        confirm_release()  # Cancelling here must leave pre-existing torque unchanged.
        for n in NAMES:
            bus.write('Torque_Enable', n, 0, normalize=False)
            if bus.read('Torque_Enable', n, normalize=False) != 0:
                raise RuntimeError(f'{n}: torque OFF unverified; support arm and disconnect motor power')
    check_torque('after_release')
    while True:
        if confirm_pose is not None:
            confirm_pose()
        check_torque('after_pose_prompt')
        positions = read_all('Present_Position')
        invalid = {n: {'position': pos, 'min': saved_calibration[n]['range_min'],
                       'max': saved_calibration[n]['range_max']}
                   for n, pos in positions.items()
                   if not saved_calibration[n]['range_min'] <= pos <= saved_calibration[n]['range_max']}
        with trace_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'time': time.time(), 'stage': 'pose_validation',
                                'positions': positions, 'outside_range': invalid})+'\n')
        if not invalid:
            break
        detail = ', '.join(f"{n}: 현재 {p['position']}, 범위 {p['min']}~{p['max']}"
                           for n, p in invalid.items())
        if confirm_pose is None:
            raise RuntimeError('Starting pose outside calibration: '+detail)
        check_torque('before_pose_retry')
        print('시작 자세를 다시 맞춰 주세요. 모터 힘은 OFF이고 목표 명령은 아직 보내지 않았습니다.', flush=True)
        print(detail, flush=True)
        print('해당 관절을 끝까지 접거나 펴지 말고 학습 때의 시작 자세로 옮겨 주세요. Ctrl+C로 취소 가능합니다.', flush=True)
    # Complete ALL speed/limit changes before writing ANY position target.
    for n in NAMES:
        check_torque(f'before_ram:{n}')
        # Match the other real joints and the stored power-on torque limits.
        for reg, value in (('Goal_Velocity', 0), ('Torque_Limit', expected_max[n])):
            if bus.read(reg, n, normalize=False) != value:
                bus.write(reg, n, value, normalize=False)
                check_torque(f'after_ram:{n}:{reg}')
            if bus.read(reg, n, normalize=False) != value:
                raise RuntimeError(f'{n} {reg}: readback failed')
    check_torque('ram_complete')
    latched = []
    for n in NAMES:
        # A servo may engage on a goal-position write. Only that servo can be ON
        # from this point; every not-yet-latched servo must still be OFF.
        check_torque(f'before_latch:{n}', latched)
        pos = bus.read('Present_Position', n, normalize=False)
        c = saved_calibration[n]
        if not c['range_min'] <= pos <= c['range_max']:
            raise RuntimeError(f'{n}: pose moved outside calibrated range '
                               f'(position={pos}, range={c["range_min"]}..{c["range_max"]})')
        bus._real_target_write_started = True
        bus.write('Goal_Position', n, pos, normalize=False)
        if bus.read('Goal_Position', n, normalize=False) != pos:
            raise RuntimeError(f'{n}: current-position target readback failed')
        latched.append(n)
        torque = check_torque(f'after_latch:{n}', latched)
        with trace_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'time': time.time(), 'stage': 'latched_position',
                               'motor': n, 'raw_target': pos, 'torque': torque[n]})+'\n')
    if {n: asdict(c) for n, c in bus.read_calibration().items()} != actual_calibration:
        raise RuntimeError('Calibration changed unexpectedly')
    torque = check_torque('prepare_complete', latched)
    print('시작 자세 기록 완료 | 모터 ON:', ', '.join(n for n, v in torque.items() if v) or '없음', flush=True)
    return before


def encode_camera(frame):
    """Losslessly transmit native RGB; keep training geometry and pixel values."""
    import cv2
    import numpy as np
    if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
        raise RuntimeError(f'Expected native 640x480 RGB uint8, got {frame.shape} {frame.dtype}')
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError('Camera PNG encoding failed')
    return {'encoding': 'png_rgb_v1', 'data': encoded.tobytes()}


def make_client_class(base):
    class RealClient(base):
        def __init__(self, config, event_path):
            super().__init__(config)
            self.io_lock = threading.RLock()
            self.log_lock = threading.Lock()
            self.capture_lock = threading.Lock()
            self.latest_state = {}
            self.last_target = {}
            self.upload_seconds = 0.0
            self.observer = None
            self.error = None
            self.outcome = None
            self.stop_reason = None
            self.first_action_time = None
            self.action_count = 0
            self._dequeued_step = -1
            self.recording = None
            self.episode_seconds = 30.0
            self.poll_key = lambda: None
            self.events = event_path.open('x', encoding='utf-8', buffering=1)
            # Reads and writes share one half-duplex motor bus. Cameras/network
            # operate outside this lock and cannot hold up action execution.
            for name in ('read', 'write', 'sync_read', 'sync_write'):
                original = getattr(self.robot.bus, name)

                @functools.wraps(original)
                def locked(*args, _method=original, **kwargs):
                    with self.io_lock:
                        return _method(*args, **kwargs)

                setattr(self.robot.bus, name, locked)

        def event(self, kind, **values):
            with self.log_lock:
                self.events.write(json.dumps({'time': time.time(), 'kind': kind, **values}) + '\n')

        def start(self):
            from lerobot.transport import services_pb2
            _, call = self.stub.Ready.with_call(services_pb2.Empty(), timeout=10)
            metadata = dict(call.trailing_metadata())
            if metadata.get('real-image-pipeline') != 'native-rgb-v1':
                raise RuntimeError('Wrong policy server: native real-image pipeline is required')
            self.stub.SendPolicyInstructions(
                services_pb2.PolicySetup(data=pickle.dumps(self.policy_config)), timeout=120)
            self.event('server', metadata=metadata)
            self.shutdown_event.clear()
            return True

        def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
            """Keep the queue snapshot/merge atomic relative to motor ticks."""
            import torch
            from lerobot.async_inference.helpers import TimedAction
            with self.action_queue_lock:
                with self.latest_action_lock:
                    cutoff = max(self.latest_action, self._dequeued_step)
                previous = {a.get_timestep(): a.get_action() for a in self.action_queue.queue}
                merged = Queue()
                for action in incoming_actions:
                    step, value = action.get_timestep(), action.get_action()
                    if value.shape != (6,) or not torch.isfinite(value).all():
                        self.error = 'Invalid policy action; no further actions will be executed'
                        self.shutdown_event.set()
                        raise ValueError(self.error)
                    if step <= cutoff:
                        continue
                    if aggregate_fn is not None and step in previous:
                        value = aggregate_fn(previous[step], value)
                    merged.put(TimedAction(action.get_timestamp(), step, value))
                self.action_queue = merged
            if incoming_actions:
                self.event('prediction', first_step=incoming_actions[0].get_timestep(),
                           actions=[a.get_action().detach().cpu().tolist() for a in incoming_actions])

        def observation_loop(self):
            from lerobot.async_inference.helpers import TimedObservation
            from lerobot.transport import services_pb2
            from lerobot.transport.utils import send_bytes_in_chunks
            try:
                while self.running:
                    if not self._ready_to_send_observation():
                        self.shutdown_event.wait(0.01)
                        continue
                    with self.capture_lock:
                        if not self.running:
                            break
                        with self.latest_action_lock:
                            step = max(self.latest_action, 0)
                        captured_at = time.time()
                        raw = self.robot.get_observation()
                    self.latest_state = {n: float(raw[f'{n}.pos']) for n in NAMES}
                    raw['task'] = self.config.task
                    for key in CAMERAS:
                        raw[key] = encode_camera(raw[key])
                    with self.action_queue_lock:
                        must_go = self.must_go.is_set() and self.action_queue.empty()
                    observation = TimedObservation(captured_at, step, raw, must_go=must_go)
                    payload = pickle.dumps(observation)
                    start = time.perf_counter()
                    self.stub.SendObservations(
                        send_bytes_in_chunks(payload, services_pb2.Observation), timeout=10,
                    )
                    self.upload_seconds = time.perf_counter() - start
                    self.event('observation', step=step, state=self.latest_state,
                               captured_at=captured_at,
                               upload_seconds=self.upload_seconds, bytes=len(payload))
                    if must_go:
                        self.must_go.clear()
                    self.shutdown_event.wait(0.01)
            except Exception as exc:
                if self.running:
                    self.error = repr(exc)
                    print('Observation stream failed:', exc, flush=True)
                    self.shutdown_event.set()

        def control_loop_action(self, verbose=False):
            with self.action_queue_lock:
                try:
                    timed = self.action_queue.get_nowait()
                except Empty:
                    return None
                self._dequeued_step = timed.get_timestep()
            requested = self._action_tensor_to_action_dict(timed.get_action())
            if not all(math.isfinite(v) for v in requested.values()):
                raise RuntimeError('Nonfinite action rejected')
            action = self.robot.send_action(requested)
            with self.latest_action_lock:
                self.latest_action = timed.get_timestep()
            if self.first_action_time is None:
                self.first_action_time = time.perf_counter()
                self.event('policy_motion_started')
            self.action_count += 1
            self.last_target = {n: float(action[f'{n}.pos']) for n in NAMES}
            self.event('action', step=self.latest_action, requested=requested, target=self.last_target)
            return action

        def control_loop(self, task=None, verbose=False):
            self.start_barrier.wait()
            self.observer = threading.Thread(target=self.observation_loop, daemon=True)
            self.observer.start()
            last_report = time.perf_counter()
            performed = 0
            waiting_since = time.perf_counter()
            while self.running:
                start = time.perf_counter()
                key = self.poll_key()
                if key in ('s', 'f', 'q', '\x1b'):
                    self.outcome = {'s': 'success', 'f': 'failure'}.get(key, 'aborted')
                    self.stop_reason = 'operator_' + self.outcome
                    break
                if self.recording is not None and self.recording.error:
                    self.error = 'Video capture failed: ' + self.recording.error
                    self.stop_reason = 'recording_error'
                    break
                if self.first_action_time is not None and start-self.first_action_time >= self.episode_seconds:
                    self.outcome = 'timeout'
                    self.stop_reason = 'time_limit'
                    break
                if self.first_action_time is None and start-waiting_since > 30:
                    self.error = 'No policy actions received within 30 seconds'
                    self.stop_reason = 'no_actions'
                    break
                if self.actions_available():
                    if self.control_loop_action(verbose) is not None:
                        performed += 1
                elapsed = start - last_report
                if elapsed >= 1:
                    with self.action_queue_lock:
                        queued = self.action_queue.qsize()
                    actual = self.latest_state.get('gripper', float('nan'))
                    target = self.last_target.get('gripper', float('nan'))
                    print(f'Actions {performed/elapsed:.1f} Hz | queue {queued} | '
                          f'upload {self.upload_seconds:.2f}s | '
                          f'gripper actual {actual:.1f} -> target {target:.1f}', flush=True)
                    performed = 0
                    last_report = start
                self.shutdown_event.wait(max(0, self.config.environment_dt - (time.perf_counter() - start)))

        def quiesce(self):
            self.shutdown_event.set()
            self.channel.close()  # Unblock both pending network calls before joining.
            if self.observer:
                self.observer.join(timeout=5)

        def hold_current_pose(self):
            self.quiesce()
            with self.io_lock:
                raw = self.robot.bus.sync_read('Present_Position', normalize=False)
                for n, pos in raw.items():
                    c = self.robot.calibration[n]
                    if not c.range_min <= pos <= c.range_max:
                        raise RuntimeError(f'{n}: cannot hold a reading outside calibration')
                self.robot.bus.sync_write('Goal_Position', raw, normalize=False)
            self.event('stop_hold', raw_positions=raw)

        def stop(self):
            self.quiesce()
            errors = []
            if self.recording is not None:
                self.recording.close()
            try:
                with self.capture_lock:
                    with self.io_lock:
                        if self.robot.bus.is_connected:
                            try:
                                self.robot.bus.disable_torque(num_retry=2)
                                torque = {n: self.robot.bus.read('Torque_Enable', n, normalize=False) for n in NAMES}
                                if any(torque.values()):
                                    errors.append(f'Torque OFF not verified: {torque}')
                            except Exception as exc:
                                errors.append(f'Torque OFF: {exc}')
                            finally:
                                try:
                                    self.robot.bus.disconnect(disable_torque=False)
                                except Exception as exc:
                                    errors.append(f'Bus disconnect: {exc}')
                    for camera in self.robot.cameras.values():
                        try:
                            if camera.is_connected:
                                camera.disconnect()
                        except Exception as exc:
                            errors.append(f'Camera disconnect: {exc}')
            finally:
                self.events.close()
            if errors:
                raise RuntimeError('; '.join(errors))

    return RealClient


def preload_policy(cfg):
    """Load and identify the remote policy before opening the physical bus."""
    import grpc
    from lerobot.utils.feature_utils import hw_to_dataset_features
    from lerobot.async_inference.helpers import RemotePolicyConfig
    from lerobot.transport import services_pb2, services_pb2_grpc
    features = hw_to_dataset_features(
        {**{n+'.pos': float for n in NAMES}, **{c: (480, 640, 3) for c in CAMERAS}},
        'observation', use_video=False)
    specs = RemotePolicyConfig(cfg.policy_type, cfg.pretrained_name_or_path,
                               features,
                               cfg.actions_per_chunk, cfg.policy_device)
    with grpc.insecure_channel(cfg.server_address) as channel:
        stub = services_pb2_grpc.AsyncInferenceStub(channel)
        _, call = stub.Ready.with_call(services_pb2.Empty(), timeout=10)
        metadata = dict(call.trailing_metadata())
        if metadata.get('real-image-pipeline') != 'native-rgb-v1':
            raise RuntimeError('Wrong policy server: native real-image pipeline is required')
        _, setup = stub.SendPolicyInstructions.with_call(
            services_pb2.PolicySetup(data=pickle.dumps(specs)), timeout=120)
        if dict(setup.trailing_metadata()).get('real-model-sha256') != metadata.get('real-model-sha256'):
            raise RuntimeError('Model identity changed during setup')
    return metadata


def support_before_release():
    while True:
        try:
            input('\n로봇 팔을 받치거나 안정적으로 내려놓은 뒤 Enter: 모터 힘을 해제합니다. ')
            return
        except KeyboardInterrupt:
            print('\n아직 모터 힘을 유지합니다. 팔을 받친 뒤 Enter를 눌러 주세요.', flush=True)


def close_partial_client(client):
    errors = []
    robot = getattr(client, 'robot', None)
    if robot is not None:
        if robot.bus.is_connected:
            try:
                robot.bus.disable_torque(num_retry=2)
                if any(robot.bus.read('Torque_Enable', n, normalize=False) for n in NAMES):
                    raise RuntimeError('Torque OFF readback failed')
            except Exception as exc:
                errors.append(str(exc))
            finally:
                try:
                    robot.bus.disconnect(disable_torque=False)
                except Exception as exc:
                    errors.append(str(exc))
        for cam in robot.cameras.values():
            try:
                if cam.is_connected:
                    cam.disconnect()
            except Exception as exc:
                errors.append(str(exc))
    if hasattr(client, 'channel'):
        client.channel.close()
    if errors:
        raise RuntimeError('; '.join(errors))


def run_trial(Client, cfg, saved, bus_factory, folder, episode_seconds):
    from real_trial_support import TrialRecorder, event_metrics, poll_key
    folder.mkdir()
    result = dict(outcome='setup_error', operator_confirmed=False, stop_reason='',
                  failure_stage='', policy_seconds=0, error='', video_error='', folder=folder.name)
    client = None
    receiver = None
    bus = bus_factory()
    try:
        try:
            bus.connect(handshake=False)
            prepare_bus(
                bus, saved, folder/'startup_backup.json',
                confirm_release=lambda: input('다른 로봇 프로그램을 종료하고 팔을 받치세요. Enter: 남아 있는 모터 힘 해제. '),
                confirm_pose=lambda: input('모터 힘 OFF. 학습 때처럼 팔/큐브/컵을 시작 위치에 놓고 손을 뗀 뒤 Enter: 현재 자세 유지. '),
            )
        except BaseException as original_error:
            # A target write may have engaged a servo before a later failure.
            # Do not leave that partially prepared arm energized unnoticed.
            import traceback
            failure = {'original_error': repr(original_error), 'traceback': traceback.format_exc()}
            failure_path = folder/'startup_failure.json'
            failure_path.write_text(json.dumps(failure, indent=2)+'\n', encoding='utf-8')
            if bus.is_connected and getattr(bus, '_real_target_write_started', False):
                try:
                    torque = {n: bus.read('Torque_Enable', n, normalize=False) for n in NAMES}
                    need_release = any(torque.values())
                except Exception as exc:
                    failure['cleanup_initial_read_error'] = repr(exc)
                    need_release = True  # Unknown state; never assume the arm is OFF.
                if need_release:
                    print('시작 자세 준비 중단. 모터 힘 해제 전 팔을 받쳐 주세요.', flush=True)
                    support_before_release()
                    try:
                        release_setup_torque(bus, folder/'startup_cleanup.json')
                        print('모터 6개 OFF 확인 완료.', flush=True)
                    except Exception as exc:
                        failure['cleanup_error'] = repr(exc)
                        failure_path.write_text(json.dumps(failure, indent=2)+'\n', encoding='utf-8')
                        raise RuntimeError(f'Original startup error: {original_error}; cleanup error: {exc}') from original_error
            failure_path.write_text(json.dumps(failure, indent=2)+'\n', encoding='utf-8')
            raise
        finally:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
        client = Client.__new__(Client)
        Client.__init__(client, cfg, folder/'events.jsonl')
        client.episode_seconds = episode_seconds
        client.poll_key = poll_key
        client.start()
        # Start camera background readers before recording cached frames.
        client.robot.get_observation()
        input('손을 치우고 Enter: 추론 시작. S=성공, F=실패, Q=중단 (Enter 없이 키만 누르세요). ')
        while poll_key() is not None:
            pass
        client.recording = TrialRecorder(client.robot.cameras, folder)
        client.recording.start()
        def receive():
            try:
                client.receive_actions()
            except Exception as exc:
                if client.running:
                    client.error = repr(exc)
                    client.shutdown_event.set()
        receiver = threading.Thread(target=receive, daemon=True)
        receiver.start()
        client.control_loop()
        result['outcome'] = client.outcome or 'runtime_error'
        result['stop_reason'] = client.stop_reason or 'stream_stopped'
        if client.error:
            raise RuntimeError(client.error)
    except KeyboardInterrupt:
        result.update(outcome='aborted', stop_reason='keyboard_interrupt')
    except Exception as exc:
        result.update(outcome='runtime_error' if getattr(client, 'action_count', 0) else 'setup_error',
                      stop_reason='exception', error=repr(exc))
        print('실행 오류:', exc, flush=True)
    finally:
        if client is not None:
            started = getattr(client, 'first_action_time', None)
            result['policy_seconds'] = time.perf_counter()-started if started is not None else 0
            if hasattr(client, 'events'):
                try:
                    client.hold_current_pose()
                    if result['outcome'] == 'success':
                        # Retain the final scene for checking that the cube stays in the cup.
                        time.sleep(3)
                except KeyboardInterrupt:
                    result.update(outcome='aborted', stop_reason='keyboard_interrupt_during_stop')
                except Exception as exc:
                    result.update(outcome='runtime_error', error=result['error']+'; stop: '+repr(exc))
                    print('정지 자세 유지 오류. 팔을 받쳐 주세요:', exc, flush=True)
                if client.recording:
                    client.recording.close()
                    result['video_error'] = client.recording.error or ''
            robot = getattr(client, 'robot', None)
            if robot is not None and robot.bus.is_connected:
                support_before_release()
            try:
                if hasattr(client, 'events'):
                    client.stop()
                else:
                    close_partial_client(client)
            except Exception as exc:
                result.update(outcome='runtime_error', error=result['error']+'; cleanup: '+repr(exc))
                print('모터 OFF 확인 실패. 팔을 받치고 모터 전원을 분리하세요:', exc, flush=True)
            if receiver:
                receiver.join(timeout=5)
        result.update(event_metrics(folder/'events.jsonl'))
        if result['outcome'] == 'success' and not result['executed_actions']:
            result.update(outcome='aborted', stop_reason='success_key_before_motion')
        try:
            if result['outcome'] == 'success':
                answer = input('사람 도움 없이 집기/들기/놓기를 끝내고 큐브가 컵 안에 3초 이상 있었나요? y/N: ')
                result['operator_confirmed'] = answer.strip().lower() == 'y'
                if not result['operator_confirmed']:
                    result['outcome'] = 'failure'
            if result['executed_actions'] and result['outcome'] != 'success':
                result['failure_stage'] = input('실패 단계 입력 (접근/열기/집기/들기/놓기/기타, Enter=미분류): ').strip()
        except (KeyboardInterrupt, EOFError):
            if result['outcome'] == 'success':
                result.update(outcome='failure', operator_confirmed=False)
        (folder/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result


def main():
    import os
    import platform
    from importlib.metadata import version
    from real_trial_support import archive_session, write_summary, SUCCESS_CRITERION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--episode-seconds', type=float, default=30)
    args = parser.parse_args()
    if not args.execute:
        parser.error('Use --execute to run the real robot.')
    if args.trials < 1 or not 1 <= args.episode_seconds <= 120:
        parser.error('trials must be positive; episode-seconds must be in [1, 120].')
    if os.name == 'nt' and os.environ.get('CONDA_DEFAULT_ENV') != SETTINGS['conda_env']:
        parser.error(f"Run conda activate {SETTINGS['conda_env']} first.")
    load_settings(required=True)
    import torch
    from serial.tools import list_ports
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient

    torch.set_num_threads(1)
    result_root = Path(SETTINGS['result_root'])
    if not result_root.is_absolute():
        result_root = ROOT/result_root
    folder = result_root/f'run_{datetime.now():%Y%m%d_%H%M%S_%f}'
    folder.mkdir(parents=True)
    port = select_follower(SETTINGS, list_ports.comports()).device
    calibration_path = HF_LEROBOT_CALIBRATION/'robots/so_follower'/f"{SETTINGS['robot_id']}.json"
    calibration_bytes = calibration_path.read_bytes()
    saved = json.loads(calibration_bytes.decode('utf-8-sig'))
    (folder/'calibration.json').write_bytes(calibration_bytes)
    def bus_factory():
        return FeetechMotorsBus(port=port, motors={
            n: Motor(i+1, 'sts3215', MotorNormMode.RANGE_0_100 if n == 'gripper'
                     else MotorNormMode.DEGREES) for i, n in enumerate(NAMES)})
    cfg = RobotClientConfig(
        robot=SOFollowerRobotConfig(
            port=port, id=SETTINGS['robot_id'], use_degrees=True,
            max_relative_target=None, position_p_coefficient=16,
            position_i_coefficient=0, position_d_coefficient=32,
            cameras={key: OpenCVCameraConfig(index_or_path=index, width=640,
                        height=480, fps=30, fourcc='MJPG') for key, index in CAMERAS.items()},
        ),
        policy_type='smolvla', pretrained_name_or_path=MODEL, task=TASK,
        server_address=SETTINGS['server_address'], policy_device='cuda', client_device='cpu',
        actions_per_chunk=50, chunk_size_threshold=0.8,
        aggregate_fn_name='weighted_average', fps=30,
    )
    protocol = dict(runner_revision=RUNNER_REVISION, model=MODEL, dataset=SETTINGS['dataset_repo_id'],
                    training_episodes=SETTINGS['training_episodes'], training_steps=SETTINGS['training_steps'], config=asdict(cfg),
                    planned_trials=args.trials, episode_seconds=args.episode_seconds,
                    success_criterion=SUCCESS_CRITERION, calibration_sha256=hashlib.sha256(calibration_bytes).hexdigest(),
                    client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    versions={k: version(k) for k in ('lerobot', 'torch', 'grpcio')},
                    platform=platform.platform(), actual_execution='Windows physical SO-101',
                    image_pipeline='native 640x480 RGB, lossless PNG, original training resize/pad',
                    video_fps=15, status='awaiting_physical_trials')
    print(f'REAL SmolVLA | {RUNNER_REVISION} | 모델 로딩 중 (로봇 연결 전)...', flush=True)
    trials = []
    try:
        protocol['server'] = preload_policy(cfg)
        (folder/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        Client = make_client_class(RobotClient)
        for number in range(1, args.trials+1):
            print(f'\n시도 {number}/{args.trials} | 최대 {args.episode_seconds:g}초 | S 성공 / F 실패 / Q 중단', flush=True)
            result = run_trial(Client, cfg, saved, bus_factory, folder/f'trial_{number:03d}', args.episode_seconds)
            result['trial'] = number
            (folder/result['folder']/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            trials.append(result)
            summary = write_summary(folder, trials, protocol)
            print(f"집계: {summary['operator_labeled_successes']}/{summary['attempted_trials']} | {result['outcome']}", flush=True)
            if result['outcome'] in ('aborted', 'runtime_error', 'setup_error') or result['video_error']:
                break
    except (KeyboardInterrupt, Exception) as exc:
        protocol['session_error'] = repr(exc)
        print('세션 종료:', exc, flush=True)
    finally:
        protocol['status'] = 'operator_labeled_pending_video_review' if trials else 'no_physical_trials'
        (folder/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        write_summary(folder, trials, protocol)
        archive = archive_session(folder)
        print('결과:', folder.resolve(), flush=True)
        print('공유할 기록:', archive.resolve(), flush=True)
        print('report.html에서 결과/영상 확인 가능. 성공률은 사용자 판정이며 영상 검토 전입니다.', flush=True)


if __name__ == '__main__':
    main()
