"""Offline tests only: these never connect to or command physical motors."""
import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]/"real2real/so101_smolvla"))
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass, asdict
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from zipfile import ZipFile
from unittest.mock import patch

import numpy as np
import torch

from run_real import (NAMES, prepare_bus, encode_camera, make_client_class, run_trial,
                         install_setup_read_retries, release_setup_torque)
from policy_server import decode_camera, training_observation
from real_trial_support import TrialRecorder, write_summary, archive_session
from lerobot.async_inference.helpers import TimedAction
from lerobot.async_inference.robot_client import RobotClient


@dataclass
class Calibration:
    id: int
    drive_mode: int = 0
    homing_offset: int = 0
    range_min: int = 1000
    range_max: int = 3000


class Bus:
    def __init__(self, torque=0):
        self.calibration = {n: Calibration(i+1) for i,n in enumerate(NAMES)}
        self.regs = {r: {n: 0 for n in NAMES} for r in (
            'Torque_Enable','Operating_Mode','Status','Max_Torque_Limit','Goal_Velocity',
            'Torque_Limit','Goal_Position','Present_Position','Acceleration')}
        for n in NAMES:
            self.regs['Torque_Enable'][n] = torque
            self.regs['Max_Torque_Limit'][n] = 500 if n=='gripper' else 1000
            self.regs['Present_Position'][n] = 2048
            self.regs['Goal_Position'][n] = 2200
        self.regs['Goal_Velocity']['shoulder_pan'] = 100
        self.writes = []
        self.is_connected = False
    def connect(self, **kwargs):
        self.is_connected = True
    def disconnect(self, **kwargs):
        self.is_connected = False
    def disable_torque(self, **kwargs):
        self.regs['Torque_Enable'] = dict.fromkeys(NAMES, 0)
    def read(self, reg, n, **kwargs):
        return self.regs[reg][n]
    def read_calibration(self):
        return self.calibration
    def write(self, reg, n, value, **kwargs):
        if reg == 'Goal_Position':
            assert self.regs['Torque_Enable'][n] == 0
        elif reg != 'Torque_Enable':
            assert not any(self.regs['Torque_Enable'].values())
        self.writes.append((reg,n,value))
        self.regs[reg][n] = value


class RealRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def saved(self, bus):
        return {n:asdict(c) for n,c in bus.calibration.items()}
    def test_startup_on_cancel_does_not_write(self):
        bus = Bus(torque=1)
        def cancel():
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            prepare_bus(bus, self.saved(bus), self.root/'backup.json', confirm_release=cancel)
        self.assertEqual(bus.writes, [])
        self.assertTrue(all(bus.regs['Torque_Enable'].values()))
    def test_setup_uses_bounded_sdk_retries_for_bad_packets(self):
        from lerobot.motors.motors_bus import MotorsBus
        for replies, expected_attempts, expected_error in (
            ([(-1,0),(-1,0),(0,0)],3,None),
            ([(-1,0),(-1,0),(-1,0)],3,ConnectionError),
            ([(0,1)],1,RuntimeError),
        ):
            attempts=[]
            def response(*args):
                comm,error=replies[len(attempts)]
                attempts.append(1)
                return 0,comm,error
            backend=SimpleNamespace(packet_handler=SimpleNamespace(
                read1ByteTxRx=response,getTxRxResult=lambda c:'Incorrect status packet',
                getRxPacketError=lambda e:'Motor fault'),port_handler=None,
                _is_comm_success=lambda c:c==0,_is_error=lambda e:e!=0)
            bus=SimpleNamespace(read=lambda *a,**kw:MotorsBus._read(
                backend,40,1,2,num_retry=kw.get('num_retry',0),err_msg='test read')[0])
            install_setup_read_retries(bus)
            if expected_error:
                with self.assertRaises(expected_error): bus.read('Torque_Enable','shoulder_lift')
            else:
                self.assertEqual(bus.read('Torque_Enable','shoulder_lift'),0)
            self.assertEqual(len(attempts),expected_attempts)
    def test_release_attempts_other_joints_when_one_joint_cannot_be_read(self):
        class BrokenReadBus(Bus):
            def read(self,reg,n,**kwargs):
                if n=='shoulder_lift':
                    raise ConnectionError('Incorrect status packet')
                return super().read(reg,n,**kwargs)
        bus=BrokenReadBus(torque=1)
        with self.assertRaisesRegex(RuntimeError,'Motor OFF unverified'):
            release_setup_torque(bus,self.root/'cleanup.json')
        report=json.loads((self.root/'cleanup.json').read_text())
        self.assertEqual(len(bus.writes),6)
        self.assertFalse(report['shoulder_lift']['off_verified'])
        self.assertTrue(all(report[n]['off_verified'] for n in NAMES if n!='shoulder_lift'))
    def test_startup_on_requires_supervision(self):
        bus = Bus(torque=1)
        with self.assertRaisesRegex(RuntimeError, 'Torque remains ON'):
            prepare_bus(bus, self.saved(bus), self.root/'backup.json')
        self.assertEqual(bus.writes, [])
    def test_restore_ram_and_latch_pose(self):
        bus = Bus(torque=1)
        called = []
        prepare_bus(bus, self.saved(bus), self.root/'backup.json',
                    confirm_release=lambda:called.append('release'),
                    confirm_pose=lambda:called.append('pose'))
        self.assertEqual(called,['release','pose'])
        self.assertFalse(any(bus.regs['Torque_Enable'].values()))
        self.assertEqual(bus.regs['Torque_Limit'],bus.regs['Max_Torque_Limit'])
        self.assertEqual(bus.regs['Goal_Position'],bus.regs['Present_Position'])
        self.assertTrue(all(r in ('Torque_Enable','Torque_Limit','Goal_Velocity','Goal_Position') for r,_,_ in bus.writes))
        self.assertFalse(any(r=='Torque_Enable' and v!=0 for r,_,v in bus.writes))
    def test_goal_write_engages_only_latched_servo(self):
        class AutoEngageBus(Bus):
            def write(self, reg, n, value, **kwargs):
                super().write(reg,n,value,**kwargs)
                if reg=='Goal_Position':
                    self.regs['Torque_Enable'][n]=1
        bus=AutoEngageBus()
        prepare_bus(bus,self.saved(bus),self.root/'backup.json')
        self.assertTrue(all(bus.regs['Torque_Enable'].values()))
        self.assertEqual(bus.regs['Goal_Position'],bus.regs['Present_Position'])
        first_goal=next(i for i,w in enumerate(bus.writes) if w[0]=='Goal_Position')
        self.assertTrue(all(w[0]=='Goal_Position' for w in bus.writes[first_goal:]))
        trace=[json.loads(line) for line in (self.root/'startup_trace.jsonl').read_text().splitlines()]
        self.assertEqual(trace[-1]['stage'],'prepare_complete')
        self.assertEqual(trace[-1]['latched'],list(NAMES))
    def test_torque_before_latching_is_rejected_and_identified(self):
        bus=Bus()
        def pose():
            bus.regs['Torque_Enable']['wrist_roll']=1
        with self.assertRaisesRegex(RuntimeError,'after_pose_prompt.*wrist_roll'):
            prepare_bus(bus,self.saved(bus),self.root/'backup.json',confirm_pose=pose)
        self.assertEqual(bus.writes,[])
    def test_initial_outside_pose_can_be_repositioned_after_release(self):
        bus=Bus(torque=1)
        bus.regs['Present_Position']['elbow_flex']=3100
        prompts=[]
        def pose():
            self.assertFalse(any(bus.regs['Torque_Enable'].values()))
            self.assertFalse(any(w[0]=='Goal_Position' for w in bus.writes))
            prompts.append('pose')
            bus.regs['Present_Position']['elbow_flex']=2000
        prepare_bus(bus,self.saved(bus),self.root/'backup.json',
                    confirm_release=lambda:prompts.append('release'),confirm_pose=pose)
        self.assertEqual(prompts,['release','pose'])
        self.assertEqual(bus.regs['Goal_Position']['elbow_flex'],2000)
        saved=json.loads((self.root/'backup.json').read_text())
        self.assertEqual(saved['before']['Present_Position']['elbow_flex'],3100)
    def test_invalid_pose_reprompts_before_any_goal_or_ram_writes(self):
        bus=Bus();bus.regs['Present_Position']['elbow_flex']=3100
        calls=[]
        def pose():
            self.assertEqual(bus.writes,[])
            calls.append(1)
            if len(calls)==2:bus.regs['Present_Position']['elbow_flex']=2000
        prepare_bus(bus,self.saved(bus),self.root/'backup.json',confirm_pose=pose)
        self.assertEqual(len(calls),2)
        self.assertEqual(bus.regs['Goal_Position']['elbow_flex'],2000)
    def test_invalid_pose_without_operator_never_sends_targets(self):
        bus=Bus();bus.regs['Present_Position']['elbow_flex']=3100
        with self.assertRaisesRegex(RuntimeError,'Starting pose outside calibration.*elbow_flex'):
            prepare_bus(bus,self.saved(bus),self.root/'backup.json')
        self.assertEqual(bus.writes,[])
    def test_unlatched_servo_activation_aborts_and_releases_prepared_arm(self):
        class BadEngageBus(Bus):
            def write(self,reg,n,value,**kwargs):
                super().write(reg,n,value,**kwargs)
                if reg=='Goal_Position':
                    self.regs['Torque_Enable'][n]=1
                    self.regs['Torque_Enable']['gripper']=1
        bus=BadEngageBus()
        with patch('builtins.input',return_value=''):
            result=run_trial(None,None,self.saved(bus),lambda:bus,self.root/'trial',30)
        self.assertEqual(result['outcome'],'setup_error')
        self.assertIn('after_latch:shoulder_pan',result['error'])
        self.assertIn('gripper',result['error'])
        self.assertFalse(any(bus.regs['Torque_Enable'].values()))
        self.assertFalse(bus.is_connected)
        targets=[w for w in bus.writes if w[0]=='Goal_Position']
        self.assertEqual(len(targets),1)
    def test_mismatch_fault_or_unexpected_limits_write_nothing(self):
        for kind in ('calibration','Status','Operating_Mode','Max_Torque_Limit'):
            with self.subTest(kind=kind):
                bus=Bus(); saved=self.saved(bus)
                if kind=='calibration': saved['gripper']['homing_offset']=1
                else: bus.regs[kind]['gripper']=7
                with self.assertRaises(RuntimeError):
                    prepare_bus(bus,saved,self.root/(kind+'.json'))
                self.assertEqual(bus.writes,[])
    def test_png_rgb_exact_and_square_rejected(self):
        frame=np.random.default_rng(3).integers(0,256,(480,640,3),dtype=np.uint8)
        np.testing.assert_array_equal(frame,decode_camera(encode_camera(frame)))
        with self.assertRaises(ValueError): decode_camera(frame[:256,:256])
    def test_native_tensor_matches_training_without_changing_model_config(self):
        from lerobot.utils.feature_utils import hw_to_dataset_features
        from lerobot.configs.types import PolicyFeature, FeatureType
        feature=PolicyFeature(type=FeatureType.VISUAL, shape=(3,256,256))
        images={'observation.images.camera1':feature}
        raw={n+'.pos':float(i) for i,n in enumerate(NAMES)}
        frame=np.random.default_rng(5).integers(0,256,(480,640,3),dtype=np.uint8)
        raw['camera1']=encode_camera(frame);raw['task']='test'
        features=hw_to_dataset_features({**{n+'.pos':float for n in NAMES},'camera1':(480,640,3)},'observation',use_video=False)
        observation=training_observation(raw,features,images)
        tensor=observation['observation.images.camera1']
        torch.testing.assert_close(tensor.squeeze(0),torch.from_numpy(frame).permute(2,0,1).float()/255,rtol=0,atol=0)
        self.assertEqual(feature.shape,(3,256,256))
    def test_partial_constructor_failure_still_releases_torque(self):
        bus=Bus()
        class BrokenClient:
            def __init__(self, cfg, events):
                bus.connect()
                bus.regs['Torque_Enable']=dict.fromkeys(NAMES,1)
                self.robot=SimpleNamespace(bus=bus,cameras={})
                raise RuntimeError('simulated camera connection failure')
        with patch('builtins.input',return_value=''):
            result=run_trial(BrokenClient,None,self.saved(bus),lambda:bus,self.root/'trial',30)
        self.assertEqual(result['outcome'],'setup_error')
        self.assertEqual(result['executed_actions'],0)
        self.assertFalse(bus.is_connected)
        self.assertFalse(any(bus.regs['Torque_Enable'].values()))
    def client(self):
        cls=make_client_class(RobotClient); c=cls.__new__(cls)
        c.action_queue_lock=threading.Lock(); c.latest_action_lock=threading.Lock()
        c.action_queue=Queue(); c.latest_action=-1; c._dequeued_step=-1
        c.event=lambda *a,**kw: None
        c.shutdown_event=threading.Event(); c.error=None
        c.first_action_time=None; c.action_count=0; c.last_target={}
        c.robot=SimpleNamespace(action_features={n+'.pos':float for n in NAMES},send_action=lambda a:a)
        return c
    def test_queue_drops_inflight_and_nonfinite(self):
        c=self.client(); c.latest_action=2;c._dequeued_step=3
        c._aggregate_action_queues([TimedAction(0,i,torch.full((6,),float(i))) for i in range(2,6)])
        self.assertEqual([a.timestep for a in c.action_queue.queue],[4,5])
        c.control_loop_action(); self.assertEqual(c.latest_action,4)
        with self.assertRaises(ValueError):
            c._aggregate_action_queues([TimedAction(0,6,torch.full((6,),float('nan')))])
        self.assertTrue(c.shutdown_event.is_set())
        self.assertEqual(c.action_count,1)
    def test_action_ticks_continue_while_upload_blocked(self):
        c=self.client(); c.config=SimpleNamespace(environment_dt=1/30)
        c.start_barrier=threading.Barrier(1); c.latest_state={}; c.upload_seconds=0
        c.episode_seconds=.5;c.recording=None;c.poll_key=lambda:None
        c.outcome=None;c.stop_reason=None
        # The observer is stalled on a network response for the whole trial.
        c.observation_loop=lambda:c.shutdown_event.wait(2)
        c._aggregate_action_queues([TimedAction(0,i,torch.zeros(6)) for i in range(50)])
        c.control_loop();c.shutdown_event.set();c.observer.join(1)
        self.assertGreaterEqual(c.action_count,12)
        self.assertLessEqual(c.action_count,17)
        self.assertEqual(c.outcome,'timeout')
    def test_recording_outputs_readable_video(self):
        import cv2
        camera=SimpleNamespace(read_latest=lambda:np.zeros((480,640,3),np.uint8),latest_timestamp=1.)
        recorder=TrialRecorder({'camera1':camera},self.root)
        recorder.start();time.sleep(.3);recorder.close()
        self.assertIsNone(recorder.error)
        cap=cv2.VideoCapture(str(self.root/'camera1.mp4'));ok,frame=cap.read();cap.release()
        self.assertTrue(ok);self.assertEqual(frame.shape,(480,640,3))
        self.assertGreaterEqual(recorder.frames['camera1'],2)
    def test_denominator_keeps_abort_timeout_and_errors(self):
        trials=[dict(trial=i,folder=str(i),outcome=o,operator_confirmed=(o=='success'),executed_actions=n)
                for i,(o,n) in enumerate([('success',10),('timeout',10),('aborted',2),('runtime_error',1),('setup_error',0),('success',0)])]
        result=write_summary(self.root,trials,{'test_only':True})
        self.assertEqual(result['attempted_trials'],4)
        self.assertEqual(result['operator_labeled_successes'],1)
        self.assertEqual(result['operator_labeled_success_rate'],.25)
        self.assertEqual(result['video_review_status'],'pending')
        archive=archive_session(self.root)
        try:
            with ZipFile(archive) as z:
                saved=json.loads(z.read(self.root.name+'/summary.json'))
                self.assertEqual(saved,result)
        finally:
            archive.unlink()


if __name__=='__main__':
    unittest.main(verbosity=2)
