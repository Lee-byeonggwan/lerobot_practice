"""Recording and honest outcome accounting for supervised physical trials."""

import csv
import html
import json
import math
import os
from pathlib import Path
import threading
import time
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED


SUCCESS_CRITERION = ('Autonomous approach, gripper opening, grasp, lift and release; '
                     'cube rests inside the cup for at least 3 seconds, with no human assistance.')


def poll_key():
    if os.name != 'nt':
        return None
    import msvcrt
    if not msvcrt.kbhit():
        return None
    key = msvcrt.getwch()
    if key in ('\x00', '\xe0'):
        if msvcrt.kbhit():
            msvcrt.getwch()
        return None
    return key.lower()


class TrialRecorder:
    """Record cached frames independently of RPC calls and motor commands."""
    def __init__(self, cameras, folder, fps=15):
        self.cameras, self.folder, self.fps = cameras, Path(folder), fps
        self.done = threading.Event()
        self.thread = None
        self.error = None
        self.frames = {name: 0 for name in cameras}
        self.writers = {}
        self.timestamps = None

    def start(self):
        import cv2
        try:
            for name, camera in self.cameras.items():
                frame = camera.read_latest()
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(self.folder / f'{name}.mp4'),
                                         cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
                self.writers[name] = writer
                if not writer.isOpened():
                    raise RuntimeError(f'Cannot open video writer for {name}')
                if not cv2.imwrite(str(self.folder / f'{name}_start.png'),
                                   cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f'Cannot save starting image for {name}')
            self.timestamps = (self.folder / 'video_timestamps.csv').open('x', newline='', encoding='utf-8')
            self.csv = csv.writer(self.timestamps)
            self.csv.writerow(['camera', 'frame', 'wall_time', 'monotonic_time', 'camera_capture_monotonic'])
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        except Exception:
            self._release()
            raise

    def _run(self):
        import cv2
        try:
            while not self.done.is_set():
                tick = time.perf_counter()
                for name, camera in self.cameras.items():
                    frame = camera.read_latest()
                    self.writers[name].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    self.csv.writerow([name, self.frames[name], time.time(), time.perf_counter(),
                                       getattr(camera, 'latest_timestamp', '')])
                    self.frames[name] += 1
                self.timestamps.flush()
                self.done.wait(max(0, 1/self.fps - (time.perf_counter()-tick)))
        except Exception as exc:
            self.error = repr(exc)
        finally:
            self._release()

    def _release(self):
        for writer in self.writers.values():
            writer.release()
        self.writers.clear()
        if self.timestamps is not None:
            self.timestamps.close()
            self.timestamps = None

    def close(self):
        self.done.set()
        if self.thread:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                self.error = 'Video recorder did not stop within 5 seconds'
        else:
            self._release()


def event_metrics(path):
    actions, uploads = [], []
    if Path(path).exists():
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event['kind'] == 'action':
                actions.append(event['time'])
            elif event['kind'] == 'observation':
                uploads.append(event['upload_seconds'])
    rate = (len(actions)-1)/(actions[-1]-actions[0]) if len(actions)>1 and actions[-1]>actions[0] else None
    return {'executed_actions': len(actions), 'mean_action_hz': rate,
            'mean_upload_seconds': sum(uploads)/len(uploads) if uploads else None,
            'max_upload_seconds': max(uploads) if uploads else None}


def wilson95(successes, count):
    if not count:
        return None
    z, p = 1.959963984540054, successes/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [max(0, center-half), min(1, center+half)]


def write_summary(folder, trials, protocol):
    folder = Path(folder)
    attempted = [t for t in trials if t.get('executed_actions', 0)>0]
    successes = sum(t.get('outcome') == 'success' and t.get('operator_confirmed', False) for t in attempted)
    summary = {
        'protocol': protocol, 'success_criterion': SUCCESS_CRITERION,
        'attempted_trials': len(attempted), 'setup_failures': len(trials)-len(attempted),
        'operator_labeled_successes': successes,
        'operator_labeled_success_rate': successes/len(attempted) if attempted else None,
        'wilson_95_interval': wilson95(successes, len(attempted)),
        'video_review_status': 'pending',
        'counting_rule': 'Every trial with at least one executed policy action is in the denominator, including timeouts, aborts and runtime errors. Setup-only failures are listed separately.',
        'trials': trials,
    }
    (folder/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    columns = ['trial', 'outcome', 'operator_confirmed', 'stop_reason', 'failure_stage',
               'policy_seconds', 'executed_actions', 'mean_action_hz', 'mean_upload_seconds',
               'max_upload_seconds', 'video_error', 'error', 'folder']
    with (folder/'trials.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(trials)
    rate = f'{successes}/{len(attempted)} ({100*successes/len(attempted):.1f}%)' if attempted else 'Not measured'
    rows = ''.join('<tr>'+''.join(f'<td>{html.escape(str(t.get(k, "")))}</td>' for k in columns[:-1])+
                   f'<td><a href="{html.escape(t["folder"])}/result.json">result</a> '+
                   f'<a href="{html.escape(t["folder"])}/camera1.mp4">overview video</a></td></tr>' for t in trials)
    document = ('<!doctype html><meta charset="utf-8"><title>SO-101 real SmolVLA evaluation</title>'
                '<style>body{font:16px system-ui;margin:32px;line-height:1.5}table{border-collapse:collapse;font-size:13px}'
                'th,td{border:1px solid #ccc;padding:7px}pre{white-space:pre-wrap}</style>'
                '<h1>SO-101 real SmolVLA evaluation</h1>'
                f'<p>Operator-labeled success rate: <strong>{rate}</strong>. Video review: pending.</p>'
                f'<p>{html.escape(SUCCESS_CRITERION)}</p><p>{html.escape(summary["counting_rule"])}</p>'
                '<table><tr>'+''.join(f'<th>{c}</th>' for c in columns)+'</tr>'+rows+'</table>'
                '<h2>Reproduction settings</h2><pre>'+html.escape(json.dumps(protocol, indent=2))+'</pre>')
    (folder/'report.html').write_text(document, encoding='utf-8')
    return summary


def archive_session(folder):
    folder = Path(folder)
    destination = folder.with_suffix('.zip')
    with ZipFile(destination, 'x', compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                archive.write(path, arcname=str(Path(folder.name)/path.relative_to(folder)),
                              compress_type=ZIP_STORED if path.suffix=='.mp4' else ZIP_DEFLATED)
    return destination
