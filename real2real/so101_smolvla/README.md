# 실제 SO-101: real-to-real

실제 로봇 시연 20개로 SmolVLA LoRA를 20,000 step 학습하고 같은 실제 로봇 설정으로 추론합니다. Windows는 SO-101 follower와 USB 카메라 3개에 연결하고, Linux GPU 서버는 모델을 실행합니다. 시연 수집에는 SO-101 leader도 필요합니다.

## 1. 환경 설치

Windows **PowerShell**에서 저장소 루트로 이동한 뒤 실행합니다.

```powershell
conda create -n lerobot python=3.12 -y
conda activate lerobot
python -m pip install -r envs/requirements-windows.txt
python -c "import lerobot, cv2; print('lerobot / OpenCV OK')"
Copy-Item real2real/so101_smolvla/config.example.json real2real/so101_smolvla/config.local.json
```

카메라용 OpenCV는 requirements에 포함되어 있습니다. `(base)`가 아닌 `(lerobot)` 환경을 사용합니다. PowerShell의 홈 이동은 `cd $env:USERPROFILE`이며 CMD의 `cd /d` 구문을 사용하지 않습니다.

Linux 서버에서도 같은 저장소를 준비합니다. 저장소 루트에서:

```bash
python3.12 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r envs/requirements-train.txt
cp real2real/so101_smolvla/config.example.json real2real/so101_smolvla/config.local.json
```

Linux에는 `ffmpeg`도 설치합니다. 모델 다운로드가 필요한 첫 실행에는 인터넷 연결이 필요합니다. GPU 드라이버는 CUDA 13.0 wheel을 지원해야 합니다. Windows client는 CPU로 동작합니다.

## 2. 로컬 설정과 실제 calibration

`config.local.json`을 각 PC에 맞게 편집합니다. 예시 카메라 번호 `1/3/0`은 원래 실험의 번호이며 다른 PC에서도 같다는 의미는 아닙니다.

| 키 | 의미 |
|---|---|
| `follower_usb_serial` / `follower_port` | follower USB serial 또는 COM 포트. 둘 다 넣으면 둘 다 일치해야 함 |
| `robot_id` | 실제 데이터 수집 때 사용한 follower ID |
| `leader_port` / `leader_id` | 시연 수집에 사용할 leader |
| `cameras` | 학습과 추론에서 동일한 `camera1/2/3` 대응 |
| `server_address` | SSH 터널을 쓰면 `127.0.0.1:8765` |
| `server_model_path` | **Linux 서버의** 병합 모델 절대 경로. Windows에도 같은 Linux 경로를 입력 |
| `task` | 데이터 수집·학습에서 쓰는 task 문자열 |
| `dataset_repo_id` | 새 로컬 데이터의 ID. 예: `local/so101_pickplace` |
| `result_root` | 평가 기록 위치. 상대 경로는 저장소 루트 기준 |

포트 목록은 Windows에서 다음 명령으로 확인합니다.

```powershell
python -c "from serial.tools import list_ports; print('\n'.join(f'{p.device} | {p.description} | serial={p.serial_number}' for p in list_ports.comports()))"
```

처음 사용하는 로봇이라면 실제 follower/leader를 LeRobot으로 보정한 뒤 시연을 수집합니다. 아래 `COM_FOLLOWER`, `COM_LEADER`와 ID를 자신의 값으로 바꿉니다.

```powershell
python -m lerobot.scripts.lerobot_calibrate --robot.type=so101_follower --robot.port=COM_FOLLOWER --robot.id=my_follower
python -m lerobot.scripts.lerobot_calibrate --teleop.type=so101_leader --teleop.port=COM_LEADER --teleop.id=my_leader
```

이미 실제 시연에 사용한 calibration이 있다면 그대로 사용합니다. 기본 follower 파일은 `%USERPROFILE%\.cache\huggingface\lerobot\calibration\robots\so_follower\<robot_id>.json`입니다. 다른 로봇의 calibration이나 sim 값을 복사하지 않습니다. 추론 코드는 파일과 모터의 보정값이 다르면 중단하며 자동 재보정하지 않습니다. 재보정으로 좌표계가 바뀌면 기존 데이터/가중치와의 일관성도 다시 확보해야 합니다.

상태/행동 순서는 `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`입니다. 팔은 **degrees**, gripper는 **0–100**입니다. sim의 6개 radians와 혼용하지 않습니다.

## 3. 실제 시연 20개 기록

로봇과 카메라를 사용하는 다른 프로그램을 종료합니다. Windows PowerShell에서 저장소 루트 기준:

```powershell
conda activate lerobot
python real2real/so101_smolvla/record_real.py --root datasets/real_train20 --episode-seconds 30 --reset-seconds 30
```

leader로 follower를 조작해 큐브를 집어 컵에 넣습니다. 수집은 LeRobot의 키보드 제어를 사용합니다. 오른쪽 화살표로 현재 episode/reset 대기를 끝내고, 왼쪽 화살표로 현재 episode를 다시 수집하며, Esc로 중단합니다. **기록의 reset은 지정한 초 단위 대기**이고 아래 평가 프로그램의 Enter 대기와 다릅니다. 시연의 시작 자세, 큐브/컵 위치, 카메라 위치를 추론 때도 맞춥니다.

예시 task `Pick up the cube and place it in the basket`은 원래 real 학습 문자열을 보존한 것입니다. 실제 목표 물체는 컵입니다. 새 시연에서 문장을 바꾸려면 수집 전에 `task`를 변경하고 추론에도 동일하게 사용합니다.

생성된 `datasets/real_train20` 폴더 전체를 Linux 서버의 같은 상대 경로로 복사합니다. `meta`, `data`, `videos`를 함께 보존합니다. Hugging Face로 올릴 필요는 없습니다. 데이터와 calibration은 이 저장소에 커밋하지 않습니다.

## 4. 학습 및 LoRA 병합

Linux 저장소 루트, 학습 환경에서:

```bash
source .venv-train/bin/activate
python real2real/so101_smolvla/train_real.py \
  --dataset local/so101_pickplace --dataset-root datasets/real_train20 \
  --output artifacts/real_smolvla/train --gpu 0
python real2real/so101_smolvla/merge_lora.py \
  --checkpoint artifacts/real_smolvla/train/checkpoints/020000/pretrained_model \
  --output artifacts/real_smolvla/merged \
  --dataset local/so101_pickplace --dataset-root datasets/real_train20 --device cpu
```

`--dataset`은 기록할 때의 `dataset_repo_id`와 맞춥니다. 학습은 20 episodes/30 FPS를 요구합니다. batch 64, LoRA rank/alpha 64, LR 0.001, 20,000 steps를 사용합니다. 병합은 동일 난수의 두 프레임 예측으로 adapter/merged 차이를 검사하고 전·후처리 파일도 복사합니다. 이는 배포 파일 검증이며 실제 작업 성공률 검증은 아닙니다.

양쪽 PC의 `server_model_path`를 실제 `artifacts/real_smolvla/merged`의 **Linux 절대 경로**로 바꿉니다. 어댑터만 있는 `pretrained_model` 경로를 넣지 않습니다.

선택적으로 `eval_offline.py`로 기록 데이터의 미래 action MAE를 계산할 수 있습니다. 이 수치는 실제 큐브 집기 성공률과 다릅니다.

## 5. 서버 및 터널

Linux에서 학습 환경을 켜고 서버를 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=0 python real2real/so101_smolvla/policy_server.py --host=127.0.0.1 --port=8765 --fps=30
```

Windows의 **별도 PowerShell 창**에서 SSH 접속 정보를 바꿔 실행하고 창을 유지합니다. `22`는 예시 포트이며 서버의 실제 SSH 포트를 사용합니다.

```powershell
ssh -N -p 22 -o ExitOnForwardFailure=yes -L 8765:127.0.0.1:8765 USER@HOST
```

터널이 정상 연결되면 출력 없이 계속 대기할 수 있습니다. `Connection refused`는 비밀번호 문제가 아니라 해당 주소/포트의 SSH 연결 실패입니다. 서버는 pickle을 사용하는 LeRobot async 프로토콜이므로 localhost + 본인 SSH 터널로 사용하며 공개 인터넷에 포트를 열지 않습니다.

## 6. 실제 평가 연속 20회

Windows의 다른 창, 저장소 루트에서 실행합니다. 이 명령은 실제 모터를 움직입니다.

```powershell
conda activate lerobot
python real2real/so101_smolvla/run_real.py --execute --trials 20 --episode-seconds 30
```

각 회차의 순서는 다음과 같습니다.

1. 팔을 받치고 Enter → 남아 있는 모터 힘 해제.
2. 모터 OFF 상태에서 팔/큐브/컵을 학습 시작 위치로 배치하고 손을 뗀 뒤 Enter → 현재 자세 유지.
3. 준비가 끝나면 다시 Enter → 추론 시작. 이때부터 최대 30초.
4. 성공 시 **S**, 실패 시 **F**, 중단 시 **Q**를 Enter 없이 누름.
5. 종료 안내에 따라 팔을 받친 뒤 Enter → 모터 힘 해제. 성공 여부/실패 단계 질문에 응답.
6. 다음 회차의 시작 위치를 다시 준비. **사용자가 Enter를 누를 때까지 기다립니다.**

따라서 사람이 환경을 정리하는 시간은 30초에 포함되지 않습니다. 성공했어도 S를 누르지 않으면 `timeout`으로 기록됩니다. 원래 성공 기준은 큐브를 들어 컵 안에 놓고 gripper와 분리된 채 유지하는 것이며, 최종 성공은 사용자의 판정입니다. 자동 영상 판정은 하지 않습니다.

모델 action 자체에 별도 sim 보정, gripper 제한, `max_relative_target` 제한을 추가하지 않습니다. LeRobot의 중첩 chunk 평균은 old 0.3 + new 0.7이며 chunk 50, threshold 0.8, 목표 제어 30 Hz입니다. lossless PNG 전송을 별도 thread로 처리하고 서버에서 원래 640×480 비율을 유지한 뒤 SmolVLA의 학습용 resize/pad를 적용합니다. 실제 달성 Hz는 PC/카메라/네트워크에 따라 달라지며 로그에 남깁니다.

시작 시 RAM의 Goal_Velocity=0, Torque_Limit=팔 1000/gripper 500을 복원하고 현재 자세를 목표로 기록합니다. PID는 16/0/32입니다. EEPROM/calibration은 바꾸지 않습니다. calibration 범위·모터 통신·fault 검사는 유지합니다.

## 7. 중단/오류와 결과

- `outside calibration range`: 모터 OFF 안내에서 해당 관절을 학습 시작 자세의 보정 범위 안으로 옮긴 뒤 Enter를 누릅니다. 범위 검사를 삭제하지 않습니다.
- `Incorrect status packet`: 다른 `robot_client`, teleop, serial 프로그램을 종료하고 USB/전원/배선을 확인한 뒤 재실행합니다. 코드에는 제한된 재시도가 포함되어 있습니다.
- 거의 움직이지 않을 때: 원래와 같은 시작 자세·카메라 대응·task·real merged 경로인지 확인합니다. 작은 모델 목표값 자체를 모터 속도 문제로 단정하지 않습니다.
- 준비 오류가 나면 그 회차는 추론 시도로 집계하지 않고 세션을 멈춥니다. 실제 완료한 횟수를 기준으로 `--trials 남은횟수`로 재시작합니다. 예: 완료 7회에서 총 20회를 채우려면 13회.

결과는 `results/real_smolvla/run_*`에 저장됩니다. `report.html`, `summary.json`, 회차별 영상/이벤트와 ZIP이 생성됩니다. 이 결과에는 로컬 경로와 calibration 사본이 포함될 수 있으므로 코드 저장소와 구분해 보관합니다. Git에서는 모두 제외됩니다.
