# Isaac Sim SO-101: sim-to-sim

2 cm 큐브를 컵에 넣는 scripted expert 시연 20개를 생성하고, SmolVLA LoRA를 20,000 step 학습한 뒤 시뮬레이션에서 평가합니다. 로봇 USB나 real calibration은 사용하지 않습니다.

## 1. 설치

Linux + NVIDIA RTX 지원 GPU를 사용합니다. 원래 수집/최종 학습/평가는 GPU 두 개를 사용했습니다. Python/Isaac Sim/IsaacLab 버전은 [환경 기록](../../docs/ENVIRONMENT.md)에 있습니다. `git`, `ffmpeg`, `cmake`, C/C++ 빌드 도구와 `uv==0.11.28`을 먼저 설치합니다.

아래 명령은 저장소 루트에서 시작합니다.

```bash
cd sim2sim/so101_isaac
bash setup_sources.sh
bash setup_sim.sh
```

`setup_sim.sh`는 고정 IsaacLab 커밋의 lockfile로 `.venv-sim`을 만들고 패치된 LeIsaac을 설치합니다. Isaac Sim 에셋/확장 설치에는 큰 디스크 공간과 다운로드 시간이 필요합니다. NVIDIA Isaac Sim 라이선스를 확인하고 동의한 환경에서 다음을 설정합니다.

```bash
export OMNI_KIT_ACCEPT_EULA=YES
source activate_sim.sh
python scripts/fetch_assets.py
python scripts/prepare_assets.py --device cuda:0 --livestream 0
```

기본 로봇 USD와 Table with Cube ZIP의 SHA-256을 검증하고, 목재 큐브 → 컵 → 흰 테이블/보라색 로봇 → 수집용 장면 순으로 생성합니다. 출력이 이미 있으면 덮어쓰지 않습니다. `prepare_real_style_layout`의 이름은 시각적 배치를 뜻하며 실물 카메라와 보정된 sim-to-real 설정을 의미하지 않습니다.

학습 환경은 별도로 만듭니다. 아래 명령도 `sim2sim/so101_isaac`에서 실행합니다.

```bash
uv venv --python 3.12.3 .venv-train
uv pip install --python .venv-train/bin/python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv-train/bin/python -r ../../envs/requirements-train.txt
```

이후 `.venv-sim/bin/python`과 `.venv-train/bin/python`을 구분해 사용합니다. 첫 모델 로딩 때 Hugging Face에서 기반 모델을 받으므로 오프라인 변수를 미리 설정하지 않습니다.

## 2. 시연 20개 생성 및 검증

`configs/collection_plan.json`은 데이터가 아니라 초기 위치 설정입니다. 기준 큐브 위치 `(0.19, -0.43)` m 주변 5개 그룹에서 각 4개 위치를 사용합니다. cup은 `(0.25, -0.60)` m, 로봇 시작 관절각은 `[0,-95,70,85,0,10]` degrees를 radians로 변환합니다.

```bash
.venv-sim/bin/python scripts/init_collection.py --dataset datasets/scripted_train20
.venv-sim/bin/python scripts/collect_scripted_batch.py \
  --dataset datasets/scripted_train20 \
  --plan datasets/scripted_train20/collection_plan.json \
  --gpus 0 1 --center-release --gentle-approach --grasp-x 0.002
.venv-sim/bin/python scripts/audit_scripted_dataset.py datasets/scripted_train20
```

GPU 한 개로 수집하려면 `--gpus 0`을 씁니다. 각 episode는 최대 4번의 높이 설정을 시도하고 독립 검증에 통과한 성공 시연만 채택합니다. 중간에 종료됐다면 `init_collection.py`를 다시 실행하지 않고 수집 명령을 재실행합니다. 완료된 episode는 건너뜁니다. 20개가 모두 준비되지 않으면 audit/변환 단계가 실패합니다.

expert는 simulator의 물체 위치를 사용해 IK/접촉 제어를 합니다. 이 시연의 성공률은 학습 정책의 성공률이 아닙니다. 생성된 데이터의 영상·관절·물체 trace, 컵 테두리 여유, 집기 전 큐브 이동, 마지막 안정 상태를 검사합니다. 성공 20개 × 840 frames = 16,800 frames, 30 FPS입니다.

## 3. LeRobot 포맷 변환

```bash
.venv-train/bin/python scripts/convert_cup_to_lerobot.py \
  --source datasets/scripted_train20 --output datasets/lerobot_train20 \
  --repo-id local/so101_cup_train20
```

학습 입력은 6개 관절 상태와 영상 3개입니다. privileged 물체 상태는 정책 입력에 넣지 않습니다. 시연 영상 `top/wrist/front`는 데이터셋 `top/wrist/belly`, 모델 `camera1/camera2/camera3`으로 연결됩니다. task 문자열은 `Pick up the wooden cube and place it inside the cup.`입니다.

## 4. SmolVLA 학습

원래 GPU 배치 변경을 재현하는 순서입니다. 첫 학습이 정상 종료된 후 resume합니다. optimizer/RNG 상태가 포함된 checkpoint 전체를 유지하세요.

```bash
.venv-train/bin/python scripts/train_cup_smolvla.py \
  --dataset datasets/lerobot_train20 --run outputs/train_stage1 --gpu 0 --steps 2000
.venv-train/bin/python scripts/resume_training.py \
  --checkpoint outputs/train_stage1/training/checkpoints/002000 \
  --run outputs/train_final --gpus 0 1
```

처음 2,000 step은 batch 64, 이후에는 GPU당 batch 32로 전체 batch 64를 유지합니다. scheduler decay는 첫 단계부터 20,000으로 설정합니다. resume의 `--steps=20000`은 추가 20,000 step이 아니라 최종 누적 step입니다. 원래 실행 중 프로세스를 전환하던 관리 코드는 제거하고 완료된 checkpoint에서 이어가도록 정리했습니다.

GPU 한 개로 전체 학습하려면 아래 대안을 사용합니다. 모델·학습률 설정은 같지만 원래 2-GPU 실행과 RNG/데이터 순서가 같지는 않습니다.

```bash
.venv-train/bin/python scripts/train_cup_smolvla.py \
  --dataset datasets/lerobot_train20 --run outputs/train_single --gpu 0 --steps 20000
```

로그는 각 run의 `train.log`, 최종 확인은 `result.json`입니다. 출력 폴더가 이미 있으면 중단합니다. 이 sim 평가 서버는 LoRA adapter를 직접 로딩하므로 real용 merge 과정이 필요 없습니다.

## 5. 학습 정책 평가 20회

```bash
.venv-train/bin/python scripts/run_cup_evaluation.py \
  --checkpoint outputs/train_final/training/checkpoints/020000/pretrained_model \
  --dataset datasets/lerobot_train20 --output outputs/eval20 \
  --sim-device cuda:0 --policy-device cuda:1 --episodes 20 --seconds 40
.venv-train/bin/python scripts/score_cup_rollouts.py outputs/eval20
```

single-GPU 학습을 썼다면 checkpoint 경로를 `outputs/train_single/...`로 바꿉니다. 평가의 두 device를 `cuda:0`으로 같게 설정하는 것도 가능하지만 메모리 여유와 실행 가능성은 별도 확인이 필요하며 이번 배포 검증에는 포함하지 않았습니다. Unix socket 경로 길이 제한 때문에 저장소/출력 경로를 짧게 두세요.

평가는 새 jitter 위치, 고정 cup, 동일한 접힌 시작 자세에서 최대 40 **simulation seconds**를 사용합니다. 50개 action chunk를 예측하며 추론하는 동안 시뮬레이션 시간은 멈춥니다. 실제 로봇의 실시간 지연 조건과 다릅니다. 정책은 RGB/관절만 보고 물체 상태는 채점에만 사용합니다. 관절 제한은 시뮬레이터의 joint limits를 적용합니다.

`summary.json`의 online 진단과 `score_cup_rollouts.py`의 최종 자세 기반 판정을 구분합니다. 후자는 마지막 1초 동안 큐브가 컵 안에 있고, 로봇과 분리되며, 들어 올린 이력이 있고, cube/cup 이동폭 ≤2 mm 및 cube 회전폭 ≤5°인지를 검사합니다. 결과/영상은 `outputs/`에만 생성되고 Git에서 제외됩니다.
