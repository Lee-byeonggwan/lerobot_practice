# 환경 및 재현 범위

## 실험 환경에서 확인한 버전

| 항목 | Isaac Sim 실행 | Linux SmolVLA 학습·서버 | Windows 로봇 |
|---|---|---|---|
| OS | Ubuntu 24.04 | Ubuntu 24.04 | Windows 11 |
| Python | 3.12.13 | 3.12.3 | 원본 로그에 정확한 버전 없음; 설치 안내는 3.12 |
| PyTorch | 2.11.0+cu130 | 2.11.0, CUDA runtime 13.0 | 2.11.0 |
| torchvision | 0.26.0+cu130 | 0.26.0 | 별도 기록 없음; 안내는 0.26.0 |
| Isaac Sim | 6.2.0.0 | 사용하지 않음 | 사용하지 않음 |
| IsaacLab | 커밋 `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4` | — | — |
| LeIsaac | 커밋 `24d3bcd3f1e4585740fc79921782c41617237812` + 포함 패치 | — | — |
| LeRobot | 설치하지 않음 | 0.6.1 | 0.6.1 |
| NumPy | 2.5.1 | 2.2.6 | 설치 안내는 2.2.6 |
| gRPC | 사용하지 않음 | 1.84.0 | 1.84.1 |

Linux 학습용 주요 패키지는 `envs/requirements-train.txt`에 고정했습니다. 시뮬레이션 환경은 위 IsaacLab 커밋의 `uv.lock`을 `uv sync --frozen`으로 설치합니다. LeIsaac의 오래된 `isaaclab`/`lerobot` extras를 함께 설치하면 버전이 바뀌므로 사용하지 않습니다. Windows requirements는 원본의 핵심 버전에 맞춘 이식용 설치 명세이며, 원본 전체 환경을 freeze한 파일은 아닙니다.

시뮬레이션 수집/평가는 NVIDIA RTX 렌더링과 해당 Isaac Sim/CUDA 버전을 지원하는 드라이버가 필요합니다. 원래 시뮬레이션 학습은 2,000 step까지 GPU 1개, 이후 20,000 step까지 GPU 2개를 사용했습니다. 평가도 시뮬레이터와 정책을 GPU 0/1로 나눴습니다. 최소 VRAM 및 다른 GPU에서의 처리량은 측정하지 않았습니다.

## 학습 설정

| 설정 | Sim-to-sim | Real-to-real |
|---|---|---|
| 기반 모델 | `lerobot/smolvla_base` | `lerobot/smolvla_base` |
| episode 수 | 성공 20개, 모두 train | 20개, 모두 train |
| 제어/데이터 FPS | 30 | 30 |
| 상태/행동 | 관절 6개, absolute radians | 팔 5개 degrees + gripper 0–100 |
| 영상 | RGB 640×480, 3개 | RGB 640×480, 3개 |
| camera1/2/3 | top / wrist / front(belly) | 실제 기록의 camera1 / camera2 / camera3 |
| steps | 20,000 | 20,000 |
| batch | 64 → GPU당 32 × 2 | 64 × 1 GPU |
| LoRA rank / alpha | 64 / 64 | 64 / 64 |
| optimizer LR / decay LR | 0.001 / 0.0001 | 0.001 / 0.0001 |
| warmup / decay steps | 1,000 / 20,000 | 1,000 / 30,000 (0.6.1 기본값) |
| seed | 20261101 | 1000 (0.6.1 기본값) |
| checkpoint 주기 | 2,000 | 5,000 |

원래 작업의 시연·가중치는 배포하지 않습니다. 시뮬레이션 physics/rendering, GPU 개수 변경에 따른 RNG와 데이터 순서, 새 실제 시연 차이 때문에 비트 단위 재현이나 같은 성공률을 약속하지 않습니다. Sim 평가는 학습 위치 주변의 새 jitter를 사용하며 분포 밖 일반화 평가는 아닙니다.

원래 로컬 캐시에 있던 기반 모델 revision은 [MODEL_REFERENCES.json](MODEL_REFERENCES.json)에 별도로 기록했습니다. 실행 스크립트는 모델 ID를 사용하므로 Hub의 `main` 변경까지 고정한 배포본은 아닙니다.

## 배포본 검증

검증 결과와 미실행 범위는 [VALIDATION.md](VALIDATION.md)에 기록합니다. 설치 명세 작성만으로 새 PC에서 전체 수집·20,000-step 학습·물리 로봇 평가를 재실행했다고 간주하지 않습니다.
