# lerobot_practice

SO-101이 큐브를 집어 컵에 넣는 두 실험의 코드와 환경 설정입니다.

| 실험 | 데이터 수집 | 학습 | 실행 안내 |
|---|---|---|---|
| Sim-to-sim | Isaac Sim SO-101 scripted expert, 성공 20 episodes | SmolVLA LoRA, 20,000 steps | [시뮬레이션 README](sim2sim/so101_isaac/README.md) |
| Real-to-real | 실제 SO-101 leader/follower 원격조작, 20 episodes | SmolVLA LoRA, 20,000 steps | [실제 로봇 README](real2real/so101_smolvla/README.md) |

ACT/PushT와 sim-to-real 실험은 포함하지 않습니다. 두 실험은 각각 수집한 데이터로 별도 모델을 학습합니다. 시뮬레이션 체크포인트를 실제 로봇에 실행하는 저장소가 아닙니다.

## 포함 범위

```text
sim2sim/so101_isaac/       장면 생성, 시연 수집·검증, 변환, 학습, 시뮬레이션 평가
  configs/               큐브 초기 위치 20개의 실험 설정
  patches/               고정 LeIsaac 커밋에 적용할 IsaacLab 호환 패치
real2real/so101_smolvla/   실제 시연 수집, 학습, LoRA 병합, 서버, Windows 추론
envs/                    학습/Windows 의존성 버전
tests/                   로봇에 연결하지 않는 추론 코드 검증
docs/                    버전·재현 범위·소스 구성 기록
licenses/                포함된 외부 코드의 라이선스
```

데이터셋, 학습 가중치, 영상, 실험 결과, 개인 calibration, 전체 가상환경은 포함하지 않습니다. 시뮬레이션 USD 에셋도 저장하지 않으며 공식 배포처에서 다운로드해 SHA-256을 검증합니다. 처음 실행할 때 SmolVLA 기반 모델 다운로드가 필요합니다.

이 저장소는 **실험 절차를 재현**하기 위한 코드입니다. 공개되지 않은 원래 real 데이터나 최종 가중치의 동일한 결과를 보장하지 않습니다. 실제 데이터는 사용자의 로봇에서 새로 수집하고, 동일한 카메라 이름·배치·단위로 학습과 추론을 연결해야 합니다.

## 시작 순서

1. 위 표에서 원하는 실험의 README를 엽니다.
2. 환경 설치 → 데이터 수집 → 학습 → 평가 순서로 진행합니다.
3. 실제 로봇 설정은 `config.example.json`을 복사한 `config.local.json`에 입력합니다. 이 파일은 Git에서 제외됩니다.

Linux 학습 환경과 Isaac Sim 환경은 **별도 가상환경**입니다. Isaac Sim의 NumPy 및 USD 의존성과 LeRobot의 요구 버전이 다릅니다. 실제 로봇 실행은 Windows, GPU 학습·추론 서버는 Linux로 구성했습니다.

실험에서 관측된 버전과 이 배포본의 검증 범위는 [환경 기록](docs/ENVIRONMENT.md), 원본 대비 정리 내용은 [소스 안내](docs/SOURCES.md)에 있습니다. 외부 코드와 에셋의 출처는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 참고하세요.

## 코드 검증

Linux 학습 환경에서 저장소 루트 기준으로 실행합니다. 실제 로봇에는 연결하지 않습니다.

```bash
python tests/test_real_runtime.py
python -m compileall -q real2real sim2sim tests
```

## GitHub에 올리기

이 폴더 자체를 저장소 루트로 사용합니다. 상위의 원래 실험 폴더 전체를 올리지 않습니다. GitHub에 빈 저장소를 만든 뒤 다음 명령의 `OWNER/lerobot_practice`를 바꿉니다.

```bash
git init -b main
git add .
git status --short
git commit -m "Add SO-101 sim-to-sim and real-to-real experiment code"
git remote add origin https://github.com/OWNER/lerobot_practice.git
git push -u origin main
```

프로젝트 자체 코드의 공개 라이선스는 소유자가 선택해야 합니다. 현재 `licenses/`는 외부 코드의 라이선스이며 전체 프로젝트에 대한 라이선스 선언은 아닙니다.
