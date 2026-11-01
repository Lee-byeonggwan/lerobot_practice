# 소스 구성과 이식 변경

## 가져온 범위

- 원래 `leisaac_practice/scripts`: SO-101 cup 장면 생성, scripted 시연 수집·검증, LeRobot 변환, SmolVLA 학습, IPC 추론, 시뮬레이션 평가·채점.
- 원래 `leisaac_practice/leisaac`: 고정 upstream 커밋 대비 15개 파일의 호환 패치. 전체 upstream 저장소는 포함하지 않음.
- 원래 `so101_smolvla`: real v3 학습 설정, `merge_lora.py`, `eval_offline.py`, 최종 `real_eval/run_real_v3.py`의 startup-v4, 영상/평가 지원 코드, native RGB 서버, 오프라인 테스트.
- `lerobot_practice`의 ACT/PushT는 요청 범위 밖이므로 포함하지 않음.

## 이식하면서 바꾼 부분

- `run_real_v3.py` → `run_real.py`, `serve_real_v3.py` → `policy_server.py`: 개인 경로·USB serial·COM·카메라 번호·dataset ID를 설정 파일로 이동. 모터 준비 및 action 실행 논리는 유지.
- 실제 시연 수집/학습을 `record_real.py`와 `train_real.py` CLI로 정리. 데이터/출력 경로는 인자로 지정하고 출력 덮어쓰기는 거부.
- sim 수집 초기 manifest를 생성하는 `init_collection.py`와 정확한 20개 위치 설정 추가.
- 이전 interactive editor가 만들던 초기 USD prim을 `prepare_editor_seed.py`로 생성. 최종 collection scene의 물리·배치 설정은 유지.
- sim 평가의 checkpoint/dataset/output/device/회차를 CLI로 이동. GPU 번호 0/1 전용 제한을 일반화하고 평가의 렌더 GPU도 지정 device에 맞춤.
- sim 학습의 offline 강제 설정 제거. 2,000-step 첫 단계 후 완료된 checkpoint를 두 GPU로 resume하도록 정리. 다른 프로세스를 종료하는 기존 운영 스크립트는 제외.
- 학습 환경이 원래 다른 가상환경의 site-packages를 참조하던 방식은 독립 설치 명세로 대체.
- 실제 평가 데이터/영상/개인 calibration/가중치/실험 ZIP은 제외. `configs/collection_plan.json`은 초기 위치 실험 설정만 포함.

현재 소스 파일의 해시는 `SOURCE_MANIFEST.json`에 기록합니다. 문서/코드의 원본 출처 목록이며 학습 데이터나 성공률 기록이 아닙니다.
