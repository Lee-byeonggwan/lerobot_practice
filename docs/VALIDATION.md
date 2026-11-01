# 배포본 검증 기록

2026-11-02, 기존 실험 환경에서 수행했습니다.

- 실제 모터를 mock으로 대체한 runtime 테스트 **19개 통과**. 시작 자세 범위, torque 해제/부분 실패 정리, 목표 기록 순서, 통신 재시도, 비정상 action 제외, PNG RGB 무손실, 영상 기록, 전송 중 action loop 유지 등을 검사했습니다.
- 모든 Python/JSON 파일 구문 검사, shell 설치/활성화 스크립트 `bash -n` 통과.
- 경량 CLI 7개의 `--help`와 빈 collection manifest/20개 위치 초기화 확인.
- 원래 LeIsaac 고정 커밋의 깨끗한 임시 소스에 `git apply --check` 통과.
- 원본 공식 에셋만 복사한 임시 폴더에서 `prepare_wood_cube`, `prepare_cup_scene`, `prepare_real_style_layout`, `prepare_collection_scene` **네 단계 정상 완료**. 수동 editor 파일 없이 seed를 생성했습니다.
- 생성한 collection scene의 활성 physics/xform/points/extent 속성 **243개**, robot layer 속성 **529개**가 원래 실험의 대응 속성과 모두 일치했습니다. 이 검사는 USD 구성 비교이며 rendering/동적 rollout 재검증은 아닙니다.
- 공식 GitHub API에서 IsaacLab 커밋 및 Table with Cube 다운로드 항목 확인. 공식 Hugging Face 파일 메타데이터의 robot USD SHA-256이 로컬 원본과 일치함을 확인했습니다.
- `uv pip compile`로 Python 3.12 Linux/Windows requirements의 의존성 해석 통과. 패키지 설치나 DLL/CUDA 실행 검증은 별도입니다.
- Markdown 상대 링크, 데이터/가중치/설정/결과/외부 소스의 Git 제외 규칙, 개인 경로·원래 서버 주소·USB serial·토큰 패턴 검사 통과.

새 환경 전체 설치, 새로운 시연 수집, 20,000-step 재학습, GPU rendering/평가 rollout, Windows 물리 로봇 동작은 이번 정리 작업에서 다시 실행하지 않았습니다. Windows 설치 명세는 원본 핵심 버전과 API에 맞춘 이식 명세입니다. 성공률 기록과 데이터/가중치는 포함하지 않았습니다.
