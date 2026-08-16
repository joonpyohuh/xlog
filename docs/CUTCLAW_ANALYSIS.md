# CutClaw 종합 분석 (GVCLab/CutClaw)

> xlog 설계의 참고를 위한 리서치 노트. 분석 시점: 2026-08-16.

## 1. 개요

CutClaw는 "Agentic Hours-Long Video Editing via Music Synchronization" (arXiv:2603.29664, Beijing Jiaotong Univ. · GVC Lab · Tencent ARC Lab)의 공식 구현이다. **수 시간짜리 롱폼 영상 + 음악**을 입력하면 텍스트 지시(instruction) 하나로 음악 비트에 맞춘 몽타주를 만들어주는 연구용 시스템이다. Streamlit UI(`app.py`)와 CLI(`local_run.py`)를 제공하며, LiteLLM을 게이트웨이로 써서 비디오/오디오/에이전트 모델을 각각 다른 프로바이더로 지정할 수 있다.

## 2. 파이프라인 구조 (핵심)

```
[전처리 — 3개 스레드 병렬]
 A. ASR + 화자분리 + 등장인물 식별 (film 모드)
 B. 샷 감지(scenedetect) → 프레임 샘플링(decord) → VLM 캡셔닝
    → 샷을 씬으로 병합(scene_merge) → 씬 단위 분석(scene JSON DB)
 C. 오디오 분석: madmom 기반 다운비트/피치/멜에너지 키포인트
    → 음악 구조(Level-1) + 키포인트 캡션(Level-2)

[에이전트 3인 체제]
 1. Screenwriter (Screenwriter_scene_short.py)
    - 음악 구간 선택 → 전체 구조 제안 → shot_plan JSON 생성
    - 섹션별 테마/내러티브/추천 씬 인덱스 지정
 2. Editor (core.py, EditorCoreAgent)
    - shot_plan의 각 shot에 대해 tool-calling 루프 실행. 도구 4개:
      * semantic_neighborhood_retrieval: 추천 씬 ±3 범위의 샷 정보 조회
      * fine_grained_shot_trimming: 특정 구간 프레임을 VLM에 보내 세밀 분석
      * review_clip: 이미 사용한 구간과 중복 여부 검사
      * commit: 최종 구간 확정(포맷 검증, 자동 트림, JSON 저장)
    - ParallelShotOrchestrator로 shot 단위 병렬 처리(worker 4)
 3. Reviewer (Reviewer.py)
    - commit 전 게이트: VLM 주인공 등장 비율 검사(MIN_PROTAGONIST_RATIO 0.7),
      길이/중복 검증. 불합격 시 피드백과 함께 재시도

[렌더링] render/render_video.py (1,613줄)
 - shot_point JSON → ffmpeg 컷/컨캣, 9:16 크롭, 훅 대사(자막) 오버레이,
   ending.mp4 브랜딩 카드 첨부
```

## 3. 주목할 설계 패턴

| 패턴 | 내용 | xlog 반영 |
|---|---|---|
| **캡션 DB 캐싱** | 첫 실행 때 영상을 구조화된 JSON DB(샷/씬 캡션)로 "해체"하고 이후 편집은 캐시 재사용 | 반영 예정(2차). 현재는 매 작업마다 분석 |
| **에이전트 역할 분리** | 작가(계획)→편집자(선택)→검토자(검증) 3단 루프 | xlog는 screenwriter→judge 2단 + 인간 평가로 단순화 |
| **탐색 범위 제한** | 에이전트가 추천 씬 ±3 밖을 검색하면 도구가 거부 → 환각 방지 | moments 목록 안에서만 자르도록 프롬프트 제약 |
| **중복 호출 감지** | 같은 time_range 3회 반복 시 대화 리셋(RESTART) | 스켈레톤에선 미구현, 운영 시 필요 |
| **형식 강제 파싱** | `[shot: HH:MM:SS to HH:MM:SS]` 정규식 + 재시도 | xlog는 structured outputs(json_schema)로 대체 — 더 견고 |
| **컨텍스트 관리** | 도구 결과 truncation, internal_scenes 상한, 스냅샷 롤백 | complete_json 단발 호출 구조라 현재 불필요 |
| **설정 중앙화** | config.py 단일 모듈 + CLI `--config.X` 오버라이드 | app/config.py로 동일 패턴 채택 |
| **레이트리밋 백오프** | retry-after 파싱 + 지수 백오프 | Anthropic SDK 기본 재시도에 위임 |

## 4. CutClaw의 한계 (xlog 차별점)

1. **음악 중심**: 모든 컷 타이밍이 음악 키포인트에 종속. xlog는 음악 없이 내용(핵심 모먼트) 중심.
2. **무거운 의존성**: madmom, decord, torch, whisper 등 GPU 셋업 필요. xlog는 ffmpeg + Claude API만.
3. **단일 영상 입력**: 영상 1개 + 오디오 1개. xlog는 영상 1~3개 조합.
4. **학습 루프 없음**: 사용자 피드백이 시스템에 축적되지 않음. xlog의 핵심 차별점 = rubric 학습 루프.
5. **A/B 비교 없음**: 결과물 1개. xlog는 2안 생성 → AI 심사 → 인간 최종 선택.
6. **크롭**: CutClaw는 VLM 기반 주체 인식 크롭(문서상) — xlog 스켈레톤은 센터 크롭(추후 업그레이드 포인트).

## 5. xlog가 나중에 가져올 만한 것

- **scene_merge**: 샷을 의미 단위 씬으로 묶어 분석 비용 절감 (롱폼 지원 시 필수)
- **Reviewer 게이트**: 렌더 전 자동 품질 검증(주인공 비율, 길이, 중복)
- **fine_grained_shot_trimming**: 선택 구간을 다시 VLM로 확대 분석해 컷 포인트 정밀화
- **훅 대사 자막**: SRT 기반 오프닝 훅 텍스트 오버레이 (render_video.py의 hook_dialogue 로직)
- **캐시 재사용**: 같은 원본으로 여러 숏츠를 만들 때 분석 결과 재사용
