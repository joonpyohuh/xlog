# xlog

**주 3회 이상 숏폼을 만드는 1인 크리에이터**를 위한 도구.
Raw video(1~3개) + 자유 요청 한 줄 → AI가 핵심을 추출 → 30~60초 세로형(9:16) 숏츠 2편(A/B) 제작 → AI 심사 + **파일럿 크리에이터(당신)의 선택**으로 편집 기준을 학습하는 로컬 웹앱.

> 현재 로컬 버전의 정체성: **"내가 xlog를 만들면서 쓰는, 내가 기준을 만드는 앱."**
> 내 평가(어느 편집이 좋은지 + 이유)와 내가 주는 YouTube 레퍼런스 링크로부터 기준(rubric)이 계속 진화한다.

CutClaw(GVCLab) 분석 기반 설계 — [docs/CUTCLAW_ANALYSIS.md](docs/CUTCLAW_ANALYSIS.md).

## 제품 원칙 (V1)

1. **타깃**: 일반 사용자 X, 주 3회+ 숏폼 제작 1인 크리에이터 O
2. **AI 심사 기준**: "가장 보편적이고 무난한 편집 형태" — 실제 인기 숏츠의 관행을 기준으로 삼는다
3. **속도**: 결과물이 나오기까지의 시간을 최소화 (아래 '속도 설계' 참고)
4. **추상적 요청 수용**: "재밌는 자막 넣고 눈에 확 들어오는 효과 넣어줘" 같은 프롬프트를 그대로 받는다
5. **출력**: 유튜브 쇼츠/릴스 표준 9:16 (1080×1920)

## 요구사항 매핑

| 기능 | 구현 위치 |
|---|---|
| 입력: raw video 1~3개 + 자유 프롬프트 | `app/api/routes.py`, `app/pipeline/ingest.py` |
| AI 핵심 추출 → 30~60초 숏츠 | `app/pipeline/highlight.py` → `screenwriter.py` |
| 자막 번인(메인스트림 스타일 normal/emphasis) | `app/pipeline/captions.py` + `render.py` |
| 무료버전 브랜딩: "directed by xlog" | `app/pipeline/outro.py` |
| 숏츠 form (LLM이 학습·갱신) | `app/knowledge/shorts_form.py` |
| A/B 2편 제작 + AI 심사 | `app/evaluation/judge.py` (기준: 보편적 편집) |
| 내 평가로 기준 학습 | `app/evaluation/feedback.py` → `rubric.py` (버전 관리) |
| **YouTube 링크로 스타일 학습** | `app/knowledge/reference.py` (yt-dlp + Claude vision) |
| 로컬 사이트 | `python run.py` → http://127.0.0.1:8321 |

## 속도 설계

- 프레임 분석: 전 영상의 청크를 **동시 병렬** LLM 호출 (`MAX_PARALLEL_LLM=4`)
- 분석 단계는 `effort="low"`(빠름), 편집안 작성만 `high`
- 렌더링 A · 렌더링 B · AI 심사를 **3-way 병렬** 실행 (심사는 편집안 메타데이터만 필요)
- ffmpeg `veryfast` 프리셋, 자막 오버레이를 컷 인코딩과 같은 패스에서 처리

## 환각(hallucination) 방지 — 3중 방어

1. **결정론적 검증** (`app/pipeline/verify.py`): 편집안의 모든 컷 범위를 실제 영상 길이로 클램핑, 1초 미만·잘못된 인덱스 샷 제거 (LLM 아님, 코드)
2. **모먼트 교차 검증**: Claude가 추출한 모먼트를 **GPT가 같은 프레임으로 독립 재검증** — 프레임이 뒷받침하지 않는 모먼트(환각)는 편집안 작성 전에 제거
3. **이중 심사**: Claude 심사와 병렬로 GPT가 독립 심사 → 일치하면 "2모델 일치" 배지, 불일치하면 양쪽 추천을 모두 표시(숨기지 않음)

OpenAI 호출은 전부 **fail-open**: 오류가 나도 파이프라인은 Claude 단독으로 계속 진행. `OPENAI_API_KEY` 미설정 시 자동 비활성화. 모델은 `XLOG_OPENAI_MODEL`(기본 `gpt-5`)로 변경 가능.

## 기준(rubric) 학습 — 두 가지 경로

1. **내 평가**: A/B 중 선택 + 코멘트 → LLM이 rubric 개정 (AI 심사와 내 선택이 불일치하면 가장 강한 학습 신호로 강조됨)
2. **레퍼런스 링크**: 좋아하는 숏츠의 YouTube 링크 + "왜 좋은지" 메모 → 다운로드 → 편집 스타일 역분석(훅 구성, 컷 리듬, 자막 사용) → 구체적 규칙으로 증류되어 rubric `preferences`에 병합

모든 rubric 버전은 `data/rubric/history/`에 스냅샷 보존. 레퍼런스 기록은 `data/rubric/references.jsonl`.

## 설치 및 실행

```bash
# 사전 조건: Python 3.11+, ffmpeg/ffprobe가 PATH에 있어야 함
cd xlog
pip install -r requirements.txt
# .env에 ANTHROPIC_API_KEY 설정 (이미 생성됨)
python run.py     # → http://127.0.0.1:8321
```

## 현재 한계 (다음 단계)

- 크롭이 센터 크롭 — 주체 인식(face-aware) 크롭으로 업그레이드 예정
- 심사가 편집안 메타데이터 기반 — 렌더 결과 프레임 재분석 심사 예정
- "효과"는 자막 2종만 — 줌/펀치인·트랜지션·사운드 이펙트는 다음 단계
- 분석 캐시 없음(같은 원본 재사용 시 재분석)
- ASR(음성 인식) 미연동 — 대사 기반 모먼트 추출·자막 싱크는 다음 단계
- 레퍼런스 학습은 1fps 프레임 기반 근사 — 정밀 컷 감지(scenedetect)는 다음 단계

## 보안 주의

`.env`는 gitignore에 포함. **채팅/문서에 노출된 API 키는 반드시 재발급하세요.**
