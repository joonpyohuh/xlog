"""Global configuration for xlog.

Target user: solo creators producing 3+ shorts per week. Every default
below optimizes for (a) turnaround time and (b) the most universally
accepted mainstream shorts editing style.

CutClaw-style philosophy: one config module drives the whole pipeline.
Override via environment variables where noted.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
# Vercel Functions are read-only except /tmp.
ON_VERCEL = bool(os.environ.get("VERCEL"))

# ------------------ API ------------------ #
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# Second model for cross-verification (hallucination reduction). Override with
# XLOG_OPENAI_MODEL if this id is unavailable on your account.
OPENAI_MODEL = os.environ.get("XLOG_OPENAI_MODEL", "gpt-5")
# Cross-check Claude's outputs with OpenAI: moment verification + second
# judging opinion. Auto-disabled when no OpenAI key is configured.
# All cross-checks FAIL OPEN: an OpenAI error never blocks the pipeline.
CROSS_CHECK = bool(OPENAI_API_KEY)
# Claude model used for all LLM stages (vision analysis, screenwriting, judging).
CLAUDE_MODEL = os.environ.get("XLOG_CLAUDE_MODEL", "claude-opus-5")
# Effort per stage: extraction runs at low effort (speed), creative stages higher.
ANALYSIS_EFFORT = "low"
WRITER_EFFORT = "high"
JUDGE_EFFORT = "medium"
# Concurrent LLM calls during analysis (frame chunks / multiple videos).
MAX_PARALLEL_LLM = 4

# ------------------ Input constraints ------------------ #
MIN_VIDEOS = 1          # requirement 1: at least 1 raw video
MAX_VIDEOS = 3          # requirement 1: at most 3 raw videos
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
MAX_UPLOAD_MB = 2048    # per file

# ------------------ Output (shorts) constraints ------------------ #
SHORT_MIN_SEC = 30      # requirement 2: 30s ~ 60s shorts
SHORT_MAX_SEC = 60
# Standard YouTube Shorts / Instagram Reels format: 9:16 vertical.
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_FPS = 30
VARIANTS_PER_JOB = 2    # requirement 7: produce two candidate edits (A / B)

# ------------------ Free-version branding ------------------ #
# requirement 3: black card, white text, appended at the end of every short.
OUTRO_TEXT = "directed by xlog"
OUTRO_DURATION_SEC = 2.0

# ------------------ Analysis ------------------ #
# Frame sampling rate for the coarse pass. Dense refine (REFINE_FPS) then
# re-reads only the windows the coarse pass marked as worth cutting.
ANALYSIS_FPS = 0.5
REFINE_FPS = 8.0          # 5–10 fps on candidate windows only
REFINE_PAD_SEC = 1.0
REFINE_MAX_WINDOWS = 8    # per source video
REFINE_MIN_SCORE = 6      # intensity or hook_potential
JUDGE_FPS = 1.0           # sample rendered shorts for the pixel judge
JUDGE_MAX_FRAMES = 20     # per variant; keeps the vision call bounded
ANALYSIS_FRAME_LONG_EDGE = 768
MAX_FRAMES_PER_REQUEST = 50

# ------------------ Paths ------------------ #
DATA_DIR = Path("/tmp/xlog") if ON_VERCEL else (BASE_DIR / "data")
UPLOAD_DIR = DATA_DIR / "uploads"
JOBS_DIR = DATA_DIR / "jobs"
RUBRIC_DIR = DATA_DIR / "rubric"
REFERENCE_DIR = DATA_DIR / "references"   # downloaded YouTube style references
TMP_DIR = DATA_DIR / "tmp"

for _d in (UPLOAD_DIR, JOBS_DIR, RUBRIC_DIR, REFERENCE_DIR, TMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# yt-dlp binary/module for YouTube source + style-reference download
# Empty = `python -m yt_dlp` (the pip package). Set XLOG_YTDLP to a binary if you have one.
YTDLP_BIN = os.environ.get("XLOG_YTDLP", "")
REFERENCE_ANALYSIS_FPS = 1.0   # denser sampling: we're studying the *editing*
REFERENCE_MAX_SEC = 180        # only learn from short-form references
SOURCE_MAX_SEC = 30 * 60       # long-form YouTube source cap
# OpenAI DPO base. gpt-5 is not fine-tunable; 4.1-mini is.
FT_BASE_MODEL = os.environ.get("XLOG_FT_BASE_MODEL", "gpt-4.1-mini-2025-04-14")
DPO_BETA = float(os.environ.get("XLOG_DPO_BETA", "0.1"))

# ------------------ ffmpeg ------------------ #
FFMPEG_BIN = os.environ.get("XLOG_FFMPEG", "ffmpeg")
FFPROBE_BIN = os.environ.get("XLOG_FFPROBE", "ffprobe")
RENDER_PRESET = "veryfast"   # speed over marginal size — turnaround matters
RENDER_CRF = "21"

# ------------------ Supabase (xlog cloud DB for learning memory) ------------------ #
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")  # server-side only; anon or service_role
