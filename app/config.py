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
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# GPT Terra: footage frames + shot plans (instruction-following, not Sol reasoning).
# Grok: live X search + judge. Gemini: fallback if GPT is down. GPT-mini: verifier.
GEMINI_MODEL = os.environ.get("XLOG_GEMINI_MODEL", "gemini-3.5-flash-lite")
GROK_MODEL = os.environ.get("XLOG_GROK_MODEL", "grok-4.6")
OPENAI_EDITOR_MODEL = os.environ.get("XLOG_OPENAI_EDITOR_MODEL", "gpt-5.6-terra")
# none = no extra thinking tokens. Editing is quoting the frame, not solving.
OPENAI_EDITOR_EFFORT = os.environ.get("XLOG_OPENAI_EDITOR_EFFORT", "none")
# Set 1 to force Grok shot plans even when GPT is available.
USE_GROK_FOR_WRITER = os.environ.get("XLOG_USE_GROK_FOR_WRITER", "0") != "0"
# x_search before writing: on = always (meme voice needs live slang).
# auto = only when the brief sounds trendy/funny. off = never.
GROK_TRENDS = os.environ.get("XLOG_GROK_TRENDS", "on")
# Local default: both quality chips work. Paid Pro later sets this to 0.
PRO_UNLOCKED = os.environ.get("XLOG_PRO_UNLOCKED", "1") != "0"
QUALITY_DEFAULT = os.environ.get("XLOG_QUALITY", "fast")
OPENAI_MODEL = os.environ.get("XLOG_OPENAI_MODEL", "gpt-4.1-mini")
CROSS_CHECK = bool(OPENAI_API_KEY or XAI_API_KEY)
# Grok reasoning_effort. Default on the API is high — always set it.
WRITER_EFFORT = os.environ.get("XLOG_WRITER_EFFORT", "high")
JUDGE_EFFORT = os.environ.get("XLOG_JUDGE_EFFORT", "medium")
# Kept for the JPEG fallback path if Gemini video upload fails.
COARSE_EFFORT = "medium"
ANALYSIS_EFFORT = "high"
# Concurrent LLM calls during analysis (frame chunks / multiple videos).
MAX_PARALLEL_LLM = 4

# ------------------ Input constraints ------------------ #
MIN_VIDEOS = 1          # requirement 1: at least 1 raw video
MAX_VIDEOS = 3          # requirement 1: at most 3 raw videos
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
# Local vlog .mov from a phone/camera is often 4–12GB. 2GB was choking demos.
MAX_UPLOAD_MB = int(os.environ.get("XLOG_MAX_UPLOAD_MB", "16384"))
# Rendered A/B shorts stay on disk this long after a job finishes. Source
# footage is dropped immediately — that is what actually fills the disk.
SHORTS_RETENTION_SEC = int(os.getenv("XLOG_SHORTS_RETENTION_SEC", str(7 * 24 * 3600)))

# ------------------ Output (shorts) constraints ------------------ #
SHORT_MIN_SEC = 30      # requirement 2: 30s ~ 60s shorts
SHORT_MAX_SEC = 60
# Hard floor: a plan that cannot be stretched this far is a broken edit, not a
# short. validate_variants() stretches and gap-fills until it clears this.
SHORT_FLOOR_SEC = 20
# Standard YouTube Shorts / Instagram Reels format: 9:16 vertical.
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_FPS = 30
# Longform 16:9: shrink to fit 9:16 and pad (letterbox). Never center-crop
# the picture to fake a vertical short. Set 0 to restore cover-crop.
OUTPUT_FIT = os.environ.get("XLOG_OUTPUT_FIT", "1") != "0"
VARIANTS_PER_JOB = 2    # requirement 7: produce two candidate edits (A / B)
# A longform that already went through a first edit carries burned-in subtitles
# in the bottom band. Cropping that band off is cheaper and cleaner than
# inpainting, and the 9:16 crop throws away most of the frame anyway.
SOURCE_CAPTION_BAND = 0.18

# ------------------ Free-version branding ------------------ #
# requirement 3: black card, white text, appended at the end of every short.
OUTRO_TEXT = "directed by xlog"
OUTRO_DURATION_SEC = 2.0

# ------------------ Analysis ------------------ #
# Frame sampling rate for the coarse pass. Dense refine (REFINE_FPS) then
# re-reads only the windows the coarse pass marked as worth cutting.
ANALYSIS_FPS = 0.5
COARSE_MAX_FRAMES = 96    # long vlogs: drop fps so one video stays ~4 LLM calls
REFINE_FPS = 2.0          # enough to catch a 1s beat without 8fps payload
REFINE_PAD_SEC = 1.0
REFINE_MAX_WINDOWS = 3    # per source video
REFINE_MIN_SCORE = 6      # intensity or hook_potential
JUDGE_FPS = 0.8           # sample rendered shorts for the pixel judge
JUDGE_MAX_FRAMES = 14     # per variant; keeps the vision call bounded
ANALYSIS_FRAME_LONG_EDGE = 512
# Image tokens are (w*h)/750, and the coarse pass is half input cost. It only
# has to answer "is this window worth a dense look" — the refine pass re-reads
# whatever it picks at full size, so triage runs on smaller frames.
COARSE_FRAME_LONG_EDGE = 384
MAX_FRAMES_PER_REQUEST = 24

# ------------------ Paths ------------------ #
DATA_DIR = Path("/tmp/xlog") if ON_VERCEL else (BASE_DIR / "data")
UPLOAD_DIR = DATA_DIR / "uploads"
JOBS_DIR = DATA_DIR / "jobs"
RUBRIC_DIR = DATA_DIR / "rubric"
REFERENCE_DIR = DATA_DIR / "references"   # downloaded YouTube style references
TMP_DIR = DATA_DIR / "tmp"
CREDITS_DIR = DATA_DIR / "credits"

for _d in (UPLOAD_DIR, JOBS_DIR, RUBRIC_DIR, REFERENCE_DIR, TMP_DIR, CREDITS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# yt-dlp binary/module for YouTube source + style-reference download
# Empty = `python -m yt_dlp` (the pip package). Set XLOG_YTDLP to a binary if you have one.
YTDLP_BIN = os.environ.get("XLOG_YTDLP", "")
# Comma-separated browsers to try after an anonymous 403 (chrome is often locked on Windows).
YTDLP_COOKIES_FROM_BROWSER = os.environ.get(
    "XLOG_YTDLP_COOKIES_FROM_BROWSER", "edge,chrome,firefox"
)
# A hung download must not pin a job in "ingesting" forever.
YTDLP_TIMEOUT_SEC = int(os.environ.get("XLOG_YTDLP_TIMEOUT_SEC", "300"))
REFERENCE_ANALYSIS_FPS = 1.0   # denser sampling: we're studying the *editing*
REFERENCE_MAX_SEC = 180        # only learn from short-form references
SOURCE_MAX_SEC = int(os.environ.get("XLOG_SOURCE_MAX_SEC", str(60 * 60)))  # default 60min
# Learned taste is injected as a bounded, reinforcement-ranked rule list.
# One-off rules stay out until they show up twice.
TASTE_RULES_IN_PROMPT = int(os.environ.get("XLOG_TASTE_RULES", "8"))
TASTE_MIN_SEEN = int(os.environ.get("XLOG_TASTE_MIN_SEEN", "2"))
TASTE_RULE_CHARS = 200
TASTE_WEIGHT_NUDGE = 2

# ------------------ ffmpeg ------------------ #
FFMPEG_BIN = os.environ.get("XLOG_FFMPEG", "ffmpeg")
FFPROBE_BIN = os.environ.get("XLOG_FFPROBE", "ffprobe")
# Windows default cp949 chokes on ffmpeg stderr for non-ASCII paths → stdout=None.
SUBPROCESS_TEXT = {
    "capture_output": True,
    "text": True,
    "encoding": "utf-8",
    "errors": "replace",
}
RENDER_PRESET = "veryfast"   # speed over marginal size — turnaround matters
RENDER_CRF = os.environ.get("XLOG_RENDER_CRF", "18")   # 18 ≈ visually lossless
# Shots inside one variant encode concurrently; both variants also run in parallel.
RENDER_WORKERS = int(os.environ.get("XLOG_RENDER_WORKERS", "4"))
# Two shots may not reuse the same footage; overlap beyond this is trimmed away.
MAX_SHOT_OVERLAP_SEC = 0.5

# ------------------ Pro editor time-savers (CapCut / Premiere chores) ---------- #
# Silence/filler: ffmpeg is free; Whisper is only for filler words on the
# *shot spans* (~1 min of audio), not the whole vlog.
STT_MODEL = os.environ.get("XLOG_STT_MODEL", "whisper-1")
SILENCE_NOISE_DB = float(os.environ.get("XLOG_SILENCE_DB", "-35"))
SILENCE_MIN_SEC = float(os.environ.get("XLOG_SILENCE_MIN_SEC", "0.2"))
BREATH_KEEP_SEC = 0.08
ZOOM_FATIGUE_SEC = 15.0
ZOOM_SCALES = (1.10, 1.15)
SFX_GAIN = 0.14          # linear; keeps SFX under dialogue
SFX_REPEAT_WINDOW_SEC = 12.0
# YouTube Shorts / Reels right-rail + bottom UI. Captions stay inside this.
SAFE_TOP = 0.12
SAFE_BOTTOM = 0.78
SAFE_SIDE = 0.10
SAFE_RIGHT_EXTRA = 0.08  # like/comment column

# ------------------ Narration (captions read aloud, in sync) ------------------ #
NARRATION = os.environ.get("XLOG_NARRATION", "1") != "0"
# edge-tts: free, no key.
EDGE_TTS_VOICE = os.environ.get("XLOG_EDGE_TTS_VOICE", "ko-KR-SunHiNeural")
NARRATION_DUCK = float(os.environ.get("XLOG_NARRATION_DUCK", "0.28"))
NARRATION_MAX_TEMPO = 1.6   # speed-up ceiling when a line overruns its shot

# ------------------ Supabase (xlog cloud DB for learning memory) ------------------ #
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")  # server-side only; anon or service_role
