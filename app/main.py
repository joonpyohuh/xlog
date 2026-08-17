"""xlog — raw video in, shorts out. The pilot creator's local
criteria-building tool."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.routes import router
from app.storage import jobs as job_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    stranded = job_store.fail_stranded()
    if stranded:
        print(f"[xlog] closed {stranded} job(s) stranded by the last restart")
    yield


app = FastAPI(title="xlog", version="0.2.0", lifespan=lifespan)
app.include_router(router)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})
