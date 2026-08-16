"""xlog — raw video in, shorts out. The pilot creator's local
criteria-building tool."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.routes import router

app = FastAPI(title="xlog", version="0.2.0")
app.include_router(router)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")
