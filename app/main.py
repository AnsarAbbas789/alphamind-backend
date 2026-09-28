"""
AlphaMind — main.py
=====================================================================
FastAPI ایپ کا مرکزی entry point۔

⚠️ 4 ستمبر 2026 اضافہ: سرور شروع ہوتے ہی Live Scanner کا background
loop خودکار شروع ہو جاتا ہے (scanner_background_loop) — یہ ہمیشہ پس
منظر میں چلتا رہتا ہے، ہر 5 منٹ بعد پوری watchlist چیک کرتا ہے۔
=====================================================================
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.core.scanner import scanner_background_loop
from app.routes import scan as scan_routes
from app.routes import signal as signal_routes

settings = get_settings()

logging.basicConfig(level=settings.LOG_LEVEL)
logging.getLogger("httpx").setLevel(logging.WARNING)  # candle-fetch کے شور کو دبائیں، صاف scanner لاگ نظر آئے
logger = logging.getLogger("alphamind")


@asynccontextmanager
async def lifespan(app: FastAPI):
    scanner_task = asyncio.create_task(scanner_background_loop())
    logger.info("Live Scanner background loop started.")
    yield
    scanner_task.cancel()


app = FastAPI(
    title="AlphaMind AI Signal Engine",
    description="Professional multi-LLM ensemble crypto signal engine",
    version="1.1.0",
    docs_url="/docs" if settings.is_dev else None,
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(signal_routes.router)
app.include_router(scan_routes.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": "Something went wrong. Please try again."},
    )


@app.get("/")
async def root():
    return {"service": "AlphaMind AI Signal Engine", "status": "running"}


@app.get("/health")
async def health():
    from app.llm.registry import build_active_providers
    active = build_active_providers(settings)
    return {
        "status": "ok",
        "active_llm_providers": [p.name for p in active],
        "active_provider_count": len(active),
        "supabase_configured": bool(settings.SUPABASE_JWT_SECRET),
        "default_exchange": settings.DEFAULT_EXCHANGE,
    }