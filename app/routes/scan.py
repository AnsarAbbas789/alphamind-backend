"""
AlphaMind — routes/scan.py
=====================================================================
Frontend یہاں سے ہر 30-60 سیکنڈ میں پوچھے گا "ابھی watchlist کی
حالت کیا ہے؟" — یہ کوئی نیا حساب نہیں لگاتا، صرف scanner.py کے
پہلے سے تیار نتیجے میں سے پڑھ کر دیتا ہے (بہت تیز، سستا)۔
=====================================================================
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.auth import AuthenticatedUser, verify_supabase_token
from app.core.scanner import get_scan_results, SCAN_INTERVAL_SECONDS

router = APIRouter()


@router.get("/api/scan-results")
async def scan_results(user: AuthenticatedUser = Depends(verify_supabase_token)):
    return {
        "results": get_scan_results(),
        "scan_interval_seconds": SCAN_INTERVAL_SECONDS,
    }