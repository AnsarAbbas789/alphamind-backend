"""
AlphaMind — core/auth.py
=====================================================================
Supabase اب دو طرح کے JWT سائننگ سپورٹ کرتا ہے:
  - Legacy: HS256 (ایک شیئرڈ secret سے sign/verify)
  - نیا: ES256 (asymmetric — public/private key pair سے)
نیا پروجیکٹ بننے پر Supabase اب ڈیفالٹ طور پر ES256 استعمال کرتا ہے۔
یہ فائل ٹوکن کا header دیکھ کر خودکار طور پر صحیح طریقہ چنتی ہے —
دونوں صورتیں سنبھالی گئی ہیں، اس لیے چاہے آپ کا پروجیکٹ کسی بھی
طریقے پر ہو، یہ کام کرے گا۔
=====================================================================
"""

from __future__ import annotations

import jwt
from fastapi import Header, HTTPException, status
from jwt import PyJWKClient

from app.config import get_settings

_jwks_client_cache: PyJWKClient | None = None


class AuthenticatedUser:
    def __init__(self, user_id: str, email: str | None):
        self.user_id = user_id
        self.email = email


def _get_jwks_client(supabase_url: str) -> PyJWKClient:
    global _jwks_client_cache
    if _jwks_client_cache is None:
        jwks_url = f"{supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"
        _jwks_client_cache = PyJWKClient(jwks_url)
    return _jwks_client_cache


async def verify_supabase_token(authorization: str = Header(default=None)) -> AuthenticatedUser:
    settings = get_settings()

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="missing_authorization_header")

    token = authorization.removeprefix("Bearer ").strip()

    try:
        header = jwt.get_unverified_header(token)
        alg = header.get("alg", "HS256")

        if alg == "ES256":
            if not settings.SUPABASE_URL:
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="server_auth_not_configured")
            jwks_client = _get_jwks_client(settings.SUPABASE_URL)
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token, signing_key.key, algorithms=["ES256"],
                audience="authenticated", options={"verify_exp": True},
            )
        else:
            if not settings.SUPABASE_JWT_SECRET:
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="server_auth_not_configured")
            payload = jwt.decode(
                token, settings.SUPABASE_JWT_SECRET, algorithms=["HS256"],
                audience="authenticated", options={"verify_exp": True},
            )
    except HTTPException:
        raise
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="token_expired")
    except Exception:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid_token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="token_missing_subject")

    return AuthenticatedUser(user_id=user_id, email=payload.get("email"))