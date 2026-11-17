import os
from slowapi import Limiter
from starlette.requests import Request

TRUSTED_PROXY_HOPS = int(os.getenv("TRUSTED_PROXY_HOPS", "1"))


def client_ip(request: Request) -> str:
    peer = request.client.host if request.client else "unknown"
    if TRUSTED_PROXY_HOPS <= 0:
        return peer
    parts = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
    if len(parts) >= TRUSTED_PROXY_HOPS:
        return parts[-TRUSTED_PROXY_HOPS]
    return peer


def rate_key(request: Request) -> str:
    uid = getattr(request.state, "uid", None)
    return f"u:{uid}" if uid else f"ip:{client_ip(request)}"


limiter = Limiter(key_func=rate_key)