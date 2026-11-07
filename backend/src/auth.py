import os
from fastapi import Header, HTTPException
from supabase import create_client

_supabase = None


def get_supabase():
    global _supabase
    if _supabase is None:
        _supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_KEY"]
        )
    return _supabase


def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    try:
        return get_supabase().auth.get_user(token).user
    except Exception:
        return None

def require_auth(authorization: str = Header(None)):
    user = get_current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail={
            "code": "UNAUTHORIZED", "message": "Login required"
        })
    return user

def require_premium(authorization: str = Header(None)):
    user = require_auth(authorization)
    res = (
        get_supabase().table("profiles")
        .select("plan").eq("id", str(user.id)).limit(1).execute()
    )
    plan = res.data[0]["plan"] if res.data else None
    if plan != "premium":
        raise HTTPException(status_code=403, detail={
            "code": "PREMIUM_REQUIRED", "message": "Premium subscription required",
        })
    return user