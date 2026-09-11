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

async def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ")[1]
    try:
        sb = get_supabase()
        user = sb.auth.get_user(token)
        return user.user
    except Exception:
        return None

async def require_auth(authorization: str = Header(None)):
    user = await get_current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail={
            "code": "UNAUTHORIZED",
            "message": "Login required"
        })
    return user

async def require_premium(authorization: str = Header(None)):
    user = await require_auth(authorization)
    sb = get_supabase()
    profile = sb.table("profiles")\
        .select("plan")\
        .eq("id", str(user.id))\
        .single()\
        .execute()
    if not profile.data or profile.data["plan"] != "premium":
        raise HTTPException(status_code=403, detail={
            "code": "PREMIUM_REQUIRED",
            "message": "Premium subscription required"
        })
    return user