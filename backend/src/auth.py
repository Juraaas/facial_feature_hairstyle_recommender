import os
from fastapi import Header, HTTPException, Request
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


def get_current_user(request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    try:
        user = get_supabase().auth.get_user(token).user
    except Exception:
        return None
    if user:
        request.state.uid = str(user.id)
    return user


def require_auth(request: Request, authorization: str = Header(None)):
    user = get_current_user(request, authorization)
    if not user:
        raise HTTPException(status_code=401, detail={
            "code": "UNAUTHORIZED", "message": "Login required"
        })
    return user