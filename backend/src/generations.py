import os
from datetime import datetime, timezone
from src.auth import get_supabase
from src.logger import get_logger

log = get_logger(__name__)

DAILY_LIMIT = int(os.getenv("MAX_GENERATIONS_PER_DAY", "300"))

def start_generation(user_id: str, style_name: str, color_id: str, gender: str):
    try:
        res = get_supabase().table("generations").insert({
            "user_id": user_id, "style_name": style_name,
            "color_id": color_id, "gender": gender, "status": "pending",
        }).execute()
        return res.data[0]["id"]
    except Exception:
        log.exception("start_generation failed")
        return None


def finish_generation(gen_id, status: str, error_code=None, duration_ms=None):
    if not gen_id:
        return
    try:
        get_supabase().table("generations").update({
            "status": status,
            "error_code": error_code,
            "duration_ms": duration_ms,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", gen_id).execute()
    except Exception:
        log.exception("finish_generation failed")


def daily_limit_reached() -> bool:
    try:
        since = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        res = (
            get_supabase().table("generations")
            .select("id", count="exact").gte("created_at", since).limit(1).execute()
        )
        return (res.count or 0) >= DAILY_LIMIT
    except Exception:
        log.exception("daily_limit_reached check failed")
        return False