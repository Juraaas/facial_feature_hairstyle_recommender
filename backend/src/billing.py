from src.auth import get_supabase
from src.logger import get_logger

log = get_logger(__name__)

COSTS = {"analysis": 1, "preview": 3}
SIGNUP_BONUS = 1
_REASONS = {"analysis": "spend_analysis", "preview": "spend_preview"}


def _rpc(fn: str, params: dict):
    return get_supabase().rpc(fn, params).execute().data


def ensure_signup_bonus(user_id: str):
    try:
        _rpc("grant_credits", {
            "p_user": user_id, "p_amount": SIGNUP_BONUS, "p_reason": "signup_bonus",
            "p_ref_type": "user", "p_ref_id": user_id,
        })
    except Exception:
        log.exception("signup bonus failed for %s", user_id)


def get_balance(user_id: str) -> int:
    res = get_supabase().table("profiles").select("credits").eq("id", user_id).limit(1).execute()
    return int(res.data[0]["credits"]) if res.data else 0


def spend(user_id: str, kind: str, ref_type: str, ref_id: str) -> int:
    return int(_rpc("spend_credits", {
        "p_user": user_id, "p_amount": COSTS[kind], "p_reason": _REASONS[kind],
        "p_ref_type": ref_type, "p_ref_id": ref_id,
    }))


def refund_generation(gen_id: str):
    try:
        return _rpc("refund_generation", {"p_gen": gen_id})
    except Exception:
        log.error("REFUND_FAILED generation=%s", gen_id, exc_info=True)
        return None


def grant(user_id: str, amount: int, reason: str, ref_type: str, ref_id: str,
          note: str | None = None, by: str | None = None) -> int:
    return int(_rpc("grant_credits", {
        "p_user": user_id, "p_amount": amount, "p_reason": reason,
        "p_ref_type": ref_type, "p_ref_id": ref_id, "p_note": note, "p_by": by,
    }))


def claim_gender_fix(user_id: str, analysis_id: str) -> bool:
    try:
        return bool(_rpc("claim_gender_fix", {"p_user": user_id, "p_analysis": analysis_id}))
    except Exception:
        log.exception("claim_gender_fix failed")
        return False


def list_packs() -> list:
    res = (get_supabase().table("credit_packs")
           .select("id,name_pl,name_en,credits,price_cents,currency")
           .eq("active", True).order("sort_order").execute())
    return res.data or []


def get_pack(pack_id: str):
    res = (get_supabase().table("credit_packs").select("*")
           .eq("id", pack_id).eq("active", True).limit(1).execute())
    return res.data[0] if res.data else None