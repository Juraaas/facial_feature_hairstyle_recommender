import re, os
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional
from uuid import UUID, uuid4
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from src import billing
from src.auth import require_auth, get_supabase
from src.generations import finish_generation
from src.limits import limiter, client_ip, TRUSTED_PROXY_HOPS
from src.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/admin")

STUCK_AFTER_MINUTES = 10


def _err(code: str, message: str, status: int) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _fail(where: str, e: Exception) -> HTTPException:
    log.error("ADMIN_ERROR where=%s code=%s message=%s", where,
              getattr(e, "code", None), getattr(e, "message", None) or str(e), exc_info=True)
    return _err("INTERNAL_ERROR", "Something went wrong", 500)


def require_admin(user=Depends(require_auth)):
    try:
        res = get_supabase().table("profiles").select("is_admin").eq("id", str(user.id)).limit(1).execute()
    except Exception as e:
        raise _fail("require_admin", e)
    if not res.data or not res.data[0].get("is_admin"):
        raise _err("FORBIDDEN", "Admin access required", 403)
    return user


def audit(admin, action: str, target_user=None, **details):
    try:
        get_supabase().table("admin_actions").insert({
            "admin_id": str(admin.id), "action": action,
            "target_user": str(target_user) if target_user else None, "details": details,
        }).execute()
    except Exception:
        log.exception("audit failed action=%s", action)


def _count(table: str, since: Optional[str] = None, **eq) -> int:
    q = get_supabase().table(table).select("id", count="exact")
    for k, v in eq.items():
        q = q.eq(k, v)
    if since:
        q = q.gte("created_at", since)
    return q.limit(1).execute().count or 0


@router.get("/stats")
@limiter.limit("30/minute")
def stats(request: Request, admin=Depends(require_admin)):
    now = datetime.now(timezone.utc)
    d1 = (now - timedelta(days=1)).isoformat()
    d7 = (now - timedelta(days=7)).isoformat()
    d30 = (now - timedelta(days=30)).isoformat()
    stuck_before = (now - timedelta(minutes=STUCK_AFTER_MINUTES)).isoformat()
    prefix = os.getenv("REVENUE_SESSION_PREFIX", "cs_")
    try:
        sb = get_supabase()
        purchases = (
            sb.table("credit_ledger").select("delta,note")
            .eq("reason", "purchase").like("ref_id", f"{prefix}%")
            .gte("created_at", d30).limit(2000).execute().data or []
        )
        revenue_cents = 0
        for p in purchases:
            m = re.match(r"(\d+)\s+pln", p.get("note") or "")
            if m:
                revenue_cents += int(m.group(1))
        stuck = (sb.table("generations").select("id", count="exact").eq("status", "pending")
                 .lt("created_at", stuck_before).limit(1).execute().count or 0)
        return {
            "users": {"total": _count("profiles"), "new_7d": _count("profiles", d7)},
            "analyses": {"total": _count("analyses"), "last_24h": _count("analyses", d1)},
            "generations": {
                "total": _count("generations"),
                "last_24h": {s: _count("generations", d1, status=s) for s in ("succeeded", "failed", "pending")},
                "stuck": stuck,
            },
            "tickets": {"open": _count("support_tickets", status="open")
                                + _count("support_tickets", status="in_progress")},
            "ratings_7d": {"good": _count("preview_ratings", d7, rating="good"),
                           "bad": _count("preview_ratings", d7, rating="bad")},
            "sales_30d": {"purchases": len(purchases),
                          "credits": sum(int(p["delta"]) for p in purchases),
                          "revenue_pln": round(revenue_cents / 100, 2)},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise _fail("stats", e)


@router.get("/users")
@limiter.limit("60/minute")
def users(request: Request, q: str = Query("", max_length=100), admin=Depends(require_admin)):
    try:
        qb = get_supabase().table("profiles").select("id,email,credits,is_admin,created_at")
        q = re.sub(r"[%_,()*]", "", q.strip())
        if q:
            qb = qb.ilike("email", f"%{q}%")
        return {"users": qb.order("created_at", desc=True).limit(30).execute().data or []}
    except Exception as e:
        raise _fail("users", e)


@router.get("/users/{user_id}")
@limiter.limit("60/minute")
def user_detail(request: Request, user_id: UUID, admin=Depends(require_admin)):
    uid = str(user_id)
    try:
        sb = get_supabase()
        prof = sb.table("profiles").select("id,email,credits,is_admin,created_at").eq("id", uid).limit(1).execute().data
        if not prof:
            raise _err("USER_NOT_FOUND", "User not found", 404)

        def last(table, cols, n):
            return (sb.table(table).select(cols).eq("user_id", uid)
                    .order("created_at", desc=True).limit(n).execute().data or [])

        audit(admin, "view_user", uid)
        return {
            "user": prof[0],
            "analyses_count": _count("analyses", user_id=uid),
            "ledger": last("credit_ledger", "id,delta,reason,note,created_at", 30),
            "generations": last("generations", "id,style_name,color_id,status,error_code,created_at", 20),
            "tickets": last("support_tickets", "id,category,message,status,credits_granted,created_at", 20),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise _fail("user_detail", e)


class CreditsBody(BaseModel):
    amount: int = Field(ge=-500, le=500)
    note: str = Field(min_length=3, max_length=200)


@router.post("/users/{user_id}/credits")
@limiter.limit("30/minute")
def adjust_credits(request: Request, user_id: UUID, body: CreditsBody, admin=Depends(require_admin)):
    if body.amount == 0:
        raise _err("INVALID_AMOUNT", "Amount must not be zero", 400)
    uid, ref = str(user_id), str(uuid4())
    try:
        sb = get_supabase()
        if not sb.table("profiles").select("id").eq("id", uid).limit(1).execute().data:
            raise _err("USER_NOT_FOUND", "User not found", 404)

        if body.amount > 0:
            balance = billing.grant(uid, body.amount, "admin_grant", "admin", ref, body.note, str(admin.id))
        else:
            balance = sb.rpc("spend_credits", {
                "p_user": uid, "p_amount": -body.amount, "p_reason": "admin_revoke",
                "p_ref_type": "admin", "p_ref_id": ref}).execute().data
            if balance is None or int(balance) < 0:
                raise _err("INSUFFICIENT_CREDITS", "User has fewer credits than you try to remove", 409)
            try:
                sb.table("credit_ledger").update({"note": body.note, "created_by": str(admin.id)}) \
                  .eq("reason", "admin_revoke").eq("ref_type", "admin").eq("ref_id", ref).execute()
            except Exception:
                log.exception("could not annotate admin_revoke ledger row")

        audit(admin, "credits_grant" if body.amount > 0 else "credits_revoke", uid,
              amount=body.amount, note=body.note, balance=int(balance))
        return {"ok": True, "balance": int(balance)}
    except HTTPException:
        raise
    except Exception as e:
        raise _fail("adjust_credits", e)


@router.get("/tickets")
@limiter.limit("60/minute")
def tickets(request: Request, status: Literal["open", "closed", "all"] = "open", admin=Depends(require_admin)):
    try:
        sb = get_supabase()
        qb = sb.table("support_tickets").select(
            "id,user_id,generation_id,category,message,status,resolution,credits_granted,admin_note,created_at,resolved_at")
        if status == "open":
            qb = qb.in_("status", ["open", "in_progress"])
        elif status == "closed":
            qb = qb.in_("status", ["resolved", "rejected"])
        items = qb.order("created_at", desc=(status != "open")).limit(100).execute().data or []

        user_ids = list({t["user_id"] for t in items if t["user_id"]})
        gen_ids = list({t["generation_id"] for t in items if t["generation_id"]})
        emails, gens, ratings, charged, refunded = {}, {}, {}, {}, {}
        if user_ids:
            for r in sb.table("profiles").select("id,email").in_("id", user_ids).execute().data or []:
                emails[r["id"]] = r["email"]
        if gen_ids:
            for g in (sb.table("generations").select("id,style_name,color_id,status,error_code,duration_ms,created_at")
                      .in_("id", gen_ids).execute().data or []):
                gens[g["id"]] = g
            for r in sb.table("preview_ratings").select("generation_id,rating").in_("generation_id", gen_ids).execute().data or []:
                ratings[r["generation_id"]] = r["rating"]
            for r in (sb.table("credit_ledger").select("reason,delta,ref_id").eq("ref_type", "generation")
                      .in_("ref_id", gen_ids).execute().data or []):
                if r["reason"] == "spend_preview":
                    charged[r["ref_id"]] = -int(r["delta"])
                elif r["reason"] in ("refund_failed", "refund_ticket"):
                    refunded[r["ref_id"]] = refunded.get(r["ref_id"], 0) + int(r["delta"])

        for t in items:
            gid = t["generation_id"]
            t["user_email"] = emails.get(t["user_id"])
            t["generation"] = gens.get(gid)
            t["rating"] = ratings.get(gid)
            t["charged"] = charged.get(gid)
            t["refunded"] = refunded.get(gid, 0)
        return {"tickets": items}
    except Exception as e:
        raise _fail("tickets", e)


class ResolveBody(BaseModel):
    action: Literal["grant", "reject"]
    credits: int = Field(default=0, ge=0, le=20)
    note: str = Field(default="", max_length=500)


_RESOLVE_ERRORS = {
    "TICKET_NOT_FOUND": 404, "TICKET_CLOSED": 409, "TICKET_NO_OWNER": 409,
    "ALREADY_REFUNDED": 409, "AMOUNT_EXCEEDS_CHARGE": 400, "INVALID_AMOUNT": 400,
}


@router.post("/tickets/{ticket_id}/resolve")
@limiter.limit("60/minute")
def resolve_ticket(request: Request, ticket_id: UUID, body: ResolveBody, admin=Depends(require_admin)):
    tid = str(ticket_id)
    try:
        sb = get_supabase()
        if body.action == "grant":
            if body.credits < 1:
                raise _err("INVALID_AMOUNT", "Credits must be at least 1", 400)
            try:
                balance = sb.rpc("resolve_ticket", {
                    "p_ticket": tid, "p_credits": body.credits,
                    "p_admin": str(admin.id), "p_note": body.note or None}).execute().data
            except Exception as e:
                msg = str(getattr(e, "message", None) or e)
                for code, http in _RESOLVE_ERRORS.items():
                    if code in msg:
                        raise _err(code, code.replace("_", " ").capitalize(), http)
                raise
            audit(admin, "ticket_grant", None, ticket=tid, credits=body.credits, note=body.note)
            return {"ok": True, "balance": balance}

        now = datetime.now(timezone.utc).isoformat()
        res = (sb.table("support_tickets")
               .update({"status": "rejected", "resolution": "no_action",
                        "admin_note": body.note or None, "resolved_at": now, "updated_at": now})
               .eq("id", tid).in_("status", ["open", "in_progress"]).execute())
        if not res.data:
            raise _err("TICKET_CLOSED", "Ticket is already closed or does not exist", 409)
        audit(admin, "ticket_reject", None, ticket=tid, note=body.note)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise _fail("resolve_ticket", e)


@router.get("/generations/stuck")
@limiter.limit("60/minute")
def stuck_generations(request: Request, admin=Depends(require_admin)):
    before = (datetime.now(timezone.utc) - timedelta(minutes=STUCK_AFTER_MINUTES)).isoformat()
    try:
        sb = get_supabase()
        items = (sb.table("generations").select("id,user_id,style_name,color_id,created_at")
                 .eq("status", "pending").lt("created_at", before)
                 .order("created_at").limit(50).execute().data or [])
        ids = list({g["user_id"] for g in items if g["user_id"]})
        emails = {}
        if ids:
            for r in sb.table("profiles").select("id,email").in_("id", ids).execute().data or []:
                emails[r["id"]] = r["email"]
        for g in items:
            g["user_email"] = emails.get(g["user_id"])
        return {"generations": items}
    except Exception as e:
        raise _fail("stuck_generations", e)


@router.post("/generations/{gen_id}/fail")
@limiter.limit("30/minute")
def fail_stuck_generation(request: Request, gen_id: UUID, admin=Depends(require_admin)):
    """Oznacza zawieszona generacje jako nieudana i zwraca kredyty (zwrot jest idempotentny)."""
    gid = str(gen_id)
    before = (datetime.now(timezone.utc) - timedelta(minutes=STUCK_AFTER_MINUTES)).isoformat()
    try:
        sb = get_supabase()
        row = (sb.table("generations").select("id,status,created_at").eq("id", gid)
               .eq("status", "pending").lt("created_at", before).limit(1).execute().data)
        if not row:
            raise _err("NOT_STUCK", "Generation is not pending for long enough", 409)
        refund = sb.rpc("refund_generation", {"p_gen": gid}).execute().data
        finish_generation(gid, "failed", "STUCK", None)
        audit(admin, "generation_fail_refund", None, generation=gid, refunded_balance=refund)
        return {"ok": True, "refunded": refund is not None}
    except HTTPException:
        raise
    except Exception as e:
        raise _fail("fail_stuck_generation", e)


@router.get("/whoami-ip")
def whoami_ip(request: Request, admin=Depends(require_admin)):
    return {
        "peer": request.client.host if request.client else None,
        "x_forwarded_for": request.headers.get("x-forwarded-for"),
        "x_real_ip": request.headers.get("x-real-ip"),
        "detected": client_ip(request),
        "hops": TRUSTED_PROXY_HOPS,
    }


@router.post("/sentry-test")
def sentry_test(admin=Depends(require_admin)):
    log.error("SENTRY_TEST logged error (powinien pojawić się w Sentry jako problem)")
    raise RuntimeError("SENTRY_TEST unhandled exception")