import json
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from src.auth import require_auth, get_supabase
from src.limits import limiter
from src.logger import get_logger

log = get_logger(__name__)
router = APIRouter()

MAX_TICKETS_PER_DAY = 5


def _err(code: str, message: str, status: int) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


class TicketBody(BaseModel):
    category: Literal["preview_quality", "technical", "billing", "other"]
    message: str = Field(min_length=10, max_length=2000)
    generation_id: Optional[UUID] = None


class DeleteBody(BaseModel):
    email: str = Field(min_length=3, max_length=320)


@router.post("/tickets")
@limiter.limit("5/hour")
def create_ticket(request: Request, body: TicketBody, user=Depends(require_auth)):
    uid = str(user.id)
    sb = get_supabase()

    gen_id = None
    if body.generation_id:
        owned = (
            sb.table("generations").select("id")
            .eq("id", str(body.generation_id)).eq("user_id", uid).limit(1).execute()
        )
        if not owned.data:
            raise _err("INVALID_GENERATION", "Generation not found", 404)
        gen_id = str(body.generation_id)

    since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    recent = (
        sb.table("support_tickets").select("id", count="exact")
        .eq("user_id", uid).gte("created_at", since).limit(1).execute()
    )
    if (recent.count or 0) >= MAX_TICKETS_PER_DAY:
        raise _err("TOO_MANY_TICKETS", "Daily ticket limit reached", 429)

    try:
        res = sb.table("support_tickets").insert({
            "user_id": uid,
            "generation_id": gen_id,
            "category": body.category,
            "message": body.message.strip(),
        }).execute()
    except Exception as e:
        if "23505" in str(e):
            raise _err("TICKET_EXISTS", "There is already an open ticket for this preview", 409)
        log.exception("create_ticket failed")
        raise _err("INTERNAL_ERROR", "Could not create ticket", 500)

    log.info("TICKET_CREATED id=%s category=%s", res.data[0]["id"], body.category)
    return {"id": res.data[0]["id"], "status": "open"}


@router.get("/tickets")
def list_tickets(user=Depends(require_auth)):
    try:
        res = (
            get_supabase().table("support_tickets")
            .select("id,category,message,status,resolution,credits_granted,generation_id,created_at,resolved_at")
            .eq("user_id", str(user.id)).order("created_at", desc=True).limit(50).execute()
        )
        return {"tickets": res.data or []}
    except Exception:
        log.exception("list_tickets failed")
        return {"tickets": []}


@router.get("/account/export")
@limiter.limit("3/hour")
def export_account(request: Request, user=Depends(require_auth)):
    uid = str(user.id)
    sb = get_supabase()

    def rows(table: str, cols: str):
        return sb.table(table).select(cols).eq("user_id", uid).order("created_at", desc=True).execute().data or []

    try:
        profile = sb.table("profiles").select("email,credits,created_at").eq("id", uid).limit(1).execute().data
        payload = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "account": (profile[0] if profile else {}) | {"id": uid},
            "analyses": rows("analyses", "id,traits,top_styles,gender,created_at"),
            "generations": rows("generations", "id,style_name,color_id,gender,status,created_at"),
            "credit_ledger": rows("credit_ledger", "delta,reason,note,created_at"),
            "support_tickets": rows("support_tickets", "category,message,status,resolution,credits_granted,created_at"),
            "preview_ratings": rows("preview_ratings", "style_name,color_id,gender,rating,created_at"),
        }
    except Exception:
        log.exception("export_account failed")
        raise _err("INTERNAL_ERROR", "Could not export data", 500)

    return Response(
        content=json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="stylizzer-data.json"'},
    )


@router.post("/account/delete")
@limiter.limit("3/hour")
def delete_account(request: Request, body: DeleteBody, user=Depends(require_auth)):
    if body.email.strip().lower() != (user.email or "").lower():
        raise _err("EMAIL_MISMATCH", "Email does not match the account", 400)

    uid = str(user.id)
    sb = get_supabase()
    try:
        sb.rpc("delete_user_data", {"p_user": uid}).execute()
        sb.auth.admin.delete_user(uid)
    except Exception:
        log.exception("delete_account failed")
        raise _err("DELETE_FAILED", "Could not delete the account, please try again", 500)

    try:
        sb.table("profiles").delete().eq("id", uid).execute()
    except Exception:
        log.warning("profile cleanup after account deletion failed")

    log.info("ACCOUNT_DELETED")
    return {"ok": True}