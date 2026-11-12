import asyncio
import os

import stripe
from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel, Field

from src import billing
from src.auth import require_auth
from src.logger import get_logger

log = get_logger(__name__)

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://stylizzer.vercel.app")

if not (stripe.api_key and WEBHOOK_SECRET):
    log.warning("Stripe is not fully configured (secret key / webhook secret)")

router = APIRouter()


class CheckoutBody(BaseModel):
    pack_id: str = Field(min_length=1, max_length=40)


@router.post("/create-checkout")
async def create_checkout(body: CheckoutBody, user=Depends(require_auth)):
    pack = await asyncio.to_thread(billing.get_pack, body.pack_id)
    if not pack or not pack.get("stripe_price_id"):
        raise HTTPException(400, detail={"code": "INVALID_PACK", "message": "Unknown pack"})
    try:
        session = await asyncio.to_thread(
            stripe.checkout.Session.create,
            mode="payment",
            payment_method_types=["card"],
            line_items=[{"price": pack["stripe_price_id"], "quantity": 1}],
            success_url=f"{FRONTEND_URL}/analyse?payment=success",
            cancel_url=f"{FRONTEND_URL}/analyse?payment=cancelled",
            customer_email=user.email,
            client_reference_id=str(user.id),
            metadata={"user_id": str(user.id), "pack_id": pack["id"], "credits": str(pack["credits"])},
        )
        return {"url": session.url}
    except Exception:
        log.exception("Stripe checkout creation failed")
        raise HTTPException(500, detail={"code": "PAYMENT_ERROR", "message": "Could not start checkout"})


@router.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig, WEBHOOK_SECRET)
    except Exception:
        log.warning("Invalid Stripe webhook signature or payload")
        raise HTTPException(400, detail="Invalid webhook")

    if event["type"] in ("checkout.session.completed", "checkout.session.async_payment_succeeded"):
        session = event["data"]["object"].to_dict()

        if session.get("payment_status") != "paid":
            log.info("Checkout %s not paid yet (%s)", session.get("id"), session.get("payment_status"))
            return {"ok": True}

        meta = session.get("metadata") or {}
        user_id = meta.get("user_id")
        try:
            amount = int(meta.get("credits", "0"))
        except ValueError:
            amount = 0
        if not user_id or not (0 < amount <= 10000):
            log.error("PAYMENT_UNPROCESSABLE session=%s metadata=%s", session.get("id"), meta)
            return {"ok": True}

        note = f"{session.get('amount_total')} {session.get('currency')} pack={meta.get('pack_id')}"
        try:
            balance = await asyncio.to_thread(
                billing.grant, user_id, amount, "purchase", "stripe_session", session["id"], note)
            log.info("Granted %s credits to %s (session %s, balance %s)",
                     amount, user_id, session["id"], balance)
        except Exception:
            log.exception("Failed to grant purchased credits (session %s)", session.get("id"))
            raise HTTPException(500, detail="Database error")

    return {"ok": True}