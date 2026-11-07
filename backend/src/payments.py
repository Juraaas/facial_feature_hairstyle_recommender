import asyncio
import os
import stripe
from fastapi import APIRouter, Request, HTTPException, Depends
from src.auth import require_auth, get_supabase
from src.logger import get_logger

log = get_logger(__name__)

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")
PRICE_ID = os.environ.get("STRIPE_PRICE_ID")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://stylizzer.vercel.app")

if not (stripe.api_key and WEBHOOK_SECRET and PRICE_ID):
    log.warning("Stripe is not fully configured (secret key / webhook secret / price id)")

router = APIRouter()


@router.post("/create-checkout")
async def create_checkout(user=Depends(require_auth)):
    try:
        session = await asyncio.to_thread(
            stripe.checkout.Session.create,
            payment_method_types=["card"],
            mode="payment",
            line_items=[{"price": PRICE_ID, "quantity": 1}],
            success_url=f"{FRONTEND_URL}/analyse?payment=success",
            cancel_url=f"{FRONTEND_URL}/analyse?payment=cancelled",
            customer_email=user.email,
            client_reference_id=str(user.id),
            metadata={"user_id": str(user.id)},
        )
        return {"url": session.url}
    except Exception:
        log.exception("Stripe checkout creation failed")
        raise HTTPException(500, detail={
            "code": "PAYMENT_ERROR", "message": "Could not start checkout",
        })


def _grant_premium(user_id: str):
    get_supabase().table("profiles").update({"plan": "premium"}).eq("id", user_id).execute()


@router.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig, WEBHOOK_SECRET)
    except Exception:
        log.warning("Invalid Stripe webhook signature or payload")
        raise HTTPException(400, detail="Invalid webhook")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"].to_dict()

        if session.get("payment_status") != "paid":
            log.info("Checkout %s not paid yet (%s), skipping",
                     session.get("id"), session.get("payment_status"))
            return {"ok": True}

        user_id = (session.get("metadata") or {}).get("user_id")
        if not user_id:
            log.warning("Webhook: no user_id in metadata (session %s)", session.get("id"))
            return {"ok": True}

        try:
            await asyncio.to_thread(_grant_premium, user_id)
            log.info("Granted premium to user %s (session %s)", user_id, session.get("id"))
        except Exception:
            log.exception("Failed to grant premium")
            raise HTTPException(500, detail="Database error")

    return {"ok": True}