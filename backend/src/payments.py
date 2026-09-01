import os
import stripe
from fastapi import APIRouter, Request, HTTPException, Depends
from src.auth import require_auth, get_supabase

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")
PRICE_ID = os.environ.get("STRIPE_PRICE_ID")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://stylizzer.vercel.app")

router = APIRouter()

@router.post("/create-checkout")
async def create_checkout(user=Depends(require_auth)):
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="payment",
            line_items=[{"price": PRICE_ID, "quantity": 1}],
            success_url=f"{FRONTEND_URL}/analyse?payment=success",
            cancel_url=f"{FRONTEND_URL}/analyse?payment=cancelled",
            customer_email=user.email,
            metadata={"user_id": str(user.id)},
        )
        return {"url": session.url}
    except Exception as e:
        raise HTTPException(500, detail={"code": "PAYMENT_ERROR", "message": str(e)})

@router.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig, WEBHOOK_SECRET)
    except Exception as e:
        raise HTTPException(400, detail=str(e))

    if event["type"] == "checkout.session.completed":
        user_id = event["data"]["object"]["metadata"]["user_id"]
        sb = get_supabase()
        sb.table("profiles")\
          .update({"plan": "premium"})\
          .eq("id", user_id)\
          .execute()

    return {"ok": True}