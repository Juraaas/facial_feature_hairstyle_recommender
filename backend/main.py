import asyncio
import base64
import os
import threading
from typing import Literal, Optional
import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from src.auth import require_premium, require_auth, get_current_user, get_supabase
from src.color_recommender import analyze_skin_tone
from src.exceptions import INVALID_IMAGE, NO_FACE_DETECTED, INTERNAL_ERROR
from src.feedback import save_session, save_vote
from src.hair_classifier import estimate_hair_length
from src.landmarks import FaceLandmarkDetector
from src.logger import get_logger
from src.payments import router as payments_router
from src.pipeline import run_pipeline
from src.style_generator import generate_preview, STYLE_PROMPTS_MALE, STYLE_PROMPTS_FEMALE, COLOR_PROMPTS
from src.transformation_estimator import estimate_transformation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
DEBUG_ENABLED = os.getenv("DEBUG_ENDPOINTS") == "1"
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "https://stylizzer.vercel.app").split(",")
    if o.strip()
]
MAX_FILE_SIZE = 10 * 1024 * 1024
HAIR_TYPES = {"straight", "wavy", "curly", "coily"}

log = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)

_detector = None
_norms = None
_female_norms = None
_model_lock = threading.Lock()

class VoteBody(BaseModel):
    style_name: str = Field(min_length=1, max_length=80)
    vote: Literal["up", "down"]
    features: dict[str, float] = Field(default_factory=dict, max_length=30)
    gender: Literal["Man", "Woman", ""] = ""

class FeedbackBody(BaseModel):
    features: dict[str, float] = Field(default_factory=dict, max_length=30)
    quality_score: float = Field(ge=0, le=1)
    top_styles: list[dict] = Field(default_factory=list, max_length=5)
    rating: Optional[float] = Field(default=None, ge=0.5, le=5)
    comment: str = Field(default="", max_length=1000)

class PreviewRatingBody(BaseModel):
    style_name: str = Field(min_length=1, max_length=80)
    color_id: Literal["natural", "blonde", "dark", "black", "auburn"]
    gender: Literal["Man", "Woman"]
    rating: Literal["good", "bad"]


def get_detector():
    global _detector
    if _detector is None:
        _detector = FaceLandmarkDetector(model_path=MODEL_PATH)
    return _detector

HF_NORMS_REPO = "juras3k/hairstyle-norms"

def get_norms():
    global _norms, _female_norms
    if _norms is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("Missing HF_TOKEN environment variable")

        male_path = hf_hub_download(
            repo_id=HF_NORMS_REPO,
            filename="male_norms_v3.csv",
            repo_type="dataset",
            token=token,
        )
        female_path = hf_hub_download(
            repo_id=HF_NORMS_REPO,
            filename="female_norms_v3.csv",
            repo_type="dataset",
            token=token,
        )
        _norms = pd.read_csv(male_path, index_col=0)
        _female_norms = pd.read_csv(female_path, index_col=0)
    return _norms, _female_norms

def get_gender(img):
    from src.gender import detect_gender
    return detect_gender(img) or "Unknown"

def preprocess_image(contents: bytes, target: int = 512) -> np.ndarray:
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    oh, ow = img.shape[:2]
    side = min(oh, ow)
    y0 = (oh - side) // 2
    x0 = (ow - side) // 2
    crop = img[y0:y0+side, x0:x0+side]
    return cv2.resize(crop, (target, target), interpolation=cv2.INTER_LANCZOS4)

def validate_image_bytes(contents: bytes) -> bool:
    return contents[:2] == b"\xff\xd8" or contents[:4] == b"\x89PNG"

def http_error(code: str, message: str, status: int = 422) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})

def norm_lang(lang: str) -> str:
    return "pl" if (lang or "").lower().startswith("pl") else "en"

def save_analysis(user, traits, recs, gender):
    try:
        all_styles = recs["all_styles"]
        styles_list = all_styles.get("pl", []) if isinstance(all_styles, dict) else all_styles
        top3 = [
            {"name": s.get("name"), "score": s.get("display_score", 0)}
            for s in styles_list[:3]
        ]
        get_supabase().table("analyses").insert({
            "user_id": str(user.id),
            "traits": traits,
            "top_styles": top3,
            "gender": gender,
        }).execute()
    except Exception:
        log.exception("Failed to save analysis")

app = FastAPI(
    docs_url="/docs" if DEBUG_ENABLED else None,
    redoc_url=None,
    openapi_url="/openapi.json" if DEBUG_ENABLED else None,
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

app.add_middleware(SecurityHeadersMiddleware)
app.include_router(payments_router)
app.state.limiter = limiter
app.mount("/images", StaticFiles(directory=os.path.join(BASE_DIR, "images")), name="images")


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"code": "RATE_LIMITED", "message": "Too many requests - try again later"},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.head("/health")
def health_head():
    return Response(status_code=200)


@app.get("/")
def root():
    return {"status": "ok", "service": "stylizzer-api"}


@app.head("/")
def root_head():
    return Response(status_code=200)


@app.post("/analyse")
@limiter.limit("5/minute")
def analyse(request: Request,
            file: UploadFile = File(...),
            lang: str = Query("pl"),
            gender: str = Query(None),
            debug: bool = Query(False),
            compact_debug: bool = Query(False),
            user = Depends(get_current_user)):
    lang = norm_lang(lang)
    try:
        contents = file.file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise http_error("FILE_TOO_LARGE", "File too large - max 10MB", 400)
        if not validate_image_bytes(contents):
            raise http_error("INVALID_IMAGE", "Only JPEG and PNG files are accepted", 400)
        img = preprocess_image(contents, target=512)
        if img is None:
            raise http_error(INVALID_IMAGE, "Could not decode image - try a JPG or PNG file", status=400)

        with _model_lock:
            detector = get_detector()
            norms, female_norms = get_norms()
            detected = get_gender(img)
            final_gender = gender if gender in ("Man", "Woman") else (detected if detected in ("Man", "Woman") else "Man")
            result = run_pipeline(
                img, detector, gender=final_gender, lang=lang, debug_rules=False
            )

        landmarks, features, traits, scores, recs, quality, hair_mask = result
        if landmarks is None:
            code = getattr(quality, "blocking_code", None) or NO_FACE_DETECTED
            message = getattr(quality, "blocking", None) or "No face detected"
            raise http_error(code, message)
        
        selected_norms = female_norms if final_gender == "Woman" else norms

        color_rec = None
        try:
            color_rec = analyze_skin_tone(img, hair_mask)
        except Exception:
            print(f"Color recommendation error")
        
        response = {
            "gender": final_gender,
            "features": features,
            "traits": traits,
            "quality": {
                "score": quality.score,
                "warnings": quality.warnings,
                "hair_coverage": traits.get("coverage", 0.05),
            },
            "face_analysis": recs["face_analysis"],
            "styles": recs["all_styles"],
            "norms": {
                feat: {
                    "p5": float(selected_norms.loc["p5", feat]),
                    "p95": float(selected_norms.loc["p95", feat]),
                    "mean": float(selected_norms.loc["mean", feat]),
                }
                for feat in features.keys()
                if feat in selected_norms.columns
            },
            "fade_recommendation": recs.get("fade_recommendation"),
            "color_recommendation": color_rec,
        }

        if DEBUG_ENABLED and (debug or compact_debug):
            from src.debug_payload import build_debug_payload
            payload = build_debug_payload(recs, scores, lang, compact=compact_debug)
            response["debug"] = payload
            if compact_debug:
                response["styles"] = payload["compact_ranking"]

        if user:
            save_analysis(user, traits, recs, final_gender)

        return response
    except HTTPException:
        raise
    except Exception:
        log.exception("Unexpected error in /analyse")
        raise http_error(INTERNAL_ERROR, "Something went wrong - please try again", status=500)


@app.post("/vote")
@limiter.limit("30/minute")
async def vote(request: Request, body: VoteBody):
    try:
        save_vote(body.style_name, body.vote, body.features, body.gender)
    except Exception:
        log.exception("Could not save vote")
        raise http_error(INTERNAL_ERROR, "Could not save vote", 500)
    return {"ok": True}


@app.post("/feedback")
@limiter.limit("5/minute")
async def feedback(request: Request, body: FeedbackBody):
    try:
        save_session(
            body.features,
            body.quality_score,
            {"top_styles": body.top_styles},
            rating=body.rating,
            comment=body.comment,
        )
    except Exception:
        log.exception("Could not save feedback")
        raise http_error(INTERNAL_ERROR, "Could not save feedback", 500)
    return {"ok": True}


@app.post("/preview-rating")
@limiter.limit("10/minute")
async def preview_rating(request: Request, body: PreviewRatingBody, user=Depends(require_auth)):
    try:
        get_supabase().table("preview_ratings").insert({
            "user_id": str(user.id),
            "style_name": body.style_name,
            "color_id": body.color_id,
            "gender": body.gender,
            "rating": body.rating,
        }).execute()
    except Exception:
        log.exception("Preview rating error")
    return {"ok": True}


@app.post("/style-preview")
@limiter.limit("3/minute")
async def style_preview(
    request: Request,
    file: UploadFile = File(...),
    style_name: str = Form(...),
    color_id: str = Form("natural"),
    gender: Literal["Man", "Woman"] = Form("Man"),
    hair_type: str = Form(None),
    hair_coverage: float = Form(0.05, ge=0, le=1),
    lang: Literal["pl", "en"] = Form("pl"),
    user = Depends(require_premium),
):
    lang = norm_lang(lang)

    prompts = STYLE_PROMPTS_FEMALE if gender == "Woman" else STYLE_PROMPTS_MALE
    if style_name not in prompts or color_id not in COLOR_PROMPTS:
        raise http_error("INVALID_INPUT", "Unknown style or color", 400)
    hair_type = hair_type if hair_type in HAIR_TYPES else None

    try:
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise http_error("FILE_TOO_LARGE", "File too large - max 10MB", 400)
        if not validate_image_bytes(contents):
            raise http_error("INVALID_IMAGE", "Only JPEG and PNG files are accepted", 400)
        
        img = preprocess_image(contents)
        if img is None:
            raise http_error("INVALID_IMAGE", "Cannot decode image", 400)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise http_error("INVALID_IMAGE", "Cannot encode image", 400)
        processed_bytes = buf.tobytes()

        preview_bytes, transformation = await asyncio.gather(
            generate_preview(processed_bytes, style_name, color_id, gender),
            asyncio.to_thread(
                estimate_transformation,
                current_length=estimate_hair_length(hair_coverage),
                current_hair_type=hair_type or "straight",
                target_style=style_name,
                color_change=color_id,
                gender=gender,
                lang=lang,
            ),
        )

        return {
            "image_b64": base64.b64encode(preview_bytes).decode(),
            "transformation": transformation,
        }
        
    except HTTPException:
        raise
    except Exception:
        log.exception("Preview generation failed")
        raise http_error(INTERNAL_ERROR, "Preview generation failed", 500)


@app.get("/history")
async def get_history(user=Depends(require_auth)):
    try:
        data = (
            get_supabase().table("analyses")
            .select("id, traits, top_styles, gender, created_at")
            .eq("user_id", str(user.id))
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )
        return {"analyses": data.data}
    except Exception:
        log.exception("History error")
        return {"analyses": []}


@app.delete("/history/{analysis_id}")
async def delete_analysis(analysis_id: str, user=Depends(require_auth)):
    try:
        (
            get_supabase().table("analyses")
            .delete()
            .eq("id", analysis_id)
            .eq("user_id", str(user.id))
            .execute()
        )
        return {"ok": True}
    except Exception:
        log.exception("Delete error")
        raise http_error(INTERNAL_ERROR, "Delete failed", 500)