import cv2
import numpy as np
import pandas as pd
import sys
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from src.landmarks import FaceLandmarkDetector
from src.pipeline import run_pipeline
from src.feedback import save_session, save_vote
from src.hair_segmentation import segment_face
from src.hair_classifier import classify_hair
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from huggingface_hub import hf_hub_download
from src.exceptions import (
    INVALID_IMAGE, NO_FACE_DETECTED, FACE_TOO_SMALL,
    FACE_ROTATED, FACE_TILTED, POOR_ALIGNMENT, INTERNAL_ERROR
)
from src.style_generator import generate_preview
from src.auth import require_premium, require_auth, get_current_user, get_supabase
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from src.payments import router as payments_router


sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")

limiter = Limiter(key_func=get_remote_address)

_detector = None
_norms = None
_female_norms = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = FaceLandmarkDetector(
            model_path=os.path.join(os.path.dirname(__file__), "models/face_landmarker.task")
        )
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
            filename="male_norms_v2.csv",
            repo_type="dataset",
            token=token,
        )
        female_path = hf_hub_download(
            repo_id=HF_NORMS_REPO,
            filename="female_norms_v2.csv",
            repo_type="dataset",
            token=token,
        )
        _norms = pd.read_csv(male_path, index_col=0)
        _female_norms = pd.read_csv(female_path, index_col=0)

    return _norms, _female_norms

def get_gender(img):
    from src.gender import detect_gender
    return detect_gender(img) or "Unknown"

def decode_and_resize_image(contents, max_size=640):
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    h, w = img.shape[:2]

    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    return img

def http_error(code: str, message: str, status: int = 422) -> HTTPException:
    return HTTPException(status_code=status, detail={
        "code": code,
        "message": message,
    })

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", 
                   "http://localhost:3000",
                   "https://face-fit-ai.vercel.app",
                   "https://stylizzer.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(payments_router)
app.state.limiter = limiter

app.mount("/images", StaticFiles(directory=os.path.join(BASE_DIR, "images")), name="images")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.head("/health")
def health_head():
    return Response(status_code=200)

@app.get("/")
def root():
    return {"status": "ok", "service": "facial-feature-hairstyle-recommender"}

@app.head("/")
def root_head():
    return None

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"code": "RATE_LIMITED", "message": "Too many requests — try again later"}
    )

@app.post("/analyse")
@limiter.limit("5/minute")
async def analyse(request: Request,
            file: UploadFile = File(...),
            lang: str = Query("pl"),
            debug: bool = Query(False),
            user = Depends(get_current_user)):
    try:
        MAX_FILE_SIZE = 10 * 1024 * 1024 
        contents = file.file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise http_error("FILE_TOO_LARGE", "File too large — max 10MB", 400)
        arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            raise http_error(INVALID_IMAGE,
                             "Could not decode image - try a JPG or PNG file",
                             status=400)
    
        h, w = img.shape[:2]
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        detector = get_detector()
        norms, female_norms = get_norms()
        gender = get_gender(img)
        
        result = run_pipeline(img, detector, gender=gender, lang=lang)
        landmarks, features, traits, scores, recs, quality = result

        if landmarks is None:
            code = getattr(quality, "blocking_code", None) or NO_FACE_DETECTED
            message = getattr(quality, "blocking", None) or "No face detected"
            raise http_error(code, message)
        
        selected_norms = female_norms if gender == "Woman" else norms
        
        response = {
            "gender": gender,
            "features": features,
            "traits": traits,
            "quality": {
                "score": quality.score,
                "warnings": quality.warnings,
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
        }

        if debug:
            response["debug"] = {
                "raw_scores": scores,
                "style_ranking": [
                    {
                        "rank": i + 1,
                        "name": style.get("name"),
                        "score": style.get("score"),
                        "image": style.get("image"),
                        "category": style.get("category"),
                        "tags": style.get("tags", []),
                        "description": style.get("description", ""),
                        "contributions": style.get("contributions", []),
                        "negatives": style.get("negatives", []),
                        "missing": style.get("missing", []),
                    }
                    for i, style in enumerate(recs["all_styles"])
                ],
                "top_styles": [
                    {
                        "rank": i + 1,
                        "name": style.get("name"),
                        "score": style.get("score"),
                        "contributions": style.get("contributions", []),
                        "negatives": style.get("negatives", []),
                    }
                    for i, style in enumerate(recs.get("top_styles", []))
                ],
            }
        if user and landmarks is not None:
            try:
                sb = get_supabase()
                top3 = []
                styles_list = recs["all_styles"].get("pl", []) if isinstance(recs["all_styles"], dict) else recs["all_styles"]
                for s in styles_list[:3]:
                    top3.append({"name": s.get("name"), "score": s.get("display_score", 0)})
                sb.table("analyses").insert({
                    "user_id": str(user.id),
                    "traits": traits,
                    "top_styles": top3,
                    "gender": gender,
                }).execute()
            except Exception as e:
                print(f"Failed to save analysis: {e}")
        
        return response
    except HTTPException:
        raise
    except Exception:
        import traceback
        print(f"UNEXPECTED ERROR:\n{traceback.format_exc()}")
        raise http_error(INTERNAL_ERROR,
                         "Something went wrong - please try again",
                         status=500)

@app.post("/vote")
async def vote(body: dict):
    try:
        save_vote(
            body["style_name"],
            body["vote"],
            body["features"],
            body.get("gender", "")
        )
    except Exception as e:
        raise http_error(INTERNAL_ERROR, str(e), status=500)
    return {"ok": True}

@app.post("/feedback")
async def feedback(body: dict):
    try:
        save_session(
            body["features"],
            body["quality_score"],
            {"top_styles": body["top_styles"]},
            rating=body.get("rating"),
            comment=body.get("comment", "")
        )
    except Exception as e:
        raise http_error(INTERNAL_ERROR, str(e), status=500)
    return {"ok": True}

@app.post("/debug-hair")
async def debug_hair(file: UploadFile = File(...)):
    try:
        MAX_FILE_SIZE = 10 * 1024 * 1024 
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise http_error("FILE_TOO_LARGE", "File too large — max 10MB", 400)
        img = decode_and_resize_image(contents)
        if img is None:
            raise http_error(INVALID_IMAGE, "Could not decode image", status=400)

        hair_mask, _ = segment_face(img)

        if hair_mask is None:
            raise http_error(INTERNAL_ERROR, "Hair segmentation failed", status=500)

        coverage = float(np.sum(hair_mask > 0) / hair_mask.size)
        result = classify_hair(img, hair_mask)

        return {
            "coverage": round(coverage, 4),
            "hair_type": result["hair_type"],
            "hairline": result["hairline"],
            "hair_conf": result["hair_conf"],
            "hairline_conf": result["hairline_conf"],
            "mask_shape": {
                "height": hair_mask.shape[0],
                "width": hair_mask.shape[1],
            }
        }
    
    except HTTPException:
        raise
    except Exception:
        import traceback
        print(traceback.format_exc())
        raise http_error(INTERNAL_ERROR, "Hair debug failed", status=500)

@app.post("/style-preview")
async def style_preview(
    file: UploadFile = File(...),
    style_name: str = Form(...),
    color_id: str = Form("natural"),
    user = Depends(require_premium),
):
    try:
        contents = await file.read()
        result = await generate_preview(contents, style_name, color_id)
        return Response(content=result, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception:
        import traceback
        print(traceback.format_exc())
        raise http_error(INTERNAL_ERROR, "Preview generation failed", 500)

@app.get("/history")
async def get_history(user = Depends(require_auth)):
    sb = get_supabase()
    data = sb.table("analyses")\
        .select("id, traits, top_styles, gender, created_at")\
        .eq("user_id", str(user.id))\
        .order("created_at", desc=True)\
        .limit(20)\
        .execute()
    return {"analyses": data.data}