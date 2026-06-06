import cv2
import numpy as np
import pandas as pd
import sys
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.landmarks import FaceLandmarkDetector
from src.pipeline import run_pipeline
from src.feedback import save_session, save_vote
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from src.drawing import draw_landmarks
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")

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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", 
                   "http://localhost:3000",
                   "https://facial-feature-hairstyle-recommende.vercel.app",
                   "https://*.hf.space",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/analyse")
def analyse(file: UploadFile = File(...)):
    contents = file.file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    
    h, w = img.shape[:2]
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    detector = get_detector()
    norms, female_norms = get_norms()
    gender = get_gender(img)
    
    result = run_pipeline(img, detector, gender=gender)
    landmarks, features, traits, scores, recs, quality = result

    if landmarks is None:
        raise HTTPException(
            status_code=422,
            detail=quality.blocking if quality and quality.blocking else "No face detected",
        )
    
    selected_norms = female_norms if gender == "Woman" else norms
    
    return {
        "gender":   gender,
        "features": features,
        "traits":   traits,
        "quality":  {
            "score":    quality.score,
            "warnings": quality.warnings,
        },
        "analysis": recs["face_analysis"],
        "styles":   recs["all_styles"],
        "norms": {
            feat: {
                "p5": float(selected_norms.loc["p5", feat]),
                "p95": float(selected_norms.loc["p95", feat]),
                "mean": float(selected_norms.loc["mean", feat]),
            }
            for feat in features.keys()
            if feat in selected_norms.columns
        }
    }

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
        raise HTTPException(status_code=500, detail=str(e))
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
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True}

@app.post("/landmarks-overlay")
def landmarks_overlay(file: UploadFile = File(...)):
    contents = file.file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    h, w = img.shape[:2]
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    detector = get_detector()
    
    landmarks = detector.detect(img)
    if landmarks is not None:
        img = draw_landmarks(img, landmarks, draw_indices=False)
    
    _, buf = cv2.imencode('.jpg', img)
    return Response(content=buf.tobytes(), media_type="image/jpeg")
