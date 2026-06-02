import cv2
import numpy as np
import pandas as pd
import sys
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.landmarks import FaceLandmarkDetector
from src.pipeline import run_pipeline
from src.gender import detect_gender
from src.feedback import save_session, save_vote
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from src.drawing import draw_landmarks

sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

detector = None
norms = None
female_norms = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, norms, female_norms
    detector = FaceLandmarkDetector(model_path="models/face_landmarker.task")
    norms = pd.read_csv("data/norms/male_norms_v2.csv", index_col=0)
    female_norms = pd.read_csv("data/norms/female_norms_v2.csv", index_col=0)
    print("Models loaded")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory="images"), name="images")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyse")
async def analyse(file: UploadFile = File(...)):
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    
    h, w = img.shape[:2]
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    gender = detect_gender(img) or "Unknown"
    result = run_pipeline(img, detector, gender=gender)
    landmarks, features, traits, scores, recs, quality = result

    if landmarks is None:
        raise HTTPException(
            status_code=422,
            detail=quality.blocking if quality and quality.blocking else "No face detected",
        )
    
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
                "p5":   float(norms.loc["p5",   feat]) if gender != "Woman"
                        else float(female_norms.loc["p5",   feat]),
                "p95":  float(norms.loc["p95",  feat]) if gender != "Woman"
                        else float(female_norms.loc["p95",  feat]),
                "mean": float(norms.loc["mean", feat]) if gender != "Woman"
                        else float(female_norms.loc["mean", feat]),
            }
            for feat in features.keys()
            if feat in norms.columns
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
async def landmarks_overlay(file: UploadFile = File(...)):
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    h, w = img.shape[:2]
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    landmarks = detector.detect(img)
    if landmarks is not None:
        img = draw_landmarks(img, landmarks, draw_indices=False)
    
    _, buf = cv2.imencode('.jpg', img)
    return Response(content=buf.tobytes(), media_type="image/jpeg")
