import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks.python import vision
from src.geometry import FaceGeometry
from src.features import extract_features
from src.landmarks import FaceLandmarkDetector

detector = FaceLandmarkDetector(
    model_path="models/face_landmarker.task"
)

def process_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    landmarks = detector.detect(img)
    if landmarks is None:
        return None
    
    try:
        return extract_features(landmarks)
    except Exception:
        return None
    
def compute_norms(dataset_dir):
    all_paths = Path(dataset_dir).rglob("*.jpg")
    paths = []
    for p in all_paths:
        parts = p.name.split("_")
        if len(parts) >= 2 and parts[1] == "1":
            paths.append(p)

    print(f"Images to be processed: {len(paths)}")

    rows = []
    for i, p in enumerate(paths):
        if i % 100 == 0:
            print(f"{i}/{len(paths)}")
        f = process_image(p)
        if f:
            rows.append(f)
    
    print(f"Faces detected: {len(rows)}/{len(paths)}")

    df = pd.DataFrame(rows)
    for col in df.columns:
        mean, std = df[col].mean(), df[col].std()
        df = df[(df[col] > mean - 3*std) & (df[col] < mean + 3*std)]

    stats = df.agg([
        "mean", "std",
        lambda x: x.quantile(0.05),
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.95),
    ])
    stats.index = ["mean", "std", "p5", "p25", "p75", "p95"]

    return stats, df

stats, df = compute_norms("dataset/part1/")
stats.to_csv("women_norms_p123.csv")
print(stats)