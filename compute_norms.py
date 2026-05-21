import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from src.geometry import FaceGeometry
from src.features import extract_features
from src.landmarks import FaceLandmarkDetector
from src.hair_segmentation import find_hairline_y

HAIR_CLASS = 13
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def load_models():
    detector  = FaceLandmarkDetector(model_path="models/face_landmarker.task")
    processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    model     = SegformerForSemanticSegmentation.from_pretrained(
        "jonathandinu/face-parsing"
    ).to(DEVICE)
    model.eval()
    return detector, processor, model

def segment_batch(imgs_rgb, processor, model, device):
    results = []
    for img in imgs_rgb:
        h, w = img.shape[:2]
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        upsampled = torch.nn.functional.interpolate(
            outputs.logits, size=(h, w), mode="bilinear", align_corners=False
        )
        seg_map   = upsampled.argmax(dim=1).squeeze().cpu().numpy()
        hair_mask = (seg_map == HAIR_CLASS).astype(np.uint8) * 255
        results.append(hair_mask)
    return results

def process_batch(paths, detector, processor, seg_model, device):
    imgs     = []
    imgs_rgb = []

    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        imgs.append(img)
        imgs_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not imgs:
        return []

    hair_masks = segment_batch(imgs_rgb, processor, seg_model, device)

    results = []
    for img, hair_mask in zip(imgs, hair_masks):
        landmarks = detector.detect(img)
        if landmarks is None:
            continue
        try:
            geo = FaceGeometry(landmarks)
            hairline_y = find_hairline_y(hair_mask, geo.face_width())
            if hairline_y is not None:
                h_img  = img.shape[0]
                brow_y = geo.brow_line_y()
                if hairline_y > brow_y or hairline_y > h_img * 0.4:
                    hairline_y = None

            if hairline_y is None:
                continue
            f = extract_features(landmarks, hairline_y=hairline_y)

            BOUNDS = {
                "face_ratio":       (0.5, 2.5),
                "nose_position":    (0.2, 0.8),
                "jaw_to_height":    (0.3, 1.5),
                "lower_face_ratio": (0.1, 0.8),
                "chin_prominence":  (0.0, 0.6),
                "upper_third":      (0.05, 0.65),
                "middle_third":     (0.05, 0.65),
                "lower_third":      (0.05, 0.65),
            }
            valid = True
            for key, (lo, hi) in BOUNDS.items():
                if key in f and not (lo <= f[key] <= hi):
                    valid = False
                    break
            if valid:
                results.append(f)
        except Exception:
            continue
    return results

def compute_norms(dataset_dir, gender="0", batch_size=16):
    all_paths = Path(dataset_dir).rglob("*.jpg")
    paths = [
        p for p in all_paths
        if len(p.name.split("_")) >= 2 and p.name.split("_")[1] == gender
    ]

    print(f"Images to process: {len(paths)}  device: {DEVICE}")

    detector, processor, seg_model = load_models()

    rows = []
    total_processed = 0

    for i in range(0, len(paths), batch_size):
        batch  = paths[i:i + batch_size]
        results = process_batch(batch, detector, processor, seg_model, DEVICE)
        rows.extend(results)
        total_processed += len(batch)
        if i % 500 == 0:
            print(f"{total_processed}/{len(paths)}  collected: {len(rows)}  "
                  f"({len(rows)/max(total_processed,1)*100:.0f}% retention)")

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


if __name__ == "__main__":
    stats, df = compute_norms("dataset/part123/", gender="0", batch_size=16)
    stats.to_csv("data/norms/male_norms_v2.csv")
    print("=== MALE ===")
    print(stats)

    stats, df = compute_norms("dataset/part123/", gender="1", batch_size=16)
    stats.to_csv("data/norms/female_norms_v2.csv")
    print("=== FEMALE ===")
    print(stats)