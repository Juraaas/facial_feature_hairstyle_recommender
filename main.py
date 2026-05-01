import cv2
import numpy as np
from src.landmarks import FaceLandmarkDetector
from src.features import extract_features
from src.drawing import (
    draw_landmarks,
    draw_geometry,
    draw_features,
    draw_feature_debug,
)
from src.face_traits import interpret_face

def run_pipeline(img, detector):
    landmarks = detector.detect(img)

    if landmarks is None:
        return None, None

    features = extract_features(landmarks)
    return landmarks, features

def compare_traits(t1, t2):
    print("\n=== TRAITS COMPARISON ===")

    all_keys = set(t1.keys()).union(t2.keys())

    for k in all_keys:
        v1 = t1.get(k, None)
        v2 = t2.get(k, None)

        if v1 != v2:
            print(f"{k}: IMG1={v1} | IMG2={v2}")


img1 = cv2.imread("dataset/test_images/004455.jpg")
img2 = cv2.imread("dataset/test_images/fotka_test.jpg")

detector = FaceLandmarkDetector(
    model_path="models/face_landmarker.task"
)

landmarks, features = run_pipeline(img1, detector)
landmarks2, features2 = run_pipeline(img2, detector)

if landmarks is None or landmarks2 is None:
    print("No face detected")
    exit()

print("\n=== FEATURES (IMG1) ===")
for k, v in features.items():
    print(f"{k}: {v:.4f}")

print("\n=== FEATURES (IMG2) ===")
for k, v in features2.items():
    print(f"{k}: {v:.4f}")

traits = interpret_face(features)
traits2 = interpret_face(features2)

print("\n=== TRAITS (IMG1) ===")
for k, v in traits.items():
    print(f"{k}: {v}")

print("\n=== TRAITS (IMG2) ===")
for k, v in traits2.items():
    print(f"{k}: {v}")


compare_traits(traits, traits2)

img_resized = cv2.resize(img1, None, fx=1.5, fy=1.5)
_, f_resized = run_pipeline(img_resized, detector)

print("\n=== SELF CONSISTENCY TEST ===")
for k in features.keys():
    diff = abs(features[k] - f_resized[k])
    print(f"{k}: diff = {diff:.6f}")


_, f2 = run_pipeline(img2, detector)

print("\n=== CROSS IMAGE COMPARISON ===")
for k in features.keys():
    diff = abs(features[k] - f2[k])
    print(f"{k}: diff = {diff:.6f}")


img_landmarks = draw_landmarks(img1, landmarks, draw_indices=False)
img_geometry = draw_geometry(img_landmarks, landmarks)
img_features = draw_features(img_geometry, features)

img_final = cv2.resize(img_features, None, fx=2, fy=2)

img2_landmarks = draw_landmarks(img2, landmarks2, draw_indices=False)
img2_geometry = draw_geometry(img2_landmarks, landmarks2)
img2_features = draw_features(img2_geometry, features2)

cv2.imshow("IMG1 - FULL PIPELINE", img_final)
cv2.imshow("IMG2 - FEATURES + TRAITS", img2_features)

cv2.waitKey(0)
cv2.destroyAllWindows()