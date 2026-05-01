import cv2
import numpy as np
from src.landmarks import FaceLandmarkDetector
from src.features import extract_features
from src.visualization import (
    draw_landmarks,
    draw_geometry,
    draw_features,
    draw_feature_debug,
)

def run_pipeline(img, detector):
    landmarks = detector.detect(img)

    if landmarks is None:
        return None, None

    features = extract_features(landmarks)
    return landmarks, features


img1 = cv2.imread("dataset/test_images/004455.jpg")
img2 = cv2.imread("dataset/test_images/fotka_test.jpg")

detector = FaceLandmarkDetector(
    model_path="models/face_landmarker.task"
)

landmarks, features = run_pipeline(img1, detector)
landmarks2, features2 = run_pipeline(img2, detector)

if landmarks is None:
    print("No face detected")
    exit()

print("\n=== FEATURES (IMG1) ===")
for k, v in features.items():
    print(f"{k}: {v:.4f}")

img_landmarks = draw_landmarks(img1, landmarks, draw_indices=False)
img_landmarks2 = draw_landmarks(img2, landmarks2, draw_indices=False)
img_geometry = draw_geometry(img_landmarks, landmarks)
img_features = draw_features(img_geometry, features)
img_debug = draw_feature_debug(img_features, features)

img_final = cv2.resize(img_debug, None, fx=2, fy=2)

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

cv2.imshow("1 - Landmarks", img_landmarks)
cv2.imshow("2 - Geometry", img_geometry)
cv2.imshow("3 - Features Raw", img_features)
cv2.imshow("4 - Debug Overlay", img_debug)
cv2.imshow("5 - FINAL", img_final)
cv2.imshow("img2 - Landmarks", img_landmarks2)
cv2.waitKey(0)
cv2.destroyAllWindows()