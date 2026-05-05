import cv2
import numpy as np
from src.landmarks import FaceLandmarkDetector
from src.drawing import (
    draw_landmarks,
    draw_geometry,
    draw_features,
)
from src.pipeline import run_pipeline


def print_features(features, label):
    print(f"\n=== FEATURES ({label}) ===")
    for k, v in features.items():
        print(f"{k}: {v:.4f}")


def print_traits(traits, label):
    print(f"\n=== TRAITS ({label}) ===")
    for k, v in traits.items():
        print(f"{k}: {v}")


def print_scores(scores, label):
    print(f"\n=== STYLE SCORES ({label}) ===")
    for k, v in scores.items():
        print(f"{k}: {v:.3f}")


def print_recommendations(recs, label):
    print(f"\n=== RECOMMENDATIONS ({label}) ===")

    print("\nFace analysis:")
    for exp in recs["face_analysis"]:
        print(f"   - {exp}")

    print("\nTop hairstyles:")

    for style in recs["top_styles"]:
        print(f"\n{style['name']} (score: {style['score']:.3f})")

        if style["contributions"]:
            print("Key factors:")

            for c in style["contributions"][:3]:
                percent = c["percent"] * 100

                print(
                    f"   - {c['desc']} "
                    f"({percent:.1f}% influence)"
                )


def compare_traits(t1, t2):
    print("\n=== TRAITS COMPARISON ===")
    all_keys = set(t1.keys()).union(t2.keys())

    for k in all_keys:
        if t1.get(k) != t2.get(k):
            print(f"{k}: IMG1={t1.get(k)} | IMG2={t2.get(k)}")


img1 = cv2.imread("dataset/test_images/004455.jpg")
img2 = cv2.imread("dataset/test_images/konar.jpg")

detector = FaceLandmarkDetector(
    model_path="models/face_landmarker.task"
)

l1, f1, t1, s1, r1 = run_pipeline(img1, detector)
l2, f2, t2, s2, r2 = run_pipeline(img2, detector)

if l1 is None or l2 is None:
    print("No face detected")
    exit()


#print_features(f1, "IMG1")
#print_features(f2, "IMG2")

#print_traits(t1, "IMG1")
#print_traits(t2, "IMG2")

#compare_traits(t1, t2)

#print_scores(s1, "IMG1")
#print_scores(s2, "IMG2")

print_recommendations(r1, "IMG1")
print_recommendations(r2, "IMG2")


img_resized = cv2.resize(img1, None, fx=1.5, fy=1.5)
_, f_resized, _, _, _ = run_pipeline(img_resized, detector)

print("\n=== SELF CONSISTENCY TEST ===")
for k in f1.keys():
    diff = abs(f1[k] - f_resized[k])
    print(f"{k}: diff = {diff:.6f}")

print("\n=== CROSS IMAGE COMPARISON ===")
for k in f1.keys():
    diff = abs(f1[k] - f2[k])
    print(f"{k}: diff = {diff:.6f}")

img1_vis = draw_landmarks(img1, l1, draw_indices=False)
img1_vis = draw_geometry(img1_vis, l1)
img1_vis = draw_features(img1_vis, f1)
img1_vis = cv2.resize(img1_vis, None, fx=2, fy=2)

img2_vis = draw_landmarks(img2, l2, draw_indices=False)
img2_vis = draw_geometry(img2_vis, l2)
img2_vis = draw_features(img2_vis, f2)

cv2.imshow("IMG1 - FULL PIPELINE", img1_vis)
cv2.imshow("IMG2 - FEATURES", img2_vis)

cv2.waitKey(0)
cv2.destroyAllWindows()