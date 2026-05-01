import cv2
import numpy as np


def draw_landmarks(image, landmarks, draw_indices=False):
    img = image.copy()

    for i, (x, y, z) in enumerate(landmarks):
        x, y = int(x), int(y)

        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

        if draw_indices:
            cv2.putText(
                img,
                str(i),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    return img

def draw_geometry(image, landmarks):
    img = image.copy()

    def p(idx):
        return tuple(map(int, landmarks[idx][:2]))
    
    cv2.line(img, p(10), p(152), (255, 0, 0), 2)
    cv2.line(img, p(234), p(454), (0, 255, 255), 2)
    cv2.line(img, p(172), p(397), (0, 0, 255), 2)
    cv2.line(img, p(33), p(263), (255, 255, 0), 2)

    return img

def draw_features(image, features):
    img = image.copy()
    y = 30

    for k, v in features.items():
        text = f"{k}: {v:.3f}"

        cv2.putText(
            img,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 25

    return img

def draw_feature_debug(image, features):
    img = image.copy()

    cv2.putText(
        img,
        f"FR: {features['face_ratio']:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        img,
        f"JR: {features['jaw_ratio']:.2f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        img,
        f"SYM: {features['symmetry']:.3f}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    return img