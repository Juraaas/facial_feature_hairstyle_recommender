import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.geometry import FaceGeometry

@dataclass
class QualityReport:
    passed: bool
    score: float
    warnings: list[str]
    blocking: Optional[str] = None

def assess_quality(landmarks, img_shape) -> QualityReport:
    h, w = img_shape[:2]
    geo = FaceGeometry(landmarks)
    warnings = []

    face_width_px = geo.face_width()
    min_face_px = min(w, h) * 0.15

    if face_width_px < min_face_px:
        return QualityReport(
            passed=False, score=0.0,
            blocking="Face too small — move closer to the camera"
        )
    
    nose_x = geo.nose()[0]
    left_x = geo.left_eye()[0]
    right_x = geo.right_eye()[0]
    eye_mid_x = (left_x + right_x) / 2
    eye_dist = abs(left_x - right_x)

    yaw_offset = abs(nose_x - eye_mid_x) / eye_dist if eye_dist > 0 else 0
    if yaw_offset > 0.25:
        warnings.append("Head is turned — results may be less accurate")
    
    chin_y = geo.chin()[1]
    forehead_y = geo.forehead_top()[1]
    face_h = abs(chin_y - forehead_y)
    nose_y = geo.nose()[1]
    expected_y = (chin_y + forehead_y) / 2

    pitch_offset = abs(nose_y - expected_y) / face_h if face_h > 0 else 0
    if pitch_offset > 0.15:
        warnings.append("Head is tilted up or down - try a straight-on photo")

    left_cheek_x = geo.left_cheek()[0]
    right_cheek_x = geo.right_cheek()[0]
    mid_x = (left_cheek_x + right_cheek_x) / 2
    
    nose_offset = abs(geo.nose()[0] - mid_x) / face_width_px
    if nose_offset > 0.12:
        warnings.append("Unusual landmark alignment - lighting or angle may be off")
    
    score = max(0.1, 1.0 - len(warnings) * 0.2)

    return QualityReport(passed=True, score=round(score, 2), warnings=warnings)
