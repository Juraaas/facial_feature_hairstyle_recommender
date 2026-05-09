import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class ValidationResult:
    valid: bool
    error: Optional[str] = None

def validate_landmarks(landmarks) -> ValidationResult:
    if landmarks is None:
        return ValidationResult(False, "No face detected")
    
    if len(landmarks) != 478:
        return ValidationResult(False, f"Expected 478 landmarks, got {len(landmarks)}")
    
    if np.any(np.isnan(landmarks)) or np.any(np.isinf(landmarks)):
        return ValidationResult(False, "Landmarks contain NaN or inf values")
    
    return ValidationResult(True)

def validate_features(features: dict) -> ValidationResult:
    EXPECTED_KEYS = {
        "face_ratio", "jaw_ratio", "jaw_to_height", "eye_ratio",
        "eye_height", "lip_ratio", "nose_position", "lower_face_ratio",
        "chin_prominence", "symmetry",
    }

    missing = EXPECTED_KEYS - set(features.keys())
    if missing:
        return ValidationResult(False, f"Missing features: {missing}")
    
    for key, value in features.items():
        if not np.isfinite(value):
            return ValidationResult(False, f"Feature '{key}' is {value}")
        if value < 0:
            return ValidationResult(False, f"Feature '{key}' is negative: {value}")
        
    SANITY_BOUNDS = {
        "face_ratio":       (0.5, 2.5),
        "jaw_ratio":        (0.3, 1.2),
        "eye_ratio":        (0.2, 0.9),
        "symmetry":         (0.0, 1.0),
        "nose_position":    (0.2, 0.8),
    }

    for key, (lo, hi) in SANITY_BOUNDS.items():
        if key in features and not (lo <= features[key] <= hi):
            return ValidationResult(
                False,
                f"Feature '{key}' = {features[key]:.3f} is outside expected range [{lo}, {hi}]"
            )
    
    return ValidationResult(True)