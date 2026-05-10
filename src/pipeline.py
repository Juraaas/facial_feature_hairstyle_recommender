from src.features import extract_features
from src.face_traits import interpret_face
from src.rules import apply_rules
from src.recommender import generate_recommendations
from src.validation import validate_landmarks, validate_features
from src.logger import get_logger

log = get_logger(__name__) 

def run_pipeline(img, detector):
    landmarks = detector.detect(img)
    lm_check = validate_landmarks(landmarks)

    if not lm_check.valid:
        log.error(f"Landmarks validation failed: {lm_check.error}")
        return None, None, None, None, None

    features = extract_features(landmarks)
    feat_check = validate_features(features)

    if not feat_check.valid:
        log.error(f"Feature validation failed: {feat_check.error}")
        return None, None, None, None, None
    
    log.info(f"Features extracted: { {k: f'{v:.3f}' for k, v in features.items()} }")

    traits = interpret_face(features)
    scores = apply_rules(traits)
    recs = generate_recommendations(scores, traits)

    return landmarks, features, traits, scores, recs