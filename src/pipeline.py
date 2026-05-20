from src.features import extract_features
from src.face_traits import interpret_face
from src.rules import apply_rules
from src.face_traits_female import interpret_face_female
from src.rules_female import apply_rules_female
from src.recommender import generate_recommendations
from src.validation import validate_landmarks, validate_features
from src.logger import get_logger
from src.quality import assess_quality
from src.hair_segmentation import segment_face, find_hairline_y
from src.geometry import FaceGeometry

log = get_logger(__name__) 

def run_pipeline(img, detector, gender=None):
    landmarks = detector.detect(img)
    lm_check = validate_landmarks(landmarks)

    if not lm_check.valid:
        log.error(f"Landmarks validation failed: {lm_check.error}")
        return None, None, None, None, None, None
    
    quality = assess_quality(landmarks, img.shape)

    if quality.blocking:
        log.warning(f"Quality check blocked: {quality.blocking}")
        return None, None, None, None, None, quality
    
    hair_mask, skin_mask = segment_face(img)
    geo_temp = FaceGeometry(landmarks)
    face_w_px = geo_temp.face_width()
    hairline_y = find_hairline_y(hair_mask, face_w_px)

    features = extract_features(landmarks, hairline_y=hairline_y)
    feat_check = validate_features(features)

    if not feat_check.valid:
        log.error(f"Feature validation failed: {feat_check.error}")
        return None, None, None, None, None, quality
    
    log.info(f"Features extracted: { {k: f'{v:.3f}' for k, v in features.items()} }")

    if gender == "Woman":
        traits = interpret_face_female(features)
        scores = apply_rules_female(traits)
        recs   = generate_recommendations(scores, traits,
                     hairstyles_path="data/hairstyles_female.json")
    else:
        traits = interpret_face(features)
        scores = apply_rules(traits)
        recs   = generate_recommendations(scores, traits,
                     hairstyles_path="data/hairstyles.json")

    return landmarks, features, traits, scores, recs, quality