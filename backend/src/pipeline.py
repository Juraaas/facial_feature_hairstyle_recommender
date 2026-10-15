from src.features import extract_features
from src.face_traits import interpret_face
from src.rules import apply_rules
from src.recommender import generate_recommendations
from src.validation import validate_landmarks, validate_features
from src.logger import get_logger
from src.quality import assess_quality
from src.hair_segmentation import segment_face, find_hairline_y
from src.geometry import FaceGeometry
from src.hair_classifier import classify_hair

log = get_logger(__name__) 

def run_pipeline(img, detector, gender=None, lang="pl", debug_rules=False):
    landmarks = detector.detect(img)
    lm_check = validate_landmarks(landmarks)

    if not lm_check.valid:
        log.error(f"Landmarks validation failed: {lm_check.error}")
        return None, None, None, None, None, None
    
    quality = assess_quality(landmarks, img)

    if quality.blocking:
        log.warning(f"Quality check blocked: {quality.blocking}")
        return None, None, None, None, None, quality
    
    hair_mask, _ = segment_face(img)
    geo_temp = FaceGeometry(landmarks)
    face_w_px = geo_temp.face_width()
    hairline_y = find_hairline_y(hair_mask, face_w_px)

    features = extract_features(landmarks, hairline_y=hairline_y)
    feat_check = validate_features(features)

    if not feat_check.valid:
        log.error(f"Feature validation failed: {feat_check.error}")
        return None, None, None, None, None, quality
    
    log.info(f"Features extracted: { {k: f'{v:.3f}' for k, v in features.items()} }")
    
    hair_result = classify_hair(img, hair_mask, gender=gender or "Man", landmarks=landmarks)
    log.info(f"Hair: type={hair_result['hair_type']} "
             f"({hair_result['hair_conf']:.2f}), "
             f"hairline={hair_result['hairline']} "
             f"({hair_result['hairline_conf']:.2f}), "
             f"coverage={hair_result['coverage']:.3f}")
    
    if gender == "Woman":
        traits = interpret_face(features, gender="Woman")
        if hair_result["hair_type"] is not None:
            traits["hair_type"] = hair_result["hair_type"]
        if hair_result["hairline"] is not None:
            traits["hairline"] = hair_result["hairline"]
        if hair_result["coverage"] is not None:
            traits["coverage"] = hair_result["coverage"]

    else:
        traits = interpret_face(features, gender="Man")
        if hair_result["hair_type"] is not None:
            traits["hair_type"] = hair_result["hair_type"]
        if hair_result["hairline"] is not None:
            traits["hairline"] = hair_result["hairline"]
        if hair_result["coverage"] is not None:
            traits["coverage"] = hair_result["coverage"]    

    if debug_rules:
        debug_result = apply_rules(traits, gender=gender or "Man", debug=True)
        scores = debug_result["scores"]
        trace = debug_result["trace"]

        print("\n" + "=" * 60)
        print(f"RULES DEBUG | gender={gender or 'Man'}")
        print("Traits:", traits)

        for stage in (
            "base",
            "after_interactions",
            "after_symmetry",
            "after_constraints",
            "after_clamp",
        ):
            print(f"\n{stage}:")
            print(trace[stage])

        print("\nDELTAS:")
        for stage, delta in trace["deltas"].items():
            print(f"{stage}: {delta}")

        print("\nCLAMPED DIMENSIONS:")
        print(trace["clamped_dimensions"])
        print("=" * 60)

    else:
        scores = apply_rules(traits, gender=gender or "Man")

    if gender == "Woman":
        recs = generate_recommendations(scores, traits, gender="Woman", hairstyles_path="data/hairstyles_female.json", lang=lang)
    else:
        recs = generate_recommendations(scores, traits, gender="Man", hairstyles_path="data/hairstyles.json", lang=lang)

    return landmarks, features, traits, scores, recs, quality, hair_mask