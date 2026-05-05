from src.features import extract_features
from src.face_traits import interpret_face
from src.rules import apply_rules
from src.recommender import generate_recommendations

def run_pipeline(img, detector):
    landmarks = detector.detect(img)

    if landmarks is None:
        return None, None, None, None, None

    features = extract_features(landmarks)
    traits = interpret_face(features)
    scores = apply_rules(traits)
    recs = generate_recommendations(scores, traits)

    return landmarks, features, traits, scores, recs