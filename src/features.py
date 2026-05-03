from src.geometry import FaceGeometry

def extract_features(landmarks):
    geo = FaceGeometry(landmarks)

    return {
        "face_ratio": geo.face_ratio(),
        "jaw_ratio": geo.jaw_ratio(),
        "jaw_to_height": geo.jaw_to_height(),
        "jaw_projection": geo.jaw_projection(),
        "eye_ratio": geo.eye_ratio(),
        "nose_position": geo.nose_position_ratio(),
        "symmetry": geo.symmetry_score(),
    }