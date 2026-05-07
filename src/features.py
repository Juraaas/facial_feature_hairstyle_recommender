from src.geometry import FaceGeometry

def extract_features(landmarks):
    geo = FaceGeometry(landmarks)

    return {
        "face_ratio": geo.face_ratio(),
        "jaw_ratio": geo.jaw_ratio(),
        "jaw_to_height": geo.jaw_to_height(),
        "eye_ratio": geo.eye_ratio(),
        "eye_height": geo.eye_height(),
        "lip_ratio": geo.lip_ratio(),
        "nose_position": geo.nose_position_ratio(),
        "lower_face_ratio": geo.lower_face_ratio(),
        "chin_prominence": geo.chin_prominence(),
        "symmetry": geo.symmetry_score(),
    }