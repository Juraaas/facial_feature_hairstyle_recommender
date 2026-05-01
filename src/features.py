from src.geometry import FaceGeometry

def extract_features(landmarks):
    geo = FaceGeometry(landmarks)

    return {
        "face_ratio": geo.face_ratio(),
        "face_width": geo.face_width(),
        "face_height": geo.face_height(),
        "jaw_ratio": geo.jaw_ratio(),
        "eye_distance": geo.eye_dist(),
        "nose_position": geo.nose_position_ratio(),
        "symmetry": geo.symmetry_score(),
    }