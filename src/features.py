from src.geometry import FaceGeometry

def extract_features(landmarks, hairline_y=None):
    geo = FaceGeometry(landmarks, hairline_y=hairline_y)
    thirds = geo.facial_thirds_ratio()

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
        "upper_third": thirds["upper"]  if thirds else 0.33,
        "middle_third": thirds["middle"] if thirds else 0.33,
        "lower_third": thirds["lower"]  if thirds else 0.33,
        "mid_lower_ratio": geo.middle_lower_ratio(),
    }