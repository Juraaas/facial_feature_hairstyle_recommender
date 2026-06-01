from backend.src.geometry import FaceGeometry

def extract_features(landmarks, hairline_y=None):
    geo = FaceGeometry(landmarks, hairline_y=hairline_y)
    thirds = geo.facial_thirds_ratio()

    if thirds:
        u = thirds["upper"]
        m = thirds["middle"]
        l = thirds["lower"]
        max_dev   = max(abs(u - 0.333), abs(m - 0.333), abs(l - 0.333))
        if max_dev < 0.05:
            thirds_balance = 0.0
        elif max_dev < 0.10:
            thirds_balance = 0.5
        else:
            thirds_balance = 1.0
    else:
        u = m = l = 0.33
        thirds_balance = 0.0

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
        "upper_third": u,
        "middle_third": m,
        "lower_third": l,
        "mid_lower_ratio": geo.middle_lower_ratio(),
        "thirds_balance": thirds_balance,
    }