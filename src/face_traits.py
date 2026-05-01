

def interpret_face(features):
    traits = {}

    fr = features["face_ratio"]
    jr = features["jaw_ratio"]
    er = features["eye_ratio"]
    nose_pos = features["nose_position"]
    sym = features["symmetry"]

    if fr > 1.25:
        traits["face_length"] = "long"
    elif fr < 0.95:
        traits["face_length"] = "short"
    else:
        traits["face_length"] = "balanced"

    if jr > 0.9:
        traits["jaw"] = "wide"
    else:
        traits["jaw"] = "narrow"

    if er > 0.5:
        traits["eyes"] = "wide"
    elif er < 0.4:
        traits["eyes"] = "close"
    else:
        traits["eyes"] = "normal"

    if nose_pos > 0.55:
        traits["nose"] = "upper-dominant"
    else:
        traits["nose"] = "lower-dominant"

    if sym < 0.02:
        traits["symmetry"] = "high"
    elif sym < 0.05:
        traits["symmetry"] = "medium"
    else:
        traits["symmetry"] = "low"

    return traits