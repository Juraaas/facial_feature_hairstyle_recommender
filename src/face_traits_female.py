def interpret_face_female(features):
    traits = {}

    fr = features["face_ratio"]
    if fr > 1.22:
        traits["face_length"] = "long"
    elif fr < 1.13:
        traits["face_length"] = "short"
    else:
        traits["face_length"] = "balanced"

    jr = features["jaw_ratio"]
    if jr > 0.81:
        traits["jaw"] = "wide"
    elif jr < 0.77:
        traits["jaw"] = "narrow"
    else:
        traits["jaw"] = "normal"

    jh = features["jaw_to_height"]
    if jh > 0.7:
        traits["jaw_height"] = "high"
    elif jh < 0.64:
        traits["jaw_height"] = "low"
    else:
        traits["jaw_height"] = "normal"

    er = features["eye_ratio"]
    if er > 0.66:
        traits["eyes"] = "wide"
    elif er < 0.62:
        traits["eyes"] = "close"
    else:
        traits["eyes"] = "normal"

    eh = features["eye_height"]
    if eh > 0.4:
        traits["eye_openness"] = "open"
    elif eh < 0.3:
        traits["eye_openness"] = "narrow"
    else:
        traits["eye_openness"] = "normal"

    lr = features["lip_ratio"]
    if lr > 0.45:
        traits["lips"] = "wide"
    elif lr < 0.36:
        traits["lips"] = "narrow"
    else:
        traits["lips"] = "normal"

    np_ = features["nose_position"]
    if np_ > 0.6:
        traits["nose"] = "lower-dominant"
    elif np_ < 0.55:
        traits["nose"] = "upper-dominant"
    else:
        traits["nose"] = "balanced"

    lfr = features["lower_face_ratio"]
    if lfr > 0.34:
        traits["lower_face"] = "long"
    elif lfr < 0.29:
        traits["lower_face"] = "short"
    else:
        traits["lower_face"] = "normal"

    cp = features["chin_prominence"]
    if cp > 0.25:
        traits["chin"] = "prominent"
    elif cp < 0.2:
        traits["chin"] = "recessed"
    else:
        traits["chin"] = "normal"

    sym = features["symmetry"]
    if sym < 0.041:
        traits["symmetry"] = "high"
    elif sym < 0.218:
        traits["symmetry"] = "medium"
    else:
        traits["symmetry"] = "low"

    return traits