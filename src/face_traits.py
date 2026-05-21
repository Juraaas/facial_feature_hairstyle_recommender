
def interpret_face(features):
    traits = {}

    fr = features["face_ratio"]
    if fr > 1.22:
        traits["face_length"] = "long"
    elif fr < 1.13:
        traits["face_length"] = "short"
    else:
        traits["face_length"] = "balanced"

    jr = features["jaw_ratio"]
    if jr > 0.825:
        traits["jaw"] = "wide"
    elif jr < 0.8:
        traits["jaw"] = "narrow"
    else:
        traits["jaw"] = "normal"


    jh = features["jaw_to_height"]
    if jh > 0.71:
        traits["jaw_height"] = "high"
    elif jh < 0.65:
        traits["jaw_height"] = "low"
    else:
        traits["jaw_height"] = "normal"

    er = features["eye_ratio"]
    if er > 0.64:
        traits["eyes"] = "wide"
    elif er < 0.6:
        traits["eyes"] = "close"
    else:
        traits["eyes"] = "normal"

    eh = features["eye_height"]
    if eh > 0.36:
        traits["eye_openness"] = "open"
    elif eh < 0.27:
        traits["eye_openness"] = "narrow"
    else:
        traits["eye_openness"] = "normal"

    lr = features["lip_ratio"]
    if lr > 0.43:
        traits["lips"] = "wide"
    elif lr < 0.35:
        traits["lips"] = "narrow"
    else:
        traits["lips"] = "normal"

    np_ = features["nose_position"]
    if np_ > 0.58:
        traits["nose"] = "lower-dominant"
    elif np_ < 0.53:
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
    if cp > 0.24:
        traits["chin"] = "prominent"
    elif cp < 0.19:
        traits["chin"] = "recessed"
    else:
        traits["chin"] = "normal"

    sym = features["symmetry"]
    if sym < 0.032:
        traits["symmetry"] = "high"
    elif sym < 0.173:
        traits["symmetry"] = "medium"
    else:
        traits["symmetry"] = "low"

    mlr = features["mid_lower_ratio"]
    if mlr < 0.80:
        traits["facial_thirds"] = "lower_dominant"
    elif mlr > 1.20:
        traits["facial_thirds"] = "middle_dominant"
    else:
        traits["facial_thirds"] = "balanced"

    ut = features["upper_third"]
    if ut < 0.27:
        traits["forehead"] = "low"
    elif ut > 0.38:
        traits["forehead"] = "high"
    else:
        traits["forehead"] = "normal"

    u = features["upper_third"]
    m = features["middle_third"]
    l = features["lower_third"]
    max_dev = max(abs(u - 0.333), abs(m - 0.333), abs(l - 0.333))
    if max_dev < 0.05:
        traits["thirds_balance"] = "balanced"
    elif max_dev < 0.10:
        traits["thirds_balance"] = "slight_imbalance"
    else:
        traits["thirds_balance"] = "imbalanced"

    return traits