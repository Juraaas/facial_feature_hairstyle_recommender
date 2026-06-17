THRESHOLDS = {
    "Man": {
        "face_ratio": {"long": 1.38, "short": 1.215},
        "jaw_ratio": {"wide": 0.825, "narrow": 0.8},
        "jaw_to_height": {"high": 0.66,  "low": 0.58},
        "eye_ratio": {"wide": 0.64,  "close": 0.6},
        "eye_height": {"open": 0.36,  "narrow": 0.27},
        "lip_ratio": {"wide": 0.43,  "narrow": 0.35},
        "nose_position": {"lower": 0.628,"upper": 0.568},
        "lower_face_ratio": {"long": 0.316, "short": 0.269},
        "chin_prominence": {"prom": 0.219, "rec": 0.173},
        "symmetry": {"high": 0.032, "medium": 0.163},
        "mid_lower_ratio": {"lower_dom": 0.78, "middle_dom": 0.913},
        "upper_third": {"high": 0.356, "low": 0.286},
    },
    "Woman": {
        "face_ratio": {"long": 1.3, "short": 1.085},
        "jaw_ratio": {"wide": 0.81, "narrow": 0.77},
        "jaw_to_height": {"high": 0.735, "low": 0.61},
        "eye_ratio": {"wide": 0.66,  "close": 0.62},
        "eye_height": {"open": 0.4,   "narrow": 0.3},
        "lip_ratio": {"wide": 0.45,  "narrow": 0.36},
        "nose_position": {"lower": 0.62, "upper": 0.53},
        "lower_face_ratio": {"long": 0.348, "short": 0.28},
        "chin_prominence":  {"prom": 0.25,  "rec": 0.2},
        "symmetry": {"high": 0.038, "medium": 0.21},
        "mid_lower_ratio": {"lower_dom": 0.80, "middle_dom": 1.0},
        "upper_third": {"high": 0.325, "low": 0.195},
    },
}

def interpret_face(features, gender="Man"):
    t = THRESHOLDS.get(gender, THRESHOLDS["Man"])
    traits = {}

    fr = features["face_ratio"]
    if fr > t["face_ratio"]["long"]:
        traits["face_length"] = "long"
    elif fr < t["face_ratio"]["short"]:
        traits["face_length"] = "short"
    else:
        traits["face_length"] = "balanced"

    jr = features["jaw_ratio"]
    if jr > t["jaw_ratio"]["wide"]:
        traits["jaw"] = "wide"
    elif jr < t["jaw_ratio"]["narrow"]:
        traits["jaw"] = "narrow"
    else:
        traits["jaw"] = "normal"

    jh = features["jaw_to_height"]
    if jh > t["jaw_to_height"]["high"]:
        traits["jaw_height"] = "high"
    elif jh < t["jaw_to_height"]["low"]:
        traits["jaw_height"] = "low"
    else:
        traits["jaw_height"] = "normal"

    er = features["eye_ratio"]
    if er > t["eye_ratio"]["wide"]:
        traits["eyes"] = "wide"
    elif er < t["eye_ratio"]["close"]:
        traits["eyes"] = "close"
    else:
        traits["eyes"] = "normal"

    eh = features["eye_height"]
    if eh > t["eye_height"]["open"]:
        traits["eye_openness"] = "open"
    elif eh < t["eye_height"]["narrow"]:
        traits["eye_openness"] = "narrow"
    else:
        traits["eye_openness"] = "normal"

    lr = features["lip_ratio"]
    if lr > t["lip_ratio"]["wide"]:
        traits["lips"] = "wide"
    elif lr < t["lip_ratio"]["narrow"]:
        traits["lips"] = "narrow"
    else:
        traits["lips"] = "normal"

    np_ = features["nose_position"]
    if np_ > t["nose_position"]["lower"]:
        traits["nose"] = "lower-dominant"
    elif np_ < t["nose_position"]["upper"]:
        traits["nose"] = "upper-dominant"
    else:
        traits["nose"] = "balanced"

    lfr = features["lower_face_ratio"]
    if lfr > t["lower_face_ratio"]["long"]:
        traits["lower_face"] = "long"
    elif lfr < t["lower_face_ratio"]["short"]:
        traits["lower_face"] = "short"
    else:
        traits["lower_face"] = "normal"

    cp = features["chin_prominence"]
    if cp > t["chin_prominence"]["prom"]:
        traits["chin"] = "prominent"
    elif cp < t["chin_prominence"]["rec"]:
        traits["chin"] = "recessed"
    else:
        traits["chin"] = "normal"

    sym = features["symmetry"]
    if sym < t["symmetry"]["high"]:
        traits["symmetry"] = "high"
    elif sym < t["symmetry"]["medium"]:
        traits["symmetry"] = "medium"
    else:
        traits["symmetry"] = "low"

    mlr = features["mid_lower_ratio"]
    if mlr < t["mid_lower_ratio"]["lower_dom"]:
        traits["facial_thirds"] = "lower_dominant"
    elif mlr > t["mid_lower_ratio"]["middle_dom"]:
        traits["facial_thirds"] = "middle_dominant"
    else:
        traits["facial_thirds"] = "balanced"

    ut = features["upper_third"]
    if ut > t["upper_third"]["high"]:
        traits["forehead"] = "high"
    elif ut < t["upper_third"]["low"]:
        traits["forehead"] = "low"
    else:
        traits["forehead"] = "normal"

    u = features["upper_third"]
    m = features["middle_third"]
    l = features["lower_third"]

    thirds = {"upper": u, "middle": m, "lower": l}
    dominant = max(thirds, key=thirds.get)
    dominant_val = thirds[dominant]

    traits["dominant_third"] = dominant if dominant_val > 0.38 else "balanced"

    if u > 0.35 and l < 0.34:
        traits["thirds_vertical"] = "top_heavy"
    elif u < 0.27 and l > 0.38:
        traits["thirds_vertical"] = "bottom_heavy"
    else:
        traits["thirds_vertical"] = "balanced"

    max_dev = max(abs(u - 0.333), abs(m - 0.333), abs(l - 0.333))
    if max_dev < 0.05:
        traits["thirds_balance"] = "balanced"
    elif max_dev < 0.10:
        traits["thirds_balance"] = "slight_imbalance"
    else:
        traits["thirds_balance"] = "imbalanced"

    traits.setdefault("hair_type", None)
    traits.setdefault("hairline",  None)

    return traits