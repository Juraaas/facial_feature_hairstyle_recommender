def apply_rules(traits):
    scores = {
        "volume_top": 0,
        "volume_sides": 0,
        "short_sides": 0,
        "longer_hair": 0,
        "fringe": 0,
        "clean_lines": 0,
        "textured_top": 0,
        "soft_texture": 0,
    }

    if traits["face_length"] == "long":
        scores["volume_sides"] += 2
        scores["fringe"] += 2
        scores["volume_top"] -= 1
        scores["longer_hair"] -= 1

    elif traits["face_length"] == "short":
        scores["volume_top"] += 2
        scores["longer_hair"] += 1
        scores["volume_sides"] -= 1

    if traits["jaw"] == "wide":
        scores["soft_texture"] += 2
        scores["volume_top"] += 1
        scores["short_sides"] -= 1
    else:
        scores["volume_sides"] += 1
        scores["short_sides"] += 1

    if traits["jaw_height"] == "high":
        scores["longer_hair"] += 1
        scores["volume_top"] -= 1
    elif traits["jaw_height"] == "low":
        scores["volume_top"] += 1
        scores["fringe"] += 1

    if traits["eyes"] == "wide":
        scores["volume_top"] += 1
        scores["clean_lines"] += 1
    elif traits["eyes"] == "close":
        scores["volume_sides"] += 1
        scores["fringe"] -= 1

    if traits["eye_openness"] == "narrow":
        scores["fringe"] -= 1
        scores["volume_top"] += 1
    elif traits["eye_openness"] == "open":
        scores["fringe"] += 1

    if traits["lips"] == "wide":
        scores["soft_texture"] += 1
        scores["volume_top"] += 1
    elif traits["lips"] == "narrow":
        scores["clean_lines"] += 1

    if traits["nose"] == "upper-dominant":
        scores["fringe"] += 1 
        scores["volume_top"] -= 1
    elif traits["nose"] == "lower-dominant":
        scores["volume_top"] += 1
        scores["fringe"] -= 1

    if traits["lower_face"] == "long":
        scores["longer_hair"] += 1
        scores["soft_texture"] += 1
    elif traits["lower_face"] == "short":
        scores["short_sides"] += 1
        scores["clean_lines"] += 1

    if traits["chin"] == "prominent":
        scores["longer_hair"] += 1
        scores["soft_texture"] += 1
        scores["textured_top"] += 2
    elif traits["chin"] == "recessed":
        scores["volume_top"] += 1
        scores["short_sides"] += 1
        scores["textured_top"] -= 1

    scores = apply_symmetry_modulation(scores, traits)

    return scores

def apply_symmetry_modulation(scores, traits):
    TEXTURE_KEYS = ["soft_texture", "volume_sides", "fringe"]
    CLEAN_KEYS   = ["clean_lines", "short_sides"]

    if traits["symmetry"] == "low":
        for k in TEXTURE_KEYS:
            scores[k] = scores[k] * 1.3
        for k in CLEAN_KEYS:
            scores[k] = scores[k] * 0.7

    elif traits["symmetry"] == "high":
        for k in CLEAN_KEYS:
            scores[k] = scores[k] * 1.2
        scores["clean_lines"] += 1

    return scores