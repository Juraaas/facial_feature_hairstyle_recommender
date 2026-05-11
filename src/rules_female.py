STRONG = 3
MEDIUM = 2
WEAK   = 1

def apply_rules_female(traits):
    scores = {
        "volume_top": 0,
        "volume_sides": 0,
        "layers": 0,
        "longer_hair": 0,
        "fringe": 0,
        "clean_lines": 0,
        "textured_top": 0,
        "soft_texture": 0,
        "updo": 0,
        "curtain_fringe": 0,
    }

    if traits["face_length"] == "long":
        scores["volume_sides"] += STRONG
        scores["fringe"] += STRONG
        scores["curtain_fringe"] += MEDIUM
        scores["layers"] += MEDIUM
        scores["volume_top"] -= MEDIUM
        scores["longer_hair"] -= WEAK
        scores["updo"] -= WEAK

    elif traits["face_length"] == "short":
        scores["volume_top"] += STRONG
        scores["longer_hair"] += STRONG
        scores["updo"] += MEDIUM
        scores["fringe"] -= MEDIUM
        scores["volume_sides"] -= WEAK

    if traits["jaw"] == "wide":
        scores["longer_hair"] += STRONG
        scores["soft_texture"] += STRONG
        scores["layers"] += MEDIUM
        scores["curtain_fringe"] += MEDIUM
        scores["volume_sides"] -= MEDIUM
        scores["clean_lines"] -= MEDIUM

    elif traits["jaw"] == "narrow":
        scores["volume_sides"] += STRONG
        scores["layers"] += MEDIUM
        scores["clean_lines"] += WEAK
        scores["soft_texture"] += WEAK
        

    if traits["jaw_height"] == "high":
        scores["longer_hair"] += MEDIUM
        scores["layers"] += WEAK
        scores["volume_top"] -= MEDIUM

    elif traits["jaw_height"] == "low":
        scores["volume_top"] += MEDIUM
        scores["updo"] += WEAK
        scores["longer_hair"] -= WEAK

    if traits["eyes"] == "wide":
        scores["clean_lines"] += MEDIUM
        scores["volume_top"] += WEAK
        scores["curtain_fringe"] -= WEAK

    elif traits["eyes"] == "close":
        scores["curtain_fringe"] += STRONG
        scores["volume_sides"] += MEDIUM
        scores["fringe"] -= MEDIUM

    if traits["eye_openness"] == "narrow":
        scores["volume_top"] += MEDIUM
        scores["curtain_fringe"] -= MEDIUM
        scores["fringe"] -= STRONG

    elif traits["eye_openness"] == "open":
        scores["fringe"] += MEDIUM
        scores["curtain_fringe"] += MEDIUM
        scores["textured_top"] += WEAK

    if traits["lips"] == "wide":
        scores["soft_texture"] += MEDIUM
        scores["layers"] += WEAK
        scores["clean_lines"] -= WEAK

    elif traits["lips"] == "narrow":
        scores["clean_lines"] += MEDIUM
        scores["updo"] += WEAK

    if traits["nose"] == "upper-dominant":
        scores["fringe"] += STRONG
        scores["curtain_fringe"] += MEDIUM
        scores["volume_top"] -= STRONG

    elif traits["nose"] == "lower-dominant":
        scores["volume_top"] += STRONG
        scores["updo"] += MEDIUM
        scores["fringe"] -= MEDIUM

    if traits["lower_face"] == "long":
        scores["longer_hair"] += MEDIUM
        scores["soft_texture"] += MEDIUM
        scores["layers"] += WEAK

    elif traits["lower_face"] == "short":
        scores["clean_lines"] += MEDIUM
        scores["updo"] += WEAK
        scores["longer_hair"] -= WEAK

    if traits["chin"] == "prominent":
        scores["longer_hair"] += STRONG
        scores["soft_texture"] += STRONG
        scores["layers"] += MEDIUM
        scores["clean_lines"] -= MEDIUM

    elif traits["chin"] == "recessed":
        scores["volume_top"] += STRONG
        scores["updo"] += MEDIUM
        scores["clean_lines"] += WEAK
        scores["longer_hair"] -= WEAK

    scores = apply_symmetry_modulation_female(scores, traits)

    return scores

def apply_symmetry_modulation_female(scores, traits):
    TEXTURE_KEYS = ["soft_texture", "volume_sides", "fringe", "layers"]
    CLEAN_KEYS   = ["clean_lines", "updo"]

    if traits["symmetry"] == "low":
        for k in TEXTURE_KEYS:
            scores[k] = scores[k] * 1.3 + 1
        for k in CLEAN_KEYS:
            scores[k] = scores[k] * 0.7

    elif traits["symmetry"] == "high":
        for k in CLEAN_KEYS:
            scores[k] = scores[k] * 1.2
        scores["clean_lines"] += WEAK
        for k in TEXTURE_KEYS:
            scores[k] = scores[k] * 0.9

    return scores