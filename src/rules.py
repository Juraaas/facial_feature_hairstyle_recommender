STRONG = 3
MEDIUM = 2
WEAK = 1

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
        scores["volume_sides"] += STRONG
        scores["fringe"] += STRONG
        scores["volume_top"] -= MEDIUM
        scores["longer_hair"] -= MEDIUM
        scores["clean_lines"] -= WEAK

    elif traits["face_length"] == "short":
        scores["volume_top"] += STRONG
        scores["longer_hair"] += MEDIUM
        scores["fringe"] -= MEDIUM
        scores["volume_sides"] -= WEAK

    if traits["jaw"] == "wide":
        scores["soft_texture"] += STRONG
        scores["volume_top"] += MEDIUM
        scores["longer_hair"] += WEAK
        scores["short_sides"] -= STRONG
        scores["clean_lines"] -= MEDIUM
        
    elif traits["jaw"] == "narrow":
        scores["short_sides"] += STRONG
        scores["volume_sides"] += MEDIUM
        scores["clean_lines"] += WEAK
        scores["soft_texture"] -= WEAK
        

    if traits["jaw_height"] == "high":
        scores["longer_hair"] += MEDIUM
        scores["soft_texture"] += WEAK
        scores["volume_top"] -= MEDIUM
        scores["short_sides"] -= WEAK

    elif traits["jaw_height"] == "low":
        scores["volume_top"] += MEDIUM
        scores["fringe"] += WEAK
        scores["longer_hair"] -= WEAK

    if traits["eyes"] == "wide":
        scores["volume_top"] += MEDIUM
        scores["clean_lines"] += MEDIUM
        scores["volume_sides"] -= WEAK

    elif traits["eyes"] == "close":
        scores["volume_sides"] += MEDIUM
        scores["soft_texture"] += WEAK
        scores["fringe"] -= MEDIUM
        scores["clean_lines"] -= WEAK

    if traits["eye_openness"] == "narrow":
        scores["volume_top"] += MEDIUM
        scores["clean_lines"] += WEAK
        scores["fringe"] -= STRONG

    elif traits["eye_openness"] == "open":
        scores["fringe"] += MEDIUM
        scores["textured_top"] += WEAK

    if traits["lips"] == "wide":
        scores["soft_texture"] += MEDIUM
        scores["volume_top"] += WEAK
        scores["clean_lines"] -= WEAK

    elif traits["lips"] == "narrow":
        scores["clean_lines"] += MEDIUM
        scores["short_sides"] += WEAK
        scores["soft_texture"] -= WEAK

    if traits["nose"] == "upper-dominant":
        scores["fringe"] += MEDIUM
        scores["volume_sides"] += WEAK
        scores["volume_top"] -= STRONG

    elif traits["nose"] == "lower-dominant":
        scores["volume_top"] += STRONG
        scores["longer_hair"] += WEAK
        scores["fringe"] -= MEDIUM

    if traits["lower_face"] == "long":
        scores["longer_hair"] += MEDIUM
        scores["soft_texture"] += MEDIUM
        scores["short_sides"] -= WEAK

    elif traits["lower_face"] == "short":
        scores["short_sides"] += MEDIUM
        scores["clean_lines"] += MEDIUM
        scores["longer_hair"] -= WEAK

    if traits["chin"] == "prominent":
        scores["textured_top"] += STRONG
        scores["longer_hair"] += MEDIUM
        scores["soft_texture"] += MEDIUM
        scores["short_sides"] -= WEAK
        scores["clean_lines"] -= WEAK

    elif traits["chin"] == "recessed":
        scores["volume_top"] += STRONG
        scores["short_sides"] += MEDIUM
        scores["clean_lines"] += WEAK
        scores["textured_top"] -= MEDIUM
        scores["longer_hair"] -= WEAK

    if traits.get("facial_thirds") == "lower_dominant":
        scores["volume_top"] += STRONG
        scores["fringe"] -= MEDIUM
        scores["longer_hair"] -= WEAK

    elif traits.get("facial_thirds") == "middle_dominant":
        scores["fringe"] += MEDIUM
        scores["volume_sides"] += WEAK
        scores["volume_top"] -= MEDIUM

    if traits.get("forehead") == "high":
        scores["fringe"] += STRONG
        scores["volume_top"] -= MEDIUM

    elif traits.get("forehead") == "low":
        scores["volume_top"] += WEAK
        scores["textured_top"] += WEAK
        scores["fringe"] -= STRONG

    if traits.get("thirds_balance") == "imbalanced":
        scores["soft_texture"] += WEAK
        scores["clean_lines"] -= WEAK

    scores = apply_symmetry_modulation(scores, traits)

    return scores

def apply_symmetry_modulation(scores, traits):
    TEXTURE_KEYS = ["soft_texture", "volume_sides", "fringe"]
    CLEAN_KEYS   = ["clean_lines", "short_sides"]

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