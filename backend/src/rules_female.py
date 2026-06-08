from rules import clamp_scores

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

    if traits.get("face_length") == "long":
        scores["volume_sides"] += STRONG
        scores["fringe"] += STRONG
        scores["curtain_fringe"] += MEDIUM
        scores["layers"] += MEDIUM
        scores["volume_top"] -= MEDIUM
        scores["updo"] -= MEDIUM
        scores["longer_hair"] -= WEAK

    elif traits.get("face_length") == "short":
        scores["volume_top"] += STRONG
        scores["longer_hair"] += STRONG
        scores["updo"] += MEDIUM
        scores["fringe"] -= MEDIUM
        scores["volume_sides"] -= WEAK

    if traits.get("jaw") == "wide":
        scores["longer_hair"] += STRONG
        scores["soft_texture"] += STRONG
        scores["layers"] += MEDIUM
        scores["curtain_fringe"] += MEDIUM
        scores["volume_sides"] -= MEDIUM
        scores["clean_lines"] -= MEDIUM

    elif traits.get("jaw") == "narrow":
        scores["volume_sides"] += STRONG
        scores["layers"] += MEDIUM
        scores["clean_lines"] += WEAK
        scores["soft_texture"] += WEAK    

    if traits.get("jaw_height") == "high":
        scores["longer_hair"] += MEDIUM
        scores["layers"] += WEAK
        scores["volume_top"] -= MEDIUM

    elif traits.get("jaw_height") == "low":
        scores["volume_top"] += MEDIUM
        scores["longer_hair"] -= WEAK

    if traits.get("eyes") == "wide":
        scores["clean_lines"] += MEDIUM
        scores["volume_top"] += WEAK
        scores["curtain_fringe"] -= WEAK

    elif traits.get("eyes") == "close":
        scores["curtain_fringe"] += STRONG
        scores["volume_sides"] += MEDIUM
        scores["fringe"] -= MEDIUM

    if traits.get("eye_openness") == "narrow":
        scores["volume_top"] += MEDIUM
        scores["fringe"] -= STRONG
        scores["curtain_fringe"] -= MEDIUM

    elif traits.get("eye_openness") == "open":
        scores["fringe"] += MEDIUM
        scores["curtain_fringe"] += MEDIUM
        scores["textured_top"] += WEAK

    if traits.get("lips") == "wide":
        scores["soft_texture"] += MEDIUM
        scores["layers"] += WEAK
        scores["clean_lines"] -= WEAK

    elif traits.get("lips") == "narrow":
        scores["clean_lines"] += MEDIUM
        scores["updo"] += WEAK

    if traits.get("nose") == "upper-dominant":
        scores["fringe"] += STRONG
        scores["curtain_fringe"] += MEDIUM
        scores["volume_top"] -= STRONG

    elif traits.get("nose") == "lower-dominant":
        scores["volume_top"] += STRONG
        scores["updo"] += WEAK
        scores["fringe"] -= MEDIUM

    if traits.get("lower_face") == "long":
        scores["longer_hair"] += MEDIUM
        scores["soft_texture"] += MEDIUM
        scores["layers"] += WEAK

    elif traits.get("lower_face") == "short":
        scores["clean_lines"] += MEDIUM
        scores["longer_hair"] -= WEAK

    if traits.get("chin") == "prominent":
        scores["longer_hair"] += STRONG
        scores["soft_texture"] += STRONG
        scores["layers"] += MEDIUM
        scores["clean_lines"] -= MEDIUM

    elif traits.get("chin") == "recessed":
        scores["volume_top"] += STRONG
        scores["updo"] += WEAK
        scores["clean_lines"] += WEAK
        scores["longer_hair"] -= WEAK

    if traits.get("facial_thirds") == "lower_dominant":
        scores["volume_top"] += STRONG
        scores["updo"] += MEDIUM
        scores["fringe"] -= MEDIUM
        scores["longer_hair"] -= WEAK

    elif traits.get("facial_thirds") == "middle_dominant":
        scores["fringe"] += MEDIUM
        scores["curtain_fringe"] += MEDIUM
        scores["volume_top"] -= MEDIUM

    if traits.get("forehead") == "high":
        scores["fringe"] += STRONG
        scores["curtain_fringe"] += MEDIUM
        scores["volume_top"] -= MEDIUM
        scores["updo"] -= MEDIUM

    elif traits.get("forehead") == "low":
        scores["updo"] += MEDIUM
        scores["fringe"] -= STRONG
        scores["curtain_fringe"] -= MEDIUM

    if traits.get("thirds_balance") == "imbalanced":
        scores["soft_texture"] += MEDIUM
        scores["layers"] += MEDIUM
        scores["clean_lines"] -= MEDIUM

    scores = apply_interaction_rules_female(scores, traits)
    scores = apply_symmetry_modulation_female(scores, traits)
    scores = clamp_scores(scores)

    return scores

def apply_interaction_rules_female(scores, traits):
    if traits.get("face_length") == "long" and traits.get("forehead") == "high":
        scores["fringe"] += WEAK
        scores["curtain_fringe"] += WEAK
        scores["volume_sides"] += WEAK
        scores["volume_top"] -= MEDIUM
        scores["updo"] -= WEAK

    if traits.get("face_length") == "long" and traits.get("jaw") == "narrow":
        scores["volume_sides"] += MEDIUM
        scores["layers"] += MEDIUM
        scores["curtain_fringe"] += WEAK
        scores["volume_top"] -= WEAK
        scores["updo"] -= WEAK

    if traits.get("face_length") == "short" and traits.get("jaw") == "narrow":
        scores["volume_top"] += MEDIUM
        scores["updo"] += MEDIUM
        scores["clean_lines"] += WEAK
        scores["fringe"] -= MEDIUM
        scores["volume_sides"] -= WEAK

    if traits.get("jaw") == "wide" and traits.get("chin") == "prominent":
        scores["soft_texture"] += STRONG
        scores["longer_hair"] += MEDIUM
        scores["layers"] += MEDIUM
        scores["curtain_fringe"] += WEAK
        scores["clean_lines"] -= MEDIUM
        scores["updo"] -= WEAK

    if traits.get("jaw") == "narrow" and traits.get("chin") == "recessed":
        scores["volume_top"] += WEAK
        scores["clean_lines"] += WEAK
        scores["longer_hair"] -= WEAK
        scores["soft_texture"] -= WEAK

        if traits.get("face_length") != "long" and traits.get("forehead") != "high":
            scores["updo"] += WEAK

    if traits.get("eyes") == "close" and traits.get("forehead") == "low":
        scores["curtain_fringe"] += MEDIUM
        scores["volume_sides"] += MEDIUM
        scores["volume_top"] += WEAK
        scores["fringe"] -= STRONG

    if traits.get("eyes") == "wide" and traits.get("forehead") == "high":
        scores["fringe"] += WEAK
        scores["volume_top"] -= WEAK

    if traits.get("eye_openness") == "narrow" and traits.get("face_length") == "long":
        scores["volume_sides"] += MEDIUM
        scores["layers"] += WEAK
        scores["fringe"] -= MEDIUM
        scores["curtain_fringe"] -= WEAK

    if traits.get("eye_openness") == "narrow" and traits.get("forehead") == "high":
        scores["curtain_fringe"] += WEAK
        scores["soft_texture"] += WEAK
        scores["volume_top"] -= WEAK
    
    if traits.get("facial_thirds") == "lower_dominant" and traits.get("chin") == "prominent":
        scores["soft_texture"] += MEDIUM
        scores["layers"] += MEDIUM
        scores["clean_lines"] -= WEAK

        if traits.get("face_length") != "long":
            scores["volume_top"] += MEDIUM
            scores["updo"] += WEAK
        else:
            scores["volume_sides"] += MEDIUM
            scores["curtain_fringe"] += WEAK
    
    if traits.get("facial_thirds") == "middle_dominant" and traits.get("face_length") == "long":
        scores["fringe"] += WEAK
        scores["curtain_fringe"] += WEAK
        scores["volume_sides"] += WEAK
        scores["volume_top"] -= WEAK

    if traits.get("symmetry") == "low" and traits.get("jaw") == "wide":
        scores["layers"] += MEDIUM
        scores["soft_texture"] += WEAK
        scores["clean_lines"] -= WEAK
        scores["updo"] -= WEAK
    
    return scores

def apply_symmetry_modulation_female(scores, traits):
    TEXTURE_KEYS = ["soft_texture", "volume_sides", "fringe", "curtain_fringe", "layers"]
    CLEAN_KEYS   = ["clean_lines", "updo"]

    if traits.get("symmetry") == "low":
        for k in TEXTURE_KEYS:
            scores[k] = scores[k] * 1.3 + 1
        for k in CLEAN_KEYS:
            scores[k] = scores[k] * 0.7

    elif traits.get("symmetry") == "high":
        for k in CLEAN_KEYS:
            scores[k] = scores[k] * 1.2
        scores["clean_lines"] += WEAK
        for k in TEXTURE_KEYS:
            scores[k] = scores[k] * 0.9

    return scores