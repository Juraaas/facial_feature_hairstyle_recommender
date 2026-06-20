STRONG = 3
MEDIUM = 2
WEAK = 1

FEMALE_DIMS = {"layers", "updo", "curtain_fringe"}

def _base_scores(gender):
    scores = {
        "volume_top":   0,
        "volume_sides": 0,
        "longer_hair":  0,
        "fringe":       0,
        "clean_lines":  0,
        "textured_top": 0,
        "soft_texture": 0,
    }
    if gender == "Woman":
        scores.update({"layers": 0, "updo": 0, "curtain_fringe": 0})
    else:
        scores["short_sides"] = 0
    return scores


def apply_rules(traits, gender="Man"):
    scores = _base_scores(gender)
    is_female = gender == "Woman"

    if traits.get("face_length") == "long":
        scores["volume_sides"] += STRONG
        scores["fringe"] += STRONG
        scores["volume_top"] -= MEDIUM
        scores["longer_hair"] -= MEDIUM if not is_female else WEAK
        scores["clean_lines"] -= WEAK
        if is_female:
            scores["curtain_fringe"] += MEDIUM
            scores["layers"] += MEDIUM
            scores["updo"] -= MEDIUM
    
    elif traits.get("face_length") == "short":
        scores["volume_top"] += STRONG
        scores["longer_hair"] += MEDIUM if not is_female else STRONG
        scores["fringe"] -= MEDIUM
        scores["volume_sides"] -= WEAK
        if is_female:
            scores["updo"] += MEDIUM

    if traits.get("jaw") == "wide":
        scores["soft_texture"] += STRONG
        scores["longer_hair"] += WEAK if not is_female else MEDIUM
        scores["clean_lines"] -= MEDIUM
        if is_female:
            scores["layers"] += MEDIUM
            scores["curtain_fringe"] += MEDIUM
            scores["volume_sides"] -= MEDIUM
        else:
            scores["volume_top"] += MEDIUM
            scores["short_sides"] -= STRONG

    elif traits.get("jaw") == "narrow":
        scores["volume_sides"] += MEDIUM if not is_female else STRONG
        scores["clean_lines"] += WEAK
        scores["soft_texture"] -= WEAK
        if is_female:
            scores["layers"] += MEDIUM
        else:
            scores["short_sides"] += STRONG

    if traits.get("jaw_height") == "high":
        scores["longer_hair"] += MEDIUM
        scores["volume_top"] -= MEDIUM
        if is_female:
            scores["layers"] += WEAK
        else:
            scores["soft_texture"] += WEAK
            scores["short_sides"] -= WEAK

    elif traits.get("jaw_height") == "low":
        scores["volume_top"] += MEDIUM
        scores["fringe"] += WEAK
        scores["longer_hair"] -= WEAK

    if traits.get("eyes") == "wide":
        scores["clean_lines"] += MEDIUM
        scores["volume_top"] += MEDIUM if not is_female else WEAK
        scores["volume_sides"] -= WEAK
        if is_female:
            scores["curtain_fringe"] -= WEAK

    elif traits.get("eyes") == "close":
        scores["volume_sides"] += MEDIUM
        scores["fringe"] -= MEDIUM
        if is_female:
            scores["curtain_fringe"] += STRONG
        else:
            scores["soft_texture"] += WEAK
            scores["clean_lines"] -= WEAK
    
    if traits.get("eye_openness") == "narrow":
        scores["volume_top"] += MEDIUM
        scores["fringe"] -= STRONG
        if is_female:
            scores["curtain_fringe"] -= MEDIUM
        else:
            scores["clean_lines"] += WEAK

    elif traits.get("eye_openness") == "open":
        scores["fringe"] += MEDIUM
        scores["textured_top"] += WEAK
        if is_female:
            scores["curtain_fringe"] += MEDIUM

    if traits.get("lips") == "wide":
        scores["soft_texture"] += MEDIUM
        scores["clean_lines"] -= WEAK
        if is_female:
            scores["layers"] += WEAK
        else:
            scores["volume_top"] += WEAK

    elif traits.get("lips") == "narrow":
        scores["clean_lines"] += MEDIUM
        if is_female:
            scores["updo"] += WEAK
        else:
            scores["short_sides"] += WEAK
            scores["soft_texture"] -= WEAK

    if traits.get("nose") == "upper-dominant":
        scores["fringe"] += MEDIUM if not is_female else STRONG
        scores["volume_top"] -= STRONG
        if is_female:
            scores["curtain_fringe"] += MEDIUM
        else:
            scores["volume_sides"] += WEAK

    elif traits.get("nose") == "lower-dominant":
        scores["volume_top"] += STRONG
        scores["fringe"] -= MEDIUM
        if is_female:
            scores["updo"] += WEAK
        else:
            scores["longer_hair"] += WEAK

    if traits.get("lower_face") == "long":
        scores["longer_hair"] += MEDIUM
        scores["soft_texture"] += MEDIUM
        if is_female:
            scores["layers"] += WEAK
        else:
            scores["short_sides"] -= WEAK

    elif traits.get("lower_face") == "short":
        scores["clean_lines"] += MEDIUM
        scores["longer_hair"] -= WEAK
        if not is_female:
            scores["short_sides"] += MEDIUM

    if traits.get("chin") == "prominent":
        scores["longer_hair"] += MEDIUM if not is_female else STRONG
        scores["soft_texture"] += MEDIUM if not is_female else STRONG
        scores["clean_lines"] -= WEAK if not is_female else MEDIUM
        if is_female:
            scores["layers"] += MEDIUM
        else:
            scores["textured_top"] += STRONG
            scores["short_sides"] -= WEAK

    elif traits.get("chin") == "recessed":
        scores["volume_top"] += STRONG
        scores["clean_lines"] += WEAK
        scores["longer_hair"] -= WEAK
        if is_female:
            scores["updo"] += WEAK
        else:
            scores["short_sides"] += MEDIUM
            scores["textured_top"] -= MEDIUM

    if traits.get("facial_thirds") == "lower_dominant":
        scores["volume_top"] += STRONG
        scores["fringe"] -= MEDIUM
        scores["longer_hair"] -= WEAK
        if is_female:
            scores["updo"] += MEDIUM

    elif traits.get("facial_thirds") == "middle_dominant":
        scores["fringe"] += MEDIUM
        scores["volume_top"] -= MEDIUM
        if is_female:
            scores["curtain_fringe"] += MEDIUM
        else:
            scores["volume_sides"] += WEAK

    if traits.get("forehead") == "high":
        scores["fringe"] += STRONG
        scores["volume_top"] -= MEDIUM
        if is_female:
            scores["curtain_fringe"] += MEDIUM
            scores["updo"] -= MEDIUM

    elif traits.get("forehead") == "low":
        scores["fringe"] -= STRONG
        if is_female:
            scores["updo"] += MEDIUM
            scores["curtain_fringe"] -= MEDIUM
        else:
            scores["volume_top"] += WEAK
            scores["textured_top"] += WEAK

    if traits.get("thirds_balance") == "imbalanced":
        scores["soft_texture"] += MEDIUM
        scores["clean_lines"] -= MEDIUM
        if is_female:
            scores["layers"] += MEDIUM

    if traits.get("hair_type") == "curly":
        scores["soft_texture"] += MEDIUM
        scores["textured_top"] += MEDIUM
        scores["clean_lines"] -= WEAK
        scores["longer_hair"] -= WEAK

    elif traits.get("hair_type") in ("straight", "wavy"):
        scores["clean_lines"] += WEAK
        scores["longer_hair"] += WEAK

    elif traits.get("hair_type") == "coily":
        scores["soft_texture"] += STRONG
        scores["textured_top"] += STRONG
        scores["volume_top"] += MEDIUM
        scores["clean_lines"] -= MEDIUM
        scores["longer_hair"] -= MEDIUM

    if traits.get("hairline") == "receding":
        scores["fringe"] -= STRONG
        scores["textured_top"] += WEAK
        scores["volume_top"] -= WEAK
        if not is_female:
            scores["short_sides"] += WEAK

    elif traits.get("hairline") == "uneven":
        scores["fringe"] -= MEDIUM
        scores["soft_texture"] += WEAK

    scores = _apply_interaction_rules(scores, traits, gender)
    scores = _apply_symmetry_modulation(scores, traits, gender)
    scores = clamp_scores(scores)

    return scores

def _apply_interaction_rules(scores, traits, gender):
    is_female = gender == "Woman"

    if traits.get("face_length") == "long" and traits.get("forehead") == "high":
        scores["fringe"] += MEDIUM if not is_female else WEAK
        scores["volume_sides"] += WEAK
        scores["volume_top"] -= STRONG if not is_female else MEDIUM
        if is_female:
            scores["curtain_fringe"] += WEAK
            scores["updo"] -= WEAK
        else:
            scores["longer_hair"] -= MEDIUM
            scores["clean_lines"] -= WEAK

    if traits.get("face_length") == "long" and traits.get("jaw") == "narrow":
        scores["volume_sides"] += WEAK if not is_female else MEDIUM
        if is_female:
            scores["layers"] += MEDIUM
            scores["curtain_fringe"] += WEAK
            scores["volume_top"] -= WEAK
            scores["updo"] -= WEAK
        else:
            scores["short_sides"] -= WEAK

    if traits.get("face_length") == "short" and traits.get("jaw") == "narrow":
        scores["volume_top"] += MEDIUM
        scores["clean_lines"] += WEAK
        scores["fringe"] -= MEDIUM
        scores["volume_sides"] -= WEAK
        if is_female:
            scores["updo"] += MEDIUM
        else:
            scores["short_sides"] += MEDIUM

    if traits.get("jaw") == "wide" and traits.get("chin") == "prominent":
        scores["soft_texture"] += STRONG
        scores["textured_top"] += MEDIUM
        scores["longer_hair"] += WEAK if not is_female else MEDIUM
        scores["short_sides"] -= MEDIUM
        scores["clean_lines"] -= MEDIUM
        if is_female:
            scores["layers"] += MEDIUM
            scores["curtain_fringe"] += WEAK
            scores["updo"] -= WEAK

    if traits.get("jaw") == "narrow" and traits.get("chin") == "recessed":
        scores["clean_lines"] += MEDIUM if not is_female else WEAK
        scores["volume_top"] += WEAK
        scores["soft_texture"] -= WEAK
        if is_female:
            scores["longer_hair"] -= WEAK
            if traits.get("face_length") != "long" and traits.get("forehead") != "high":
                scores["updo"] += WEAK
        else:
            scores["short_sides"] += MEDIUM

    if traits.get("eyes") == "close" and traits.get("forehead") == "low":
        scores["volume_sides"] += MEDIUM
        scores["volume_top"] += WEAK
        scores["soft_texture"] += WEAK
        scores["fringe"] -= STRONG
        if is_female:
            scores["curtain_fringe"] += MEDIUM

    if traits.get("eyes") == "wide" and traits.get("forehead") == "high":
        scores["fringe"] += MEDIUM if not is_female else WEAK
        scores["volume_top"] -= MEDIUM if not is_female else WEAK

    if traits.get("eye_openness") == "narrow" and traits.get("face_length") == "long":
        scores["volume_sides"] += WEAK if not is_female else MEDIUM
        scores["fringe"] -= MEDIUM
        if is_female:
            scores["layers"] += WEAK
            scores["curtain_fringe"] -= WEAK

    if traits.get("eye_openness") == "narrow" and traits.get("forehead") == "high":
        scores["fringe"] += MEDIUM
        scores["volume_top"] -= WEAK
        if is_female:
            scores["curtain_fringe"] += WEAK
            scores["soft_texture"] += WEAK
        else:
            scores["textured_top"] += WEAK

    if (traits.get("facial_thirds") == "lower_dominant" and traits.get("chin") == "prominent"):
        scores["soft_texture"] += MEDIUM
        scores["clean_lines"] -= WEAK
        if is_female:
            scores["layers"] += MEDIUM
        else:
            scores["textured_top"] += MEDIUM

        if traits.get("face_length") != "long":
            scores["volume_top"] += MEDIUM
            if is_female:
                scores["updo"] += WEAK
        else:
            scores["volume_sides"] += MEDIUM
            if is_female:
                scores["curtain_fringe"] += WEAK

    if (traits.get("facial_thirds") == "middle_dominant" 
        and traits.get("face_length") == "long"):
        scores["volume_top"] -= WEAK
        scores["volume_sides"] += WEAK
        if is_female:
            scores["fringe"] += WEAK
            scores["curtain_fringe"] += WEAK
        else:
            scores["fringe"] += WEAK

    if traits.get("symmetry") == "low" and traits.get("jaw") == "wide":
        scores["soft_texture"] += WEAK
        scores["clean_lines"] -= WEAK
        if is_female:
            scores["layers"] += MEDIUM
            scores["updo"] -= WEAK
        else:
            scores["textured_top"] += MEDIUM
            scores["short_sides"] -= WEAK

    return scores

def _apply_symmetry_modulation(scores, traits, gender):
    is_female = gender == "Woman"
    TEXTURE_KEYS = ["soft_texture", "volume_sides", "fringe", "textured_top"]
    CLEAN_KEYS = ["clean_lines"]

    if is_female:
        TEXTURE_KEYS = ["soft_texture", "volume_sides", "fringe",
                        "curtain_fringe", "layers"]
        CLEAN_KEYS   = ["clean_lines", "updo"]
    else:
        CLEAN_KEYS = ["clean_lines", "short_sides"]

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

def _apply_hard_constraints(scores, traits):
    if traits.get("hairline") == "receding":
        scores["fringe"] = min(scores["fringe"], -2)
    
    return scores

def clamp_scores(scores, min_score=-5, max_score=10):
    if scores is None:
        raise ValueError("clamp_scores received None")
    return {
        k: max(min_score, min(max_score, v))
        for k, v in scores.items()
    }