def apply_rules(traits):
    scores = {
        "volume_top": 0,
        "volume_sides": 0,
        "short_sides": 0,
        "longer_hair": 0,
        "fringe": 0,
        "symmetry": 0,
        "soft_texture": 0,
    }

    if traits["face_length"] == "long":
        scores["volume_sides"] += 2
        scores["volume_top"] -= 1
        scores["fringe"] += 2

    elif traits["face_length"] == "short":
        scores["volume_top"] += 2
        scores["volume_sides"] -= 1

    if traits["jaw"] == "wide":
        scores["soft_texture"] += 2
        scores["short_sides"] -= 1
    else:
        scores["volume_sides"] += 1

    if traits["eyes"] == "wide":
        scores["volume_top"] += 1
    elif traits["eyes"] == "close":
        scores["volume_sides"] += 1

    if traits["jaw_projection"] == "strong":
        scores["soft_texture"] += 2
        scores["fringe"] += 1

    elif traits["jaw_projection"] == "weak":
        scores["volume_top"] += 2

    scores = apply_symmetry_modulation(scores, traits)

    return scores

def apply_symmetry_modulation(scores, traits):
    TEXTURE_KEYS = ["soft_texture", "volume_sides", "fringe"]
    CLEAN_KEYS = ["symmetry", "short_sides"]

    if traits["symmetry"] == "low":
        for k in TEXTURE_KEYS:
            scores[k] *= 1.3

        for k in CLEAN_KEYS:
            scores[k] *= 0.7

    elif traits["symmetry"] == "high":
        for k in CLEAN_KEYS:
            scores[k] *= 1.2

    return scores