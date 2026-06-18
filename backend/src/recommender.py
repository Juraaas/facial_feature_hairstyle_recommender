import json

STYLE_DESCRIPTIONS = {
    "volume_top": "height on top",
    "volume_sides": "fuller sides",
    "short_sides": "tapered sides",
    "longer_hair": "longer length",
    "fringe": "front fringe",
    "clean_lines": "clean shape",
    "soft_texture": "soft texture",
    "textured_top": "textured top",
    "layers": "layered cut",
    "updo": "lifted style",
    "curtain_fringe": "curtain fringe",
}

NEGATIVE_EXPLANATIONS = {
    "fringe": "fringe may not suit your eye proportions or add unwanted weight to the forehead",
    "volume_sides": "side volume may widen your face shape",
    "volume_top": "extra height may emphasise the length of your face",
    "short_sides": "tapered sides may draw attention to a wider jaw",
    "clean_lines": "sharp geometric cuts can highlight facial asymmetry",
    "soft_texture": "heavy texture may work against your face's natural structure",
    "longer_hair": "added length risks elongating your face further",
    "textured_top": "textured volume on top may unbalance a prominent chin",
    "layers": "heavy layering may not suit your face proportions",
    "updo": "lifted styles may elongate your face further",
    "curtain_fringe": "a centre parting may emphasise close-set eyes",
}

TRAIT_EXPLANATIONS = {
    "face_length": {
        "long": "long face shape — styles with side volume and fringe work in your favour",
        "short": "shorter face shape — height on top helps elongate proportions",
        "balanced": "face length is well balanced",
    },
    "forehead": {
        "high": "high forehead — fringe optically lowers the hairline",
        "low": "low forehead — keep the forehead open, avoid heavy fringe",
    },
    "jaw": {
        "wide": "wide jaw — soft layered styles reduce visual sharpness",
        "narrow": "narrow jaw — side volume improves overall balance",
    },
    "eyes": {
        "wide": "wide-set eyes — vertical emphasis and clean partings suit you well",
        "close": "close-set eyes — side width creates better visual spacing",
    },
    "lips": {
        "wide": "wider lips — soft texture on top balances the lower face",
        "narrow": "narrower lips — clean structured styles complement well",
    },
    "chin": {
        "prominent": "prominent chin — textured top and length balance the profile",
        "recessed": "recessed chin — volume on top draws focus upward",
    },
    "symmetry": {
        "high": "high facial symmetry — clean geometric styles suit you well",
        "low": "noticeable asymmetry — textured styles redistribute visual balance",
    },
    "eye_openness": {
        "narrow": "narrower eyes — avoid heavy fringe to keep eyes visible",
    },
    "thirds_vertical": {
        "top_heavy": "forehead dominates — fringe and side volume balance the face",
        "bottom_heavy": "lower face dominates — height on top corrects the balance",
    },
    "hair_type": {
    "curly": "your natural texture can work well with styles that embrace movement and volume",
    "coily": "your natural texture can work well with rounded shape, controlled volume, and defined texture",
    "straight": "clean and structured styles tend to complement your natural texture",
    "wavy": "soft textured styles can enhance your natural movement",
    },
    "hairline": {
        "receding": "your hairline shape may work better with styles that avoid heavy forward fringe",
        "uneven": "your hairline shape may benefit from softer texture and less rigid outlines",
    },
}

TRAIT_SCORE_MAP = {
    "face_length": ["volume_sides", "fringe", "volume_top", "longer_hair"],
    "forehead": ["fringe", "volume_top", "curtain_fringe"],
    "jaw": ["soft_texture", "short_sides", "volume_sides", "clean_lines"],
    "jaw_height": ["volume_top", "longer_hair", "fringe", "short_sides"],
    "facial_thirds": ["volume_top", "fringe", "volume_sides", "longer_hair"],
    "eyes": ["volume_sides", "fringe", "clean_lines", "curtain_fringe"],
    "eye_openness": ["fringe", "volume_top", "curtain_fringe"],
    "lips": ["soft_texture", "clean_lines", "volume_top"],
    "chin": ["textured_top", "longer_hair", "volume_top", "short_sides"],
    "symmetry": ["soft_texture", "clean_lines", "textured_top"],
    "thirds_vertical": ["fringe", "volume_top", "volume_sides"],
    "hair_type": ["soft_texture", "textured_top", "clean_lines", "longer_hair"],
    "hairline": ["fringe", "short_sides", "textured_top"],
}

MISSING_SENSITIVE_FEATURES = {
    "volume_sides",
    "fringe",
    "curtain_fringe",
    "layers",
    "short_sides",
    "longer_hair",
}

def load_hairstyles(path="data/hairstyles.json"):
    with open(path, "r") as f:
        return json.load(f)["styles"]
    

def score_hairstyle(user_scores, style):
    score = 0.0
    total_importance = 0.0
    matched_importance = 0.0

    for key, user_value in user_scores.items():
        style_value = style["attributes"].get(key, 0)
        
        importance = abs(user_value)
        total_importance += importance

        contribution = user_value * style_value
        score += contribution
        
        if contribution > 0:
            matched_importance += importance * style_value

        if key in MISSING_SENSITIVE_FEATURES and user_value >= 3 and style_value < 0.2:
            score -= user_value * 0.35

        if user_value <= -3 and style_value > 0.6:
            score -= abs(user_value) * style_value * 0.35
    
    if total_importance == 0:
        return 0.0
    
    base_score = score / total_importance
    match_concentration = matched_importance / total_importance

    return base_score * (0.75 + 0.25 * match_concentration)

def explain_match(user_scores, style, total_score):  
    positive = []
    negative = []
    missing = []

    pos_total = 0.0
    neg_total = 0.0
    missing_total = 0.0

    attributes = style.get("attributes", {})

    for key, user_value in user_scores.items():
        style_value = attributes.get(key, 0)
        contribution = user_value * style_value

        if contribution > 0:
            positive.append({
                "feature": key,
                "raw": contribution,
                "desc": STYLE_DESCRIPTIONS.get(key,key),
            })
            pos_total += contribution
        
        elif contribution < 0:
            negative.append({
                "feature": key,
                "raw": contribution,
                "desc": STYLE_DESCRIPTIONS.get(key, key),
                "reason": NEGATIVE_EXPLANATIONS.get(key, "may not suit your face profile"),
            })
            neg_total += abs(contribution)
        
        if key in MISSING_SENSITIVE_FEATURES and user_value >= 3 and style_value < 0.2:
            missing_strength = user_value * (1 - style_value)
            missing.append({
                "feature": key,
                "raw": missing_strength,
                "desc": STYLE_DESCRIPTIONS.get(key, key),
                "reason": f"this style lacks {STYLE_DESCRIPTIONS.get(key, key)}, which your analysis strongly favours",
            })
            missing_total += missing_strength

    for c in positive:
        c["percent"] = c["raw"] / pos_total if pos_total > 0 else 0.0
    
    for c in negative:
        c["percent"] = abs(c["raw"]) / neg_total if neg_total > 0 else 0.0

    for c in missing:
        c["percent"] = c["raw"] / missing_total if missing_total > 0 else 0.0

    positive.sort(key=lambda x: x["percent"], reverse=True)
    negative.sort(key=lambda x: x["percent"], reverse=True)
    missing.sort(key=lambda x: x["percent"], reverse=True)

    return positive, negative, missing


INFLUENCE_THRESHOLD = 2.0

def explain_from_traits(traits, scores=None):
    explanations = []
    skip_values  = {"normal", "balanced", "slight_imbalance", None}
    for key, value in traits.items():
        if value in skip_values:
            continue
        if key not in TRAIT_EXPLANATIONS:
            continue
        explanation = TRAIT_EXPLANATIONS[key].get(value)
        if not explanation:
            continue

        if scores is not None:
            related_dims = TRAIT_SCORE_MAP.get(key, [])
            if related_dims:
                influence = sum(abs(scores.get(dim, 0)) for dim in related_dims)
                if influence < INFLUENCE_THRESHOLD:
                    continue

        explanations.append(explanation)

    return explanations

def generate_recommendations(user_scores, traits, top_k=3, hairstyles_path="data/hairstyles.json"):
    styles = load_hairstyles(hairstyles_path)
    results = []

    for style in styles:
        score = score_hairstyle(user_scores, style)
        positive, negative, missing = explain_match(user_scores, style, score)

        results.append({
            "name": style["name"],
            "score": score,
            "category": style.get("category", ""),
            "tags": style.get("tags", []),
            "contributions": positive,
            "negatives": negative,
            "missing": missing,
            "image": style.get("image", None),
            "description": style.get("description", ""),
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "top_styles": results[:top_k],
        "all_styles": results,
        "face_analysis": explain_from_traits(traits, scores=user_scores)
    }