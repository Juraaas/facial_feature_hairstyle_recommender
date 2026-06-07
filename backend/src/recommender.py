import json

STYLE_DESCRIPTIONS = {
    "volume_top": "height on top",
    "volume_sides": "fuller sides",
    "short_sides": "faded sides",
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
    "fringe": "fringe may not suit your eye proportions",
    "volume_sides": "side volume may widen your face shape",
    "volume_top": "added height may emphasise face length",
    "short_sides": "fade may accentuate your jaw width",
    "clean_lines": "structured styles may highlight asymmetry",
    "soft_texture": "heavy texture may not complement your proportions",
    "longer_hair": "length may elongate your face further",
    "textured_top": "textured top may not balance your chin prominence",
    "layers": "heavy layers may not suit your face structure",
    "updo": "updo may elongate your face further",
    "curtain_fringe": "curtain fringe may widen close-set eyes too much",
}

TRAIT_EXPLANATIONS = {
    "face_length": {
        "long": "long face shape — styles with side volume and fringe work in your favour",
        "short": "shorter face shape — height on top helps elongate proportions",
        "balanced": "face length is well balanced",
    },
    "facial_thirds": {
        "lower_dominant": "lower face carries more visual weight — volume on top restores balance",
        "middle_dominant": "mid face is prominent — fringe and height draw the eye upward",
        "balanced": "facial thirds are evenly proportioned",
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
    if total_score <= 0:
        return [], [], []
    
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

def explain_from_traits(traits):
    explanations = []
    skip_values = {"normal", "balanced", "slight_imbalance"}

    for key, value in traits.items():
        if value in skip_values:
            continue
        if key in TRAIT_EXPLANATIONS:
            explanation = TRAIT_EXPLANATIONS[key].get(value)
            if explanation:
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
        "face_analysis": explain_from_traits(traits)
    }