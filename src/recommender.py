import json

STYLE_DESCRIPTIONS = {
    "volume_top":     "height on top",
    "volume_sides":   "fuller sides",
    "short_sides":    "faded sides",
    "longer_hair":    "longer length",
    "fringe":         "front fringe",
    "clean_lines":    "clean shape",
    "soft_texture":   "soft texture",
    "textured_top":   "textured top",
    "layers":         "layered cut",
    "updo":           "lifted style",
    "curtain_fringe": "curtain fringe",
}

NEGATIVE_EXPLANATIONS = {
    "fringe":        "fringe may not suit your eye proportions",
    "volume_sides":  "side volume may widen your face shape",
    "volume_top":    "added height may emphasise face length",
    "short_sides":   "fade may accentuate your jaw width",
    "clean_lines":   "structured styles may highlight asymmetry",
    "soft_texture":  "heavy texture may not complement your proportions",
    "longer_hair":   "length may elongate your face further",
    "textured_top":  "textured top may not balance your chin prominence",
    "layers":         "heavy layers may not suit your face structure",
    "updo":           "updo may elongate your face further",
    "curtain_fringe": "curtain fringe may widen close-set eyes too much",
}

TRAIT_EXPLANATIONS = {
    "face_length": {
        "long":     "your face is long — reducing vertical length works in your favour",
        "short":    "your face is shorter — adding height on top helps balance proportions",
        "balanced": "your face length proportions are well balanced",
    },
    "jaw": {
        "wide":   "wide jaw — softer, layered styles reduce visual harshness",
        "narrow": "narrow jaw — adding volume on the sides improves balance",
        "normal": "your jaw width is well proportioned",
    },
    "jaw_height": {
        "high":   "jaw sits high — longer styles help elongate the lower face",
        "low":    "jaw sits low — volume on top balances the lower face weight",
        "normal": "jaw height is proportionate",
    },
    "eyes": {
        "wide":   "wide-set eyes — vertical emphasis and clean partings work well",
        "close":  "close-set eyes — width on the sides creates better visual spacing",
        "normal": "eye spacing is balanced",
    },
    "eye_openness": {
        "open":   "open eyes — fringe can work well without hiding your expression",
        "narrow": "narrower eyes — avoid heavy fringe, keep the eyes visible",
        "normal": "eye openness is average",
    },
    "lips": {
        "wide":   "wider lips — soft texture on top balances the lower face",
        "narrow": "narrower lips — clean structured styles complement well",
        "normal": "lip width is proportionate",
    },
    "nose": {
        "upper-dominant": "upper face dominant — fringe can optically shorten the upper area",
        "lower-dominant": "lower face dominant — volume on top restores balance",
        "balanced":       "nose position is well centred",
    },
    "lower_face": {
        "long":   "longer lower face — soft texture around the jaw reduces length",
        "short":  "shorter lower face — clean sides and structure complement well",
        "normal": "lower face proportions are balanced",
    },
    "chin": {
        "prominent": "prominent chin — textured top and longer length balance the profile",
        "recessed":  "recessed chin — volume on top and short sides draw focus upward",
        "normal":    "chin prominence is average",
    },
    "symmetry": {
        "high":   "high facial symmetry — clean, geometric styles suit you well",
        "medium": "slight asymmetry — natural styles look most harmonious",
        "low":    "noticeable asymmetry — textured styles redistribute visual balance",
    },
    "facial_thirds": {
        "lower_dominant":  "lower face is dominant — volume on top restores balance",
        "middle_dominant": "mid face is dominant — height on top draws the eye upward",
        "balanced":        "facial thirds are well proportioned",
    },
    "forehead": {
        "high":   "high forehead — fringe can optically lower the hairline",
        "low":    "low forehead — avoid heavy fringe, keep the forehead open",
        "normal": "forehead height is proportionate",
    },
    "thirds_balance": {
        "balanced":         "facial thirds are well balanced",
        "slight_imbalance": "slight imbalance in facial thirds",
        "imbalanced":       "noticeable imbalance in facial thirds — softer styles help",
    },
}

def load_hairstyles(path="data/hairstyles.json"):
    with open(path, "r") as f:
        return json.load(f)["styles"]
    

def score_hairstyle(user_scores, style):
    score = 0.0
    total_weight = 0.0
    matched_weight = 0.0

    for key, style_value in style["attributes"].items():
        user_value = user_scores.get(key, 0)
        contribution = user_value * style_value
        score += contribution
        total_weight += abs(style_value)
        
        if contribution > 0:
            matched_weight += style_value
    
    if total_weight == 0:
        return 0.0
    
    base_score = score / total_weight
    match_concentration = matched_weight / total_weight

    return base_score * (0.7 + 0.3 * match_concentration)

def explain_match(user_scores, style, total_score):
    if total_score <= 0:
        return [], []
    
    positive = []
    negative = []
    pos_total = 0.0
    neg_total = 0.0
    for key, style_value in style["attributes"].items():
        user_value = user_scores.get(key, 0)
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
                "raw":     contribution,
                "desc":    STYLE_DESCRIPTIONS.get(key, key),
                "reason":  NEGATIVE_EXPLANATIONS.get(key, "may not suit your face profile"),
            })
            neg_total += abs(contribution)

    for c in positive:
        c["percent"] = c["raw"] / pos_total if pos_total > 0 else 0.0
    
    for c in negative:
        c["percent"] = abs(c["raw"]) / neg_total if neg_total > 0 else 0.0

    positive.sort(key=lambda x: x["percent"], reverse=True)
    negative.sort(key=lambda x: x["percent"], reverse=True)

    return positive, negative

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
        positive, negative = explain_match(user_scores, style, score)

        results.append({
            "name": style["name"],
            "score": score,
            "category": style.get("category", ""),
            "tags": style.get("tags", []),
            "contributions": positive,
            "negatives": negative,
            "image": style.get("image", None),
            "description": style.get("description", ""),
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "top_styles": results[:top_k],
        "all_styles": results,
        "face_analysis": explain_from_traits(traits)
    }