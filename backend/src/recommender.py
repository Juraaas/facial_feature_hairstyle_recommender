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
    
def compute_traits_influences(traits, gender):
    from src.rules import apply_rules
    base_scores = apply_rules(traits, gender=gender)
    influences = {}

    for key in traits:
        if traits[key] in {None, "normal", "balanced", "slight_imbalance"}:
            continue
        traits_without = {**traits, key: "normal"}
        scores_without = apply_rules(traits_without, gender=gender)
        delta = {
            dim: round(base_scores.get(dim, 0) - scores_without.get(dim, 0), 3)
            for dim in base_scores
            if abs(base_scores.get(dim, 0) - scores_without.get(dim, 0)) > 0.01
        }
        total_impact = sum(abs(v) for v in delta.values())
        if total_impact > 0.5:
            influences[key] = {
                "value": traits[key],
                "total_impact": round(total_impact, 3),
                "delta": delta,
            }
    return dict(sorted(
        influences.items(),
        key=lambda x: x[1]["total_impact"],
        reverse=True,
    ))

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

def _build_face_analysis(influences, traits):
    explanations = []
    skip_values  = { None, "normal", "balanced", "slight_imbalance"}
    seen_dims = set()

    priority_order = ["hairline", "hair_type"] + [
        k for k in influences.keys() if k not in ("hairline", "hair_type")
    ]

    for key in priority_order:
        if key not in influences:
            continue
        info = influences[key]      
        value = info["value"]
        if value in skip_values:
            continue

        exp = TRAIT_EXPLANATIONS.get(key, {}).get(value)
        if not exp:
            continue
        delta = info["delta"]
        top_dims = sorted(delta.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
        filtered_dims = [(d, c) for d, c in top_dims if d not in seen_dims]

        if not filtered_dims and top_dims:
            continue

        dim_hints = []
        for dim, change in top_dims:
            desc = STYLE_DESCRIPTIONS.get(dim, dim)
            dim_hints.append(f"favours {desc}" if change > 0 else f"works against {desc}")
            seen_dims.add(dim)
        if dim_hints:
            exp = f"{exp} ({', '.join(dim_hints)})"
        explanations.append(exp)
        
        if len(explanations) >= 5:
            break
        
    return explanations

def generate_recommendations(user_scores, traits, gender="Man", top_k=3, hairstyles_path="data/hairstyles.json"):
    styles = load_hairstyles(hairstyles_path)
    influences = compute_traits_influences(traits, gender)
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
        "face_analysis": _build_face_analysis(influences, traits),
        "trait_influences": influences,
    }