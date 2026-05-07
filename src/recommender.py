import json

STYLE_DESCRIPTIONS = {
    "volume_top": "volume on top",
    "volume_sides": "volume on sides",
    "short_sides": "short sides (fade)",
    "longer_hair": "longer hair length",
    "fringe": "fringe / bangs",
    "clean_lines": "clean, symmetrical styles",
    "soft_texture": "soft, layered texture",
    "textured_top":  "textured top",
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
}

def load_hairstyles(path="data/hairstyles.json"):
    with open(path, "r") as f:
        return json.load(f)["styles"]
    

def score_hairstyle(user_scores, style):
    score = 0.0
    total_weight = 0.0

    for key, style_value in style["attributes"].items():
        user_value = user_scores.get(key, 0)
        score += user_value * style_value
        total_weight += abs(style_value)

    return score / total_weight if total_weight > 0 else 0.0

def explain_match(user_scores, style, total_score):
    if total_score <= 0:
        return []
    
    contributions = []
    raw_total = 0.0
    for key, style_value in style["attributes"].items():
        user_value = user_scores.get(key, 0)
        contribution = user_value * style_value

        if contribution > 0:
            percent = contribution / total_score

            contributions.append({
                "feature": key,
                "raw": contribution,
                "desc": STYLE_DESCRIPTIONS.get(key,key),
            })
            raw_total += contribution

    for c in contributions:
        c["percent"] = c["raw"] / raw_total if raw_total > 0 else 0.0

    contributions.sort(key=lambda x: x["percent"], reverse=True)
    return contributions

def explain_from_traits(traits):
    explanations = []
    skip_values = {"normal", "balanced"}

    for key, value in traits.items():
        if value in skip_values:
            continue
        if key in TRAIT_EXPLANATIONS:
            explanation = TRAIT_EXPLANATIONS[key].get(value)
            if explanation:
                explanations.append(explanation)

    return explanations

def generate_recommendations(user_scores, traits, top_k=3):
    styles = load_hairstyles()
    results = []

    for style in styles:
        score = score_hairstyle(user_scores, style)
        contributions = explain_match(user_scores, style, score)

        results.append({
            "name": style["name"],
            "score": score,
            "contributions": contributions,
            "image": style.get("image", None),
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "top_styles": results[:top_k],
        "face_analysis": explain_from_traits(traits)
    }