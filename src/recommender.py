import json

STYLE_DESCRIPTIONS = {
    "volume_top": "volume on top",
    "volume_sides": "volume on sides",
    "short_sides": "short sides (fade)",
    "longer_hair": "longer hair",
    "fringe": "fringe / bangs",
    "symmetry": "clean, symmetrical styles",
    "soft_texture": "soft, layered texture",
}

TRAIT_EXPLANATIONS = {
    "face_length": {
        "long": "your face is long → reducing vertical length is beneficial",
        "short": "your face is shorter → adding height helps balance",
        "balanced": "your face proportions are balanced"
    },
    "jaw": {
        "wide": "wide jaw → softer styles reduce harshness",
        "narrow": "narrow jaw → adding volume improves balance"
    },
    "eyes": {
        "wide": "wide-set eyes → vertical emphasis works well",
        "close": "close-set eyes → adding width improves spacing",
        "normal": "eye spacing is balanced"
    },
    "nose": {
        "upper-dominant": "upper face dominant → balance upper/lower proportions",
        "lower-dominant": "lower face dominant → add structure to upper area"
    },
    "symmetry": {
        "high": "high symmetry → clean styles work best",
        "medium": "slight asymmetry → natural styles look better",
        "low": "asymmetry → textured styles improve harmony"
    },
    "jaw_projection": {
        "strong": "strong jaw → softer textures balance it",
        "weak": "weaker jaw → structured styles add definition",
        "balanced": "jaw projection is balanced"
    }
}

def load_hairstyles(path="data/hairstyles.json"):
    with open(path, "r") as f:
        return json.load(f)["styles"]
    

def score_hairstyle(user_scores, style):
    score = 0.0

    for key, style_value in style["attributes"].items():
        user_value = user_scores.get(key, 0)
        score += user_value * style_value

    return score

def explain_match(user_scores, style, total_score):
    contributions = []

    if total_score == 0:
        return contributions

    for key, style_value in style["attributes"].items():
        user_value = user_scores.get(key, 0)
        contribution = user_value * style_value

        if contribution > 0:
            percent = contribution / total_score

            contributions.append({
                "feature": key,
                "percent": percent,
                "raw": contribution,
                "desc": STYLE_DESCRIPTIONS.get(key,key),
            })

    contributions.sort(key=lambda x: x["percent"], reverse=True)
    return contributions

def explain_from_traits(traits):
    explanations = []

    for key, value in traits.items():
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
            "contributions": contributions
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "top_styles": results[:top_k],
        "face_analysis": explain_from_traits(traits)
    }