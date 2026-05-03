STYLE_DESCRIPTIONS = {
    "volume_top": "hairstyles with volume on top",
    "volume_sides": "hairstyles with volume on sides",
    "short_sides": "short sides (fade styles)",
    "longer_hair": "longer hairstyles",
    "fringe": "styles with fringe / bangs",
    "symmetry": "clean, symmetrical styles",
    "soft_texture": "soft, layered hairstyles",
}

def generate_recommendations(scores, top_k=3):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for key, score in ranked[:top_k]:
        results.append({
            "style": key,
            "description": STYLE_DESCRIPTIONS[key],
            "score": score,
        })
    return results