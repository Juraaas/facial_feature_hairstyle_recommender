def _styles(recs, key, lang):
    items = recs.get(key, [])
    return items.get(lang, []) if isinstance(items, dict) else items


def _items(lst):
    return [
        {"feature": i.get("feature"), "raw": round(float(i.get("raw", 0)), 3)}
        for i in lst
    ]


def build_debug_payload(recs, scores, lang, compact=False):
    styles = _styles(recs, "all_styles", lang)

    if compact:
        ranking = [
            {
                "rank": i + 1,
                "name": s.get("name"),
                "score": round(float(s.get("score", 0)), 4),
                "display_score": s.get("display_score"),
                "positive": _items(s.get("contributions", [])),
                "negative": _items(s.get("negatives", [])),
            }
            for i, s in enumerate(styles)
        ]
        return {"raw_scores": scores, "compact_ranking": ranking}

    top = _styles(recs, "top_styles", lang)
    return {
        "raw_scores": scores,
        "style_ranking": [
            {
                "rank": i + 1,
                "name": s.get("name"),
                "score": s.get("score"),
                "image": s.get("image"),
                "category": s.get("category"),
                "tags": s.get("tags", []),
                "description": s.get("description", ""),
                "contributions": s.get("contributions", []),
                "negatives": s.get("negatives", []),
                "missing": s.get("missing", []),
            }
            for i, s in enumerate(styles)
        ],
        "top_styles": [
            {
                "rank": i + 1,
                "name": s.get("name"),
                "score": s.get("score"),
                "contributions": s.get("contributions", []),
                "negatives": s.get("negatives", []),
            }
            for i, s in enumerate(top)
        ],
    }