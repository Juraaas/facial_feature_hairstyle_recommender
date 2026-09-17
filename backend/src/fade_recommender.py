from typing import Optional

FADE_LEVELS   = ["low", "mid", "high", "skin"]
FADE_LABELS   = {
    "low":  {"pl": "Low Fade",  "en": "Low Fade"},
    "mid":  {"pl": "Mid Fade",  "en": "Mid Fade"},
    "high": {"pl": "High Fade", "en": "High Fade"},
    "skin": {"pl": "Skin Fade", "en": "Skin Fade"},
}

FADE_DESCRIPTIONS = {
    "low": {
        "pl": (
            "Low fade zaczyna się nisko przy karku i delikatnie przechodzi w dłuższe "
            "włosy. Subtelny efekt, który zachowuje naturalną linię włosów."
        ),
        "en": (
            "A low fade starts close to the neckline and gradually blends into longer "
            "hair. Subtle and natural-looking."
        ),
    },
    "mid": {
        "pl": (
            "Mid fade zaczyna się mniej więcej na wysokości ucha i daje czysty, "
            "klasyczny wygląd bez ekstremalnego kontrastu."
        ),
        "en": (
            "A mid fade starts around ear level, giving a clean classic look "
            "without extreme contrast."
        ),
    },
    "high": {
        "pl": (
            "High fade zaczyna się powyżej uszu i tworzy wyraźny kontrast między "
            "krótkimi bokami a dłuższą górą. Mocny, nowoczesny styl."
        ),
        "en": (
            "A high fade starts above the ears, creating strong contrast between "
            "short sides and a longer top. Bold and modern."
        ),
    },
    "skin": {
        "pl": (
            "Skin fade schodzi do gołej skóry przy karku i uszach — maksymalny "
            "kontrast i najbardziej wyrazisty efekt. Wymaga regularnej pielęgnacji."
        ),
        "en": (
            "A skin fade tapers down to bare skin at the neckline and ears — "
            "maximum contrast and the boldest finish. Requires regular upkeep."
        ),
    },
}

REASONS = {
    "nape_high": {
        "pl": "Wysoko osadzona linia włosów pozwala na wyższy fade bez utraty objętości.",
        "en": "A high nape hairline allows for a higher fade without losing volume.",
    },
    "nape_low": {
        "pl": "Nisko osadzona linia włosów lepiej komponuje się z delikatnym low fade.",
        "en": "A lower nape hairline works best with a subtle low fade.",
    },
    "neck_broad": {
        "pl": "Szersze proporcje szyi zyskują na wyraźnym, wyższym fade.",
        "en": "Broader neck proportions benefit from a higher, more defined fade.",
    },
    "neck_narrow": {
        "pl": "Wąska szyja dobrze komponuje się z niższym, miękkim przejściem.",
        "en": "A narrower neck suits a softer, lower transition.",
    },
    "ear_protruding": {
        "pl": "Wyraźniejszy fade przy uszach optycznie zmniejsza ich widoczność.",
        "en": "A higher fade around the ears makes them less prominent.",
    },
    "jaw_recessed": {
        "pl": "Cofnięta szczęka lepiej wygląda z niższym fade.",
        "en": "A recessed jaw looks better with a lower fade.",
    },
    "jaw_prominent": {
        "pl": "Mocna szczęka dobrze znosi wyższy fade, który podkreśla jej strukturę.",
        "en": "A prominent jaw carries a higher fade well, emphasising its structure.",
    },
}


def recommend_fade(
    side_features: dict,
    face_traits: Optional[dict] = None,
    gender: str = "Man",
) -> Optional[dict]:
    
    if gender != "Man":
        return None

    if not side_features:
        return None

    nape = side_features.get("nape_height", 0.22)
    neck = side_features.get("neck_width", 0.30)
    jaw = side_features.get("jaw_angle", 18.0)
    ear = side_features.get("ear_protrusion", 0.07)

    reasons_pl = []
    reasons_en = []
    score = 0 

    if nape < 0.18:
        score += 2
        reasons_pl.append(REASONS["nape_high"]["pl"])
        reasons_en.append(REASONS["nape_high"]["en"])
    elif nape > 0.28:
        score -= 1
        reasons_pl.append(REASONS["nape_low"]["pl"])
        reasons_en.append(REASONS["nape_low"]["en"])

    if neck > 0.38:
        score += 1
        reasons_pl.append(REASONS["neck_broad"]["pl"])
        reasons_en.append(REASONS["neck_broad"]["en"])
    elif neck < 0.25:
        score -= 1
        reasons_pl.append(REASONS["neck_narrow"]["pl"])
        reasons_en.append(REASONS["neck_narrow"]["en"])

    if ear > 0.12:
        score += 1
        reasons_pl.append(REASONS["ear_protruding"]["pl"])
        reasons_en.append(REASONS["ear_protruding"]["en"])

    if jaw < 12:
        score -= 1
        reasons_pl.append(REASONS["jaw_recessed"]["pl"])
        reasons_en.append(REASONS["jaw_recessed"]["en"])
    elif jaw > 28:
        score += 1
        reasons_pl.append(REASONS["jaw_prominent"]["pl"])
        reasons_en.append(REASONS["jaw_prominent"]["en"])

    if score <= -1:
        level = "low"
    elif score <= 1:
        level = "mid"
    elif score <= 3:
        level = "high"
    else:
        level = "skin"

    factors = sum([
        nape != 0.22,
        neck != 0.30,
        jaw  != 18.0,
        ear  != 0.07,
    ])
    confidence = min(1.0, 0.5 + factors * 0.125)

    return {
        "level": level,
        "label_pl": FADE_LABELS[level]["pl"],
        "label_en": FADE_LABELS[level]["en"],
        "description_pl": FADE_DESCRIPTIONS[level]["pl"],
        "description_en": FADE_DESCRIPTIONS[level]["en"],
        "reasons_pl": reasons_pl,
        "reasons_en": reasons_en,
        "confidence": round(confidence, 2),
    }


HAIRSTYLE_FADE_DEFAULTS = {
    "French Crop": ("yes", "high"),
    "Messy Crop": ("yes", "mid"),
    "Buzz Cut": ("no", None),
    "Quiff": ("yes", "high"),
    "Classic Undercut": ("yes", "high"),
    "Slick Back": ("yes", "mid"),
    "Textured Fringe": ("yes", "mid"),
    "Comb Over": ("yes", "mid"),
    "Bro Flow": ("no", None),
    "Wolf Cut": ("yes",   "low"),
    "Modern Mullet": ("yes",   "low"),
    "Curtain Bangs M": ("yes",   "low"),
    "French Bob":           ("no",    None),
    "Beach Waves":          ("no",    None),
    "Layered Medium":       ("no",    None),
    "Classic Updo":         ("no",    None),
    "Soft Bun":             ("no",    None),
    "Curtain Fringe Medium":("no",    None),
    "Pixie Cut":            ("no",    None),
    "Lob":                  ("no",    None),
    "Braided Crown":        ("no",    None),
    "Wolf Cut F":           ("no",    None),
    "Soft Shag":            ("no",    None),
    "Butterfly Cut":        ("no",    None),
}


def get_style_fade_info(style_name: str, recommendation: Optional[dict]) -> dict:
    defaults = HAIRSTYLE_FADE_DEFAULTS.get(style_name, ("no", None))
    has_fade, default_level = defaults

    if has_fade == "no":
        return {"has_fade": False}

    level = recommendation["level"] if recommendation else default_level

    return {
        "has_fade": True,
        "fade_level": level,
        "from_profile": recommendation is not None,
    }