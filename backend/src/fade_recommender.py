FADE_LEVELS = ["low", "mid", "high", "skin"]

FADE_REASONS = {
    "face_wide": {
        "pl": "Szersza twarz dobrze wygląda z wyższym fade, bo wydłuża optycznie sylwetkę.",
        "en": "A wider face looks better with a higher fade. It visually elongates the silhouette.",
    },
    "face_long": {
        "pl": "Dłuższa twarz zyska na niższym fade, nie wydłużaj dodatkowo proporcji.",
        "en": "A longer face benefits from a lower fade, avoid adding further vertical length.",
    },
    "jaw_wide": {
        "pl": "Szeroka szczęka dobrze znosi wyraźny fade, który podkreśla jej strukturę.",
        "en": "A wider jaw carries a defined fade well, emphasising its structure.",
    },
    "jaw_narrow": {
        "pl": "Wąska szczęka lepiej wygląda z niższym fade, nie odsłaniaj zbytnio szyi.",
        "en": "A narrower jaw looks better with a lower fade, avoid exposing too much neck.",
    },
    "cheekbone_dom": {
        "pl": "Dominujące kości policzkowe pasują do niższego, łagodniejszego fade.",
        "en": "Dominant cheekbones suit a lower, softer fade.",
    },
    "forehead_high": {
        "pl": "Wyższe czoło balansuje się z wyraźnym fade, dodaje proporcji pionowych.",
        "en": "A higher forehead is balanced by a more defined fade, adds vertical proportion.",
    },
    "default": {
        "pl": "Klasyczny mid fade pasuje do większości typów twarzy.",
        "en": "A classic mid fade suits most face types.",
    },
}

def recommend_fade_from_front(traits: dict, gender: str = "Man") -> dict | None:
    if gender != "Man":
        return None

    score = 0
    used_keys = []

    face_len  = traits.get("face_length")
    jaw = traits.get("jaw")
    forehead = traits.get("forehead")
    face_type = traits.get("face_shape_type")

    if face_len == "wide":
        score += 1; used_keys.append("face_wide")
    elif face_len == "long":
        score -= 1; used_keys.append("face_long")

    if jaw == "wide":
        score += 1; used_keys.append("jaw_wide")
    elif jaw == "narrow":
        score -= 1; used_keys.append("jaw_narrow")

    if face_type == "triangle":
        score -= 1; used_keys.append("cheekbone_dom")

    if forehead == "high":
        score += 1; used_keys.append("forehead_high")

    if score <= -1: level = "low"
    elif score == 0: level = "mid"
    elif score <= 2: level = "high"
    else: level = "skin"

    reason_key = used_keys[0] if used_keys else "default"
    reason = FADE_REASONS[reason_key]

    return {
        "level": level,
        "reason_pl": reason["pl"],
        "reason_en": reason["en"],
    }