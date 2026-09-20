import cv2
import numpy as np


WARM_COLORS = [
    {"name_pl": "Miodowy blond", "name_en": "Honey blonde",
     "hex": "#C8A96E", "desc_pl": "Ciepły złocisty odcień który podkreśli karnację.",
     "desc_en": "Warm golden tone that enhances your complexion."},
    {"name_pl": "Karmel", "name_en": "Caramel brown",
     "hex": "#8B5A2B", "desc_pl": "Bogaty brąz z ciepłymi refleksami.",
     "desc_en": "Rich brown with warm highlights."},
    {"name_pl": "Miedziany rudy", "name_en": "Copper auburn",
     "hex": "#A0522D", "desc_pl": "Energetyczny odcień rudości który ożywi cerę.",
     "desc_en": "Vibrant auburn that adds warmth to your skin."},
    {"name_pl": "Ciemny czekoladowy", "name_en": "Dark chocolate",
     "hex": "#3B1F0A", "desc_pl": "Głęboki brąz z ciepłymi podtonami.",
     "desc_en": "Deep brown with warm undertones."},
]

COOL_COLORS = [
    {"name_pl": "Platynowy blond", "name_en": "Platinum blonde",
     "hex": "#E8E0D0", "desc_pl": "Chłodny, jasny odcień kontrastujący z karnacją.",
     "desc_en": "Cool, light tone that creates contrast with your complexion."},
    {"name_pl": "Popielaty brąz", "name_en": "Ash brown",
     "hex": "#6B5B4E", "desc_pl": "Stonowany brąz z chłodnymi szarymi refleksami.",
     "desc_en": "Muted brown with cool ashy tones."},
    {"name_pl": "Burgundy", "name_en": "Burgundy",
     "hex": "#6B1A1A", "desc_pl": "Chłodna czerwień z fioletowym podtonem.",
     "desc_en": "Cool red with violet undertone."},
    {"name_pl": "Czarny grafit", "name_en": "Graphite black",
     "hex": "#1A1A1A", "desc_pl": "Głęboka czerń harmonizująca z chłodną karnacją.",
     "desc_en": "Deep black that harmonises with cool skin tones."},
]

NEUTRAL_COLORS = [
    {"name_pl": "Naturalny brąz", "name_en": "Natural brown",
     "hex": "#5C3D2E", "desc_pl": "Klasyczny brąz pasujący do neutralnej karnacji.",
     "desc_en": "Classic brown that suits neutral undertones."},
    {"name_pl": "Ciemny blond", "name_en": "Dark blonde",
     "hex": "#A07850", "desc_pl": "Wszechstronny odcień blondu.",
     "desc_en": "Versatile blonde tone."},
]


def analyze_skin_tone(img_bgr: np.ndarray, hair_mask: np.ndarray) -> dict:
    h, w = img_bgr.shape[:2]

    face_zone = np.zeros((h, w), dtype=bool)
    y1, y2 = int(h * 0.35), int(h * 0.80)
    x1, x2 = int(w * 0.30), int(w * 0.70)
    face_zone[y1:y2, x1:x2] = True

    skin_mask = face_zone & (~hair_mask.astype(bool))

    skin_pixels = img_bgr[skin_mask]
    if len(skin_pixels) < 100:
        return {"undertone": "neutral", "confidence": 0.0, "colors": NEUTRAL_COLORS[:2]}

    avg_bgr = skin_pixels.mean(axis=0).reshape(1, 1, 3).astype(np.uint8)
    avg_lab = cv2.cvtColor(avg_bgr, cv2.COLOR_BGR2Lab)[0, 0]

    L, a, b = float(avg_lab[0]), float(avg_lab[1]), float(avg_lab[2])

    b_norm = b - 128
    a_norm = a - 128

    warmth_score = b_norm + 0.3 * a_norm

    if warmth_score > 8:
        undertone = "warm"
        confidence = min(1.0, warmth_score / 25)
        colors = WARM_COLORS
    elif warmth_score < -5:
        undertone = "cool"
        confidence = min(1.0, abs(warmth_score) / 20)
        colors = COOL_COLORS
    else:
        undertone = "neutral"
        confidence = 0.6
        colors = NEUTRAL_COLORS + WARM_COLORS[:1] + COOL_COLORS[:1]

    return {
        "undertone": undertone,
        "confidence": round(confidence, 2),
        "colors": colors[:3],
        "lab_values": {"L": round(L, 1), "a": round(a_norm, 1), "b": round(b_norm, 1)},
    }