import json
import os
from groq import Groq

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

STYLE_DESCRIPTIONS_PL = {
    "volume_top": "objętość na górze",
    "volume_sides": "pełniejsze boki",
    "short_sides": "krótkie boki",
    "longer_hair": "dłuższe włosy",
    "fringe": "grzywka",
    "clean_lines": "czysty kształt",
    "soft_texture": "miękka tekstura",
    "textured_top": "góra z teksturą",
    "layers": "warstwowe cięcie",
    "updo": "upięcie",
    "curtain_fringe": "kurtynowa grzywka",
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

NEGATIVE_EXPLANATIONS_PL = {
    "fringe": "grzywka może zasłaniać oczy lub dodawać wagi czołu",
    "volume_sides": "objętość boków może optycznie poszerzyć twarz",
    "volume_top": "dodatkowa wysokość może podkreślić długość twarzy",
    "short_sides": "krótkie boki mogą zwracać uwagę na szeroką szczękę",
    "clean_lines": "geometryczne cięcia mogą uwydatniać asymetrię",
    "soft_texture": "miękka tekstura może nie pasować do struktury twarzy",
    "longer_hair": "długość może dodatkowo wydłużyć twarz",
    "textured_top": "teksturowana góra może zaburzyć balans przy wyraźnej brodzie",
    "layers": "warstwy mogą nie pasować do proporcji twarzy",
    "updo": "upięcie może wydłużyć twarz",
    "curtain_fringe": "środkowy przedziałek może uwydatnić blisko osadzone oczy",
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

TRAIT_EXPLANATIONS_PL = {
    "face_length": {
        "long": "wydłużony kształt twarzy - objętość po bokach i grzywka pomagają zrównoważyć proporcje",
        "short": "krótszy kształt twarzy - wysokość na górze pomaga optycznie wydłużyć proporcje",
        "balanced": "długość twarzy jest dobrze zbalansowana",
    },

    "forehead": {
        "high": "wysokie czoło - grzywka pomaga optycznie obniżyć linię włosów",
        "low": "niskie czoło - warto pozostawić czoło bardziej odkryte i unikać ciężkiej grzywki",
    },

    "jaw": {
        "wide": "szeroka szczęka - miękkie, warstwowe fryzury pomagają złagodzić jej optyczną szerokość",
        "narrow": "wąska szczęka - objętość po bokach pomaga poprawić proporcje twarzy",
    },

    "eyes": {
        "wide": "szeroko rozstawione oczy - pionowe akcenty i uporządkowane przedziałki dobrze równoważą proporcje",
        "close": "oczy blisko siebie - objętość po bokach pomaga stworzyć wrażenie większego odstępu",
    },

    "lips": {
        "wide": "szersze usta - lekka tekstura na górze pomaga zrównoważyć dolną część twarzy",
        "narrow": "węższe usta - uporządkowane i strukturalne fryzury dobrze uzupełniają proporcje",
    },

    "chin": {
        "prominent": "wyraźny podbródek - tekstura na górze i odpowiednia długość pomagają zrównoważyć profil",
        "recessed": "cofnięty podbródek - objętość na górze pomaga skierować uwagę wyżej",
    },

    "symmetry": {
        "high": "wysoka symetria twarzy - uporządkowane, geometryczne fryzury dobrze współgrają z proporcjami",
        "low": "zauważalna asymetria - teksturowane fryzury pomagają rozłożyć uwagę i zrównoważyć twarz",
    },

    "eye_openness": {
        "narrow": "węższe oczy - warto unikać ciężkiej grzywki, aby nie zasłaniać oczu",
    },

    "thirds_vertical": {
        "top_heavy": "górna część twarzy dominuje - grzywka i objętość po bokach pomagają zrównoważyć proporcje",
        "bottom_heavy": "dolna część twarzy dominuje - wysokość na górze pomaga poprawić balans",
    },

    "hair_type": {
        "curly": "naturalne loki dobrze współgrają z fryzurami wykorzystującymi ruch i objętość",
        "coily": "naturalna struktura dobrze współgra z zaokrąglonym kształtem, kontrolowaną objętością i wyraźną teksturą",
        "straight": "proste włosy dobrze współgrają z uporządkowanymi i strukturalnymi fryzurami",
        "wavy": "delikatnie teksturowane fryzury mogą podkreślić naturalny ruch falowanych włosów",
    },

    "hairline": {
        "receding": "cofająca się linia włosów - lepiej sprawdzą się fryzury unikające ciężkiej grzywki zaczesanej do przodu",
        "uneven": "nierówna linia włosów - dobrze zadziała lekka tekstura",
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

HAIR_TYPE_COMPATIBILITY = {
    "straight": {
        "Curly Volume": -0.3,
        "Beach Waves": -0.15,
        "Braided Crown": -0.1,
    },
    "wavy": {
        "Long Straight Blunt": -0.2,
        "Bob Classic": -0.1,
    },
    "curly": {
        "Long Straight Blunt": -0.4,
        "Slick Back": -0.3,
        "Side Part": -0.2,
        "Comb Over": -0.2,
        "Pompadour": -0.15,
    },
    "coily": {
        "Long Straight Blunt": -0.5,
        "Slick Back": -0.4,
        "Side Part": -0.3,
        "Bro Flow": -0.3,
        "Pompadour": -0.2,
    }
}

HAIRLINE_INCOMPATIBLE = {
    "receding": [
        "French Crop",
        "Textured Fringe",
        "Curtain Fringe Medium",
        "French Bob",
        "Long with Curtain Fringe",
    ],
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

def apply_hair_compatibility(score, style_name, traits):
    hair_type = traits.get("hair_type")
    hairline = traits.get("hairline")

    if hair_type is not None and hair_type in HAIR_TYPE_COMPATIBILITY:
        penalty = HAIR_TYPE_COMPATIBILITY[hair_type].get(style_name, 0)
        score = score + penalty

    if hairline == "receding" and style_name in HAIRLINE_INCOMPATIBLE["receding"]:
        score = score - 0.4

    return max(0.0, score) 

def score_hairstyle(user_scores, style, traits=None):
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
    final_score = base_score * (0.75 + 0.25 * match_concentration)

    if traits:
        final_score = apply_hair_compatibility(
            final_score, style["name"], traits
        )

    return final_score

def explain_match(user_scores, style, total_score, lang="pl"):
    descriptions = STYLE_DESCRIPTIONS_PL if lang == "pl" else STYLE_DESCRIPTIONS
    negatives_map = NEGATIVE_EXPLANATIONS_PL if lang == "pl" else NEGATIVE_EXPLANATIONS
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
                "desc": descriptions.get(key,key),
            })
            pos_total += contribution
        
        elif contribution < 0:
            negative.append({
                "feature": key,
                "raw": contribution,
                "desc": descriptions.get(key, key),
                "reason": negatives_map.get(
                    key, 
                    "może nie pasować do profilu Twojej twarzy" if lang == "pl"
                    else "may not suit your face profile"
            ),
            })
            neg_total += abs(contribution)
        
        if key in MISSING_SENSITIVE_FEATURES and user_value >= 3 and style_value < 0.2:
            missing_strength = user_value * (1 - style_value)
            feature_desc = descriptions.get(key, key)
            if lang == "pl":
                reason = (
                    f"ten styl nie oferuje cechy „{feature_desc}”, "
                    f"którą Twoja analiza wyraźnie sugeruje"
                )
            else:
                reason = (
                    f"this style lacks {feature_desc}, "
                    f"which your analysis strongly favours"
                )

            missing.append({
                "feature": key,
                "raw": missing_strength,
                "desc": feature_desc,
                "reason": reason,
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

def _build_face_analysis(influences, traits, lang="pl"):
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

def _build_face_analysis_llm(influences, traits, gender="Man", lang="pl"):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return _build_face_analysis(influences, traits)

    trait_summary = _prepare_trait_summary(influences, traits, lang=lang)
    if not trait_summary:
        if lang == "pl":
            return [
                "Proporcje Twojej twarzy są dobrze zbalansowane.",
                "Większość fryzur powinna dobrze współgrać z Twoimi proporcjami."
            ]
        else:
            return [
                "Your facial proportions are well balanced.",
                "Most hairstyles should work well with your proportions."
            ]

    if lang == "pl":
        system_instruction = """
Jesteś profesjonalnym stylistą fryzur.

Tworzysz krótkie podsumowanie analizy twarzy dla klienta.

Pisz NATURALNYM, POPRAWNYM JĘZYKIEM POLSKIM.
Zwracaj się bezpośrednio do klienta:
"Twoja twarz", "Twoja szczęka", "dla Ciebie", "u Ciebie".

Nie używaj form:
"jego", "jej", "ich", "klient", "osoba".

Bardzo ważne:
- korzystaj WYŁĄCZNIE z informacji przekazanych w analizie,
- nie wymyślaj nowych cech,
- nie dodawaj diagnoz ani niepotwierdzonych obserwacji,
- nie tłumacz technicznych wartości,
- nie powtarzaj mechanicznie danych wejściowych,
- połącz informacje w naturalny opis,
- skup się na tym, jakie kierunki fryzur są korzystne i dlaczego,
- używaj prostego, naturalnego polskiego,
- każde zdanie powinno być krótkie.

Zwróć dokładnie 3 zdania.
Każde zdanie może mieć maksymalnie 20 słów.

Zwróć WYŁĄCZNIE poprawny JSON:
["zdanie 1", "zdanie 2", "zdanie 3"]
"""

        user_instruction = f"""
Analiza cech klienta:

{chr(10).join(trait_summary)}

Na podstawie powyższych informacji napisz krótkie podsumowanie.
"""

    else:
        system_instruction = """
You are a professional hairstylist.

Write a short facial-analysis summary directly to the client.

Use natural, professional English.
Always address the client directly:
"your face", "your jawline", "for you".

Do not use:
"his", "her", "their", "the client", "the person".

Important:
- use ONLY the information provided,
- do not invent characteristics,
- do not add unsupported observations,
- do not mention technical measurements,
- combine the provided facts into natural sentences,
- focus on which hairstyle directions suit the client and why,
- keep every sentence concise.

Return exactly 3 sentences.
Each sentence must contain at most 20 words.

Return ONLY valid JSON:
["sentence 1", "sentence 2", "sentence 3"]
"""

        user_instruction = f"""
Client's facial analysis:

{chr(10).join(trait_summary)}

Write a concise summary based only on these findings.
"""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model = "llama-3.1-8b-instant",
            messages = [
                {
                    "role": "system",
                    "content": system_instruction,
                },
                {
                    "role": "user",
                    "content": user_instruction,
                }
            ],
            max_tokens = 250,
            temperature = 0.4,
        )
        import json
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        if isinstance(result, list):
            return result[:4]
    except Exception as e:
        print(f"LLM error: {e}")

    return _build_face_analysis(influences, traits, lang=lang)

def _prepare_trait_summary(influences, traits, lang="pl"):
    skip_values   = {None, "normal", "balanced", "slight_imbalance"}
    explanations = (TRAIT_EXPLANATIONS_PL if lang == "pl" else TRAIT_EXPLANATIONS)
    style_descriptions = (STYLE_DESCRIPTIONS_PL if lang == "pl" else STYLE_DESCRIPTIONS)
    trait_summary = []
    priority_order = ["hairline", "hair_type"] + [
        k for k in influences.keys()
        if k not in ("hairline", "hair_type")
    ]

    for key in priority_order[:6]:
        if key not in influences:
            continue
        info = influences[key]
        value = info["value"]
        if value in skip_values:
            continue

        trait_explanation = explanations.get(key, {}).get(value)
        if not trait_explanation:
            trait_summary.append(f"- {key}: {value}")

        delta = info.get("delta", {})
        top_dims = sorted(delta.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
        hints = []
        for dim, change in top_dims:
            desc = style_descriptions.get(dim, dim)
            if lang == "pl":
                hint = (f"preferuje {desc}" if change > 0 else f"działa przeciwko {desc}")
            else:
                hint = (f"favours {desc}" if change > 0 else f"works against {desc}")
            hints.append(hint)
        if hints:
            trait_summary.append(f"- {trait_explanation} ({', '.join(hints)})")
        else:
            trait_summary.append(f"- {trait_explanation}")

    return trait_summary

def _build_style_result(style, user_scores, traits, score, lang):
    positive, negative, missing = explain_match(
        user_scores,
        style,
        score,
        lang=lang
    )

    return {
        "name": style["name"],
        "score": score,
        "category": style.get("category", ""),
        "tags": style.get(f"tags_{lang}", style.get("tags", [])),
        "description": style.get(
            f"description_{lang}",
            style.get("description", "")
        ),
        "contributions": positive,
        "negatives": negative,
        "missing": missing,
        "image": style.get("image"),
    }

def generate_recommendations(user_scores, traits, gender="Man", top_k=3, 
                             hairstyles_path="data/hairstyles.json", lang="pl"):
    styles = load_hairstyles(hairstyles_path)
    influences = compute_traits_influences(traits, gender)
    results_pl = []
    results_en = []

    for style in styles:
        score = score_hairstyle(user_scores, style, traits)

        results_pl.append(
            _build_style_result(
                style,
                user_scores,
                traits,
                score,
                lang="pl"
            )
        )

        results_en.append(
            _build_style_result(
                style,
                user_scores,
                traits,
                score,
                lang="en"
            )
        )

    results_pl.sort(key=lambda x: x["score"], reverse=True)
    results_en.sort(key=lambda x: x["score"], reverse=True)

    return {
        "top_styles": {
            "pl": results_pl[:top_k],
            "en": results_en[:top_k],
        },

        "all_styles": {
            "pl": results_pl,
            "en": results_en,
        },

        "face_analysis": {
            "pl": _build_face_analysis_llm(
                influences, traits, gender, lang="pl"
            ),
            "en": _build_face_analysis_llm(
                influences, traits, gender, lang="en"
            ),
        },

        "trait_influences": influences,
    }