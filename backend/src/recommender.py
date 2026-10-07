import json, re
import os
from groq import Groq
from src.fade_recommender import recommend_fade_from_front

STYLE_DESCRIPTIONS = {
    "volume_top": "height on top",
    "volume_sides": "fuller sides",
    "short_sides": "shorter sides",
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
    "short_sides": "very short sides may emphasise the width of your jaw",
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
    "short_sides": "bardzo krótkie boki mogą mocniej podkreślać szerokość szczęki",
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

def score_hairstyle(user_scores, style, traits=None):
    score = 0.0
    total_importance = 0.0

    for key, user_value in user_scores.items():
        style_value = style["attributes"].get(key, 0)
        
        importance = abs(user_value)
        total_importance += importance

        score += user_value * style_value
    
    if total_importance == 0:
        return 0.0
    
    final_score = score / total_importance

    return final_score

def explain_match(user_scores, style, total_score, lang="pl"):
    descriptions = STYLE_DESCRIPTIONS_PL if lang == "pl" else STYLE_DESCRIPTIONS
    negatives_map = NEGATIVE_EXPLANATIONS_PL if lang == "pl" else NEGATIVE_EXPLANATIONS
    positive = []
    negative = []

    pos_total = 0.0
    neg_total = 0.0

    attributes = style.get("attributes", {})

    for key, user_value in user_scores.items():
        if user_value == 0:
            continue

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
        
    for c in positive:
        c["percent"] = c["raw"] / pos_total if pos_total > 0 else 0.0
    
    for c in negative:
        c["percent"] = abs(c["raw"]) / neg_total if neg_total > 0 else 0.0

    positive.sort(key=lambda x: x["percent"], reverse=True)
    negative.sort(key=lambda x: x["percent"], reverse=True)

    return positive, negative

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

def _build_face_analysis_llm(user_scores, influences, traits, gender="Man", lang="pl"):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return _build_face_analysis(influences, traits)

    trait_summary = _prepare_trait_summary(user_scores, lang=lang)

    if not trait_summary:
        if lang == "pl":
            return [
                "Proporcje Twojej twarzy są dobrze zbalansowane.",
                "Większość fryzur powinna dobrze współgrać z Twoimi proporcjami."
            ]
        return [
            "Your facial proportions are well balanced.",
            "Most hairstyles should work well with your proportions."
        ]

    gender_pl = "Kobieta" if gender == "Woman" else "Mężczyzna"
    gender_en = "Female" if gender == "Woman" else "Male"

    if lang == "pl":
        system_msg = (
            f"Jesteś doświadczonym fryzjerem. Piszesz krótką, praktyczną analizę twarzy dla klienta.\n"
            "Zwracaj się bezpośrednio: 'Twoja twarz', 'dla Ciebie', 'u Ciebie'.\n"
            "Nigdy nie używaj: \"jego\", \"jej\", \"klient\", \"osoba\".\n"
            "Każde zdanie musi dawać konkretną wskazówkę stylistyczną - nie opisuj cech, wyjaśniaj co zrobić.\n"
            "Odpowiadasz TYLKO w JSON: {\"sentences\": [\"...\", \"...\", \"...\"]}"
        )
        user_msg = (
            f"Klient ({gender_pl}). Cechy twarzy i ich wpływ na dobór fryzury:\n\n"
            + "\n".join(trait_summary)
            + "\n\nNapisz 3 zdania które mówią klientowi CO konkretnie powinien wybrać i DLACZEGO. "
            "Unikaj ogólników. Każde zdanie = jedna konkretna rada."
            "\n{\"sentences\": [\"rada 1\", \"rada 2\", \"rada 3\"]}"
        )
    else:
        system_msg = (
            f"You are an experienced hairstylist writing a short, practical facial analysis for a client.\n"
            "Address the client directly using: \"your face\", \"for you\", \"your jawline\".\n"
            "Never use: \"his\", \"her\", \"the client\", \"this person\".\n"
            "Each sentence must provide a specific stylistic tip - don't describe characteristics, explain what to do.\n"
            "Reply ONLY in JSON format: {\"sentences\": [\"...\", \"...\", \"...\"]}"
        )
        user_msg = (
            f"({gender_en}) client. Facial features and their impact on choosing a hairstyle:\n\n"
            + "\n".join(trait_summary)
            + "\n\n Write three sentences that tell the customer EXACTLY what they should choose and WHY. "
            "Avoid generalisations. Each sentence = one specific piece of advice."
            "\n{\"sentences\": [\"advice 1\", \"advice 2\", \"advice 3\"]}"
        )

    print("\n[LLM SYSTEM PROMPT]")
    print(system_msg)

    print("\n[LLM USER PROMPT]")
    print(user_msg)

    print("\n=============================================\n")

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model = "qwen/qwen3.8-27b",
            max_tokens = 400,
            temperature = 0.7,
            top_p = 0.80,
            reasoning_effort="none",
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        text = response.choices[0].message.content.strip()
        print("\n[LLM RESPONSE]")
        print(text)
        print("\n=============================================\n")
        if not text.endswith('}'):
            matches = re.findall(r'"([^"]*)"', text)
            if matches:
                sentences = [m for m in matches if len(m) > 10]
                if sentences:
                    return sentences[:4]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            for suffix in [']}', '}', ']']:
                try:
                    parsed = json.loads(text + suffix)
                    break
                except json.JSONDecodeError:
                    continue
            else:
                print(f"LLM JSON parse failed: {text!r}")
                return _build_face_analysis(influences, traits)

        if isinstance(parsed, dict) and "sentences" in parsed:
            sentences = parsed["sentences"]
        elif isinstance(parsed, list):
            sentences = parsed
        else:
            print(f"DEBUG unexpected structure: {parsed}")
            return _build_face_analysis(influences, traits)

        if isinstance(sentences, list) and all(isinstance(s, str) for s in sentences):
            return sentences[:4]

    except Exception as e:
        print(f"LLM error: {e}")

    return _build_face_analysis(influences, traits)

def _prepare_trait_summary(user_scores, lang="pl"):
    descriptions = (
        STYLE_DESCRIPTIONS_PL
        if lang == "pl"
        else STYLE_DESCRIPTIONS
    )

    if not user_scores:
        return []

    positive = []
    negative = []

    for dim, value in user_scores.items():
        if abs(value) < 0.01:
            continue

        desc = descriptions.get(dim, dim)

        if value > 0:
            positive.append((dim, value, desc))
        else:
            negative.append((dim, value, desc))

    positive.sort(key=lambda x: x[1], reverse=True)

    negative.sort(key=lambda x: abs(x[1]), reverse=True)

    summary = []

    if lang == "pl":
        if positive:
            summary.append("ZALECANE CECHY FRYZURY:")
            for dim, value, desc in positive:
                summary.append(
                    f"- {desc} (siła: {value:+.1f})"
                )

        if negative:
            summary.append("CECHY FRYZURY, KTÓRYCH NALEŻY UNIKAĆ:")
            for dim, value, desc in negative:
                summary.append(
                    f"- {desc} (siła: {abs(value):.1f})"
                )

    else:
        if positive:
            summary.append("RECOMMENDED HAIRSTYLE FEATURES:")
            for dim, value, desc in positive:
                summary.append(
                    f"- {desc} (strength: {value:+.1f})"
                )

        if negative:
            summary.append("HAIRSTYLE FEATURES TO AVOID:")
            for dim, value, desc in negative:
                summary.append(
                    f"- {desc} (strength: {abs(value):.1f})"
                )

    return summary

def _build_style_result(style, user_scores, traits, score, lang):
    positive, negative = explain_match(
        user_scores,
        style,
        score,
        lang=lang
    )
    display_score = calculate_display_score(user_scores, style)

    return {
        "name": style["name"],
        "score": score,
        "display_score": display_score,
        "category": style.get("category", ""),
        "tags": style.get(f"tags_{lang}", style.get("tags", [])),
        "description": style.get(
            f"description_{lang}",
            style.get("description", "")
        ),
        "contributions": positive,
        "negatives": negative,
        "image": style.get("image"),
    }

def calculate_display_score(user_scores, style):
    attributes = style.get("attributes", {})

    total_importance = 0.0
    total_match = 0.0

    for key, user_value in user_scores.items():
        importance = abs(user_value)

        if importance < 0.01:
            continue

        style_value = attributes.get(key, 0.0)

        if user_value > 0:
            target = 1.0
        else:
            target = 0.0

        closeness = 1.0 - abs(style_value - target)

        total_match += importance * closeness
        total_importance += importance

    if total_importance == 0:
        return 50

    return round(
        100 * total_match / total_importance
    )

def generate_recommendations(user_scores, traits, gender="Man", top_k=3, 
                             hairstyles_path="data/hairstyles.json", lang="pl"):
    styles = load_hairstyles(hairstyles_path)
    influences = compute_traits_influences(traits, gender)
    fade_rec = recommend_fade_from_front(traits, gender)
    print(f"DEBUG influences: {list(influences.keys())}")
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
                user_scores, influences, traits, gender, lang="pl"
            ),
            "en": _build_face_analysis_llm(
                user_scores, influences, traits, gender, lang="en"
            ),
        },

        "trait_influences": influences,
        "fade_recommendation": fade_rec,
    }