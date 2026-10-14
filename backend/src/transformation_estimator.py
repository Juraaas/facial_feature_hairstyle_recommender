import os, json
from groq import Groq

HAIR_LENGTH_LABELS = {
    "very_short": "very short — currently 0.5-1 cm, like a buzz cut",
    "short": "short — currently 1-5 cm",
    "medium": "medium — currently around chin or jaw length",
    "long": "long — past the shoulders",
}

HAIR_TYPE_LABELS = {
    "straight": "straight",
    "wavy": "wavy",
    "curly": "curly",
    "coily": "coily",
    None: "unknown",
} 

def estimate_transformation(
    current_length: str,
    current_hair_type: str,
    target_style: str,
    color_change: str = "natural",
    gender: str = "Man",
    lang: str = "pl",
) -> dict | None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    color_note = (
        "" if color_change == "natural"
        else f"The client also wants to change hair color to {color_change}."
    )

    prompt = f"""You are an experienced professional hairstylist advising a client on transitioning from their current hair to a desired hairstyle.
            Your task is to provide a realistic transition estimate and useful, personalized salon advice.
            Client information:
            - Current hair length: {HAIR_LENGTH_LABELS.get(current_length, current_length)}
            - Current hair texture: {HAIR_TYPE_LABELS.get(current_hair_type, 'unknown texture')}
            - Target hairstyle: {target_style}
            - Gender: {'Woman' if gender == 'Woman' else 'Man'}
            {color_note}

            Guidelines:
            - Estimate the number of salon visits and months needed to achieve the target hairstyle.
            - Distinguish between a style that can be achieved at the next haircut and one that requires realistic time for the growing of the hair.
            - Consider the current length, target style, hair texture, trimming during the transition, and any requested color change.
            - Do not assume that hair must grow longer if the target style can be achieved by cutting or restyling the existing hair.
            - Give practical advice that directly relates to this client's transition. Avoid generic advice.
            - If the current hair texture is unknown, do not invent it.
            - Avoid false precision. Treat the estimate as approximate because actual growth, starting length, hair condition, and stylist technique vary.
            - The recommendation should explain the most useful next step, not merely repeat the target hairstyle.
            - Keep each text field concise and actionable.
            - Write in natural, friendly Polish for Polish fields and natural English for English fields.
            - Return ONLY valid JSON without Markdown.

            Return this schema:
            {{
            "visits": 1,
            "months": 0,
            "difficulty": "easy",
            "note_pl": "Jedno krótkie zdanie podsumowujące czas i trudność transformacji.",
            "note_en": "One short sentence summarizing the time and difficulty of the transition.",
            "recommendation_pl": "Konkretna porada dotycząca tej transformacji.",
            "first_step_pl": "Co konkretnie zrobić lub powiedzieć podczas pierwszej wizyty u fryzjera.",
            "maintenance_pl": "Jak utrzymać fryzurę po osiągnięciu celu."
            }}
            Use an integer from 1-8 for visits, 0-18 for months, difficulty: easy/moderate/challenging."""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model = "qwen/qwen3.8-27b",
            messages = [{"role": "user", "content": prompt}],
            max_tokens = 600,
            temperature = 0.3,
        )
        raw = response.choices[0].message.content
        print(f"TRANSFORMATION RAW: {raw[:200]}")
        data = json.loads(raw)

        required_fields = (
            "visits",
            "months",
            "difficulty",
            "note_pl",
            "note_en",
            "recommendation_pl",
            "first_step_pl",
            "maintenance_pl",
        )

        if not isinstance(data, dict):
            return None

        if not all(
            key in data and data[key] is not None
            for key in required_fields
        ):
            return None

        if data["difficulty"] not in {"easy", "moderate", "challenging"}:
            return None

        for key in required_fields[3:]:
            if not isinstance(data[key], str) or not data[key].strip():
                return None

        data["visits"] = max(1, min(8, int(data["visits"])))
        data["months"] = max(0, min(18, int(data["months"])))

        return data

    except Exception as e:
        print(f"Transformation estimator error: {e}")
        return None