import os, json
from groq import Groq

HAIR_LENGTH_LABELS = {
    "very_short": "very short (buzz/crop)",
    "short": "short",
    "medium": "medium length",
    "long": "long",
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

    prompt = f"""You are an experienced hairstylist estimating how many salon visits 
and months it would take for a client to transition to a new hairstyle safely.

Client:
- Current hair: {HAIR_LENGTH_LABELS.get(current_length, current_length)}, {HAIR_TYPE_LABELS.get(current_hair_type, 'unknown texture')}
- Target style: {target_style}
- Gender: {gender}
{color_note}

Consider: growing out time if needed, cutting/styling sessions required, 
and whether a color change adds complexity. Be realistic and practical.

Return ONLY valid JSON:
{{
  "visits": <integer 1-8>,
  "months": <integer 0-18>,
  "difficulty": "<easy|moderate|challenging>",
  "note_pl": "<one practical sentence in Polish, addressed directly to the client>",
  "note_en": "<one practical sentence in English, addressed directly to the client>"
}}"""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model = "llama-3.3-70b-versatile",
            messages = [{"role": "user", "content": prompt}],
            max_tokens = 200,
            temperature = 0.3,
            response_format = {"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)

        if not all(k in data for k in ("visits", "months", "difficulty")):
            return None

        data["visits"] = max(1, min(8,  int(data["visits"])))
        data["months"] = max(0, min(18, int(data["months"])))

        return data

    except Exception as e:
        print(f"Transformation estimator error: {e}")
        return None