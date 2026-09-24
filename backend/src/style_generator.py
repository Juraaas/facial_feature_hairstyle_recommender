import os, json, base64, httpx
import fal_client

STYLE_PROMPTS_MALE = {
    "French Crop": (
        "a French Crop: faded sides, textured top clearly longer than the sides, "
        "and a fringe resting naturally on the forehead. "
        "The fringe touching the forehead is essential."
    ),
    "Messy Crop": (
        "a Messy Crop: short disconnected sides, medium-length textured top "
        "with piece-y uneven texture and natural forward movement, "
        "relaxed undone finish with no fringe."
    ),
    "Buzz Cut": (
        "a Buzz Cut: uniformly very short hair all over — top, sides and back "
        "the same grade 1-2 clipper length. Extremely clean minimal silhouette."
    ),
    "Quiff": (
        "a modern natural Quiff with moderate volume at the front, "
        "the front hair gently lifted upward and slightly backward, "
        "with a smooth natural transition into the rest of the top "
        "and short tapered sides. "
        "The front should have visible lift and shape without looking overly high or stiff."
    ),
    "Classic Undercut": (
        "a Classic Undercut: very short disconnected sides with a sharp separation line, "
        "medium-length hair on top combed straight back. "
        "The hard disconnection between top and sides is essential."
    ),
    "Slick Back": (
        "a Slick Back hairstyle: medium-length hair swept backward "
        "from the forehead in a natural relaxed way — not wet-look or overly glossy. "
        "Matte or low-sheen finish. Short tapered sides. No fringe. "
        "Do not add any facial hair or beard. "
    ),
    "Textured Fringe": (
        "a Textured Fringe: choppy irregular fringe cut across the forehead "
        "with deliberately jagged uneven ends. Layered textured top with natural movement. "
        "The choppy irregular fringe is essential."
    ),
    "Comb Over": (
        "a Comb Over: medium-length hair on top with a clean side part, "
        "hair swept horizontally across. Skin fade on the sides. "
        "The defined side part and horizontal sweep are essential."
    ),
    "Bro Flow": (
        "a Bro Flow: hair grown past the ears toward the jaw, "
        "flowing naturally backward and outward. Medium length on sides too — "
        "no fade, no taper. Relaxed natural finish."
        "Do not add any facial hair or beard. "
    ),
    "Wolf Cut": (
        "a Wolf Cut: heavy shaggy layers throughout, curtain fringe falling "
        "on both sides of the forehead, strong volume at the crown, "
        "longer wispy layered ends at the back. "
        "The curtain fringe and shaggy crown volume are essential."
    ),
    "Modern Mullet": (
        "a modern short Mullet with short textured hair on top and sides, "
        "a short textured fringe at the front, "
        "and a clearly longer but controlled layered back reaching the lower neck. "
        "The overall silhouette should be compact, clean and contemporary, "
        "not long, dramatic or rock-inspired."
    ),
    "Curtain Bangs": (
        "men's Curtain Bangs hairstyle: medium-length hair with a clear centre part "
        "running from front to back. The fringe splits at the centre "
        "and each half falls diagonally to its respective side of the forehead, "
        "clearly framing both sides of the face. "
        "The two-sided parted fringe is essential. "
    )
}

STYLE_PROMPTS_FEMALE = {
    "French Bob": (
        "a French Bob: blunt horizontal cut at exactly jaw length, "
        "sleek smooth hair, straight heavy fringe cut straight across "
        "the forehead above the eyebrows. "
        "The blunt jaw-length cut with straight-across fringe is essential."
    ),
    "Beach Waves": (
        "Beach Waves: loose irregular waves throughout medium-length hair "
        "reaching the shoulders. Not tight curls — relaxed effortless waves "
        "with natural texture and soft volume."
    ),
    "Layered Medium": (
        "Layered Medium hair: shoulder-length with multiple soft blended layers, "
        "shorter face-framing layers at the front, feathered ends with visible movement."
    ),
    "Classic Updo": (
        "a Classic Updo: all hair gathered and pinned up at the crown or back, "
        "smooth polished surface, no loose strands. "
        "Clean structured silhouette with zero hair on face or neck."
    ),
    "Soft Bun": (
        "a Soft Bun: hair loosely gathered into a low bun at the nape, "
        "slightly puffed and relaxed with a few soft face-framing pieces left out. "
        "Casual and romantic, not tight or slicked."
    ),
    "Curtain Fringe Medium": (
        "Curtain Fringe: centre-parted fringe split into two sections "
        "falling softly on each side of the forehead and framing the face. "
        "Medium-length hair with soft layers. "
        "The centre-parted face-framing fringe is essential."
    ),
    "Pixie Cut": (
        "a Pixie Cut: very short all over — 1-2 inches. "
        "Closely tapered at nape and sides, slightly longer textured top. "
        "Significantly shorter than a bob — no hair near jaw length."
    ),
    "Lob": (
        "a Lob (long bob): hair ending at collarbone length, "
        "clean blunt or lightly layered ends, subtle natural movement. "
        "Longer than a bob, shorter than shoulder length."
    ),
    "Braided Crown": (
        "a Braided Crown: two braids wrapping around the top of the head "
        "like a halo, pinned in place. The braids are the visible focal element. "
        "Nape and back of head exposed."
    ),
    "Wolf Cut": (
        "a women's Wolf Cut: heavy curtain fringe framing the forehead, "
        "significant shaggy layering throughout with strong crown volume, "
        "wispy layered ends. "
        "The curtain fringe and shaggy high-volume crown are essential."
    ),
    "Soft Shag": (
        "a Soft Shag: medium-length feathered layers throughout, "
        "relaxed curtain fringe framing the forehead, soft crown volume, "
        "natural lived-in texture. "
        "The curtain fringe and visible feathered layering are essential."
    ),
    "Butterfly Cut": (
        "a Butterfly Cut hairstyle on a woman: long hair with soft face-framing layers "
        "that gently flip or curve outward at mid-length on both sides, "
        "creating a light airy silhouette. "
    ),
}

COLOR_PROMPTS = {
    "natural": None,
    "blonde": "dark blonde, bright and even",
    "dark": "rich dark brown, deep and natural",
    "black": "jet black, very dark and glossy",
    "auburn": "to warm auburn red, natural reddish-brown",
}

def build_prompt(style_name: str, color_id: str, gender: str = "Man") -> str:
    prompts = STYLE_PROMPTS_FEMALE if gender == "Woman" else STYLE_PROMPTS_MALE
    style = prompts.get(style_name) or STYLE_PROMPTS_MALE.get(style_name) or f"{style_name} hairstyle"
    color = COLOR_PROMPTS.get(color_id)

    prompt = (
        f"Edit only the hair of the person in the input image. "
        f"Change the hairstyle to {style}. "
    )

    if color:
        prompt += f"Also change the hair color to {color}. "
    else:
        prompt += "Keep the exact same hair color as in the original photo. "

    prompt += (
        "Preserve the exact identity and facial appearance of the person. "
        "Do not alter the face, facial proportions, eyes, eyebrows, nose, mouth, "
        "ears, skin, expression, clothing, lighting or background. "
        "Do not add, remove or change any facial hair or beard. "
    )

    print(f"SEEDREAM [{style_name} + {color_id} + {gender}]: {prompt}")
    return prompt

async def generate_preview(img_bytes: bytes, style_name: str, 
                           color_id: str, gender: str = "Man") -> bytes:

    img_b64  = base64.b64encode(img_bytes).decode()
    user_url = f"data:image/jpeg;base64,{img_b64}"
    prompt = build_prompt(style_name, color_id, gender)
    
    handler = fal_client.submit(
        "fal-ai/bytedance/seedream/v4.5/edit",
        arguments={
            "image_urls": [user_url],
            "prompt": prompt,
            "guidance_scale": 7.5,
        }
    )

    result = handler.get()
    out_url = result["images"][0]["url"]

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(out_url)
    return response.content