import os
import fal_client

STYLE_PROMPTS_MALE = {
    "French Crop": (
        "french crop hairstyle: short back and sides with a skin fade, "
        "textured top approximately 2-3 inches long, soft fringe swept forward "
        "onto the forehead, natural texture on top"
    ),
    "French Crop": (
        "French Crop hairstyle: the top section is approximately 3-4 inches long "
        "and styled forward with a soft textured fringe that falls naturally onto the forehead. "
        "The sides and back are faded short. This is NOT a buzz cut and the top must be clearly "
        "longer than the sides with visible fringe touching the forehead."
    ),
    "Messy Crop": (
        "Messy Crop hairstyle: medium-length top (3-4 inches) with deliberate piece-y, "
        "undone texture. Short disconnected undercut on the sides. Hair on top falls "
        "naturally with effortless movement — no fringe, no swept-back styling."
    ),
    "Buzz Cut": (
        "Buzz Cut: uniformly very short hair clipped to grade 1-2 length all over the head. "
        "No fade, no variation in length, no styling on top. Extremely short and clean."
    ),
    "Quiff": (
        "Quiff hairstyle: prominent volume at the front hairline swept upward and slightly back, "
        "creating a lifted peak. Medium-length top with short tapered sides. "
        "The front section is the focal point with clear height."
    ),
    "Crew Cut": (
        "crew cut: short tapered sides and back, slightly longer on top "
        "lying flat, clean fade, classic masculine silhouette"
    ),
    "Classic Undercut": (
        "Classic Undercut: a sharp disconnection between very short or shaved sides and "
        "medium-to-long hair on top (4-5 inches). Top hair is slicked or combed back. "
        "Clear contrast line between top and sides with no fade blending them."
    ),
    "Slick Back": (
        "Slick Back hairstyle: all hair on top swept straight back from the forehead, "
        "lying flat and smooth. Medium length on top (3-4 inches), short tapered sides. "
        "No volume at front, no part — all pulled directly backward."
    ),
    "Textured Fringe": (
        "Textured Fringe hairstyle: a choppy, uneven fringe cut across the forehead "
        "with deliberate irregular ends. Layered textured hair on top with natural movement. "
        "Short to medium overall length. The fringe is the defining feature."
    ),
    "Comb Over": (
        "Comb Over hairstyle: medium-length hair on top (3-4 inches) parted cleanly "
        "on one side and swept horizontally across the head. Skin fade on the sides. "
        "Polished structured finish — not casual, not slicked all the way back."
    ),
    "Bro Flow": (
        "Bro Flow hairstyle: hair grown out to jaw length or longer (5-6 inches), "
        "flowing naturally outward and backward from the crown. No styling product, "
        "natural movement. Sides are not faded — medium length throughout."
    ),
    "Wolf Cut M": (
        "Wolf Cut hairstyle for men: heavy layering throughout with a curtain fringe at the front. "
        "Volume concentrated at the crown, layers cascade down with wispy ends. "
        "Medium to long length (4-6 inches on top). Shaggy, lived-in finish "
        "reminiscent of 70s rock — clearly distinct from a standard layered cut."
    ),
    "Modern Mullet": (
        "Modern Mullet hairstyle: noticeably shorter on top and sides, "
        "with significantly longer hair at the back extending past the collar. "
        "Textured fringe at the front. Strong length contrast front-to-back is essential — "
        "the back must be clearly longer than the top."
    ),
    "Curtain Bangs": (
        "Curtain Bangs hairstyle for men: centre-parted fringe that splits and falls "
        "on both sides of the forehead, framing the face. Medium overall length (4-5 inches). "
        "Relaxed, natural movement. The centre part and face-framing fringe are the key feature."
    ),
}
STYLE_PROMPTS_FEMALE = {
    "French Bob": (
        "French Bob hairstyle: blunt horizontal cut sitting precisely at jaw length. "
        "A straight, heavy fringe cut across the forehead above the eyebrows. "
        "Hair is sleek and smooth. The blunt jaw-length cut with straight across fringe "
        "is the defining feature — not layered, not wavy."
    ),
    "Beach Waves": (
        "Beach Waves hairstyle: loose, undone waves throughout medium-length hair "
        "reaching the shoulders or collarbone. Waves are irregular and effortless — "
        "not tight ringlets, not straight. Natural texture with volume and movement."
    ),
    "Layered Medium": (
        "Layered Medium hairstyle: shoulder-length hair with multiple soft graduated layers. "
        "Face-framing front layers that are shorter than the back. Feathered, blended ends. "
        "Clear layering visible — not a blunt cut, not a shag."
    ),
    "Classic Updo": (
        "Classic Updo: all hair gathered and pinned up at the crown or back of the head. "
        "Smooth, polished surface with no loose strands. Structured and formal. "
        "Zero hair falling on the face, neck, or shoulders."
    ),
    "Soft Bun": (
        "Soft Bun hairstyle: hair loosely gathered into a low bun at the nape. "
        "Intentionally relaxed with slight puffiness, not tight or slicked. "
        "A few soft face-framing pieces left out. Casual and romantic."
    ),
    "Curtain Fringe Medium": (
        "Curtain Fringe hairstyle: centre-parted fringe that parts in the middle and "
        "falls softly on both sides framing the face. Medium-length hair with soft layers. "
        "The signature feature is the centre-parted face-framing fringe — not a straight blunt fringe."
    ),
    "Pixie Cut": (
        "Pixie Cut: very short feminine hairstyle. Hair is cut close to 1-2 inches all over. "
        "Slightly longer on top with texture, tapered closely at the nape and sides. "
        "This is significantly shorter than a bob — no hair at or near jaw length."
    ),
    "Lob": (
        "Lob (Long Bob) hairstyle: blunt or minimally layered cut sitting at collarbone length "
        "(approximately 2-3 inches below the jaw). Clean ends, slight movement. "
        "Longer than a bob, shorter than shoulder length. Simple and polished."
    ),
    "Braided Crown": (
        "Braided Crown hairstyle: two braids that wrap around the top of the head "
        "like a halo or crown, pinned in place. The nape and back of the head are exposed. "
        "A structured romantic updo — braids are the visible focal element on top."
    ),
    "Wolf Cut F": (
        "Wolf Cut hairstyle for women: heavy curtain fringe at the front, "
        "significant layering throughout with volume at the crown tapering to "
        "wispy layered ends. Medium to long length. Shaggy, high-volume finish — "
        "clearly distinct from a standard layered cut by the curtain fringe and crown volume."
    ),
    "Soft Shag": (
        "Soft Shag hairstyle: medium-length cut with a relaxed curtain fringe and "
        "extensive feathered layering throughout. Textured, lived-in finish with movement. "
        "Less dramatic than a Wolf Cut — softer layers and less crown volume."
    ),
    "Butterfly Cut": (
        "Butterfly Cut hairstyle: long hair with dramatic face-framing layers that "
        "flip outward at the midpoint creating a wing-like silhouette. "
        "Interior layers are shorter, exterior layers are longer. "
        "The distinctive outward flip of the mid-length layers defines this cut."
    ),
}

COLOR_PROMPTS = {
    "natural": "",
    "blonde": "change hair color to dark blonde, bright and even",
    "dark": "change hair color to rich dark brown, deep and natural",
    "black": "change hair color to jet black, very dark and glossy",
    "auburn": "change hair color to warm auburn red, natural reddish-brown",
}

def build_prompt(style_name: str, color_id: str, gender: str = "Man") -> str:
    prompts = STYLE_PROMPTS_FEMALE if gender == "Woman" else STYLE_PROMPTS_MALE
    style = prompts.get(style_name) or STYLE_PROMPTS_MALE.get(style_name) or f"{style_name} hairstyle"
    color = COLOR_PROMPTS.get(color_id, "")
    
    prompt = f"Change the hairstyle to {style}"
    if color:
        prompt += f".Also {color}"
    
    prompt += (
        ". Keep the person's face shape, eyebrows, ears, eye color, skin tone, facial hair, expression, clothing and background pixel-perfect identical."
    )
    
    print(f"FLUX [{style_name} + {color_id} + {gender}]: {prompt}")
    return prompt


async def generate_preview(img_bytes: bytes, style_name: str, 
                           color_id: str, gender: str = "Man") -> bytes:
    import base64
    import httpx

    prompt = build_prompt(style_name, color_id, gender)
    img_b64 = base64.b64encode(img_bytes).decode()
    image_url = f"data:image/jpeg;base64,{img_b64}"
    
    handler = fal_client.submit(
        "fal-ai/flux-pro/kontext",
        arguments={
            "image_url": image_url,
            "prompt": prompt,
            "safety_tolerance": "5",
        }
    )

    result = handler.get()
    out_url = result["images"][0]["url"]

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(out_url)
    return response.content