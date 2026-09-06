import os
import fal_client

STYLE_PROMPTS = {
    "French Crop": (
        "french crop hairstyle: short back and sides with a skin fade, "
        "textured top approximately 2-3 inches long, soft fringe swept forward "
        "onto the forehead, natural texture on top"
    ),
    "Messy Crop": (
        "messy crop hairstyle: short disconnected undercut, longer textured top "
        "with natural movement and piece-y texture, effortless undone finish, "
        "no product-heavy look"
    ),
    "Buzz Cut": (
        "buzz cut: uniform very short hair all over the head, approximately "
        "grade 2-3 clipper length, clean defined hairline, no fade"
    ),
    "Pompadour": (
        "pompadour hairstyle: hair on top swept upward and back from the forehead "
        "with significant volume, sides tapered short, smooth finish on top"
    ),
    "Quiff": (
        "quiff hairstyle: voluminous front section swept up and slightly back, "
        "textured and lifted at the front, short to medium tapered sides"
    ),
    "Crew Cut": (
        "crew cut: short tapered sides and back, slightly longer on top "
        "lying flat, clean fade, classic masculine silhouette"
    ),
    "Side Part": (
        "side part hairstyle: clean hard part on one side, hair combed "
        "over to the opposite side, medium length on top, short tapered sides"
    ),
    "Classic Undercut": (
        "classic undercut: shaved or very short sides and back with a clear "
        "disconnection, longer hair on top slicked or styled back, "
        "sharp contrast between top and sides"
    ),
    "Slick Back": (
        "slick back hairstyle: all hair swept straight back from the forehead, "
        "smooth and flat on top, medium length, short tapered sides"
    ),
    "Textured Fringe": (
        "textured fringe hairstyle: choppy uneven fringe falling across the forehead, "
        "layered textured top with natural movement, short to medium length overall"
    ),
    "Comb Over": (
        "comb over hairstyle: hair parted and combed to one side, "
        "medium length on top, clean low skin fade on the sides, "
        "polished and structured finish"
    ),
    "Bro Flow": (
        "bro flow hairstyle: medium to long hair past the ears, "
        "natural relaxed movement, hair flows back and to the sides naturally, "
        "minimal styling"
    ),
    "Curly Volume": (
        "natural curly hairstyle: defined curl pattern with volume and shape, "
        "medium length curls with good definition, not frizzy, "
        "rounded silhouette"
    ),
    "French Bob": (
        "french bob hairstyle: blunt cut at jaw length, straight heavy fringe "
        "cut across the forehead above the eyebrows, sleek and polished finish"
    ),
    "Beach Waves": (
        "beach waves hairstyle: loose effortless waves throughout medium length hair, "
        "natural texture with soft movement, undone relaxed finish, "
        "no tight curls"
    ),
    "Layered Medium": (
        "layered medium hairstyle: shoulder length hair with multiple soft layers "
        "throughout, feathered ends, movement and volume, "
        "face-framing front layers"
    ),
    "Classic Updo": (
        "classic updo hairstyle: hair gathered and pinned elegantly at the back "
        "or crown, smooth and polished finish, no loose strands, "
        "structured and formal"
    ),
    "Soft Bun": (
        "soft bun hairstyle: loosely gathered low bun at the nape, "
        "relaxed and slightly undone with soft face-framing pieces, "
        "casual elegant finish"
    ),
    "Curtain Fringe Medium": (
        "curtain fringe hairstyle: centre-parted fringe that falls on both sides "
        "of the face framing it, medium length hair with soft layers, "
        "fringe blends into the sides"
    ),
    "Pixie Cut": (
        "pixie cut hairstyle: very short all over, slightly longer on top "
        "with texture, tapered nape and sides, feminine short style"
    ),
     "Lob": (
        "lob long bob hairstyle: blunt or slightly layered cut at collarbone length, "
        "clean ends with subtle movement, versatile and polished"
    ),
     "Braided Crown": (
        "braided crown hairstyle: braids wrapped around the crown of the head "
        "like a halo, hair pinned up leaving the nape exposed, "
        "romantic and structured"
    ),
}

COLOR_PROMPTS = {
    "natural": "",
    "blonde": "change hair color to dark blonde, bright and even",
    "dark": "change hair color to rich dark brown, deep and natural",
    "black": "change hair color to jet black, very dark and glossy",
    "auburn": "change hair color to warm auburn red, natural reddish-brown",
}

def build_prompt(style_name: str, color_id: str) -> str:
    style = STYLE_PROMPTS.get(style_name, f"{style_name} hairstyle")
    color = COLOR_PROMPTS.get(color_id, "")
    
    prompt = f"Change the hairstyle to {style}"
    
    if color:
        prompt += f".Also {color}"
    
    prompt += (
        ". Keep the person's face shape, eyebrows, ears, eye color, skin tone, facial hair, expression, clothing and background pixel-perfect identical."
    )
    
    print(f"FLUX PROMPT [{style_name} + {color_id}]: {prompt}")
    return prompt


async def generate_preview(img_bytes: bytes, style_name: str, color_id: str) -> bytes:
    import base64
    import httpx

    prompt = build_prompt(style_name, color_id)
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