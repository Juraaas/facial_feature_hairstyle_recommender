import os
import fal_client

STYLE_PROMPTS = {
    "French Crop": "french crop hairstyle, short textured top with a soft fringe falling over the forehead, clean tapered sides",
    "Messy Crop": "messy crop hairstyle, short textured top, natural look",
    "Buzz Cut": "buzz cut, very short uniform hair",
    "Pompadour": "pompadour hairstyle, volume swept back from forehead",
    "Curly Volume": "natural curly hair with volume and definition",
    "French Bob": "french bob, chin length blunt cut with straight fringe",
    "Beach Waves": "beach waves, loose natural waves, effortless",
    "Layered Medium": "layered medium hair, shoulder length, soft layers",
    "Classic Updo": "classic updo, elegant pinned style",
    "Soft Bun": "soft loosely gathered bun",
}

COLOR_PROMPTS = {
    "natural": "",
    "blonde": "platinum blonde hair color",
    "dark": "dark brown hair color",
    "black": "jet black hair color",
    "auburn": "auburn red hair color",
    "grey": "silver grey hair color",
}

def build_prompt(style_name: str, color_id: str) -> str:
    style = STYLE_PROMPTS.get(style_name, f"{style_name} hairstyle")
    color = COLOR_PROMPTS.get(color_id, "")
    
    prompt = f"Change only the hairstyle to {style}"
    
    if color:
        prompt += f" with {color}"
    
    prompt += (
        ". Keep the person's face shape, eyebrows, ears, eye color, skin tone, facial hair, expression, clothing and background pixel-perfect identical."
    )
    
    print(f"FLUX PROMPT [{style_name} + {color_id}]: {prompt}")
    return prompt


async def generate_preview(img_bytes: bytes, style_name: str, color_id: str) -> bytes:
    import httpx
    prompt = build_prompt(style_name, color_id)
    url = fal_client.upload(img_bytes, content_type="image/jpeg")
    
    result = fal_client.run(
        "fal-ai/flux-pro/kontext",
        arguments={
            "image_url": url,
            "prompt": prompt,
            "safety_tolerance": "5",
        }
    )

    out_url  = result["images"][0]["url"]
    response = httpx.get(out_url, timeout=60)
    return response.content