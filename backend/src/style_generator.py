import os
import fal_client

STYLE_PROMPTS = {
    "French Crop": "french crop hairstyle, short textured top with fringe, tapered sides",
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
    
    prompt = f"Transform the hairstyle to {style}"
    
    if color:
        prompt += f" with {color}"
    
    prompt += (
        ". The new hairstyle shape, cut, and length must be clearly visible. "
        "Preserve exactly: face, skin tone, eyes, eyebrows, ears, "
        "facial hair, expression, clothing, background. "
        "Change only the hair."
    )
    
    print(f"FLUX PROMPT [{style_name} + {color_id}]: {prompt}")
    return prompt


async def generate_preview(img_bytes: bytes, style_name: str, color_id: str) -> bytes:
    import base64
    img_b64 = base64.b64encode(img_bytes).decode()
    image_url = f"data:image/jpeg;base64,{img_b64}"
    prompt = build_prompt(style_name, color_id)

    result = fal_client.run(
        "fal-ai/flux-pro/kontext",
        arguments={
            "image_url": image_url,
            "prompt": prompt,
            "num_inference_steps": 28,
            "guidance_scale": 6.0,
            "safety_tolerance": "5",
        }
    )

    out_url  = result["images"][0]["url"]
    import httpx
    response = httpx.get(out_url, timeout=30)
    return response.content