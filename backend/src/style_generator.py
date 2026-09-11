import os
import fal_client

STYLE_PROMPTS_MALE = {
    "French Crop": (
        "a French Crop with short faded sides and back, "
        "a clearly longer textured top styled forward, "
        "and a short soft fringe resting naturally on the forehead"
    ),
    "Messy Crop": (
        "a messy crop with short disconnected sides and back, "
        "a medium-length textured top with piece-y uneven texture, "
        "and natural forward movement with a relaxed, undone finish"
    ),
    "Buzz Cut": (
        "a very short uniform buzz cut with the same short clipper length "
        "across the top, sides and back, with an extremely clean minimal silhouette"
    ),
    "Quiff": (
        "a classic quiff with noticeably lifted volume at the front hairline, "
        "the front swept upward and slightly backward, "
        "a medium-length top and shorter tapered sides"
    ),
    "Crew Cut": (
        "a classic crew cut with short tapered sides and back, "
        "a short top that gradually becomes slightly longer toward the front, "
        "and a clean, neat finish"
    ),
    "Classic Undercut": (
        "a classic undercut with very short disconnected sides and back, "
        "clearly longer medium-length hair on top, "
        "with the top combed straight back and a sharp separation from the sides"
    ),
    "Slick Back": (
        "a slick back with medium-length hair swept directly backward "
        "from the forehead, a smooth controlled top, "
        "and short tapered sides with no visible fringe"
    ),
    "Textured Fringe": (
        "a textured fringe with short-to-medium layered hair on top, "
        "a clearly visible choppy irregular fringe across the forehead, "
        "and deliberately separated textured strands with natural movement"
    ),
    "Comb Over": (
        "a classic comb over with medium-length hair on top, "
        "a clearly defined side part, hair swept horizontally across the top, "
        "and short faded sides with a polished structured finish"
    ),
    "Bro Flow": (
        "a medium-to-long Bro Flow with hair grown past the ears and toward the jaw, "
        "flowing naturally backward and outward from the crown, "
        "with medium-length sides and a relaxed, natural finish"
    ),
    "Wolf Cut": (
        "a men's Wolf Cut with heavy shaggy layers, "
        "a textured curtain fringe framing the forehead, "
        "noticeable crown volume, and longer wispy layered lengths at the back"
    ),
    "Modern Mullet": (
        "a modern mullet with short textured hair on top and at the sides, "
        "a short textured fringe at the front, "
        "and distinctly longer layered hair extending down the back of the neck"
    ),
    "Curtain Bangs": (
        "men's curtain bangs with a clear center part, "
        "the fringe splitting into two sections and falling softly on both sides "
        "of the forehead, with medium-length hair and natural face-framing movement"
    ),
}
STYLE_PROMPTS_FEMALE = {
    "French Bob": (
        "a French Bob with a blunt jaw-length cut, "
        "sleek smooth hair, and a straight heavy fringe across the forehead "
        "ending above the eyebrows"
    ),
    "Beach Waves": (
        "medium-length beach waves reaching the shoulders or collarbone, "
        "with loose irregular waves throughout the hair, "
        "natural texture, soft volume and an effortless finish"
    ),
    "Layered Medium": (
        "shoulder-length layered hair with multiple soft blended layers, "
        "shorter face-framing layers around the front, "
        "and feathered ends with visible movement"
    ),
    "Classic Updo": (
        "a classic formal updo with all hair gathered and pinned at the crown "
        "or back of the head, a smooth polished surface, "
        "and a clean structured silhouette"
    ),
    "Soft Bun": (
        "a relaxed low soft bun gathered at the nape, "
        "with slightly loose volume and a natural soft texture, "
        "plus a few delicate face-framing strands around the sides"
    ),
    "Curtain Fringe Medium": (
        "medium-length hair with a clear center-parted curtain fringe, "
        "the fringe falling softly on both sides of the forehead "
        "and framing the face, with subtle soft layers and natural movement"
    ),
    "Pixie Cut": (
        "a short pixie cut with closely tapered sides and nape, "
        "a slightly longer textured top, "
        "and a short feminine silhouette with no length near the jaw"
    ),
    "Lob": (
        "a long bob ending around the collarbone, "
        "with clean blunt or lightly layered ends, "
        "subtle natural movement and a polished medium-length silhouette"
    ),
    "Braided Crown": (
        "a braided crown with two visible braids wrapping around the top "
        "of the head like a halo, pinned into place around the crown, "
        "with the braids clearly visible as the defining feature"
    ),
    "Wolf Cut": (
        "a women's Wolf Cut with heavy shaggy layers, "
        "a prominent curtain fringe framing the forehead, "
        "noticeable volume around the crown, "
        "and longer wispy layered lengths flowing toward the shoulders"
    ),
    "Soft Shag": (
        "a soft shag with medium-length feathered layers, "
        "a relaxed curtain fringe framing the forehead, "
        "soft crown volume and a natural lived-in texture"
    ),
    "Butterfly Cut": (
        "a long butterfly cut with dramatic face-framing layers, "
        "shorter interior layers and longer outer lengths, "
        "with the mid-length layers visibly flipping outward "
        "to create a soft wing-like silhouette"
    ),
}

COLOR_PROMPTS = {
    "natural": "",
    "blonde": "dark blonde, bright and even",
    "dark": "rich dark brown, deep and natural",
    "black": "jet black, very dark and glossy",
    "auburn": "to warm auburn red, natural reddish-brown",
}

def build_prompt(style_name: str, color_id: str, gender: str = "Man") -> str:
    prompts = STYLE_PROMPTS_FEMALE if gender == "Woman" else STYLE_PROMPTS_MALE
    style = prompts.get(style_name) or STYLE_PROMPTS_MALE.get(style_name) or f"{style_name} hairstyle"
    color = COLOR_PROMPTS.get(color_id, "")
    
    prompt = (
        f"Edit only the hair of the person in the input image. "
        f"Change the hairstyle to {style}. "
    )

    if color:
        prompt += f"Also change the hair color to {color}. "
    
    prompt += (
        "Preserve the exact identity and facial appearance of the person. "
        "Do not alter the face, facial proportions, eyes, eyebrows, nose, mouth, "
        "ears, skin, facial hair, expression, clothing, lighting or background."
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