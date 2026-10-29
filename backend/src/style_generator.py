import os, json, base64, httpx
import fal_client

STYLE_PROMPTS_MALE = {
    "Buzz Cut": (
        "a Buzz Cut: uniformly very short hair all over — top, sides and back "
        "the same grade 1-2 clipper length. Extremely clean minimal silhouette."
    ),
    "Crew Cut": (
        "a Crew Cut: short tapered sides, very short top that is fractionally "
        "longer than the sides with natural texture. Clean masculine finish."
    ),
    "Comb Over": (
        "an elegant Comb Over: short faded sides and medium-length hair on top " 
        "with a sharp, clearly defined side part. All the hair on top is firmly "
        "combed and swept horizontally to one side in a clean, controlled arc "
        "that follows the shape of the head. The hair lies flat and smooth against "
        "the head — no loose strands, no drooping, no falling forward. "
        "Polished, structured and well-groomed."
    ),
    "Slick Back": (
        "a classic Slick Back: medium-length hair on top combed straight backward from the forehead in one continuous, unified direction."
        "The hair lies smoothly against the head with a controlled, elegant silhouette."
        "Short tapered sides. No side-swept fringe, no hair falling over the forehead, no separated front sections." 
        "Natural matte finish, not wet or glossy."
    ),
    "Quiff": (
        "Short tapered sides and a compact, textured top. "
        "At the front hairline, the hair rises upward from the forehead "
        "and curves gently backward, creating a rounded, elevated shape. "
        "Hair from both the left and right sides of the front hairline "
        "lifts upward and blends together near the center of the top. "
        "The front must have visible height and balanced volume on both sides. "
        "No sideways sweep, no side part, no flat fringe, no long back."
    ),
    "Classic Undercut": (
        "a Classic Undercut: very short sides and back, sharply disconnected "
        "from a moderately short top. The top is approximately 5-7 cm long "
        "and combed backward as one cohesive section. "
        "Keep the nape short, with no long hair extending down the back of the neck. "
        "Moderate top volume, close-fitting sides, no flowing or shoulder-length hair. "
        "The sharp contrast between the top and sides is essential."
    ),
    "Textured Fringe": (
        "a Textured Fringe: short-to-medium hair on top with a naturally "
        "choppy fringe resting across the forehead. "
        "The top has clearly visible piecey texture and separation between "
        "individual locks, while remaining soft and realistic. "
        "Short tapered sides. Avoid exaggerated spikes or an excessively messy finish."
    ),
    "Messy Crop": (
        "a short Messy Crop: a compact haircut with a top approximately 2-3 cm long, visibly shorter than a medium-length hairstyle. " 
        "The top has subtle, rough natural texture and is styled slightly forward. The sides and back are short and tapered." 
        "Keep the overall silhouette close to the head. No long layers, no flowing hair, no extended fringe."

    ),
    "French Crop": (
        "a French Crop: faded sides, textured top clearly longer than the sides, "
        "and a fringe resting naturally on the forehead. "
        "The fringe touching the forehead is essential."
    ),
    "Wolf Cut": (
        "a Wolf Cut: shaggy layers throughout, curtain fringe on both sides of the forehead, "
        "volume at the crown, wispy layered ends. "
        "Keep the sides at medium length, not too long or exaggerated."
    ),
    "Modern Mullet": (
        "a Modern Mullet: short top and faded sides with a short fringe, "
        "and the back reaching the lower neck. "
        "The back should be longer than the top but not dramatic or rock-style. "
        "Contemporary and compact."
    ),
    "Layered Medium": (
        "a medium-length layered haircut with natural layers and soft movement. "
        "The hair has visible volume and texture, with slightly shorter layers around "
        "the face and longer layers toward the back. Natural, relaxed finish, no fringe."
    ),
    "Curtain Bangs": (
        "medium-length hair with a soft natural centre part. "
        "The front hair is divided into two loose sections — "
        "one falling softly across the left side of the forehead "
        "and one across the right side, each section grazing the eyebrow on its side. "
        "The top of the forehead is visible between the two sections. "
        "The rest of the hair flows naturally to the sides. "
        "Relaxed and slightly wispy, not theatrical or forced."
    ),
    "Bro Flow": (
        "Medium-to-long hair extending past the ears, flowing naturally "
        "backward and slightly outward from the crown. "
        "Loose, continuous layers with natural movement and moderate volume. "
        "The sides remain medium-length, without a fade or tight taper. "
        "The back follows the natural shape of the head and neck. "
        "Relaxed, natural finish, without exaggerated length or volume."
    ),
    "Long Straight": (
        "long straight hair reaching the shoulders or slightly past. "
        "Natural texture with realistic ends — not too perfect or artificial. "
        "Subtle movement."
    ),
}

STYLE_PROMPTS_FEMALE = {
    "Pixie Cut": (
        "a modern Pixie Cut: very short all over with slightly longer top. "
        "Fresh, youthful and edgy — not matronly. "
        "Clean tapered nape and sides."
    ),
    "Bob Classic": (
        "a classic Bob haircut ending around the jawline or slightly below, "
        "with a soft rounded outline and naturally curved ends. "
        "The forehead remains fully visible, with no fringe or bangs. "
        "Natural hair texture with subtle strand variation and gentle volume. "
        "The hair should look integrated with the original photograph, "
        "not overly smooth, flat, slicked down or wig-like."
    ),
    "Textured Bob": (
        "a Textured Bob: jaw-length hair with soft layers and natural movement. "
        "Slightly undone, not sleek or blunt."
    ),
    "French Bob": (
        "a French bob haircut, cut around the jawline with soft, slightly tousled texture "
        "and natural volume. Short, wispy fringe falling softly across the forehead, "
        "with slightly uneven natural ends. Effortless, textured and lived-in, "
        "not perfectly straight or geometric."
    ),
    "Wolf Cut": (
        "a natural women's Wolf Cut with shaggy layers throughout the hair, "
        "soft volume around the crown and wispy layered ends. "
        "Natural face-framing layers that follow the shape of the face without covering "
        "or flattening it. Relaxed, textured and effortless."
    ),
    "Lob": (
        "a Lob (Long Bob): hair ending at collarbone length. "
        "Clean ends with subtle movement. No fringe."
    ),
     "Soft Shag": (
        "a Soft Shag: medium-length feathered layers with a relaxed curtain fringe, "
        "soft crown volume and lived-in texture."
    ),
   "Curtain Fringe Medium": (
        "medium-length hair with a natural curtain fringe. "
        "The fringe falls softly to each side of the forehead, "
        "leaving the centre of the forehead visible. Natural and slightly wispy."
    ),
    "Layered Medium": (
        "shoulder-length layered hair with soft layers and natural movement. "
        "No fringe or bangs — forehead fully exposed. Face-framing front layers."
    ),
    "Blunt Cut Medium": (
        "a medium-length blunt haircut ending at the shoulders, with a clean, even perimeter and straight ends." 
        "The hair is worn down, with a simple natural shape and subtle movement. No fringe, no bangs, no short front pieces falling across the forehead." 
        "Keep the forehead fully visible and the front hair length consistent with the rest of the haircut."
    ),
    "Butterfly Cut": (
        "a Butterfly Cut: long hair with layers that flip gently outward at mid-length, "
        "creating a soft wing-like shape. Natural and not exaggerated."
    ),
    "Long Layers": (
        "long layered hair past the shoulders with soft flowing layers. "
        "Natural movement and volume."
    ),
    "Long Straight Blunt": (
        "a long straight blunt haircut ending well below the shoulders, "
        "with a clean even perimeter and straight ends. "
        "The hair is worn down with a simple natural shape. "
        "No fringe, no bangs, no short front pieces falling across the forehead. "
        "The forehead remains fully visible and the front hair length "
        "matches the rest of the haircut."
    ),
    "Long with Curtain Fringe": (
        "long hair with a soft natural curtain fringe. "
        "The fringe falls to each side of the forehead, "
        "leaving the centre exposed. Natural and wispy."
    ),
    "Beach Waves": (
        "medium-to-long beach waves reaching past the shoulders. "
        "Loose natural waves throughout. Not too short."
    ),
    "Classic Updo": (
        "an elegant chignon or classic bun: all hair swept up and gathered "
        "into a smooth rounded bun positioned at the back of the head "
        "or at the nape of the neck. "
        "The bun should be clearly visible as a round knot of hair. "
        "Surface is smooth and polished. No loose strands falling down. "
        "The rounded bun shape at the back is essential."
    ),
    "Soft Bun": (
        "a Soft Bun: hair loosely gathered into a low bun at the nape, "
        "slightly puffed and relaxed with a few soft face-framing pieces left out. "
        "Casual and romantic, not tight or slicked."
    ),
    "Braided Crown": (
        "a Braided Crown: two braids wrapping around the top of the head "
        "like a halo, pinned in place. The braids are the visible focal element. "
        "Nape and back of head exposed."
    ),
}

COLOR_PROMPTS = {
"natural": None,
"blonde": "blonde",
"dark": "dark brown",
"black": "black",
"auburn": "auburn",
}

def build_prompt(style_name: str, color_id: str, gender: str = "Man") -> str:
    prompts = STYLE_PROMPTS_FEMALE if gender == "Woman" else STYLE_PROMPTS_MALE
    style = prompts.get(style_name) or STYLE_PROMPTS_MALE.get(style_name) or f"{style_name} hairstyle"
    color = COLOR_PROMPTS.get(color_id)

    prompt = (
        "Edit the hairstyle of the person in the input photograph. "
        f"Change the hairstyle to {style}. "
    )

    if color:
        prompt += f"Change the hair color to {color}. "
    else:
        prompt += "Preserve the original hair color. "

    prompt += (
        "Preserve the person's exact identity and facial geometry, including "
        "face shape, jawline, chin, nose, eyes and eye spacing. "
        "Preserve the original expression, gaze, skin tone, skin texture, "
        "eye color, ears, clothing, lighting, background. "
        "IMPORTANT: The face must remain pixel-perfect identical to the input image. "
        "Do not add, remove or change any facial hair, beard, stubble or moustache — "
        "if the person is clean-shaven, the result must be clean-shaven. "
        "Do not beautify, reshape, smooth or retouch the face in any way. "
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