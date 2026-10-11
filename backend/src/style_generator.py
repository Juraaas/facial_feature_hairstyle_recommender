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
        "a classic Comb Over: moderately short sides with a subtle fade, and medium-short hair on top" 
        "A clearly visible side part separates the top into two sections. The longer section is combed horizontally across to one side, following the shape of the head." 
        "The hair must sweep sideways, not straight backward. Natural, neat finish without excessive volume."
    ),
    "Slick Back": (
        "a classic Slick Back: medium-length hair on top combed straight backward from the forehead in one continuous, unified direction."
        "The hair lies smoothly against the head with a controlled, elegant silhouette."
        "Short tapered sides. No side-swept fringe, no hair falling over the forehead, no separated front sections." 
        "Natural matte finish, not wet or glossy."
    ),
    "Quiff": (
        "a natural Quiff: the front section of the hair is directed slightly forward at the roots and then lifted upward and gently backward," 
        "forming one cohesive, rounded volume at the front hairline. The front is visibly raised but not excessively tall." 
        "The top blends naturally into the rest of the hair. Short tapered sides. "
        "No separate spikes, no split front sections, no side-swept fringe."
    ),
    "Classic Undercut": (
        "a Classic Undercut: very short sides with a clear, abrupt disconnection from the medium-length top." 
        "The top forms one cohesive section and flows backward without a prominent side part. The contrast between the short sides and longer top is clearly visible." 
        "Keep the top controlled and natural, not excessively tall or glossy."
    ),
    "Textured Fringe": (
        "a Textured Fringe: short-to-medium hair on top with a soft, naturally irregular fringe resting across the forehead." 
        "The ends are lightly separated and slightly uneven, with subtle texture rather than exaggerated spikes." 
        "Short tapered sides. The hair should look naturally styled and touchable, not heavily choppy, messy or over-textured."
    ),
    "Messy Crop": (
        "a short Messy Crop: a compact haircut with a top approximately 2-3 cm long, visibly shorter than a medium-length hairstyle." 
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
        "a Layered Medium haircut: hair ending around the chin or jaw, "
        "with visible soft layers and natural movement. "
        "Medium length, not long. No fringe."
    ),
    "Curtain Bangs": (
        "men's Curtain Bangs: medium-length hair with the front split into two sections, "
        "left half falling to the left side of the forehead, "
        "right half falling to the right side. "
        "Natural relaxed look, not perfectly symmetrical or forced."
    ),
    "Bro Flow": (
        "a Bro Flow: medium-to-long hair grown past the ears, flowing naturally backward and outward from the crown." 
        "The sides remain medium length, without a fade or tight taper. Relaxed, natural movement and realistic hair texture." 
        "Preserve the original facial hair exactly; do not add, remove or alter a beard, moustache, stubble or clean-shaven appearance."
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
        "a classic Bob ending just below the jaw, smooth and rounded. "
        "No fringe, forehead fully exposed. Natural hair texture."
        "a classic Bob haircut ending around the jawline or slightly below it, with a smooth, rounded outline and naturally inward-curving ends."
        "No fringe or bangs: the forehead remains fully visible. The haircut frames the sides of the face without changing or obscuring its shape." 
        "Natural hair texture, realistic volume and a balanced silhouette."
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
        "long straight hair with a clean blunt cut at the ends. "
        "No fringe — forehead fully exposed. Smooth and natural."
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
"blonde": "natural-looking blonde hair",
"dark": "natural-looking dark brown hair",
"black": "natural-looking black hair",
"auburn": "natural-looking auburn hair",
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
        "IMPORTANT: Preserve the person's exact identity and facial geometry, including face shape, face width-to-height ratio, "
        "jawline, chin, nose, eyes, eye spacing and natural skin texture. "
        "ears, skin, expression, clothing, lighting or background. "
        "Preserve the original facial expression, gaze, background, lighting, skin tone, eye color and existing facial hair. "
        "Do not add, remove or alter facial hair. Preserve the original hair color unless a different color is explicitly requested."
        "Keep the result photorealistic and consistent with the original photograph. Do not beautify, reshape, smooth or retouch the face."
        "The face must be pixel-perfect identical to the input. "
        "Do not change the image dimensions, aspect ratio or crop the image. "
        "The output image must be the same size and framing as the input. "
        "Only the hair on top of the head changes."
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