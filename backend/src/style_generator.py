import os, json, base64, httpx
import fal_client

STYLE_PROMPTS_MALE = {
    "Buzz Cut": (
        "a Buzz Cut: uniformly very short hair all over — top, sides and back "
        "the same grade 1-2 clipper length. Extremely clean minimal silhouette."
    ),
    "Crew Cut": (
        "a classic Crew Cut with short sides and a short, slightly longer top. "
        "The top has a natural, subtle texture and gradually tapers toward the back. "
        "Clean and masculine, with a natural finish."
    ),
    "Comb Over": (
        "a Comb Over: medium-length hair on top with a clean side part, "
        "hair swept horizontally across. Skin fade on the sides. "
        "The defined side part and horizontal sweep are essential."
    ),
    "Slick Back": (
        "a Slick Back hairstyle: medium-length hair swept backward "
        "from the forehead in a natural relaxed way — not wet-look or overly glossy. "
        "Matte or low-sheen finish. Short tapered sides. No fringe. "
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
    "Textured Fringe": (
        "a Textured Fringe: choppy irregular fringe cut across the forehead "
        "with deliberately jagged uneven ends. Layered textured top with natural movement. "
        "The choppy irregular fringe is essential."
    ),
    "Messy Crop": (
        "a Messy Crop: short disconnected sides, medium-length textured top "
        "with piece-y uneven texture and natural forward movement, "
        "relaxed undone finish with no fringe."
    ),
    "French Crop": (
        "a French Crop: faded sides, textured top clearly longer than the sides, "
        "and a fringe resting naturally on the forehead. "
        "The fringe touching the forehead is essential."
    ),
    "Wolf Cut": (
        "a Wolf Cut: heavy shaggy layers throughout, curtain fringe falling "
        "on both sides of the forehead, strong volume at the crown, "
        "longer wispy layered ends at the back. "
        "The curtain fringe and shaggy crown volume are essential."
    ),
    "Modern Mullet": (
        "a Modern Mullet: the top has short textured hair styled slightly forward, "
        "the sides are faded or tapered short, a short textured fringe at the front, "
        "and a clearly longer but controlled layered back reaching the lower neck. "
        "The overall silhouette should be compact, clean and contemporary, "
    ),
    "Layered Medium": (
        "a medium-length layered haircut with natural layers and soft movement. "
        "The hair has visible volume and texture, with slightly shorter layers around "
        "the face and longer layers toward the back. Natural, relaxed finish."
    ),
    "Curtain Bangs": (
        "men's Curtain Bangs: medium-length hair, with the front hair divided into two separate sections "
        "that each fall to their respective side of the forehead, "
        "one section falling left and one falling right, "
        "with a visible gap or parting between them at the top of the forehead. "
        "Each side drapes softly and naturally toward the temple. "
        "The two-section split at the front with hair falling on both sides is essential — "
        "this is not a fringe that falls straight across."
    ),
    "Bro Flow": (
        "a Bro Flow: hair grown past the ears toward the jaw, "
        "flowing naturally backward and outward from the crown. Medium length on sides -"
        "no fade, no taper. Relaxed natural finish."
        "The hair should look effortless and natural, not styled."
        "Preserve the person's face and facial hair exactly as in the original photo. "
        "Do not add, remove, darken or alter facial hair. "
        "If the original face is clean-shaven, the result must remain completely clean-shaven. "
        "No stubble, no beard, no moustache, no facial hair shadow."
    ),
    "Long Straight": (
        "long straight hair reaching past the shoulders, with a clean natural shape "
        "and subtle movement. The hair remains straight and smooth with natural volume "
        "and realistic texture."
    ),
}

STYLE_PROMPTS_FEMALE = {
    "Pixie Cut": (
        "a Pixie Cut: very short all over — 1-2 inches. "
        "Closely tapered at nape and sides, slightly longer textured top. "
        "Significantly shorter than a bob — no hair near jaw length."
    ),
    "Textured Bob": (
        "a Textured Bob: jaw-length hair with soft layers and natural movement "
        "throughout. Slightly undone finish with visible texture and lightness. "
        "Not sleek or blunt — the texture and movement are essential."
    ),
    "Bob Classic": (
        "a classic bob haircut, ending around the jawline and slightly below it, "
        "with a smooth rounded shape and natural inward-curving ends. "
        "No fringe or bangs, with the forehead fully visible. "
        "Natural, soft and realistic hair texture."
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
        "a Lob (long bob): hair ending at collarbone length, "
        "clean blunt or lightly layered ends, subtle natural movement. "
        "Longer than a bob, shorter than shoulder length."
    ),
    "Soft Shag": (
        "a Soft Shag: medium-length feathered layers throughout, "
        "relaxed curtain fringe framing the forehead, soft crown volume, "
        "natural lived-in texture. "
        "The curtain fringe and visible feathered layering are essential."
    ),
   "Curtain Fringe Medium": (
        "medium-length soft shag with curtain bangs. "
        "A clear center part at the forehead creates an open V-shaped fringe: "
        "the hair starts at the center of the forehead and sweeps diagonally outward "
        "to both sides, framing the face. "
        "The middle of the forehead remains visible. "
        "The two sides of the fringe blend into the layered hair naturally, "
        "with soft wispy texture and movement."
    ),
    "Layered Medium": (
        "Layered Medium hair: shoulder-length with multiple soft blended layers, "
        "shorter face-framing layers at the front, feathered ends with visible movement."
    ),
    "Blunt Cut Medium": (
        "a medium-length blunt cut ending around the shoulders, with a clean, even "
        "perimeter and straight ends. Smooth, natural hair with subtle movement and volume."
    ),
    "Butterfly Cut": (
        "a Butterfly Cut hairstyle on a woman: long hair with soft face-framing layers "
        "that gently flip or curve outward at mid-length on both sides, "
        "creating a light airy silhouette. "
    ),
    "Long Layers": (
        "long layered hair with soft, flowing layers that add natural movement and volume. "
        "The hair remains long, with subtle face-framing layers and natural texture."
    ),
    "Long Straight Blunt": (
        "long straight hair reaching below the shoulders with a clean, blunt perimeter. "
        "The ends are even and full, while the rest of the hair remains smooth and straight "
        "with natural texture and volume."
    ),
    "Long with Curtain Fringe": (
        "long hair with prominent curtain bangs and a clear center part. "
        "The fringe opens from the center of the forehead and falls diagonally "
        "outward on both sides of the face, creating two distinct face-framing sections. "
        "The center of the forehead remains clearly visible. "
        "The bangs gradually blend into the long face-framing layers, "
        "with soft wispy texture, natural movement and relaxed volume."
    ),
    "Beach Waves": (
        "Beach Waves: loose irregular waves throughout medium-length hair "
        "reaching the shoulders. Not tight curls — relaxed effortless waves "
        "with natural texture and soft volume."
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
        "IMPORTANT: Preserve the exact identity and facial appearance of the person. "
        "Do not alter the face, facial proportions, eyes, eyebrows, nose, mouth, "
        "ears, skin, expression, clothing, lighting or background. "
        "Do not add, remove or change any facial hair or beard. "
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