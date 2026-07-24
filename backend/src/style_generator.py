import requests
import base64
import os
import cv2
import numpy as np

HF_TOKEN = os.environ.get("HF_TOKEN")
INPAINT_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-inpainting"

STYLE_PROMPTS = {
    "French Crop":      "professional photo, french crop hairstyle, short textured hair with fringe, clean fade sides, natural lighting",
    "Messy Crop":       "professional photo, messy crop hairstyle, short textured top, natural effortless look",
    "Textured Fringe":  "professional photo, textured fringe hairstyle, choppy fringe over forehead, natural texture",
    "Buzz Cut":         "professional photo, buzz cut hairstyle, very short uniform hair, clean look",
    "Curly Volume":     "professional photo, natural curly hair, volume and definition, embraced curl pattern",
    "French Bob":       "professional photo, french bob hairstyle, chin length blunt cut, straight fringe",
    "Curtain Fringe Medium": "professional photo, curtain fringe hairstyle, centre parted fringe, medium length soft layers",
    "Layered Medium":   "professional photo, layered medium hairstyle, shoulder length with movement, soft layers",
    "Beach Waves":      "professional photo, beach waves hairstyle, loose natural waves, effortless texture",
    "Classic Updo":     "professional photo, classic updo hairstyle, elegant pinned up style, smooth finish",
}

NEGATIVE_PROMPT = "blurry, distorted face, changed face, different person, bad anatomy, ugly, deformed, low quality"

def generate_style_preview(img_bgr, hair_mask, style_name):
    if not HF_TOKEN:
        return None

    prompt = STYLE_PROMPTS.get(style_name)
    if not prompt:
        return None

    h, w = img_bgr.shape[:2]

    _, img_buf  = cv2.imencode(".png", img_bgr)
    _, mask_buf = cv2.imencode(".png", hair_mask)

    img_b64  = base64.b64encode(img_buf.tobytes()).decode()
    mask_b64 = base64.b64encode(mask_buf.tobytes()).decode()

    payload = {
        "inputs": prompt,
        "parameters": {
            "image": img_b64,
            "mask_image": mask_b64,
            "negative_prompt": NEGATIVE_PROMPT,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "strength": 0.85,
        }
    }

    try:
        res = requests.post(
            INPAINT_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json=payload,
            timeout=30,
        )
        if res.status_code != 200:
            print(f"HF API error {res.status_code}: {res.text[:500]}")
            return None

        result_bytes = res.content
        arr = np.frombuffer(result_bytes, np.uint8)
        img_result = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img_result

    except Exception as e:
        print(f"Style generation error: {e}")
        return None