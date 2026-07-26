import google.generativeai as genai
import base64
import os

os.environ["GEMINI_API_KEY"] = ""

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

with open("dataset/test_images/fotka_test.jpg", "rb") as f:
    img_bytes = f.read()

img_b64 = base64.b64encode(img_bytes).decode()

from google import genai as genai2
client = genai2.Client(api_key=os.environ["GEMINI_API_KEY"])

interaction = client.interactions.create(
    model="gemini-3.1-flash-image",
    input=[
        {
            "type": "image",
            "data": img_b64,
            "mime_type": "image/jpeg"
        },
        {
            "type": "text",
            "text": "Change only the hair to a french crop hairstyle with short textured top and fringe. Keep the person's face, skin tone, background and everything else exactly the same."
        }
    ],
)

with open("test_gemini_result.jpg", "wb") as f:
    f.write(base64.b64decode(interaction.output_image.data))

print("Saved test_gemini_result.jpg")