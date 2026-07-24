import cv2
import os
import sys
sys.path.insert(0, "backend")
from src.hair_segmentation import segment_face
from src.style_generator import generate_style_preview

IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else "dataset/test_images/fotka_test.jpg"

TEST_STYLES = [
    "French Crop",
    "Messy Crop", 
    "Buzz Cut",
]

img = cv2.imread(IMG_PATH)
if img is None:
    print(f"Could not read {IMG_PATH}")
    sys.exit(1)

h, w = img.shape[:2]
if max(h, w) > 512:
    scale = 512 / max(h, w)
    img   = cv2.resize(img, (int(w * scale), int(h * scale)))

print(f"Image: {img.shape}")

print("Segmenting hair...")
hair_mask, _ = segment_face(img)
if hair_mask is None:
    print("Hair segmentation failed")
    sys.exit(1)

print(f"Hair coverage: {(hair_mask > 0).mean():.3f}")

cv2.imwrite("debug_mask.png", hair_mask)
print("Saved debug_mask.png")

os.makedirs("output_previews", exist_ok=True)

for style_name in TEST_STYLES:
    print(f"\nGenerating: {style_name}...")
    result = generate_style_preview(img, hair_mask, style_name)
    
    if result is None:
        print(f"  FAILED for {style_name}")
        continue
    
    canvas = cv2.hconcat([img, result])
    out_path = f"output_previews/{style_name.replace(' ', '_')}.jpg"
    cv2.imwrite(out_path, canvas)
    print(f"  Saved: {out_path}")

print("\nDone — check output_previews/ folder")