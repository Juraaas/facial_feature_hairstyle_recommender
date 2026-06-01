import numpy as np
import cv2
from PIL import Image

HAIR_CLASS = 13
SKIN_CLASS = 1

_processor = None
_model = None

def load_segmentation_model():
    global _processor, _model
    if _processor is None: 
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        _processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        _model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
    return _processor, _model

def segment_face(img_bgr):
    try:
        import torch
        processor, model = load_segmentation_model()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        h, w = img_bgr.shape[:2]

        inputs = processor(images=pil_img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        upsampled = torch.nn.functional.interpolate(
            outputs.logits, size=(h, w), mode="bilinear", align_corners=False,
        )
        seg_map = upsampled.argmax(dim=1).squeeze().numpy()

        hair_mask = (seg_map == HAIR_CLASS).astype(np.uint8) * 255
        skin_mask = (seg_map == SKIN_CLASS).astype(np.uint8) * 255

        return hair_mask, skin_mask
    except Exception as e:
        return None, None
    
def find_hairline_y(hair_mask, face_width_px):
    if hair_mask is None:
        return None
    
    h, w = hair_mask.shape

    search_region = hair_mask[:h//2, :]
    hairline_points = []
    step = max(1, w // 40)

    for x in range(w // 4, 3 * w // 4, step):
        col = search_region[:, x]
        hair_pixels = np.where(col > 0)[0]
        if len(hair_pixels) > 0:
            hairline_points.append((x, int(np.max(hair_pixels))))
        
    if len(hairline_points) < 3:
        return None
    
    return int(np.median([p[1] for p in hairline_points]))

def get_hair_coverage(hair_mask):
    if hair_mask is None:
        return 0.0
    return np.sum(hair_mask > 0) / hair_mask.size