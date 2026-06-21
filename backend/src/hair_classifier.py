import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hair_classifier.onnx")

HAIR_TYPES = ["straight", "wavy", "curly", "coily"]
HAIRLINES = ["normal", "receding", "uneven"]

HAIR_TYPE_COVERAGE_MIN = 0.04
HAIRLINE_COVERAGE_MIN = 0.01

_session = None

def get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        model_path = os.path.join(os.path.dirname(__file__),
                                  MODEL_PATH)
        _session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
    return _session

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def prepare_crop(img_bgr, hair_mask, size=224):
    if hair_mask is None:
        return None
    coords = cv2.findNonZero(hair_mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    pad = 20
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_bgr.shape[1], x + w + pad)
    y2 = min(img_bgr.shape[0], y + h + pad)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size))

def prepare_hairline_crop(img_bgr, hair_mask, landmarks=None, size=224):
    h, w = img_bgr.shape[:2]
    if landmarks is not None:
        forehead_y = int(landmarks[10][1])
        top_bound = max(0, forehead_y - int(h * 0.15))
        bottom_bound = min(h, forehead_y + int(h * 0.1))
        region = img_bgr[top_bound:bottom_bound, :]
    else:
        region = img_bgr[:int(h * 0.4), :]
    if region.size == 0:
        return None
    return cv2.resize(region, (size, size))

def classify_hair(img_bgr, hair_mask, gender="Man", landmarks=None,
                  hair_conf_threshold=0.65, hairline_conf_threshold=0.85):
    if hair_mask is None:
        return {
            "hair_type": None,
            "hairline": None,
            "hair_conf": 0.0,
            "hairline_conf": 0.0,
        }

    coverage = np.sum(hair_mask > 0) / hair_mask.size
    session  = get_session()

    hair_type = None
    hair_conf = 0.0

    if coverage >= HAIR_TYPE_COVERAGE_MIN:
        crop = prepare_crop(img_bgr, hair_mask)
        if crop is not None:
            inp = _preprocess(crop)
            hair_logits, _ = session.run(None, {"input": inp})
            hair_probs = softmax(hair_logits[0])
            hair_conf = float(np.max(hair_probs))
            if hair_conf >= hair_conf_threshold:
                hair_type = HAIR_TYPES[np.argmax(hair_probs)]

    hairline = None
    hairline_conf = 0.0

    if gender != "Woman" and coverage >= HAIRLINE_COVERAGE_MIN:
        crop_hl = prepare_hairline_crop(img_bgr, hair_mask, landmarks=landmarks)
        if crop_hl is not None:
            inp = _preprocess(crop_hl)
            _, hairline_logits = session.run(None, {"input": inp})
            hairline_probs = softmax(hairline_logits[0])
            hairline_conf = float(np.max(hairline_probs))
            hairline_pred = HAIRLINES[np.argmax(hairline_probs)]

            if hairline_pred == "receding" and hairline_conf >= 0.90:
                hairline = "receding"
            elif hairline_pred in ("normal", "uneven") and hairline_conf >= hairline_conf_threshold:
                hairline = hairline_pred

    return {
        "hair_type": hair_type,
        "hairline": hairline,
        "hair_conf": round(hair_conf, 3),
        "hairline_conf": round(hairline_conf, 3),
        "coverage": round(float(coverage), 4),
    }

def _preprocess(crop_bgr):
    inp = crop_bgr[:, :, ::-1].astype(np.float32) / 255.0
    inp = (inp - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return inp.transpose(2, 0, 1)[None].astype(np.float32)
