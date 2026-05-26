import cv2
import numpy as np

_app = None

def get_app():
    global _app
    if _app is None:
        import insightface
        _app = insightface.app.FaceAnalysis(providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app

def detect_gender(img) -> str:
    try:
        app = get_app()
        faces = app.get(img)
        if not faces:
            return None
        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        return "Woman" if face.gender == 0 else "Man"
    except Exception:
        return None