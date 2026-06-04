import cv2
import numpy as np

_app = None

def get_app():
    global _app
    if _app is None:
        import insightface
        _app = insightface.app.FaceAnalysis(
            name="buffalo_sc",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "genderage"])
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app

def detect_gender(img) -> str:
    try:
        app = get_app()
        faces = app.get(img)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        sex = getattr(face, 'sex', None)
        if sex is not None:
            return "Woman" if sex == "F" else "Man"

        g = int(face.gender)
        return "Woman" if g == 0 else "Man"

    except Exception as e:
        print(f"DEBUG error: {e}")
        return None