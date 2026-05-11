from deepface import DeepFace

def detect_gender(img) -> str:
    try:
        result = DeepFace.analyze(img, actions=["gender"],
                                  enforce_detection=False)
        return result[0]["dominant_gender"]
    except Exception:
        return "Man"