import streamlit as st

@st.cache_resource
def load_deepface():
    from deepface import DeepFace
    return DeepFace

def detect_gender(img) -> str:
    DeepFace = load_deepface()
    try:
        result = DeepFace.analyze(img, actions=["gender"],
                                  enforce_detection=False)
        return result[0]["dominant_gender"]
    except Exception:
        return "Man"