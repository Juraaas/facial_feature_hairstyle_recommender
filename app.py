import streamlit as st
import cv2
import numpy as np
from src.landmarks import FaceLandmarkDetector
from src.pipeline import run_pipeline
from src.drawing import draw_landmarks, draw_geometry, draw_features

detector = FaceLandmarkDetector(model_path="models/face_landmarker.task")

st.title("Hairstyle AI Recommender")
uploaded = st.file_uploader("Upload image", type=["jpg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    landmarks, features, traits, scores, recs = run_pipeline(img, detector)

    if landmarks is None:
        st.error("No face detected")
        st.stop()

    st.subheader("Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Original image", channels="BGR")
    
    with col2:
        vis = img.copy()
        st.image(vis, caption="Processed view (future overlay)", channels="BGR")

    st.subheader("Face Analysis")

    for exp in recs["face_analysis"]:
        st.write("•", exp)

    st.markdown("### Facial proportions (normalized)")

    def norm(v):
        return max(0.0, min(1.0, v))

    st.write("Face ratio")
    st.progress(norm(features["face_ratio"] / 1.5))

    st.write("Jaw ratio")
    st.progress(norm(features["jaw_ratio"]))

    st.write("Eye spacing")
    st.progress(norm(features["eye_ratio"]))

    st.write("Symmetry")
    st.progress(1 - norm(features["symmetry"] * 10))
    
    st.subheader("Top Hairstyles")

    for style in recs["top_styles"]:
        st.markdown(f"## {style['name']} — {style['score']:.1f} points")

        if "image" in style:
            st.image(style["image"], width=300)

        cols = st.columns(3)

        for i, c in enumerate(style["contributions"][:3]):
            with cols[i]:
                st.metric(
                    label=c["desc"],
                    value=f"{c['percent']*100:.1f}%"
                )

def render_axis(label_left, label_right, user_value, population_value, min_v=0.0, max_v=1.0):
    st.markdown(f"### {label_left} ↔ {label_right}")
    scale = st.columns(100)

    def to_index(v):
        v = max(min_v, min(max_v, v))
        return int((v - min_v) / (max_v - min_v) * 99)
    
    user_idx = to_index(user_value)
    pop_idx = to_index(population_value)

    for i in range(100):
        if i == user_idx and i == pop_idx:
            scale[i].markdown("🔵")
        elif i == user_idx:
            scale[i].markdown("🟢")
        elif i == pop_idx:
            scale[i].markdown("⚪")
        else:
            scale[i].markdown("·")
