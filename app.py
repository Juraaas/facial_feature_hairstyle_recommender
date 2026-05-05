import streamlit as st
import cv2
import numpy as np
from src.landmarks import FaceLandmarkDetector
from src.pipeline import run_pipeline
from src.drawing import draw_landmarks, draw_geometry, draw_features
from ui_components import trait_bar

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

    trait_bar(
        title="Face length",
        value=features["face_ratio"],
        min_val=0.9,
        max_val=1.6,
        avg_val=1.2,
        min_label="Wide face",
        max_label="Long face",
    )

    trait_bar(
        title="Jaw width",
        value=features["jaw_ratio"],
        min_val=0.65,
        max_val=1.05,
        avg_val=0.85,
        min_label="Narrow jaw",
        max_label="Wide jaw",
    )

    trait_bar(
        title="Eye spacing",
        value=features["eye_ratio"],
        min_val=0.30,
        max_val=0.80,
        avg_val=0.46,
        min_label="Close-set eyes",
        max_label="Wide-set eyes",
        min_sub="min: 0.30",
        max_sub="max: 0.80",
    )

    trait_bar(
        title="Lower face proportion",
        value=features["jaw_to_height"],
        min_val=0.50,
        max_val=0.80,
        avg_val=0.65,
        min_label="Short lower face",
        max_label="Long lower face",
    )

    trait_bar(
        title="Jaw projection",
        value=features["jaw_projection"],
        min_val=0.30,
        max_val=0.60,
        avg_val=0.45,
        min_label="Weak projection",
        max_label="Strong projection",
    )

    trait_bar(
        title="Nose position",
        value=features["nose_position"],
        min_val=0.40,
        max_val=0.60,
        avg_val=0.50,
        min_label="Upper dominant",
        max_label="Lower dominant",
    )

    sym = features["symmetry"]

    trait_bar(
        title="Facial symmetry",
        value=1 - (sym * 20),
        min_val=0.0,
        max_val=1.0,
        avg_val=0.7,
        min_label="Asymmetrical",
        max_label="Highly symmetrical",
    )
    
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
