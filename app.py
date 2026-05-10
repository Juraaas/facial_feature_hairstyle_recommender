import streamlit as st
import cv2
import numpy as np
import pandas as pd
from src.landmarks import FaceLandmarkDetector
from src.pipeline import run_pipeline
from src.drawing import draw_landmarks, draw_geometry, draw_features
from ui_components import trait_bar
from src.pdf_export import generate_pdf

detector = FaceLandmarkDetector(model_path="models/face_landmarker.task")
norms = pd.read_csv("male_norms_p123.csv", index_col=0)

def n(feature):
    return {
        "min_val": float(norms.loc["p5",  feature]),
        "max_val": float(norms.loc["p95", feature]),
        "avg_val": float(norms.loc["mean", feature]),
    }

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

    trait_bar(title="Face shape", value=features["face_ratio"],
        min_label="Wide face", max_label="Long face",
        **n("face_ratio"))

    trait_bar(title="Jaw width", value=features["jaw_ratio"],
        min_label="Narrow jaw", max_label="Wide jaw",
         **n("jaw_ratio"))

    trait_bar(title="Eye spacing", value=features["eye_ratio"],
        min_label="Close-set eyes", max_label="Wide-set eyes",
        **n("eye_ratio"))

    trait_bar(title="Eye openness", value=features["eye_height"],
        min_label="Narrow eyes", max_label="Wide eyes",
        **n("eye_height"))

    trait_bar(title="Lip width", value=features["lip_ratio"],
        min_label="Narrow lips", max_label="Wide lips",
        **n("lip_ratio"))

    trait_bar(title="Nose position", value=features["nose_position"],
        min_label="High nose", max_label="Low nose",
        **n("nose_position"))

    trait_bar(title="Lower face length", value=features["lower_face_ratio"],
        min_label="Short lower face", max_label="Long lower face",
        **n("lower_face_ratio"))

    trait_bar(title="Chin prominence", value=features["chin_prominence"],
        min_label="Flat chin", max_label="Strong chin",
        **n("chin_prominence"))

    sym_n = n("symmetry")
    trait_bar(title="Facial symmetry", value=features["symmetry"],
        min_label="Symmetrical", max_label="Asymmetrical",
        **sym_n)
    
    st.subheader("Top Hairstyles")

    for style in recs["top_styles"]:
        st.markdown(f"## {style['name']} — {style['score']:.1f} points")

        if "image" in style:
            st.image(style["image"], width=300)

        if style["contributions"]:
            st.markdown("**Why it works for you:**")
            top2 = style["contributions"][:2]
            cols = st.columns(2)
            for i, c in enumerate(top2):
                with cols[i]:
                    st.metric(label=c["desc"], value=f"{c['percent']*100:.0f}%")

        if style["negatives"]:
            with st.expander("Potential drawbacks"):
                for c in style["negatives"][:2]:
                    st.write(f"• **{c['desc']}** — {c['reason']}")

        st.divider()

    st.subheader("Export")
    if st.button("📄 Generate PDF Report", key="generate_pdf"):
        pdf_bytes = generate_pdf(features, traits, recs, norms)
        st.download_button(
            label="Save Report",
            data=pdf_bytes,
            file_name="hairstyle_report.pdf",
            mime="application/pdf",
            key="download_pdf",
            icon=":material/download:",
        )
