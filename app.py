import streamlit as st
import cv2
import numpy as np
import pandas as pd
from src.landmarks import FaceLandmarkDetector
from src.pipeline import run_pipeline
from util.ui_components import trait_bar, style_card
from src.drawing import draw_landmarks, draw_geometry
from src.pdf_export import generate_pdf
from src.feedback import save_session, save_vote

def check_password():
    if st.session_state.get("authenticated"):
        return True
    
    st.title("Hairstyle AI Recommender")
    pwd = st.text_input("Enter Password", type="password")

    if st.button("Enter"):
        if pwd == st.secrets["password"]:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password")
    
    return False

if not check_password():
    st.stop()

@st.cache_resource
def load_detector():
    return FaceLandmarkDetector(model_path="models/face_landmarker.task")

@st.cache_resource
def load_norms():
    male = pd.read_csv("data/norms/male_norms_v2.csv", index_col=0)
    female = pd.read_csv("data/norms/female_norms_v2.csv", index_col=0)
    return male, female

detector = load_detector()
norms, female_norms = load_norms()

def n(feature, gender=None):
    source = female_norms if gender == "Woman" else norms
    return {
        "min_val": float(source.loc["p5",  feature]),
        "max_val": float(source.loc["p95", feature]),
        "avg_val": float(source.loc["mean", feature]),
    }

st.title("Hairstyle AI Recommender")

gender = st.radio(
    "Select gender for accurate recommendations",
    ["Man", "Woman"],
    horizontal=True,
    key="gender_select"
)

uploaded = st.file_uploader("Upload a front-facing photo", type=["jpg", "png"])

if uploaded:
    if "last_file" not in st.session_state or \
    st.session_state["last_file"] != uploaded.name:
        st.session_state["session_saved"] = False
        st.session_state["feedback_saved"] = False
        st.session_state["last_file"] = uploaded.name
        st.session_state["displayed_styles"] = None
        st.session_state["queue"] = None
        st.session_state["votes"] = {}

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    landmarks, features, traits, scores, recs, quality = run_pipeline(img, detector, gender=gender)

    if landmarks is None:
        if quality is not None and quality.blocking:
            st.error(f"⚠️ {quality.blocking}")
        else:
            st.error("No face detected - try a clearer front-facing photo")
        st.stop()
    
    det_col1, det_col2 = st.columns([1, 3])
    with det_col1:
        gender_icon = "👩" if gender == "Woman" else "👨"
        st.markdown(
            f'<div style="display:inline-flex;align-items:center;gap:6px;'
            f'font-size:13px;padding:4px 12px;border-radius:20px;'
            f'border:.5px solid #ddd;color:#555;margin-top:4px">'
            f'{gender_icon} {gender or "Unknown"} detected </div>',
            unsafe_allow_html=True
        )
    with det_col2:
        if quality is not None:
            score_pct = quality.score * 100
            bar_color = "#2d8f4e" if score_pct > 70 else \
                        "#e6a817" if score_pct > 40 else "#c0392b"
            st.markdown(
                f'<div style="padding-top:6px">'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:12px;color:#888;margin-bottom:3px">'
                f'<span>Detection confidence</span>'
                f'<span style="color:{bar_color};font-weight:500">'
                f'{score_pct:.0f}%</span></div>'
                f'<div style="height:4px;border-radius:2px;background:#eee">'
                f'<div style="width:{score_pct:.0f}%;height:100%;'
                f'border-radius:2px;background:{bar_color}"></div>'
                f'</div></div>',
                unsafe_allow_html=True
            )

    if quality is not None and quality.warnings:
        for w in quality.warnings:
            st.warning(f"⚠️ {w}")

    st.divider()

    st.subheader("Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Original image", channels="BGR", use_container_width=True)
    
    with col2:
        vis = draw_landmarks(img.copy(), landmarks, draw_indices=False)
        st.image(vis, caption="Detected landmarks", channels="BGR",
                use_container_width=True)

    st.divider()

    if recs["face_analysis"]:
        for exp in recs["face_analysis"]:
            st.markdown(
                f'<div style="font-size:13px;color:#555;padding:5px 0 5px 12px;'
                f'border-left:2px solid #ddd;margin-bottom:6px">{exp}</div>',
                unsafe_allow_html=True
            )
    else:
        st.caption("No notable traits detected — proportions are well balanced.")

    st.divider()

    st.subheader("Facial proportions")

    BARS = [
    ("face_ratio",       "Face shape",          "Wide face",        "Long face"),
    ("jaw_ratio",        "Jaw width",            "Narrow jaw",       "Wide jaw"),
    ("eye_ratio",        "Eye spacing",          "Close-set eyes",   "Wide-set eyes"),
    ("eye_height",       "Eye openness",         "Narrow eyes",      "Wide eyes"),
    ("lip_ratio",        "Lip width",            "Narrow lips",      "Wide lips"),
    ("nose_position",    "Nose position",        "High nose",        "Low nose"),
    ("lower_face_ratio", "Lower face length",    "Short lower face", "Long lower face"),
    ("chin_prominence",  "Chin prominence",      "Flat chin",        "Strong chin"),
    ("symmetry",         "Facial symmetry",      "Symmetrical",      "Asymmetrical"),
    ("upper_third",      "Forehead",             "Low forehead",     "High forehead"),
    ("middle_third",     "Mid face",             "Short mid face",   "Long mid face"),
    ("lower_third",      "Lower face thirds",    "Short lower",      "Long lower"),
]

    for feat, title, min_label, max_label in BARS:
        trait_bar(
            title=title,
            value=features[feat],
            min_label=min_label,
            max_label=max_label,
            **n(feat, gender)
        )

    st.divider()

    if st.session_state["displayed_styles"] is None:
        all_styles = recs["all_styles"]
        st.session_state["displayed_styles"] = all_styles[:3]
        st.session_state["queue"] = all_styles[3:]
    
    st.subheader("Top hairstyles")

    displayed = st.session_state["displayed_styles"]
    cols = st.columns(len(displayed))

    for i, style in enumerate(displayed):
        with cols[i]:
            vote = style_card(style, rank=i, card_key=f"{i}_{style['name']}")

            if vote is not None:
                save_vote(style["name"], vote, features, gender or "")

                if vote == "down" and st.session_state["queue"]:
                    next_style = st.session_state["queue"].pop(0)
                    st.session_state["displayed_styles"][i] = next_style
                st.rerun()

    st.divider()  

    if not st.session_state.get("session_saved"):
        save_session(features, quality.score, recs)
        st.session_state["session_saved"] = True 

    st.subheader("How accurate were our recommendations?")
    rating = st.feedback("stars", key="user_rating")

    if rating is not None:
        comment = st.text_input("Any comments? (optional)", key="user_comment")
        if st.button("Submit feedback", key="submit_feedback"):
            if not st.session_state.get("feedback_saved"):
                save_session(features, quality.score, recs,
                        rating=rating + 1,
                        comment=comment)
                st.session_state["feedback_saved"] = True

    if st.session_state.get("feedback_saved"):
        st.success("Thanks for your feedback!")

    st.subheader("Export")
    if st.button("📄 Generate PDF Report", key="generate_pdf"):
        pdf_bytes = generate_pdf(features, traits, recs, 
                                 female_norms if gender == "Woman" else norms)
        st.download_button(
            label="Save Report",
            data=pdf_bytes,
            file_name="hairstyle_report.pdf",
            mime="application/pdf",
            key="download_pdf",
            icon=":material/download:",
        )
