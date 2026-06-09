# Face Geometry-Based Hairstyle Recommendation System

An explainable computer vision pipeline that analyzes facial geometry and recommends hairstyles based on anthropometric proportions, symmetry, and structural traits. Built around **interpretable features** and **transparent decision logic** — no black-box models.

> **Live demo:** [face-fit-ai.vercel.app](https://face-fit-ai.vercel.app)

---

## How it works
```
Image → Landmark Detection → Hair Segmentation → Geometric Feature Extraction → Trait Interpretation → Rule-Based Scoring → Ranked Recommendations
```
Each step is fully inspectable. Features are normalized ratios derived from MediaPipe Face Mesh landmarks, calibrated against a population dataset of ~21,000 faces (UTKFace). Recommendations include per-feature contribution scores and explicit drawback analysis.

---

## Key design decisions

**Geometry over deep learning** — all facial features are computed analytically from landmark coordinates (eye spacing ratio, jaw-to-face-width ratio, facial thirds, symmetry score etc.), making the system auditable and easy to iterate on.

**Hair segmentation for accurate facial thirds** — a SegFormer model (jonathandinu/face-parsing) detects the hairline from the photo, enabling accurate facial thirds analysis including the upper third (hairline→brow). Photos without a detectable hairline are excluded from norm calibration to maintain methodological consistency.

**Interaction-aware scoring** — rules encode not just individual traits but combinations: long face + high forehead triggers a stronger fringe recommendation than either trait alone. A separate symmetry modulation layer adjusts texture vs. clean-line styles based on facial asymmetry.

**Explainability layer** — each recommendation includes positive contributions (why it fits), negative contributions (what works against it), and missing features (what the style lacks that your analysis favours).

**Gender-aware pipeline** — automatic gender detection via InsightFace routes the analysis through separate trait classifiers, scoring rules, and hairstyle databases calibrated for male and female facial proportions.

**React + FastAPI architecture** — CV backend exposed via REST API, React frontend deployed independently. Clean separation enables smooth interactions, real-time overlays, and future generative features without framework constraints.

---

## Pipeline

### 1. Landmark Detection
MediaPipe Face Mesh (478 landmarks) with quality assessment — checks face size, yaw/pitch rotation, brightness, and landmark alignment confidence before processing.

### 2. Hair Segmentation
SegFormer face-parsing model detects the hairline position. Used to compute the upper facial third accurately. Falls back to landmark 10 if hairline detection fails (hat, poor lighting, extreme angle).

### 3. Geometric Feature Extraction
15 normalized ratios computed from landmark coordinates:

| Feature | Description |
|---|---|
| `face_ratio` | Face height / width |
| `jaw_ratio` | Jaw width / face width |
| `eye_ratio` | Inter-eye distance / face width |
| `eye_height` | Eye opening height / width |
| `lip_ratio` | Mouth width / face width |
| `nose_position` | Vertical nose position ratio |
| `lower_face_ratio` | Mouth-to-chin / face height |
| `chin_prominence` | Chin projection from jaw midpoint |
| `jaw_to_height` | Jaw width / face height |
| `symmetry` | Lateral landmark asymmetry score |
| `upper_third` | Hairline-to-brow / face height |
| `middle_third` | Brow-to-nose / face height |
| `lower_third` | Nose-to-chin / face height |
| `mid_lower_ratio` | Middle third / lower third |
| `thirds_balance` | Deviation from ideal 1:1:1 thirds ratio |

### 4. Trait Interpretation
Features are classified against population percentiles (p25/p75 as normal range boundaries) into semantic labels — e.g. `jaw: wide`, `eyes: close-set`, `face_length: long`, `forehead: high`, `facial_thirds: lower_dominant`. Separate classifiers for male and female norms.

### 5. Rule-Based Scoring
Weighted adjustments across hairstyle dimensions with interaction rules for trait combinations and symmetry modulation. Male pipeline: `volume_top`, `volume_sides`, `short_sides`, `longer_hair`, `fringe`, `clean_lines`, `soft_texture`, `textured_top`. Female pipeline adds `layers`, `updo`, `curtain_fringe`.

### 6. Hairstyle Matching
Scores are matched against a JSON database of 15 male / 16 female hairstyles, each annotated with attribute weights and descriptions. Final score is normalized by total attribute weight to ensure comparability across styles.

### 7. Explainability Layer
- Positive contributions with percentage influence
- Negative contributions with human-readable reasons
- Missing feature flags for strongly-favoured but absent attributes
- Face analysis summary from non-neutral traits

---

## Project structure
```
├── backend/
│   ├── main.py               # FastAPI app — analyse, vote, feedback endpoints
│   └── src/
│       ├── geometry.py       # landmark-based geometric calculations
│       ├── features.py       # feature extraction pipeline
│       ├── hair_segmentation.py  # SegFormer hairline detection
│       ├── face_traits.py    # male trait classification
│       ├── face_traits_female.py
│       ├── rules.py          # male scoring rules
│       ├── rules_female.py
│       ├── recommender.py    # scoring, matching, explainability
│       ├── pipeline.py       # end-to-end orchestration
│       ├── landmarks.py      # MediaPipe wrapper
│       ├── quality.py        # image quality assessment
│       ├── validation.py     # feature sanity checks
│       ├── gender.py         # InsightFace gender detection
│       ├── feedback.py       # session and vote logging (Google Sheets)
│       └── drawing.py        # visualization overlays
│
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── api/client.js     # fetch wrappers for backend endpoints
│       ├── hooks/useAnalysis.js
│       └── components/
│           ├── TraitBar.jsx
│           ├── StyleCard.jsx
│           ├── StylesSection.jsx
│           ├── FaceAnalysis.jsx
│           ├── FaceProportions.jsx
│           └── FeedbackSection.jsx
│
├── data/
│   ├── hairstyles.json
│   ├── hairstyles_female.json
│   └── norms/
│       ├── male_norms_v2.csv
│       └── female_norms_v2.csv
│
├── tests/
│   ├── test_rules.py
│   └── test_recommender.py
│
├── util/
│   └── compute_norms.py      # GPU-accelerated dataset norm computation
│
└── main.py                   # CLI pipeline runner
```
---

## Tech stack

| Layer | Tools |
|---|---|
| Backend | FastAPI, Uvicorn |
| Frontend | React, Vite |
| Deployment | Hugging Face Spaces (backend), Vercel (frontend) |
| Landmark detection | MediaPipe Face Mesh |
| Hair segmentation | SegFormer (jonathandinu/face-parsing) |
| Gender detection | InsightFace |
| Feature computation | NumPy, OpenCV |
| Norm calibration | pandas, PyTorch (GPU), UTKFace dataset |
| Feedback storage | Google Sheets (gspread) |
| Testing | pytest |

---

## Feedback loop

Every session logs features, quality score, and top recommendations to Google Sheets. Per-recommendation thumbs up/down votes are stored separately. Collected data enables empirical weight calibration rather than manual tuning.

---

## Visual assets disclaimer

Example hairstyle visualizations used in the demo UI were generated with Google's Gemini image generation tools for illustrative and non-commercial research purposes. Some images were cropped and resized to maintain consistent framing and UI presentation.

---

## Roadmap

- **Multi-photo pipeline** — front + profile + jaw-angle photos for 3D facial analysis via InsightFace landmarks
- **Receding hairline detection** — hairline shape analysis from segmentation mask
- **Hair type classification** — straight / wavy / curly / coily from hair mask texture
- **Hairstyle simulation** — generative preview via HairFAST or SAM inpainting
- **User tutorial** — guided photo capture with examples for optimal accuracy
- **Weight optimization** — vote data → empirical attribute weight calibration
