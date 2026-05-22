# Face Geometry-Based Hairstyle Recommendation System

An explainable computer vision system that analyzes facial geometry and recommends hairstyles based on anthropometric proportions, symmetry, and structural traits. Built around **interpretable features** and **transparent decision logic** — no black-box models.

---

## How it works
```
Image → Landmark Detection → Hair Segmentation → Geometric Feature Extraction → Trait Interpretation → Rule-Based Scoring → Ranked Recommendations
```
Each step is fully inspectable. Features are normalized ratios derived from MediaPipe Face Mesh landmarks, calibrated against a population dataset of ~21,000 faces (UTKFace). Recommendations include per-feature contribution scores and explicit drawback analysis.


---

## Key design decisions

**Geometry over deep learning** — all facial features are computed analytically from landmark coordinates (eye spacing ratio, jaw-to-face-width ratio, facial thirds, symmetry score etc.), making the system auditable and easy to iterate on.

**Population-calibrated norms** — feature thresholds (p5/p25/p75/p95 percentiles) are derived from a dataset split by gender, not hand-tuned constants. This makes trait classification statistically grounded.

**Hair segmentation for accurate facial thirds** — a SegFormer model (jonathandinu/face-parsing) detects the hairline from the photo, enabling accurate facial thirds analysis including the upper third (hairline→brow). Photos without a detectable hairline are excluded from norm calibration to maintain methodological consistency.

**Explainable scoring** — each hairstyle recommendation includes positive contributions (why it fits) and negative contributions (what works against it), weighted by feature importance.

**Gender-aware pipeline** — automatic gender detection routes the analysis through separate trait classifiers, scoring rules, and hairstyle databases calibrated for male and female facial proportions.

---

### Pipeline

### 1. Landmark Detection
MediaPipe Face Mesh (478 landmarks) with quality assessment — checks face size, yaw/pitch rotation, and landmark alignment confidence before processing.

### 2. Hair Segmentation
SegFormer face-parsing model detects the hairline position. Used to compute the upper facial third accurately. If hairline detection fails (hat, poor lighting, extreme angle), the pipeline falls back to landmark 10.

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
Features are classified against population percentiles (p25/p75 as normal range boundaries) into semantic labels — e.g. `jaw: wide`, `eyes: close-set`, `face_length: long`. Separate classifiers for male and female norms.

### 5. Rule-Based Scoring
Each trait triggers weighted scoring adjustments across 8 hairstyle dimensions (`volume_top`, `volume_sides`, `short_sides`, `longer_hair`, `fringe`, `clean_lines`, `soft_texture`, `textured_top`). Rules encode established hairstyling principles — e.g. wide jaw → penalize short sides, boost soft texture. Female pipeline adds `layers`, `updo`, and `curtain_fringe` dimensions.

### 6. Hairstyle Matching
Scores are matched against a JSON database of 15 male / 16 female hairstyles, each annotated with attribute weights. Final score is normalized by total attribute weight to ensure comparability across styles with different numbers of attributes.

### 7. Explainability Layer
Per-recommendation output includes:
- positive contributions with percentage influence
- negative contributions with human-readable reasons
- face analysis summary derived from non-neutral traits
- per-feature trait bars showing user position relative to population norms

---

### Project Structure
```
├── src/
│   ├── geometry.py          # landmark-based geometric calculations
│   ├── features.py          # feature extraction pipeline
│   ├── face_traits.py       # male trait classification
│   ├── face_traits_female.py
│   ├── hair_segmentation.py # SegFormer hairline detection
│   ├── rules.py             # male scoring rules
│   ├── rules_female.py
│   ├── recommender.py       # scoring, matching, explainability
│   ├── pipeline.py          # end-to-end orchestration
│   ├── landmarks.py         # MediaPipe wrapper
│   ├── quality.py           # image quality assessment
│   ├── validation.py        # feature sanity checks
│   ├── gender.py            # gender detection (DeepFace)
│   ├── feedback.py          # session logging and vote tracking (Google Sheets)
│   ├── pdf_export.py        # report generation
│   └── drawing.py           # visualization overlays
│
├── data/
│   ├── hairstyles.json
│   ├── hairstyles_female.json
│   └── norms/
│       ├── male_norms_v2.csv
│       └── female_norms_v2.csv
├── tests/
│   ├── test_rules.py
│   └── test_recommender.py
│
├── compute_norms.py     # GPU-accelerated dataset norm computation
├── ui_components.py     # trait bars, style cards
├── app.py               # Streamlit UI
└── main.py              # CLI pipeline runner
```
---

## Tech stack

| Layer | Tools |
|---|---|
| Landmark detection | MediaPipe Face Mesh |
| Hair segmentation | SegFormer (jonathandinu/face-parsing) |
| Gender detection | manual select (planned -> InsightFace) |
| Feature computation | NumPy, OpenCV |
| Norm calibration | pandas, PyTorch (GPU), UTKFace dataset |
| UI | Streamlit |
| Report export | ReportLab |
| Feedback storage | Google Sheets (gspread) |
| Testing | pytest |

---

## Feedback loop

Every session is logged to Google Sheets with extracted features, quality score, and top recommendations. Users can vote on individual recommendations (thumbs up/down), stored in a separate sheet. This enables future weight optimization based on real preference data rather than manual tuning.

---

## Visual Assets Disclaimer

Example hairstyle visualizations used in the demo UI were generated with Google's Gemini image generation tools for illustrative and non-commercial research purposes. Some images were cropped and resized to maintain consistent framing and UI presentation.

---

## Roadmap

- **Automatic gender detection** — InsightFace integration for automatic gender classification without user input
- **Profile view support** — InsightFace 3D landmarks for jaw angle, chin projection, and lateral nose profile from side-view photos
- **Multi-photo pipeline** — upload front + profile + jaw-angle photos for comprehensive 3D facial analysis
- **Hair type classification** — straight / wavy / curly / coily detection from hair mask texture
- **Weight optimization** — use collected vote data to calibrate hairstyle attribute weights empirically
- **Hairstyle simulation** — generative preview via HairFAST or SAM-based inpainting
- **User tutorial** — guided photo capture flow with examples for optimal analysis accuracy
