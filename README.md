## Face Geometry-Based Hairstyle Recommendation System

An explainable computer vision system that analyzes facial geometry and proportions to recommend suitable hairstyles. The project focuses on extracting meaningful facial features (such as proportions, symmetry, and structure) and mapping them to hairstyle characteristics using interpretable, rule-based logic.

--- 

### Project Goal

The goal of this project is to build a system that:
- detects facial landmarks from an input image
- extracts geometric and proportional facial features
- interprets facial structure (e.g., face shape, symmetry, proportions)
- recommends hairstyles based on visual balancing principles
- provides clear explanations for each recommendation

Unlike typical black-box approaches, this system emphasizes interpretability and feature-based reasoning. The system is based on three main pillars.

Facial Geometry & Proportions
- facial thirds (vertical proportions)
- width-to-height ratios
- jaw, cheekbone, and forehead relationships

Deviation from Average
- identifying dominant or extreme facial features

Perceptual Balancing
- recommending hairstyles that visually:
elongate, widen, soften or balance facial structure

---

### Planned Pipeline
Input Image
    ↓
Face Detection
    ↓
Facial Landmarks Extraction
    ↓
Feature Extraction (geometry & ratios)
    ↓
Face Traits Interpretation
    ↓
Hairstyle Matching & Scoring
    ↓
Recommendation + Explanation

---

### Feature Extraction

The system derives interpretable, normalized facial features based on geometric relationships between landmarks. Current feature set includes:
- **face_ratio** — vertical to horizontal face proportion
- **jaw_ratio** — jaw width relative to face width
- **eye_ratio** — inter-eye distance normalized by face width
- **nose_position** — vertical position of the nose within the face
- **symmetry** — normalized left-right facial asymmetry
All features are scale-invariant, ensuring consistency across different image resolutions and sizes. Raw pixel measurements (e.g., face width, eye distance) are used internally but not exposed as final features.

### Debug & Validation

To ensure robustness and correctness, the system includes multiple validation mechanisms:
- **Visual debugging**
  - landmark visualization
  - facial geometry overlays
  - feature value rendering on image

- **Self-consistency tests**
  - feature stability under image resizing

- **Cross-image comparison**
  - evaluation of feature discriminative power across different faces

---

### Project Structure
```
face-geometry-hairstyle-recommender/
│
├── src/
│   ├── landmarks.py
│   ├── features.py
│   ├── geometry.py
│   ├── visualization.py
│
│
├── test_landmark.py
├── requirements.txt
└── README.md
```
---

### Tech Stack
- Python
- OpenCV
- MediaPipe
- NumPy
- scikit-learn
- Streamlit (demo UI)

---

### Current Status

Core CV pipeline implemented:
- facial landmark detection using MediaPipe Face Landmarker
- extraction of key geometric measurements
- normalized feature engineering (scale-invariant ratios)
- symmetry estimation and facial proportion metrics
- visual debugging tools (landmarks, geometry lines, feature overlays)
- feature validation:
  - self-consistency tests (scale invariance)
  - cross-image comparison (discriminative power)

The system is now capable of producing stable and interpretable facial feature vectors from input images. Next steps focus on improving feature completeness and semantic understanding of facial structure.

---

### Future Improvements (planned)
- machine learning-based face shape classification
- hairstyle visualization (overlay / simulation)
- real-time webcam support
- dataset-driven optimization