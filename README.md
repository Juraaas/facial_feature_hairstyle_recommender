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

### Project Structure
```
face-geometry-hairstyle-recommender/
│
├── src/
│   ├── landmarks.py
│   ├── features.py
│   ├── face_shape.py
│   ├── recommender.py
│
├── data/
│   ├── hairstyles.json
│
├── demo/
│   ├── app.py
│
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

Project in early development stage. Next planned steps:
- facial landmark detection
- feature extraction
- rule-based face interpretation
- hairstyle database creation
- recommendation engine
- explainability module
- demo application

---

### Future Improvements (planned)
- machine learning-based face shape classification
- hairstyle visualization (overlay / simulation)
- real-time webcam support
- dataset-driven optimization