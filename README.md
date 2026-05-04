## Face Geometry-Based Hairstyle Recommendation System

An explainable computer vision system that analyzes facial geometry and recommends hairstyles based on proportions, symmetry, and structural features. Unlike black-box models, this project focuses on **interpretable features** and **transparent decision logic**.

---

### What it does

- detects facial landmarks from an image  
- extracts normalized geometric features (ratios & proportions)  
- interprets facial traits (e.g. face length, symmetry, jaw structure)  
- scores and ranks hairstyles based on visual balance principles  
- explains *why* a given hairstyle fits (feature-level contributions)

---

### Pipeline
```
Image → Landmarks → Features → Traits → Scoring → Recommendations
```

---

### Core Features

- **Scale-invariant geometry**
  - face ratio, jaw ratio, eye ratio, symmetry, nose position  
- **Semantic traits layer**
  - transforms raw geometry into human-readable attributes  
- **Explainable recommendations**
  - each hairstyle is scored using weighted features  
  - outputs percentage contribution of each factor  
- **Debug & validation tools**
  - visual overlays (landmarks, geometry, features)  
  - self-consistency tests (resize invariance)  
  - cross-image comparison

---

### Project Structure
```
face-geometry-hairstyle-recommender/
│
├── src/
│   ├── drawing.py
│   ├── face_traits.py
│   ├── features.py
│   ├── geometry.py
│   ├── landmarks.py
│   ├── rules.py
│   └── recommender.py
│
data/
└── hairstyles.json
│
└── main.py
```
---

### Tech Stack
- Python
- OpenCV
- MediaPipe (Face landmarker)
- NumPy

---

### Status

Core pipeline and explainable recommendation engine implemented.  
System produces stable, interpretable feature vectors and meaningful hairstyle rankings.

---

### Next Steps
- negative contributions (why a style *doesn't* fit)  
- weight tuning & rule refinement  
- hair / forehead segmentation (facial thirds)  
- UI (Streamlit demo)  

---

### Vision

Move towards a fully explainable facial analysis system:
- facial proportions & thirds  
- masculinity / femininity spectrum  
- feature-level analysis (eyes, jaw, cheekbones, etc.)  
- personalized visual recommendations

---