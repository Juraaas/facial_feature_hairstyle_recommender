## Face Geometry-Based Hairstyle Recommendation System

An explainable computer vision system that analyzes facial geometry and recommends hairstyles based on proportions, symmetry, and structural features. Unlike black-box models, this project focuses on **interpretable features** and **transparent decision logic**.

---

### Features
- facial landmark detection
- normalized geometric feature extraction
- semantic face trait interpretation
- rule-based hairstyle scoring
- explainable recommendations with feature contributions
- hairstyle database with visual examples
- interactive Streamlit UI

---

### Pipeline
```
Image → Landmarks → Features → Traits → Scoring → Recommendations
```

---

### Current status

Implemented:
- explainable CV pipeline
- facial geometry extraction
- semantic trait layer
- hairstyle recommendation engine
- contribution-based explanations
- hairstyle database with images
- Streamlit demo UI

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
│   ├── pipeline.py
│   ├── rules.py
│   └── recommender.py
│
├──data/
│   ├──hairstyles.json
│
├── app.py
├── compute_norms.py
└── main.py
```
---

### Tech Stack
- Python
- OpenCV
- MediaPipe (Face landmarker)
- NumPy
- Streamlit

---

### Next Steps
- negative contribution analysis
- hair / forehead segmentation
- advanced facial analysis
- hairstyle visualization / simulation  

---

### Vision

Build a fully explainable facial analysis system capable of:
- detailed facial structure analysis
- masculinity / femininity estimation
- feature-level interpretation
- personalized aesthetic recommendations
- interpretable visual balancing

---