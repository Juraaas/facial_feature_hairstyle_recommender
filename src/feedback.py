import csv 
import json
from pathlib import Path
from datetime import datetime

FEEDBACK_PATH = Path("data/feedback/feedback.csv")
FIELDNAMES = [
    "timestamp", "face_ratio", "jaw_ratio", "jaw_to_height", "eye_ratio", "eye_height",
    "lip_ratio", "nose_position", "lower_face_ratio", "chin_prominence",
    "symmetry", "quality_score",
    "rec_1", "rec_2", "rec_3",
    "rating",
    "comment",
]

VOTES_PATH = Path("data/feedback/votes.csv")
VOTE_FIELDS = ["timestamp", "style_name", "vote", "face_ratio", "jaw_ratio",
               "jaw_to_height", "eye_ratio", "eye_height", "lip_ratio", "nose_position",
               "lower_face_ratio", "chin_prominence", "symmetry", "gender"]

def save_session(features, quality_score, recs, rating=None, comment=""):
    FEEDBACK_PATH.parent.mkdir(exist_ok=True)
    exists = FEEDBACK_PATH.exists()
    top_styles = recs["top_styles"]

    row = {
        "timestamp": datetime.now().isoformat(),
        "quality_score": quality_score,
        "rec_1": top_styles[0]["name"] if len(top_styles) > 0 else "",
        "rec_2": top_styles[1]["name"] if len(top_styles) > 1 else "",
        "rec_3": top_styles[2]["name"] if len(top_styles) > 2 else "",
        "rating": rating or "",
        "comment": comment,
        **{k: round(v, 4) for k, v in features.items()},
    }

    with open(FEEDBACK_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def load_feedback():
    if not FEEDBACK_PATH.exists():
        return []
    with open(FEEDBACK_PATH, "r") as f:
        return list(csv.DictReader(f))

def save_vote(style_name: str, vote: str, features: dict, gender: str = ""):
    VOTES_PATH.parent.mkdir(exist_ok=True)
    exists = VOTES_PATH.exists()

    row = {
        "timestamp":  datetime.now().isoformat(),
        "style_name": style_name,
        "vote":       vote,
        "gender":     gender,
        **{k: round(v, 4) for k, v in features.items()},
    }

    with open(VOTES_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VOTE_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)