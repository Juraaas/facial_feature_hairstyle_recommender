import gspread
import streamlit as st
from google.oauth2.service_account import Credentials
from datetime import datetime

FIELDNAMES = [
    "timestamp", "face_ratio", "jaw_ratio", "jaw_to_height", "eye_ratio", "eye_height",
    "lip_ratio", "nose_position", "lower_face_ratio", "chin_prominence",
    "symmetry", "quality_score",
    "rec_1", "rec_2", "rec_3",
    "rating",
    "comment",
]

VOTE_FIELDS = ["timestamp", "style_name", "vote", "face_ratio", "jaw_ratio",
               "jaw_to_height", "eye_ratio", "eye_height", "lip_ratio", "nose_position",
               "lower_face_ratio", "chin_prominence", "symmetry", "gender"]

def _get_sheet(sheet_name):
    creds_dict = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(st.secrets["spreadsheet_id"])
    try:
        return spreadsheet.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        return spreadsheet.add_worksheet(sheet_name, rows=1000, cols=20)
    
def _ensure_header(sheet, headers):
    existing = sheet.row_values(1)
    if existing and existing[0] != "timestamp":
        return
    if not existing:
        sheet.insert_row(headers, 1)

def save_session(features, quality_score, recs, rating=None, comment=""):
    try:
        sheet = _get_sheet("feedback")
        _ensure_header(sheet, FIELDNAMES)
        top_styles = recs["top_styles"]

        row = [
            datetime.now().isoformat(),
            round(features.get("face_ratio", 0), 4),
            round(features.get("jaw_ratio", 0), 4),
            round(features.get("jaw_to_height", 0), 4),
            round(features.get("eye_ratio", 0), 4),
            round(features.get("eye_height", 0), 4),
            round(features.get("lip_ratio", 0), 4),
            round(features.get("nose_position", 0), 4),
            round(features.get("lower_face_ratio", 0), 4),
            round(features.get("chin_prominence", 0), 4),
            round(features.get("symmetry", 0), 4),
            round(quality_score, 4),
            top_styles[0]["name"] if len(top_styles) > 0 else "",
            top_styles[1]["name"] if len(top_styles) > 1 else "",
            top_styles[2]["name"] if len(top_styles) > 2 else "",
            rating or "",
            comment,
        ]
        sheet.append_row(row)
    except Exception as e:
        pass


def save_vote(style_name: str, vote: str, features: dict, gender: str = ""):
    try:
        sheet = _get_sheet("votes")
        _ensure_header(sheet, VOTE_FIELDS)
        row = [
            datetime.now().isoformat(),
            style_name,
            vote,
            round(features.get("face_ratio", 0), 4),
            round(features.get("jaw_ratio", 0), 4),
            round(features.get("eye_ratio", 0), 4),
            round(features.get("eye_height", 0), 4),
            round(features.get("lip_ratio", 0), 4),
            round(features.get("nose_position", 0), 4),
            round(features.get("lower_face_ratio", 0), 4),
            round(features.get("chin_prominence", 0), 4),
            round(features.get("symmetry", 0), 4),
            gender,
        ]
        sheet.append_row(row)
    except Exception as e:
        pass

def load_feedback():
    try:
        sheet = _get_sheet("feedback")
        return sheet.get_all_records()
    except Exception:
        return []