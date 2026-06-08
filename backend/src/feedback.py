import gspread
import os
import json
from google.oauth2.service_account import Credentials
from datetime import datetime

FIELDNAMES = [
    "timestamp", "face_ratio", "jaw_ratio", "jaw_to_height", "eye_ratio", "eye_height",
    "lip_ratio", "nose_position", "lower_face_ratio", "chin_prominence",
    "symmetry", "upper_third", "middle_third", "lower_third",
    "mid_lower_ratio", "thirds_balance", "quality_score",
    "rec_1", "rec_2", "rec_3", "rating", "comment",
]

VOTE_FIELDS = [
    "timestamp", "style_name", "vote", "face_ratio", "jaw_ratio",
    "jaw_to_height", "eye_ratio", "eye_height", "lip_ratio", "nose_position",
    "lower_face_ratio", "chin_prominence", "symmetry",
    "upper_third", "middle_third", "lower_third", "mid_lower_ratio",
    "thirds_balance", "gender"
]

def _get_sheet(sheet_name):
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT")
    spreadsheet_id = os.environ.get("SPREADSHEET_ID")

    if not creds_json:
        raise RuntimeError("GCP_SERVICE_ACCOUNT env var not set")
    
    if not spreadsheet_id:
        raise RuntimeError("SPREADSHEET_ID env var not set")
    
    creds_dict = json.loads(creds_json)
    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(spreadsheet_id)
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
            round(features.get("upper_third", 0), 4),
            round(features.get("middle_third", 0), 4),
            round(features.get("lower_third", 0), 4),
            round(features.get("mid_lower_ratio", 0), 4),
            round(features.get("thirds_balance", 0), 4),
            round(quality_score, 4),
            top_styles[0]["name"] if len(top_styles) > 0 else "",
            top_styles[1]["name"] if len(top_styles) > 1 else "",
            top_styles[2]["name"] if len(top_styles) > 2 else "",
            rating or "",
            comment,
        ]
        sheet.append_row(row)
    except Exception as e:
        print(f"Google Sheets save_session error: {repr(e)}")
        raise


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
            round(features.get("jaw_to_height", 0), 4),
            round(features.get("eye_ratio", 0), 4),
            round(features.get("eye_height", 0), 4),
            round(features.get("lip_ratio", 0), 4),
            round(features.get("nose_position", 0), 4),
            round(features.get("lower_face_ratio", 0), 4),
            round(features.get("chin_prominence", 0), 4),
            round(features.get("symmetry", 0), 4),
            round(features.get("upper_third", 0), 4),
            round(features.get("middle_third", 0), 4),
            round(features.get("lower_third", 0), 4),
            round(features.get("mid_lower_ratio", 0), 4),
            round(features.get("thirds_balance", 0), 4),
            gender,
        ]
        sheet.append_row(row)
    except Exception as e:
        print(f"Google Sheets save_vote error: {repr(e)}")
        raise

def load_feedback():
    try:
        sheet = _get_sheet("feedback")
        return sheet.get_all_records()
    except Exception:
        return []