import pytest
from src.recommender import score_hairstyle, explain_match

def test_score_between_zero_and_one():
    user_scores = {"volume_top": 3, "clean_lines": 2, "short_sides": -1}
    style = {
        "attributes": {"volume_top": 1.0, "clean_lines": 0.8, "short_sides": 0.5}
    }
    score = score_hairstyle(user_scores, style)
    assert score >= 0

def test_contributions_sum_to_100():
    user_scores = {"volume_top": 2, "clean_lines": 1}
    style = {
        "attributes": {"volume_top": 1.0, "clean_lines": 0.5, "fringe": 0.0}
    }
    score = score_hairstyle(user_scores, style)
    contributions, _ = explain_match(user_scores, style, score)

    total_pct = sum(c["percent"] for c in contributions)
    assert abs(total_pct - 1.0) < 0.01, \
        f"Contributions sum to {total_pct:.3f}, expected 1.0"
    
def test_no_contributions_when_score_zero():
    user_scores = {"volume_top": 0, "fringe": 0}
    style = {"attributes": {"volume_top": 0.0, "fringe": 0.0}}
    positive, negative = explain_match(user_scores, style, 0)
    assert positive == []
    assert negative == []