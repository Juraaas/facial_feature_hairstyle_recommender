import pytest
from src.recommender import score_hairstyle, explain_match, calculate_display_score

def test_score_is_between_minus_one_and_one():
    user_scores = {
        "volume_top": 3,
        "clean_lines": 2,
        "short_sides": -1,
    }

    style = {
        "attributes": {
            "volume_top": 1.0,
            "clean_lines": 0.8,
            "short_sides": 0.5,
        }
    }

    score = score_hairstyle(user_scores, style)

    assert -1.0 <= score <= 1.0


def test_negative_contribution_is_preserved():
    user_scores = {
        "longer_hair": -3,
    }

    style = {
        "attributes": {
            "longer_hair": 1.0,
        }
    }

    score = score_hairstyle(user_scores, style)

    assert score == -1.0


def test_positive_contribution_is_preserved():
    user_scores = {
        "fringe": 4,
    }

    style = {
        "attributes": {
            "fringe": 1.0,
        }
    }

    score = score_hairstyle(user_scores, style)

    assert score == 1.0


def test_contributions_sum_to_one():
    user_scores = {
        "volume_top": 2,
        "clean_lines": 1,
    }

    style = {
        "attributes": {
            "volume_top": 1.0,
            "clean_lines": 0.5,
            "fringe": 0.0,
        }
    }

    score = score_hairstyle(user_scores, style)

    contributions, _ = explain_match(
        user_scores,
        style,
        score,
    )

    total_pct = sum(c["percent"] for c in contributions)

    assert abs(total_pct - 1.0) < 0.01, (
        f"Contributions sum to {total_pct:.3f}, expected 1.0"
    )


def test_negative_contributions_are_reported_separately():
    user_scores = {
        "fringe": -3,
    }

    style = {
        "attributes": {
            "fringe": 1.0,
        }
    }

    score = score_hairstyle(user_scores, style)

    positive, negative = explain_match(
        user_scores,
        style,
        score,
    )

    assert positive == []
    assert len(negative) == 1
    assert negative[0]["raw"] == -3


def test_no_contributions_when_all_scores_are_zero():
    user_scores = {
        "volume_top": 0,
        "fringe": 0,
    }

    style = {
        "attributes": {
            "volume_top": 0.0,
            "fringe": 0.0,
        }
    }

    positive, negative = explain_match(
        user_scores,
        style,
        0,
    )

    assert positive == []
    assert negative == []


def test_display_score_rewards_positive_target_match():
    user_scores = {
        "fringe": 4,
    }

    matching_style = {
        "attributes": {
            "fringe": 1.0,
        }
    }

    non_matching_style = {
        "attributes": {
            "fringe": 0.0,
        }
    }

    matching_score = calculate_display_score(
        user_scores,
        matching_style,
    )

    non_matching_score = calculate_display_score(
        user_scores,
        non_matching_style,
    )

    assert matching_score == 100
    assert non_matching_score == 0


def test_display_score_rewards_avoiding_negative_feature():
    user_scores = {
        "longer_hair": -4,
    }

    ideal_style = {
        "attributes": {
            "longer_hair": 0.0,
        }
    }

    bad_style = {
        "attributes": {
            "longer_hair": 1.0,
        }
    }

    ideal_score = calculate_display_score(
        user_scores,
        ideal_style,
    )

    bad_score = calculate_display_score(
        user_scores,
        bad_style,
    )

    assert ideal_score == 100
    assert bad_score == 0


def test_display_score_is_between_zero_and_hundred():
    user_scores = {
        "volume_top": 3,
        "fringe": 4,
        "longer_hair": -3,
        "clean_lines": 2,
    }

    style = {
        "attributes": {
            "volume_top": 0.5,
            "fringe": 0.8,
            "longer_hair": 0.2,
            "clean_lines": 0.7,
        }
    }

    score = calculate_display_score(user_scores, style)

    assert 0 <= score <= 100