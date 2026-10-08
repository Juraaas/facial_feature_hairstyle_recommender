import pytest
from src.rules import apply_rules

def make_neutral_traits():
    return {
        "face_length": "balanced",
        "jaw": "normal",
        "jaw_height": "normal",
        "eyes": "normal",
        "lips": "normal",
        "nose": "balanced",
        "lower_face": "normal",
        "chin": "normal",
        "face_shape_type": "normal",
        "forehead": "normal",
        "thirds_balance": "balanced",
        "dominant_third": "normal",
        "hair_type": "normal",
        "hairline": "normal",
        "symmetry": "medium",
    }


def test_neutral_traits_produce_neutral_scores():
    traits = make_neutral_traits()
    scores = apply_rules(traits)

    assert all(abs(v) < 2 for v in scores.values()), (
        f"Neutral traits produced strong scores: {scores}"
    )


@pytest.mark.parametrize(
    "trait, value, expected",
    [
        (
            "face_length",
            "long",
            {
                "volume_top": lambda x: x < 0,
                "volume_sides": lambda x: x > 0,
                "fringe": lambda x: x > 0,
                "longer_hair": lambda x: x < 0,
            },
        ),
        (
            "face_length",
            "short",
            {
                "volume_top": lambda x: x > 0,
                "longer_hair": lambda x: x > 0,
                "fringe": lambda x: x < 0,
                "volume_sides": lambda x: x < 0,
            },
        ),
        (
            "jaw",
            "wide",
            {
                "soft_texture": lambda x: x > 0,
                "volume_sides": lambda x: x < 0,
                "short_sides": lambda x: x < 0,
            },
        ),
        (
            "jaw",
            "narrow",
            {
                "volume_sides": lambda x: x > 0,
                "soft_texture": lambda x: x > 0,
            },
        ),
        (
            "jaw_height",
            "high",
            {
                "longer_hair": lambda x: x > 0,
                "volume_top": lambda x: x < 0,
            },
        ),
        (
            "jaw_height",
            "low",
            {
                "volume_top": lambda x: x > 0,
                "fringe": lambda x: x > 0,
                "longer_hair": lambda x: x < 0,
            },
        ),
        (
            "eyes",
            "wide",
            {
                "clean_lines": lambda x: x > 0,
                "volume_top": lambda x: x > 0,
            },
        ),
        (
            "eyes",
            "close",
            {
                "volume_sides": lambda x: x > 0,
                "fringe": lambda x: x < 0,
            },
        ),
        (
            "lips",
            "wide",
            {
                "soft_texture": lambda x: x > 0,
                "clean_lines": lambda x: x < 0,
            },
        ),
        (
            "lips",
            "narrow",
            {
                "clean_lines": lambda x: x > 0,
            },
        ),
        (
            "nose",
            "upper-dominant",
            {
                "fringe": lambda x: x > 0,
                "volume_top": lambda x: x < 0,
            },
        ),
        (
            "nose",
            "lower-dominant",
            {
                "volume_top": lambda x: x > 0,
                "fringe": lambda x: x < 0,
            },
        ),
        (
            "lower_face",
            "long",
            {
                "soft_texture": lambda x: x > 0,
                "short_sides": lambda x: x < 0,
            },
        ),
        (
            "lower_face",
            "short",
            {
                "clean_lines": lambda x: x > 0,
                "longer_hair": lambda x: x < 0,
            },
        ),
        (
            "chin",
            "prominent",
            {
                "soft_texture": lambda x: x > 0,
                "longer_hair": lambda x: x > 0,
                "clean_lines": lambda x: x < 0,
            },
        ),
        (
            "chin",
            "recessed",
            {
                "clean_lines": lambda x: x > 0,
                "longer_hair": lambda x: x < 0,
            },
        ),
        (
            "face_shape_type",
            "triangle",
            {
                "volume_top": lambda x: x > 0,
                "fringe": lambda x: x > 0,
            },
        ),
        (
            "face_shape_type",
            "square",
            {
                "soft_texture": lambda x: x > 0,
                "clean_lines": lambda x: x < 0,
            },
        ),
        (
            "forehead",
            "high",
            {
                "fringe": lambda x: x > 0,
                "volume_top": lambda x: x < 0,
            },
        ),
        (
            "forehead",
            "low",
            {
                "fringe": lambda x: x < 0,
                "volume_top": lambda x: x > 0,
            },
        ),
        (
            "thirds_balance",
            "imbalanced",
            {
                "soft_texture": lambda x: x > 0,
                "clean_lines": lambda x: x < 0,
            },
        ),
        (
            "dominant_third",
            "upper",
            {
                "fringe": lambda x: x > 0,
                "volume_top": lambda x: x < 0,
            },
        ),
        (
            "dominant_third",
            "lower",
            {
                "volume_top": lambda x: x > 0,
                "fringe": lambda x: x < 0,
            },
        ),
        (
            "dominant_third",
            "middle",
            {
                "fringe": lambda x: x > 0,
            },
        ),
        (
            "hair_type",
            "curly",
            {
                "soft_texture": lambda x: x > 0,
                "textured_top": lambda x: x > 0,
                "clean_lines": lambda x: x < 0,
            },
        ),
        (
            "hair_type",
            "straight",
            {
                "clean_lines": lambda x: x > 0,
                "longer_hair": lambda x: x > 0,
            },
        ),
        (
            "hair_type",
            "coily",
            {
                "soft_texture": lambda x: x > 0,
                "textured_top": lambda x: x > 0,
                "clean_lines": lambda x: x < 0,
            },
        ),
        (
            "hairline",
            "receding",
            {
                "textured_top": lambda x: x > 0,
                "fringe": lambda x: x <= -2,
                "volume_top": lambda x: x < 0,
            },
        ),
        (
            "hairline",
            "uneven",
            {
                "fringe": lambda x: x < 0,
                "soft_texture": lambda x: x > 0,
            },
        ),
    ],
)
def test_single_trait_matrix(trait, value, expected):
    traits = {
        **make_neutral_traits(),
        trait: value,
    }

    scores = apply_rules(traits)

    for dimension, predicate in expected.items():
        assert predicate(scores[dimension]), (
            f"{trait}={value}: unexpected {dimension} score "
            f"{scores[dimension]}; full scores={scores}"
        )


def test_low_symmetry_boosts_texture_and_reduces_clean_lines():
    low = {
        **make_neutral_traits(),
        "symmetry": "low",
        "jaw": "wide",
    }

    high = {
        **make_neutral_traits(),
        "symmetry": "high",
        "jaw": "wide",
    }

    scores_low = apply_rules(low)
    scores_high = apply_rules(high)

    assert scores_low["soft_texture"] > scores_high["soft_texture"]
    assert scores_low["volume_sides"] < scores_high["volume_sides"]
    assert scores_low["clean_lines"] < scores_high["clean_lines"]


def test_high_symmetry_favors_clean_lines():
    normal = {
        **make_neutral_traits(),
        "symmetry": "medium",
    }

    high = {
        **make_neutral_traits(),
        "symmetry": "high",
    }

    normal_scores = apply_rules(normal)
    high_scores = apply_rules(high)

    assert high_scores["clean_lines"] > normal_scores["clean_lines"]


@pytest.mark.parametrize(
    "trait_a, value_a, trait_b, value_b, dimension",
    [
        (
            "face_length",
            "long",
            "forehead",
            "high",
            "volume_sides",
        ),
        (
            "face_length",
            "long",
            "jaw",
            "narrow",
            "soft_texture",
        ),
        (
            "face_length",
            "short",
            "jaw",
            "narrow",
            "volume_top",
        ),
        (
            "jaw",
            "wide",
            "chin",
            "prominent",
            "soft_texture",
        ),
        (
            "jaw",
            "narrow",
            "chin",
            "recessed",
            "volume_sides",
        ),
        (
            "symmetry",
            "low",
            "jaw",
            "wide",
            "soft_texture",
        ),
    ],
)
def test_interaction_adds_only_a_small_additional_effect(
    trait_a,
    value_a,
    trait_b,
    value_b,
    dimension,
):
    trait_a_only = {
        **make_neutral_traits(),
        trait_a: value_a,
    }

    trait_b_only = {
        **make_neutral_traits(),
        trait_b: value_b,
    }

    combined = {
        **make_neutral_traits(),
        trait_a: value_a,
        trait_b: value_b,
    }

    score_a = apply_rules(trait_a_only)[dimension]
    score_b = apply_rules(trait_b_only)[dimension]
    score_combined = apply_rules(combined)[dimension]

    additional_effect = score_combined - score_a - score_b

    assert abs(additional_effect) <= 1.5, (
        f"Interaction {trait_a}={value_a} + "
        f"{trait_b}={value_b} produced too strong an additional effect "
        f"on {dimension}: {additional_effect:.2f}"
    )


@pytest.mark.parametrize(
    "extra_traits",
    [
        {"forehead": "high"},
        {"face_length": "long"},
        {"symmetry": "low"},
        {"nose": "upper-dominant"},
        {"dominant_third": "upper"},
    ],
)
def test_receding_hairline_always_keeps_fringe_strongly_negative(extra_traits):
    traits = {
        **make_neutral_traits(),
        "hairline": "receding",
        **extra_traits,
    }

    scores = apply_rules(traits)

    assert scores["fringe"] <= -2, (
        f"Receding hairline violated hard constraint: {scores}"
    )


def test_long_and_short_face_produce_different_rule_profiles():
    long_traits = {
        **make_neutral_traits(),
        "face_length": "long",
    }

    short_traits = {
        **make_neutral_traits(),
        "face_length": "short",
    }

    long_scores = apply_rules(long_traits)
    short_scores = apply_rules(short_traits)

    assert long_scores != short_scores
    assert long_scores["volume_top"] < short_scores["volume_top"]
    assert long_scores["volume_sides"] > short_scores["volume_sides"]