import pytest 
from src.rules import apply_rules

def make_neutral_traits():
    return {
        "face_length": "balanced",
        "jaw": "normal",
        "jaw_height": "normal",
        "eyes": "normal",
        "eye_openness": "normal",
        "lips": "normal",
        "nose": "balanced",
        "lower_face": "normal",
        "chin": "normal",
        "symmetry": "medium",
        "facial_thirds": "balanced",
        "forehead": "normal",
        "thirds_balance": "balanced",
    }

def test_neutral_traits_produce_zero_scores():
    traits = make_neutral_traits()
    scores = apply_rules(traits)

    assert all(abs(v) < 2 for v in scores.values()), \
        f"Neutral traits produces strong scores: {scores}"
    
def test_long_face_penalizes_volume_top():
    traits = {**make_neutral_traits(), "face_length": "long"}
    scores = apply_rules(traits)

    assert scores["volume_top"] < 0
    assert scores["volume_sides"] > scores["volume_top"]

def test_wide_jaw_penalizes_short_sides():
    traits = {**make_neutral_traits(), "jaw": "wide"}
    scores = apply_rules(traits)
    assert scores["short_sides"] < 0
    assert scores["soft_texture"] > 0

def test_narrow_eyes_penalizes_fringe():
    traits = {**make_neutral_traits(), "eye_openness": "narrow"}
    scores = apply_rules(traits)
    assert scores["fringe"] < 0

def test_low_symmetry_boosts_texture():
    traits = {**make_neutral_traits(), "symmetry": "low"}
    scores_low = apply_rules(traits)
    
    traits_high = {**make_neutral_traits(), "symmetry": "high"}
    scores_high = apply_rules(traits_high)
    
    assert scores_low["soft_texture"] > scores_high["soft_texture"]

def test_opposing_face_lengths_produce_different_top_styles():
    from src.recommender import generate_recommendations
    from src.rules import apply_rules
    long_traits  = {**make_neutral_traits(), "face_length": "long"}
    short_traits = {**make_neutral_traits(), "face_length": "short"}

    long_scores  = apply_rules(long_traits)
    short_scores = apply_rules(short_traits)

    long_recs  = generate_recommendations(long_scores,  long_traits)
    short_recs = generate_recommendations(short_scores, short_traits)

    long_top  = long_recs["top_styles"][0]["name"]
    short_top = short_recs["top_styles"][0]["name"]

    assert long_top != short_top, \
        f"Long and short face got same top recommendation: {long_top}"
    
def test_lower_dominant_thirds_boost_volume_top():
    traits = {**make_neutral_traits(), "facial_thirds": "lower_dominant"}
    scores = apply_rules(traits)
    assert scores["volume_top"] > 0 

def test_high_forehead_boosts_fringe():
    traits = {**make_neutral_traits(), "forehead": "high"}
    scores = apply_rules(traits)
    assert scores["fringe"] > 0

def test_low_forehead_penalizes_fringe():
    traits = {**make_neutral_traits(), "forehead": "low"}
    scores = apply_rules(traits)
    assert scores["fringe"] < 0
