from __future__ import annotations
import inspect
import json
import sys
from pathlib import Path
from collections import Counter

from src.rules import apply_rules
from src.recommender import score_hairstyle, calculate_display_score

ROOT = Path(__file__).resolve().parent
CATALOGUES = [
    (ROOT / "data" / "hairstyles.json", "Man"),
    (ROOT / "data" / "hairstyles_female.json", "Woman"),
]

PROFILES = {
    "balanced_high_symmetry": {
        "face_length": "balanced", "forehead": "normal", "jaw": "normal",
        "jaw_height": "normal", "eyes": "normal", "lips": "normal",
        "chin": "normal", "symmetry": "high", "thirds_balance": "balanced",
        "dominant_third": "normal", "face_shape_type": "normal",
        "hair_type": "straight", "hairline": "normal",
    },
    "long_face_high_forehead": {
        "face_length": "long", "forehead": "high", "jaw": "normal",
        "jaw_height": "normal", "eyes": "normal", "lips": "normal",
        "chin": "normal", "symmetry": "normal", "thirds_balance": "balanced",
        "dominant_third": "upper", "face_shape_type": "normal",
        "hair_type": "straight", "hairline": "normal",
    },
    "short_face_low_forehead": {
        "face_length": "short", "forehead": "low", "jaw": "normal",
        "jaw_height": "normal", "eyes": "normal", "lips": "normal",
        "chin": "normal", "symmetry": "normal", "thirds_balance": "balanced",
        "dominant_third": "lower", "face_shape_type": "normal",
        "hair_type": "straight", "hairline": "normal",
    },
    "wide_jaw_prominent_chin": {
        "face_length": "normal", "forehead": "normal", "jaw": "wide",
        "jaw_height": "high", "eyes": "normal", "lips": "normal",
        "chin": "prominent", "symmetry": "normal", "thirds_balance": "balanced",
        "dominant_third": "normal", "face_shape_type": "square",
        "hair_type": "straight", "hairline": "normal",
    },
    "narrow_jaw_recessed_chin": {
        "face_length": "normal", "forehead": "normal", "jaw": "narrow",
        "jaw_height": "low", "eyes": "normal", "lips": "normal",
        "chin": "recessed", "symmetry": "normal", "thirds_balance": "balanced",
        "dominant_third": "normal", "face_shape_type": "normal",
        "hair_type": "straight", "hairline": "normal",
    },
    "curly_hair": {
        "face_length": "normal", "forehead": "normal", "jaw": "normal",
        "jaw_height": "normal", "eyes": "normal", "lips": "normal",
        "chin": "normal", "symmetry": "normal", "thirds_balance": "balanced",
        "dominant_third": "normal", "face_shape_type": "normal",
        "hair_type": "curly", "hairline": "normal",
    },
    "coily_hair": {
        "face_length": "normal", "forehead": "normal", "jaw": "normal",
        "jaw_height": "normal", "eyes": "normal", "lips": "normal",
        "chin": "normal", "symmetry": "normal", "thirds_balance": "balanced",
        "dominant_third": "normal", "face_shape_type": "normal",
        "hair_type": "coily", "hairline": "normal",
    },
    "receding_hairline": {
        "face_length": "normal", "forehead": "normal", "jaw": "normal",
        "jaw_height": "normal", "eyes": "normal", "lips": "normal",
        "chin": "normal", "symmetry": "normal", "thirds_balance": "balanced",
        "dominant_third": "normal", "face_shape_type": "normal",
        "hair_type": "straight", "hairline": "receding",
    },
    "low_symmetry_wide_jaw": {
        "face_length": "normal", "forehead": "normal", "jaw": "wide",
        "jaw_height": "normal", "eyes": "normal", "lips": "normal",
        "chin": "normal", "symmetry": "low", "thirds_balance": "balanced",
        "dominant_third": "normal", "face_shape_type": "square",
        "hair_type": "wavy", "hairline": "normal",
    },
    "imbalanced_thirds": {
        "face_length": "normal", "forehead": "normal", "jaw": "normal",
        "jaw_height": "normal", "eyes": "normal", "lips": "normal",
        "chin": "normal", "symmetry": "normal", "thirds_balance": "imbalanced",
        "dominant_third": "upper", "face_shape_type": "normal",
        "hair_type": "wavy", "hairline": "normal",
    },
}

def load_catalog(path):
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    styles = payload["styles"] if isinstance(payload, dict) else payload
    if not isinstance(styles, list):
        raise ValueError(f"Unexpected catalogue format: {path}")
    return styles

def score_style(user_scores, style, traits):
    if len(inspect.signature(score_hairstyle).parameters) >= 3:
        return float(score_hairstyle(user_scores, style, traits))
    return float(score_hairstyle(user_scores, style))

def rank_styles(styles, user_scores, traits):
    rows = []
    for style in styles:
        rows.append({
            "name": style.get("name", "<unnamed>"),
            "score": score_style(user_scores, style, traits),
            "display": calculate_display_score(user_scores, style),
            "attributes": style.get("attributes", {}),
        })
    return sorted(rows, key=lambda row: row["score"], reverse=True)

def rank_of(rows, name):
    for i, row in enumerate(rows, 1):
        if row["name"].casefold() == name.casefold():
            return i
    return None

BASELINE = {
    "face_length": "normal", "forehead": "normal", "jaw": "normal",
    "jaw_height": "normal", "eyes": "normal", "lips": "normal",
    "nose": "normal", "lower_face": "normal", "chin": "normal",
    "symmetry": "normal", "thirds_balance": "balanced",
    "dominant_third": "normal", "face_shape_type": "normal",
    "hair_type": "straight", "hairline": "normal",
}

CONTRASTS = [
    ("face_length", "normal", "long", "long face"),
    ("face_length", "normal", "short", "short face"),
    ("forehead", "normal", "high", "high forehead"),
    ("forehead", "normal", "low", "low forehead"),
    ("jaw", "normal", "wide", "wide jaw"),
    ("jaw", "normal", "narrow", "narrow jaw"),
    ("chin", "normal", "prominent", "prominent chin"),
    ("chin", "normal", "recessed", "recessed chin"),
    ("symmetry", "normal", "high", "high symmetry"),
    ("symmetry", "normal", "low", "low symmetry"),
    ("hair_type", "straight", "wavy", "wavy hair (vs straight)"),
    ("hair_type", "straight", "curly", "curly hair (vs straight)"),
    ("hair_type", "straight", "coily", "coily hair (vs straight)"),
    ("hairline", "normal", "receding", "receding hairline"),
    ("hairline", "normal", "uneven", "uneven hairline"),
    ("thirds_balance", "balanced", "imbalanced", "imbalanced thirds"),
    ("dominant_third", "normal", "upper", "upper-dominant third"),
    ("dominant_third", "normal", "lower", "lower-dominant third"),
    ("face_shape_type", "normal", "square", "square face shape"),
    ("face_shape_type", "normal", "triangle", "triangle face shape"),
]

def compare_scores(before, after):
    keys = set(before) | set(after)
    return {
        key: round(after.get(key, 0) - before.get(key, 0), 3)
        for key in sorted(keys)
        if abs(after.get(key, 0) - before.get(key, 0)) > 1e-9
    }

def contrast_report(styles, gender):
    print("\\n" + "=" * 78)
    print(f"ONE-TRAIT CONTRAST TESTS | gender={gender}")
    print("Each row changes one trait from a neutral control profile.")
    print("=" * 78)
    baseline_traits = dict(BASELINE)
    base_scores = apply_rules(baseline_traits, gender=gender)
    baseline_rows = rank_styles(styles, base_scores, baseline_traits)
    base_rank = {row["name"]: i for i, row in enumerate(baseline_rows, 1)}
    print("Control uses straight hair to give the reference ranking nonzero scores;")
    print("each contrast still changes exactly one trait. Rankings are diagnostic, not style-quality verdicts.")

    for trait, baseline_value, changed_value, label in CONTRASTS:
        before_traits = dict(BASELINE)
        before_traits[trait] = baseline_value
        after_traits = dict(BASELINE)
        after_traits[trait] = changed_value
        before_scores = apply_rules(before_traits, gender=gender)
        after_scores = apply_rules(after_traits, gender=gender)
        rule_delta = compare_scores(before_scores, after_scores)
        before_rows = rank_styles(styles, before_scores, before_traits)
        after_rows = rank_styles(styles, after_scores, after_traits)
        before_score_by_name = {row["name"]: row["score"] for row in before_rows}
        after_score_by_name = {row["name"]: row["score"] for row in after_rows}

        movement = []
        for rank, row in enumerate(after_rows, 1):
            old_rank = base_rank.get(row["name"])
            if old_rank is not None:
                movement.append((rank - old_rank, row["name"], old_rank, rank))
        movers_up = sorted((x for x in movement if x[0] < 0))[:3]
        movers_down = sorted((x for x in movement if x[0] > 0), reverse=True)[:3]

        print(f"\\n[{label}] {trait}: {before_traits[trait]!r} -> {changed_value!r}")
        print("  rule delta:", rule_delta or "NO CHANGE")
        print("  top 3 after change:")
        for rank, row in enumerate(after_rows[:3], 1):
            print(f"    {rank}. {row['name']} | score={row['score']:.4f} | rank {base_rank.get(row['name'])} -> {rank}")
        if movers_up:
            print("  biggest rank rises:", ", ".join(
                f"{name} ({old}->{new})" for _, name, old, new in movers_up))
        if movers_down:
            print("  biggest rank drops:", ", ".join(
                f"{name} ({old}->{new})" for _, name, old, new in movers_down))

        if trait == "hairline" and changed_value == "receding":
            fringe = after_scores.get("fringe", 0)
            print(f"  hard constraint: fringe={fringe} (expected <= -2): "
                  f"{'PASS' if fringe <= -2 else 'FAIL'}")

        score_deltas = [
            (after_score_by_name[name] - before_score_by_name[name], name)
            for name in before_score_by_name.keys() & after_score_by_name.keys()
        ]
        changed_style_scores = [item for item in score_deltas if abs(item[0]) > 1e-8]
        if changed_style_scores:
            gains = sorted((item for item in score_deltas if item[0] > 1e-8), reverse=True)[:3]
            losses = sorted((item for item in score_deltas if item[0] < -1e-8))[:3]
            if gains:
                print("  largest raw-score gains:", ", ".join(
                    f"{name} ({delta:+.4f})" for delta, name in gains))
            if losses:
                print("  largest raw-score losses:", ", ".join(
                    f"{name} ({delta:+.4f})" for delta, name in losses))
        if rule_delta and not changed_style_scores:
            print("  WARNING: rules changed but no hairstyle score changed; inspect score pipeline.")
        elif not rule_delta:
            print("  NOTE: this trait/value caused no rule-score delta for this gender.")

def scoring_consistency_report(styles, gender):
    print("\\n" + "=" * 78)
    print(f"SCORING CONSISTENCY CHECK | gender={gender}")
    print("=" * 78)
    traits = dict(BASELINE)
    traits.update({
        "face_length": "long", "forehead": "high", "jaw": "wide",
        "chin": "prominent", "hairline": "receding", "hair_type": "curly",
    })
    user_scores = apply_rules(traits, gender=gender)
    denominator = sum(abs(value) for value in user_scores.values())
    failures = []
    for style in styles:
        attrs = style.get("attributes", {})
        expected = (
            sum(value * attrs.get(key, 0) for key, value in user_scores.items()) / denominator
            if denominator else 0.0
        )
        actual = score_style(user_scores, style, traits)
        if abs(expected - actual) > 1e-8:
            failures.append((style.get("name"), expected, actual))
    print(f"Manual weighted-score agreement: {'PASS' if not failures else 'FAIL'}")
    if failures:
        for name, expected, actual in failures[:10]:
            print(f"  {name}: expected={expected:.6f}, actual={actual:.6f}")
    else:
        print("  score_hairstyle agrees with the normalized weighted sum.")
    print("  display_score is a separate metric and is not expected to equal ranking score.")

def print_catalog(path, gender):
    print("\n" + "=" * 78)
    print(f"CATALOGUE: {path.name} | gender={gender}")
    print("=" * 78)
    styles = load_catalog(path)
    names = [s.get("name", "") for s in styles]
    duplicates = [n for n, count in Counter(names).items() if count > 1]
    print(f"Styles: {len(styles)} | unique names: {len(set(names))}/{len(names)}")
    print("DUPLICATES:", duplicates if duplicates else "none")

    missing = []
    bad_values = []
    all_keys = set()
    for style in styles:
        attrs = style.get("attributes")
        if not isinstance(attrs, dict):
            missing.append(style.get("name"))
            continue
        all_keys.update(attrs)
        for key, value in attrs.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                bad_values.append((style.get("name"), key, value))
    print("Attribute dimensions:", sorted(all_keys))
    print("Styles without attribute dict:", missing or "none")
    print("Non-numeric attribute values:", bad_values or "none")

    ranked_by_profile = {}
    for profile_name, traits in PROFILES.items():
        user_scores = apply_rules(traits, gender=gender)
        rows = rank_styles(styles, user_scores, traits)
        ranked_by_profile[profile_name] = (rows, user_scores)
        print(f"\n[{profile_name}]")
        print("  nonzero rule scores:", {k: v for k, v in user_scores.items() if v})
        print("  TOP 5:")
        for rank, row in enumerate(rows[:5], 1):
            print(f"   {rank:2}. {row['name']:<28} raw={row['score']: .4f} display={row['display']}")

        inversions = []
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if a["score"] > b["score"] and a["display"] < b["display"]:
                    inversions.append((a["name"], b["name"]))
        if inversions:
            print(f"  NOTE: display-score inversions relative to raw ranking: {len(inversions)}")

    frequency = Counter()
    for rows, _ in ranked_by_profile.values():
        frequency.update(row["name"] for row in rows[:8])
    print("\nTOP-8 frequency across controlled probes (descriptive, not a bias verdict):")
    for name, count in frequency.most_common():
        print(f"  {count}/{len(PROFILES)}  {name}")

    target_name = "Crew Cut" if gender == "Man" else "Lob (Long Bob)"
    print(f"\nTARGET RANKS: {target_name}")
    for profile_name, (rows, _) in ranked_by_profile.items():
        rank = rank_of(rows, target_name)
        if rank is not None:
            row = rows[rank - 1]
            print(f"  {profile_name:<28} {rank:2}/{len(rows)} raw={row['score']: .4f} display={row['display']}")

    focus_profile = "wide_jaw_prominent_chin" if gender == "Man" else "narrow_jaw_recessed_chin"
    rows, user_scores = ranked_by_profile[focus_profile]
    target = next((r for r in rows if r["name"].casefold() == target_name.casefold()), None)
    if target:
        contributions = []
        for dim, user_value in user_scores.items():
            attr = target["attributes"].get(dim, 0)
            contribution = user_value * attr
            if abs(contribution) > 1e-9:
                contributions.append((dim, user_value, attr, contribution))
        negative_contribs = sorted((item for item in contributions if item[3] < 0), key=lambda x: x[3])
        positive_contribs = sorted((item for item in contributions if item[3] > 0), key=lambda x: x[3], reverse=True)
        print(f"\\nCONTRIBUTION BREAKDOWN: {target_name} / {focus_profile}")
        print("  Most negative:")
        if not negative_contribs:
            print("   none")
        for dim, user_value, attr, contribution in negative_contribs[:6]:
            print(f"   {dim:<20} rule={user_value:>6.2f} attr={attr:>5.2f} product={contribution:>7.3f}")
        print("  Most positive:")
        if not positive_contribs:
            print("   none")
        for dim, user_value, attr, contribution in positive_contribs[:6]:
            print(f"   {dim:<20} rule={user_value:>6.2f} attr={attr:>5.2f} product={contribution:>7.3f}")

    print("\nDIRECTIONAL RULE CHECKS (CHECK means inspect manually, not necessarily a bug):")
    if gender == "Man":
        long_s = apply_rules(PROFILES["long_face_high_forehead"], gender=gender)
        short_s = apply_rules(PROFILES["short_face_low_forehead"], gender=gender)
        jaw_s = apply_rules(PROFILES["wide_jaw_prominent_chin"], gender=gender)
        receding_s = apply_rules(PROFILES["receding_hairline"], gender=gender)
        checks = [
            ("long face/high forehead => positive fringe", long_s.get("fringe", 0) > 0),
            ("long face/high forehead => negative top volume", long_s.get("volume_top", 0) < 0),
            ("short face/low forehead => positive top volume", short_s.get("volume_top", 0) > 0),
            ("wide jaw/prominent chin => positive soft texture", jaw_s.get("soft_texture", 0) > 0),
            ("wide jaw/prominent chin => negative clean lines", jaw_s.get("clean_lines", 0) < 0),
            ("receding hairline => negative fringe", receding_s.get("fringe", 0) < 0),
        ]
    else:
        long_s = apply_rules(PROFILES["long_face_high_forehead"], gender=gender)
        short_s = apply_rules(PROFILES["short_face_low_forehead"], gender=gender)
        jaw_s = apply_rules(PROFILES["wide_jaw_prominent_chin"], gender=gender)
        checks = [
            ("long face/high forehead => positive curtain fringe", long_s.get("curtain_fringe", 0) > 0),
            ("long face/high forehead => negative updo", long_s.get("updo", 0) < 0),
            ("short face => positive updo", short_s.get("updo", 0) > 0),
            ("wide jaw/prominent chin => positive soft texture", jaw_s.get("soft_texture", 0) > 0),
            ("wide jaw/prominent chin => positive layers", jaw_s.get("layers", 0) > 0),
        ]
    for label, passed in checks:
        print(f"  {'PASS' if passed else 'CHECK'} {label}")

    print("\nATTRIBUTE PROFILE OF FREQUENT TOP-8 STYLES:")
    for name, count in frequency.most_common(5):
        style = next((s for s in styles if s.get("name") == name), None)
        if style:
            attrs = style.get("attributes", {})
            strongest = sorted(attrs.items(), key=lambda item: abs(float(item[1])), reverse=True)[:6]
            print(f"  {name} ({count}/{len(PROFILES)}): " +
                  ", ".join(f"{k}={v}" for k, v in strongest))

def main():
    found = 0
    for path, gender in CATALOGUES:
        if path.exists():
            found += 1
            styles = load_catalog(path)
            print_catalog(path, gender)
            contrast_report(styles, gender)
            scoring_consistency_report(styles, gender)
    if not found:
        print("ERROR: no catalogue files found.")
        print("Place this script in backend/ and run it from there.")
        sys.exit(2)

if __name__ == "__main__":
    main()
