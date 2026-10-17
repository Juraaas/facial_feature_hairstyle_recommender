from __future__ import annotations
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from src.rules import apply_rules
from src.recommender import score_hairstyle 

MALE_DIMS = {
    "volume_top", "volume_sides", "short_sides", "longer_hair",
    "fringe", "clean_lines", "soft_texture", "textured_top",
}
FEMALE_DIMS = {
    "volume_top", "volume_sides", "layers", "longer_hair", "fringe",
    "clean_lines", "soft_texture", "textured_top", "updo", "curtain_fringe",
}

BASE_TRAITS = {
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

PROFILE_CASES = {
    "balanced_high_symmetry": {
        "symmetry": "high",
    },
    "long_face_high_forehead": {
        "face_length": "long",
        "forehead": "high",
    },
    "short_face_low_forehead": {
        "face_length": "short",
        "forehead": "low",
    },
    "wide_jaw_prominent_chin": {
        "jaw": "wide",
        "chin": "prominent",
    },
    "narrow_jaw_recessed_chin": {
        "jaw": "narrow",
        "chin": "recessed",
    },
    "close_set_eyes": {
        "eyes": "close",
    },
    "curly_hair": {
        "hair_type": "curly",
    },
    "coily_hair": {
        "hair_type": "coily",
    },
    "receding_hairline": {
        "hairline": "receding",
    },
    "imbalanced_thirds": {
        "thirds_balance": "imbalanced",
    },
}

def _load_style_catalogues() -> list[tuple[str, str, list[dict[str, Any]], Path]]:
    data_dir = Path(__file__).resolve().parents[1] / "backend" / "data"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Nie znaleziono katalogu {data_dir}. Uruchom skrypt z repozytorium "
            "Stylizzer i sprawdź położenie folderu data/."
        )

    preferred_names = [
        "hairstyles.json",
        "hairstyles_female.json",
    ]
    candidates: list[Path] = []
    for name in preferred_names:
        path = data_dir / name
        if path.exists():
            candidates.append(path)
    for path in sorted(data_dir.rglob("*.json")):
        if path not in candidates:
            candidates.append(path)

    catalogues = []
    seen_paths = set()
    for path in candidates:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        styles = payload.get("styles") if isinstance(payload, dict) else None
        if not isinstance(styles, list) or not styles:
            continue
        if not all(
            isinstance(style, dict)
            and isinstance(style.get("name"), str)
            and isinstance(style.get("attributes"), dict)
            for style in styles
        ):
            continue

        all_keys = set().union(*(set(s["attributes"]) for s in styles))
        female_signal = bool(all_keys & {"layers", "updo", "curtain_fringe"})
        gender = "Woman" if female_signal else "Man"
        catalogues.append((path.name, gender, styles, path))

    if not catalogues:
        raise FileNotFoundError(
            f"Nie znalazłem katalogu fryzur w formacie "
            '{"styles": [{"name": "...", "attributes": {...}}]} '
            f"w {data_dir}."
        )
    return catalogues


def _print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _check_catalogue(
    label: str, gender: str, styles: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    expected = FEMALE_DIMS if gender == "Woman" else MALE_DIMS
    names = [style["name"].strip() for style in styles]

    if len(set(names)) != len(names):
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        errors.append(f"{label}: powtarzające się nazwy fryzur: {duplicates}")

    for style in styles:
        name = style["name"]
        attrs = style["attributes"]
        missing = sorted(expected - set(attrs))
        if missing:
            warnings.append(f"{label}/{name}: brak atrybutów {missing}")

        for key, value in attrs.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"{label}/{name}: {key} nie jest liczbą: {value!r}")
            elif not math.isfinite(float(value)):
                errors.append(f"{label}/{name}: {key} nie jest wartością skończoną")
            elif not 0.0 <= float(value) <= 1.0:
                warnings.append(
                    f"{label}/{name}: {key}={value} poza oczekiwanym zakresem 0–1"
                )

    print(f"\nKatalog: {label} | płeć reguł: {gender} | fryzury: {len(styles)}")
    print(f"Unikalne nazwy: {len(set(names))}/{len(names)}")
    if warnings:
        print(f"Ostrzeżenia strukturalne: {len(warnings)}")
        for warning in warnings:
            print(f"  [WARN] {warning}")
    else:
        print("Struktura atrybutów: bez wykrytych braków / odchyleń.")
    return errors, warnings


def _rank_profile(
    gender: str,
    styles: list[dict[str, Any]],
    overrides: dict[str, str],
) -> tuple[list[tuple[float, str]], dict[str, int], dict[str, float]]:
    traits = {**BASE_TRAITS, **overrides}
    user_scores = apply_rules(traits, gender=gender)

    ranked = []
    for style in styles:
        score = score_hairstyle(user_scores, style)
        ranked.append((float(score), style["name"]))
    ranked.sort(key=lambda item: item[0], reverse=True)

    score_counts = Counter(round(score, 8) for score, _ in ranked)
    zero_count = sum(1 for score, _ in ranked if abs(score) < 1e-9)
    stats = {
        "zero_share": zero_count / len(ranked) if ranked else 0.0,
        "top_score": ranked[0][0] if ranked else 0.0,
        "bottom_score": ranked[-1][0] if ranked else 0.0,
        "unique_scores": float(len(score_counts)),
        "largest_tie": float(max(score_counts.values(), default=0)),
    }
    ranks = {name: index + 1 for index, (_, name) in enumerate(ranked)}
    return ranked, ranks, stats


def _find_rank(ranks: dict[str, int], candidates: list[str]) -> str | None:
    for candidate in candidates:
        for name in ranks:
            if name.casefold() == candidate.casefold():
                return name
    return None


def _evaluate_catalogue(
    label: str, gender: str, styles: list[dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    appearance_counts = Counter()
    ranks_by_profile: dict[str, dict[str, int]] = {}
    stats_by_profile = {}

    _print_section(f"PROFILE RANKINGS — {label}")
    for profile_name, overrides in PROFILE_CASES.items():
        ranked, ranks, stats = _rank_profile(gender, styles, overrides)
        ranks_by_profile[profile_name] = ranks
        stats_by_profile[profile_name] = stats
        for _, name in ranked[:8]:
            appearance_counts[name] += 1

        print(f"\n[{profile_name}]")
        print(
            f"  score range: {stats['bottom_score']:.3f} .. {stats['top_score']:.3f}"
            f" | score=0: {stats['zero_share']:.0%}"
            f" | unikalne wyniki: {int(stats['unique_scores'])}"
            f" | największy remis: {int(stats['largest_tie'])}"
        )
        print("  TOP 8:")
        for position, (score, name) in enumerate(ranked[:8], start=1):
            print(f"    {position:>2}. {name:<30} {score: .4f}")

        if len(ranked) != len(styles):
            errors.append(
                f"{label}/{profile_name}: ranking ma {len(ranked)} pozycji, "
                f"a katalog {len(styles)} fryzur."
            )
        if any(not math.isfinite(score) for score, _ in ranked):
            errors.append(f"{label}/{profile_name}: wykryto NaN/inf w wynikach.")

    _print_section(f"FREQUENCY ACROSS PROFILES — {label}")
    print(
        "Liczba wejść do TOP 8 spośród "
        f"{len(PROFILE_CASES)} profili (maks. {len(PROFILE_CASES)}):"
    )
    for name, count in appearance_counts.most_common():
        print(f"  {count:>2}/{len(PROFILE_CASES)}  {name}")

    _print_section(f"TARGETED CHECKS — {label}")
    if gender == "Man":
        targets = [
            ("Buzz Cut", ["Buzz Cut"]),
            ("Crew Cut", ["Crew Cut"]),
        ]
    else:
        targets = [
            ("Lob", ["Lob (Long Bob)", "Lob", "Long Bob"])
        ]

    for display_name, aliases in targets:
        found = False
        for profile_name, ranks in ranks_by_profile.items():
            actual_name = _find_rank(ranks, aliases)
            if actual_name:
                found = True
                rank = ranks[actual_name]
                print(f"{display_name:<12} | {profile_name:<28} | rank {rank:>2}/{len(styles)}")
        if not found:
            print(f"{display_name}: NOT FOUND in this catalogue")
            errors.append(f"{label}: nie znaleziono oczekiwanej fryzury {display_name}.")

    high_zero_profiles = [
        name for name, stats in stats_by_profile.items() if stats["zero_share"] >= 0.5
    ]
    if high_zero_profiles:
        print(
            "\n[REVIEW] Co najmniej połowa wyników jest równa 0 dla profili: "
            + ", ".join(high_zero_profiles)
            + ". Sprawdź, czy clipping w apply_hair_compatibility nie spłaszcza rankingu."
        )

    if gender == "Man":
        for target in ("Buzz Cut", "Crew Cut"):
            ranks = [
                _find_rank(profile_ranks, [target])
                for profile_ranks in ranks_by_profile.values()
            ]
            present_ranks = [
                profile_ranks[name]
                for profile_ranks in ranks_by_profile.values()
                if (name := _find_rank(profile_ranks, [target])) is not None
            ]
            if present_ranks and all(rank > 8 for rank in present_ranks):
                print(
                    f"[REVIEW] {target} nie trafia do TOP 8 w żadnym profilu testowym. "
                    "To sygnał do oceny, nie automatyczny dowód błędu."
                )
    else:
        present_ranks = [
            profile_ranks[name]
            for profile_ranks in ranks_by_profile.values()
            if (name := _find_rank(profile_ranks, ["Lob", "Long Bob"])) is not None
        ]
        if present_ranks and all(rank > 8 for rank in present_ranks):
            print(
                "[REVIEW] Lob/Long Bob nie trafia do TOP 8 w żadnym profilu testowym. "
                "To sygnał do oceny, nie automatyczny dowód błędu."
            )

    return errors


def main() -> int:
    try:
        catalogues = _load_style_catalogues()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2

    _print_section("STYLIZZER — RECOMMENDATION VALIDATION")
    print(f"Znalezione katalogi: {len(catalogues)}")
    print(
        "Uwaga: testy profili są kontrolowanymi sondami diagnostycznymi, "
        "a nie oceną fryzur dla prawdziwej populacji."
    )

    errors: list[str] = []
    warnings: list[str] = []
    for label, gender, styles, _path in catalogues:
        catalogue_errors, catalogue_warnings = _check_catalogue(label, gender, styles)
        errors.extend(catalogue_errors)
        warnings.extend(catalogue_warnings)
        errors.extend(_evaluate_catalogue(label, gender, styles))

    _print_section("FINAL SUMMARY")
    print(f"Katalogi przeanalizowane: {len(catalogues)}")
    print(f"Błędy blokujące: {len(errors)}")
    print(f"Ostrzeżenia strukturalne: {len(warnings)}")
    if errors:
        for error in errors:
            print(f"  [ERROR] {error}")
        print("\nWynik: wykryto problemy wymagające sprawdzenia.")
        return 1

    print(
        "\nWynik: nie wykryto błędów blokujących. Przejrzyj sekcje [REVIEW] "
        "i rankingi — same częstotliwości TOP 8 nie uzasadniają jeszcze zmiany wag."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
