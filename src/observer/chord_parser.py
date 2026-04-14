from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from src.dataloader.theory_helpers import build_theory_context

BODY_TYPES = (5, 7, 9, 11, 13)
BODY_TYPE_TO_TONES = {5: 3, 7: 4, 9: 5, 11: 6, 13: 7}
DEGREE_TO_LABEL = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 9: "9", 11: "11", 13: "13"}


@dataclass
class ChordCandidate:
    mode_name: str
    borrowed: bool
    root_degree_raw: int
    type_raw: int
    inversion_raw: int | None
    body_pcs: list[int]
    add_degrees: list[int]
    suspension_degrees: list[int]
    omit_degrees: list[int]
    alteration_tokens: list[str]
    explained_pcs: list[int]
    unexplained_pcs: list[int]
    missing_core_pcs: list[int]
    score: int = 0


def load_midi_notes(midi_path: str) -> list[dict[str, float | int]]:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(midi_path)
    notes: list[dict[str, float | int]] = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append({"start": float(note.start), "end": float(note.end), "pitch": int(note.pitch)})
    return notes


def select_target_instrument(pm: Any, instrument_name: str = "chords") -> Any:
    matches = [instrument for instrument in pm.instruments if not instrument.is_drum and instrument.name == instrument_name]
    if not matches:
        available = [instrument.name for instrument in pm.instruments if not instrument.is_drum]
        raise ValueError(f"Instrument '{instrument_name}' not found. Available non-drum instruments: {available}")
    if len(matches) > 1:
        raise ValueError(f"Instrument '{instrument_name}' is ambiguous: found {len(matches)} non-drum tracks with this name.")
    return matches[0]


def extract_harmonic_onsets(instrument: Any) -> list[float]:
    onsets = {float(note.start) for note in instrument.notes}
    return sorted(onsets)


def build_sounding_sonority(instrument: Any, onset_time: float) -> dict[str, Any]:
    observed_pitches: list[int] = []
    for note in instrument.notes:
        if note.start <= onset_time < note.end:
            observed_pitches.append(int(note.pitch))

    observed_pcs = sorted({p % 12 for p in observed_pitches})
    bass_pitch = min(observed_pitches) if observed_pitches else None
    bass_pc = bass_pitch % 12 if bass_pitch is not None else None
    return {
        "observed_pitches": observed_pitches,
        "observed_pcs": observed_pcs,
        "bass_pitch": bass_pitch,
        "bass_pc": bass_pc,
    }


def _mode_template(mode_name: str, theory_ctx: dict) -> list[int]:
    template = theory_ctx["mode_to_pcset"].get(mode_name)
    if template is None or len(template) < 7:
        raise ValueError(f"Unsupported mode: {mode_name}")
    return list(template[:7])


def _resolve_candidate_root_anchor(mode_name: str, root_raw: int, theory_ctx: dict) -> tuple[int, int]:
    template = _mode_template(mode_name, theory_ctx)
    if 0 <= int(root_raw) <= 6:
        return int(root_raw), template[int(root_raw)] % 12
    if int(root_raw) == 7:
        return 6, 10
    raise ValueError("root_degree_raw must be in [0, 7]")


def build_tertian_row(mode_name: str, root_degree_raw: int, theory_ctx: dict) -> list[int]:
    template = _mode_template(mode_name, theory_ctx)
    anchor_degree_idx, root_pc = _resolve_candidate_root_anchor(mode_name, root_degree_raw, theory_ctx)
    row = [root_pc]
    for tone_idx in range(7):
        if tone_idx == 0:
            continue
        degree_idx = (anchor_degree_idx + 2 * tone_idx) % 7
        row.append(template[degree_idx] % 12)
    return row


def build_body_from_tertian_row(tertian_row: list[int], type_raw: int) -> list[int]:
    if type_raw not in BODY_TYPE_TO_TONES:
        raise ValueError(f"Unsupported type_raw: {type_raw}")
    return list(tertian_row[: BODY_TYPE_TO_TONES[type_raw]])


def _build_mode_scale_degrees(mode_name: str, root_degree_raw: int, theory_ctx: dict) -> dict[int, int]:
    template = _mode_template(mode_name, theory_ctx)
    anchor_degree_idx, root_pc = _resolve_candidate_root_anchor(mode_name, root_degree_raw, theory_ctx)
    return {
        1: root_pc,
        2: template[(anchor_degree_idx + 1) % 7] % 12,
        3: template[(anchor_degree_idx + 2) % 7] % 12,
        4: template[(anchor_degree_idx + 3) % 7] % 12,
        5: template[(anchor_degree_idx + 4) % 7] % 12,
        6: template[(anchor_degree_idx + 5) % 7] % 12,
        7: template[(anchor_degree_idx + 6) % 7] % 12,
    }


def _mode_distance(mode_name: str, main_mode: str, theory_ctx: dict) -> int:
    mode_set = set(theory_ctx["mode_to_pcset"][mode_name])
    main_set = set(theory_ctx["mode_to_pcset"][main_mode])
    return len(mode_set.symmetric_difference(main_set))


def _classify_leftover(
    observed_pcs: set[int],
    body_pcs: list[int],
    tertian_row: list[int],
    mode_name: str,
    root_degree_raw: int,
    theory_ctx: dict,
) -> tuple[list[int], list[int], list[str], list[int], list[int], set[int]]:
    explained_set = set(body_pcs).intersection(observed_pcs)
    leftover = observed_pcs - explained_set

    scale_degrees = _build_mode_scale_degrees(mode_name, root_degree_raw, theory_ctx)
    expected_third = scale_degrees[3]
    sus2_pc = scale_degrees[2]
    sus4_pc = scale_degrees[4]

    suspension_degrees: list[int] = []
    if expected_third not in observed_pcs:
        if sus2_pc in leftover:
            suspension_degrees.append(2)
            explained_set.add(sus2_pc)
            leftover.discard(sus2_pc)
        if sus4_pc in leftover:
            suspension_degrees.append(4)
            explained_set.add(sus4_pc)
            leftover.discard(sus4_pc)

    extensions = {9: tertian_row[4], 11: tertian_row[5], 13: tertian_row[6]}
    add_degrees: list[int] = []
    for deg, pc in extensions.items():
        if pc in leftover:
            add_degrees.append(deg)
            explained_set.add(pc)
            leftover.discard(pc)

    all_targets = {
        1: tertian_row[0],
        3: tertian_row[1],
        5: tertian_row[2],
        7: tertian_row[3],
        9: tertian_row[4],
        11: tertian_row[5],
        13: tertian_row[6],
    }

    alteration_tokens: list[str] = []
    unresolved = sorted(leftover)
    for pc in unresolved:
        matched = False
        for degree, expected_pc in all_targets.items():
            if pc == (expected_pc + 1) % 12:
                alteration_tokens.append(f"#{DEGREE_TO_LABEL[degree]}")
                explained_set.add(pc)
                matched = True
                break
            if pc == (expected_pc - 1) % 12:
                alteration_tokens.append(f"b{DEGREE_TO_LABEL[degree]}")
                explained_set.add(pc)
                matched = True
                break
        if matched:
            leftover.discard(pc)

    omissions: list[int] = []
    missing_core_pcs: list[int] = []
    body_degree_map = {1: tertian_row[0], 3: tertian_row[1], 5: tertian_row[2], 7: tertian_row[3]}
    for degree in (1, 3, 5, 7):
        body_pc = body_degree_map[degree]
        if body_pc not in body_pcs:
            continue
        if degree == 3 and suspension_degrees:
            continue
        if body_pc not in observed_pcs:
            omissions.append(degree)
            missing_core_pcs.append(body_pc)

    unexplained_pcs = sorted(leftover)
    return (
        sorted(set(add_degrees)),
        sorted(set(suspension_degrees)),
        alteration_tokens,
        sorted(set(omissions)),
        sorted(set(missing_core_pcs)),
        explained_set,
    )


def _resolve_inversion(bass_pc: int | None, body_pcs: list[int]) -> int | None:
    if bass_pc is None:
        return None
    for inversion_raw, pc in enumerate(body_pcs[:4]):
        if bass_pc == pc:
            return inversion_raw
    return None


def explain_score_candidate(
    candidate: ChordCandidate,
    observed_pcs: list[int],
    bass_pc: int | None,
    main_mode: str,
    theory_ctx: dict,
) -> dict[str, Any]:
    observed_set = set(observed_pcs)
    body_set = set(candidate.body_pcs)
    extras_explained = set(candidate.explained_pcs) - body_set
    mode_distance = _mode_distance(candidate.mode_name, main_mode, theory_ctx)
    borrowed_mode_penalty = 1 if candidate.mode_name != main_mode else 0
    body_size_penalty = BODY_TYPES.index(candidate.type_raw)

    positive_terms = {
        "body_match_count": len(observed_set.intersection(body_set)),
        "extras_explained_count": len(extras_explained),
        "bass_matches_body": 1 if bass_pc in body_set else 0,
        "mode_equals_main": 1 if candidate.mode_name == main_mode else 0,
    }
    negative_terms = {
        "unexplained_pcs_count": len(candidate.unexplained_pcs),
        "missing_core_pcs_count": len(candidate.missing_core_pcs),
        "borrowed_mode_penalty": borrowed_mode_penalty,
        "mode_distance_penalty": mode_distance,
        "add_penalty": len(candidate.add_degrees),
        "suspension_penalty": len(candidate.suspension_degrees),
        "alteration_penalty": len(candidate.alteration_tokens),
        "omit_penalty": len(candidate.omit_degrees),
        "body_size_penalty": body_size_penalty,
    }

    human_readable_terms: list[str] = []
    for key, val in positive_terms.items():
        if val > 0:
            human_readable_terms.append(f"+{val} {key}")
    for key, val in negative_terms.items():
        if val > 0:
            human_readable_terms.append(f"-{val} {key}")

    score = 0
    score += sum(positive_terms.values())
    score -= sum(negative_terms.values())
    return {
        "total": score,
        "positive_terms": positive_terms,
        "negative_terms": negative_terms,
        "human_readable_terms": human_readable_terms,
    }


def score_candidate(candidate: ChordCandidate, observed_pcs: list[int], bass_pc: int | None, main_mode: str, theory_ctx: dict) -> int:
    return int(explain_score_candidate(candidate, observed_pcs, bass_pc, main_mode, theory_ctx)["total"])


def generate_candidates_for_mode_and_degree(
    observed_pcs: list[int],
    bass_pc: int | None,
    mode_name: str,
    degree_raw: int,
    theory_ctx: dict,
    main_mode: str,
) -> list[ChordCandidate]:
    observed_set = set(observed_pcs)
    tertian_row = build_tertian_row(mode_name, degree_raw, theory_ctx)
    candidates: list[ChordCandidate] = []

    for type_raw in BODY_TYPES:
        body_pcs = build_body_from_tertian_row(tertian_row, type_raw)
        add_degrees, suspension_degrees, alteration_tokens, omit_degrees, missing_core_pcs, explained_set = _classify_leftover(
            observed_pcs=observed_set,
            body_pcs=body_pcs,
            tertian_row=tertian_row,
            mode_name=mode_name,
            root_degree_raw=degree_raw,
            theory_ctx=theory_ctx,
        )
        unexplained_pcs = sorted(observed_set - explained_set)
        candidate = ChordCandidate(
            mode_name=mode_name,
            borrowed=mode_name != main_mode,
            root_degree_raw=degree_raw,
            type_raw=type_raw,
            inversion_raw=_resolve_inversion(bass_pc, body_pcs),
            body_pcs=body_pcs,
            add_degrees=add_degrees,
            suspension_degrees=suspension_degrees,
            omit_degrees=omit_degrees,
            alteration_tokens=sorted(alteration_tokens),
            explained_pcs=sorted(explained_set),
            unexplained_pcs=unexplained_pcs,
            missing_core_pcs=missing_core_pcs,
        )
        candidate.score = score_candidate(candidate, observed_pcs, bass_pc, main_mode, theory_ctx)
        candidates.append(candidate)

    return candidates


def generate_all_candidates(observed_pcs: list[int], bass_pc: int | None, main_mode: str, theory_ctx: dict) -> list[ChordCandidate]:
    all_modes = list(theory_ctx["mode_to_pcset"].keys())
    if main_mode not in all_modes:
        raise ValueError(f"Unknown main_mode: {main_mode}")

    candidates: list[ChordCandidate] = []
    for mode_name in all_modes:
        for degree_raw in range(8):
            candidates.extend(
                generate_candidates_for_mode_and_degree(
                    observed_pcs=observed_pcs,
                    bass_pc=bass_pc,
                    mode_name=mode_name,
                    degree_raw=degree_raw,
                    theory_ctx=theory_ctx,
                    main_mode=main_mode,
                )
            )
    return candidates


def _candidate_sort_key(candidate: ChordCandidate, main_mode: str, theory_ctx: dict) -> tuple:
    return (
        -candidate.score,
        0 if candidate.mode_name == main_mode else 1,
        _mode_distance(candidate.mode_name, main_mode, theory_ctx),
        BODY_TYPES.index(candidate.type_raw),
        len(candidate.unexplained_pcs),
        candidate.root_degree_raw,
        candidate.mode_name,
    )


def select_best_candidates(candidates: list[ChordCandidate]) -> list[ChordCandidate]:
    if not candidates:
        return []
    best_score = max(c.score for c in candidates)
    return [c for c in candidates if c.score == best_score]


def load_learned_weights(weights_yaml: str | None) -> dict[str, Any] | None:
    if weights_yaml is None:
        return None
    yaml_path = Path(weights_yaml)
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Learned weights YAML not found: {weights_yaml}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse learned weights YAML: {weights_yaml}") from exc
    if payload is None:
        raise ValueError(f"Learned weights YAML is empty: {weights_yaml}")
    if not isinstance(payload, dict):
        raise ValueError(f"Learned weights YAML must contain a mapping: {weights_yaml}")
    return payload


def _build_weighted_score_breakdown(feature_dict: dict[str, float], learned_weights: dict[str, Any]) -> dict[str, Any]:
    positive_weights = learned_weights.get("positive", {})
    negative_weights = learned_weights.get("negative", {})
    bias = float(learned_weights.get("bias", 0.0))
    positive_terms = {name: float(positive_weights.get(name, 0.0)) * float(feature_dict.get(name, 0.0)) for name in positive_weights}
    negative_terms = {name: float(negative_weights.get(name, 0.0)) * float(feature_dict.get(name, 0.0)) for name in negative_weights}
    total = bias + sum(positive_terms.values()) - sum(negative_terms.values())
    return {
        "total": float(total),
        "bias": bias,
        "positive_terms": positive_terms,
        "negative_terms": negative_terms,
    }


def rerank_candidates_with_learned_weights(
    candidates: list[ChordCandidate],
    observed_pcs: list[int],
    bass_pc: int | None,
    main_mode: str,
    theory_ctx: dict[str, Any],
    learned_weights: dict[str, Any] | None,
) -> tuple[list[ChordCandidate], dict[int, dict[str, Any]], str]:
    if learned_weights is None:
        sorted_candidates = sorted(candidates, key=lambda c: _candidate_sort_key(c, main_mode, theory_ctx))
        return sorted_candidates, {}, "manual"

    from src.observer.chord_score_fitting import compute_weighted_candidate_score, extract_candidate_feature_dict

    weighted_breakdowns: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        feature_dict = extract_candidate_feature_dict(candidate, observed_pcs, bass_pc, main_mode, theory_ctx)
        candidate.score = float(compute_weighted_candidate_score(feature_dict, learned_weights))
        weighted_breakdowns[id(candidate)] = _build_weighted_score_breakdown(feature_dict, learned_weights)
    sorted_candidates = sorted(candidates, key=lambda c: _candidate_sort_key(c, main_mode, theory_ctx))
    return sorted_candidates, weighted_breakdowns, "learned_weights"


def _serialize_candidate(
    candidate: ChordCandidate,
    observed_pcs: list[int] | None = None,
    bass_pc: int | None = None,
    main_mode: str | None = None,
    theory_ctx: dict[str, Any] | None = None,
    include_score_breakdown: bool = False,
    score_source: str = "manual",
    weighted_breakdown: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["score_source"] = score_source
    if include_score_breakdown:
        if observed_pcs is None or main_mode is None or theory_ctx is None:
            raise ValueError("observed_pcs, main_mode, and theory_ctx are required when include_score_breakdown=True")
        manual_breakdown = explain_score_candidate(candidate, observed_pcs, bass_pc, main_mode, theory_ctx)
        if weighted_breakdown is None:
            payload["score_breakdown"] = manual_breakdown
        else:
            payload["score_breakdown_manual"] = manual_breakdown
            payload["score_breakdown_weighted"] = weighted_breakdown
    return payload


def _observer_event_from_candidate(
    candidate: ChordCandidate,
    onset_time: float,
    offset_time: float,
    score_source: str,
) -> dict[str, Any]:
    return {
        "onset_time": float(onset_time),
        "offset_time": float(offset_time),
        "root_degree_raw": int(candidate.root_degree_raw),
        "type_raw": int(candidate.type_raw),
        "inversion_raw": None if candidate.inversion_raw is None else int(candidate.inversion_raw),
        "mode_name": candidate.mode_name,
        "borrowed": bool(candidate.borrowed),
        "add_degrees": [int(v) for v in candidate.add_degrees],
        "suspension_degrees": [int(v) for v in candidate.suspension_degrees],
        "omit_degrees": [int(v) for v in candidate.omit_degrees],
        "alteration_tokens": list(candidate.alteration_tokens),
        "score": float(candidate.score),
        "score_source": score_source,
    }


def _merge_signature(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event["root_degree_raw"],
        event["type_raw"],
        event["inversion_raw"],
        event["mode_name"],
        event["borrowed"],
        tuple(event["add_degrees"]),
        tuple(event["suspension_degrees"]),
        tuple(event["omit_degrees"]),
        tuple(event["alteration_tokens"]),
    )


def merge_consecutive_observer_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not events:
        return []
    merged = [dict(events[0])]
    for current in events[1:]:
        if _merge_signature(merged[-1]) == _merge_signature(current):
            # Keep the first event score/score_source for stable, deterministic merges.
            merged[-1]["offset_time"] = float(current["offset_time"])
        else:
            merged.append(dict(current))
    return merged


def _predict_debug_onsets(
    target_instrument: Any,
    tonic_pc: int,
    main_mode: str,
    theory_ctx: dict[str, Any],
    include_all_candidates: bool,
    include_score_breakdown: bool,
    learned_weights: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for onset_time in extract_harmonic_onsets(target_instrument):
        sonority = build_sounding_sonority(target_instrument, onset_time)
        observed_pcs = sonority["observed_pcs"]
        if len(observed_pcs) < 3:
            continue

        rel_observed_pcs = sorted({(pc - tonic_pc) % 12 for pc in observed_pcs})
        bass_pc = sonority["bass_pc"]
        rel_bass_pc = None if bass_pc is None else (bass_pc - tonic_pc) % 12

        candidates = generate_all_candidates(rel_observed_pcs, rel_bass_pc, main_mode, theory_ctx)
        sorted_candidates, weighted_breakdowns, score_source = rerank_candidates_with_learned_weights(
            candidates,
            observed_pcs=rel_observed_pcs,
            bass_pc=rel_bass_pc,
            main_mode=main_mode,
            theory_ctx=theory_ctx,
            learned_weights=learned_weights,
        )
        best_candidates = sorted(
            select_best_candidates(sorted_candidates),
            key=lambda c: _candidate_sort_key(c, main_mode, theory_ctx),
        )
        candidate_key = "candidates" if include_all_candidates else "best_candidates"
        selected = sorted_candidates if include_all_candidates else best_candidates

        results.append(
            {
                "onset_time": float(onset_time),
                "observed_pcs": rel_observed_pcs,
                candidate_key: [
                    _serialize_candidate(
                        c,
                        observed_pcs=rel_observed_pcs,
                        bass_pc=rel_bass_pc,
                        main_mode=main_mode,
                        theory_ctx=theory_ctx,
                        include_score_breakdown=include_score_breakdown,
                        score_source=score_source,
                        weighted_breakdown=weighted_breakdowns.get(id(c)),
                    )
                    for c in selected
                ],
            }
        )
    return results


def _predict_observer_events(
    target_instrument: Any,
    tonic_pc: int,
    main_mode: str,
    theory_ctx: dict[str, Any],
    learned_weights: dict[str, Any] | None,
    merge_consecutive: bool,
) -> list[dict[str, Any]]:
    # Observer events are emitted only on note onsets; each onset is evaluated with
    # the full sounding sonority at that moment. Harmonic changes caused only by
    # note releases (without a new onset) do not create additional events.
    onsets = extract_harmonic_onsets(target_instrument)
    provisional_events: list[tuple[float, ChordCandidate, str]] = []
    for onset_time in onsets:
        sonority = build_sounding_sonority(target_instrument, onset_time)
        observed_pcs = sonority["observed_pcs"]
        if len(observed_pcs) < 3:
            continue

        rel_observed_pcs = sorted({(pc - tonic_pc) % 12 for pc in observed_pcs})
        bass_pc = sonority["bass_pc"]
        rel_bass_pc = None if bass_pc is None else (bass_pc - tonic_pc) % 12

        candidates = generate_all_candidates(rel_observed_pcs, rel_bass_pc, main_mode, theory_ctx)
        sorted_candidates, _, score_source = rerank_candidates_with_learned_weights(
            candidates,
            observed_pcs=rel_observed_pcs,
            bass_pc=rel_bass_pc,
            main_mode=main_mode,
            theory_ctx=theory_ctx,
            learned_weights=learned_weights,
        )
        if not sorted_candidates:
            continue
        top1 = sorted_candidates[0]
        provisional_events.append((float(onset_time), top1, score_source))

    events: list[dict[str, Any]] = []
    if provisional_events:
        max_end = max((float(note.end) for note in target_instrument.notes), default=float(provisional_events[-1][0]))
        for idx, (onset_time, candidate, score_source) in enumerate(provisional_events):
            if idx + 1 < len(provisional_events):
                offset_time = float(provisional_events[idx + 1][0])
            else:
                offset_time = max_end
            events.append(
                _observer_event_from_candidate(
                    candidate=candidate,
                    onset_time=onset_time,
                    offset_time=offset_time,
                    score_source=score_source,
                )
            )
    if merge_consecutive:
        return merge_consecutive_observer_events(events)
    return events


def predict_observer_chords_for_midi(
    midi_path: str,
    tonic_pc: int,
    main_mode: str,
    instrument_name: str = "chords",
    weights_yaml: str | None = None,
    merge_consecutive: bool = True,
) -> list[dict[str, Any]]:
    """
    Predict observer-level chord events from a target MIDI instrument.

    Events are considered only at note onset times of the target instrument, and
    each onset is analyzed using the sounding sonority (all currently sounding
    notes). This means harmony changes caused only by note releases are not
    emitted as separate events until a subsequent onset appears.
    """
    if not 0 <= int(tonic_pc) <= 11:
        raise ValueError("tonic_pc must be in [0, 11]")

    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    theory_ctx = build_theory_context()
    if main_mode not in theory_ctx["mode_to_pcset"]:
        raise ValueError(f"Unknown main_mode: {main_mode}")
    learned_weights = load_learned_weights(weights_yaml)
    target_instrument = select_target_instrument(pm, instrument_name=instrument_name)
    return _predict_observer_events(
        target_instrument=target_instrument,
        tonic_pc=tonic_pc,
        main_mode=main_mode,
        theory_ctx=theory_ctx,
        learned_weights=learned_weights,
        merge_consecutive=merge_consecutive,
    )


def predict_chords_for_midi(
    midi_path: str,
    tonic_pc: int,
    main_mode: str,
    instrument_name: str = "chords",
    include_all_candidates: bool = False,
    include_score_breakdown: bool = False,
    weights_yaml: str | None = None,
) -> list[dict[str, Any]]:
    if not 0 <= int(tonic_pc) <= 11:
        raise ValueError("tonic_pc must be in [0, 11]")

    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    theory_ctx = build_theory_context()
    if main_mode not in theory_ctx["mode_to_pcset"]:
        raise ValueError(f"Unknown main_mode: {main_mode}")
    learned_weights = load_learned_weights(weights_yaml)

    target_instrument = select_target_instrument(pm, instrument_name=instrument_name)
    return _predict_debug_onsets(
        target_instrument=target_instrument,
        tonic_pc=tonic_pc,
        main_mode=main_mode,
        theory_ctx=theory_ctx,
        include_all_candidates=include_all_candidates,
        include_score_breakdown=include_score_breakdown,
        learned_weights=learned_weights,
    )


def _cli() -> None:
    theory_ctx = build_theory_context()
    parser = argparse.ArgumentParser(description="Predict onset-level chord candidates from MIDI.")
    parser.add_argument("--midi-path", required=True)
    parser.add_argument("--tonic-pc", required=True, type=int)
    parser.add_argument("--mode", required=True, choices=sorted(theory_ctx["mode_to_pcset"].keys()))
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--instrument-name", type=str, default="chords")
    parser.add_argument("--weights-yaml", type=str, default=None)
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args()

    predictions = predict_observer_chords_for_midi(
        args.midi_path,
        args.tonic_pc,
        args.mode,
        instrument_name=args.instrument_name,
        weights_yaml=args.weights_yaml,
        merge_consecutive=not args.no_merge,
    )

    if args.pretty:
        for idx, event in enumerate(predictions, start=1):
            print(
                f"[{idx}] onset={event['onset_time']:.3f}s offset={event['offset_time']:.3f}s "
                f"mode={event['mode_name']} root={event['root_degree_raw']} type={event['type_raw']} "
                f"inv={event['inversion_raw']} borrowed={event['borrowed']} score={event['score']} ({event['score_source']})"
            )
    else:
        print(json.dumps(predictions, ensure_ascii=False, indent=2))

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    _cli()
