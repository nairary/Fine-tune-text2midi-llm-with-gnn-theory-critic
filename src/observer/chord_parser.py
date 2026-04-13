from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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


def _serialize_candidate(
    candidate: ChordCandidate,
    observed_pcs: list[int] | None = None,
    bass_pc: int | None = None,
    main_mode: str | None = None,
    theory_ctx: dict[str, Any] | None = None,
    include_score_breakdown: bool = False,
) -> dict[str, Any]:
    payload = asdict(candidate)
    if include_score_breakdown:
        if observed_pcs is None or main_mode is None or theory_ctx is None:
            raise ValueError("observed_pcs, main_mode, and theory_ctx are required when include_score_breakdown=True")
        payload["score_breakdown"] = explain_score_candidate(candidate, observed_pcs, bass_pc, main_mode, theory_ctx)
    return payload


def predict_chords_for_midi(
    midi_path: str,
    tonic_pc: int,
    main_mode: str,
    instrument_name: str = "chords",
    include_all_candidates: bool = False,
    include_score_breakdown: bool = False,
) -> list[dict[str, Any]]:
    if not 0 <= int(tonic_pc) <= 11:
        raise ValueError("tonic_pc must be in [0, 11]")

    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    theory_ctx = build_theory_context()

    target_instrument = select_target_instrument(pm, instrument_name=instrument_name)
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
        sorted_candidates = sorted(candidates, key=lambda c: _candidate_sort_key(c, main_mode, theory_ctx))
        best_candidates = sorted(select_best_candidates(candidates), key=lambda c: _candidate_sort_key(c, main_mode, theory_ctx))
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
                    )
                    for c in selected
                ],
            }
        )

    return results


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Predict onset-level chord candidates from MIDI.")
    parser.add_argument("--midi-path", required=True)
    parser.add_argument("--tonic-pc", required=True, type=int)
    parser.add_argument("--mode", required=True, choices=["major", "minor"])
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--instrument-name", type=str, default="chords")
    parser.add_argument("--all-candidates", action="store_true")
    parser.add_argument("--debug-score", action="store_true")
    args = parser.parse_args()

    predictions = predict_chords_for_midi(
        args.midi_path,
        args.tonic_pc,
        args.mode,
        instrument_name=args.instrument_name,
        include_all_candidates=args.all_candidates,
        include_score_breakdown=args.debug_score,
    )

    if args.top_k is not None and args.top_k > 0:
        for onset_payload in predictions:
            candidate_key = "candidates" if args.all_candidates else "best_candidates"
            onset_payload[candidate_key] = onset_payload[candidate_key][: args.top_k]

    if args.pretty:
        for onset_payload in predictions:
            print(f"onset={onset_payload['onset_time']:.3f}s pcs={onset_payload['observed_pcs']}")
            candidate_key = "candidates" if args.all_candidates else "best_candidates"
            for idx, candidate in enumerate(onset_payload[candidate_key], start=1):
                print(
                    f"  [{idx}] mode={candidate['mode_name']} borrowed={candidate['borrowed']} "
                    f"root={candidate['root_degree_raw']} type={candidate['type_raw']} inv={candidate['inversion_raw']} "
                    f"adds={candidate['add_degrees']} sus={candidate['suspension_degrees']} "
                    f"omits={candidate['omit_degrees']} alt={candidate['alteration_tokens']} score={candidate['score']}"
                )
                if args.debug_score and "score_breakdown" in candidate:
                    for term in candidate["score_breakdown"]["human_readable_terms"]:
                        print(f"      {term}")
    else:
        print(json.dumps(predictions, ensure_ascii=False, indent=2))

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    _cli()
