from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataloader.theory_helpers import build_theory_context
from src.observer.chord_parser import (
    ChordCandidate,
    _candidate_sort_key,
    build_sounding_sonority,
    explain_score_candidate,
    extract_harmonic_onsets,
    generate_all_candidates,
    predict_chords_for_midi,
    score_candidate,
    select_best_candidates,
    select_target_instrument,
)


class ChordParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = build_theory_context()

    def _best(self, observed_pcs: list[int], bass_pc: int, main_mode: str = "major") -> list[ChordCandidate]:
        candidates = generate_all_candidates(observed_pcs, bass_pc, main_mode, self.ctx)
        return select_best_candidates(candidates)

    def test_simple_triad(self):
        best = self._best([0, 4, 7], bass_pc=0, main_mode="major")
        self.assertTrue(any(c.root_degree_raw == 0 and c.type_raw == 5 and c.inversion_raw == 0 for c in best))

    def test_bvii_candidate_appears(self):
        best = self._best([10, 2, 5], bass_pc=10, main_mode="major")
        self.assertTrue(any(c.root_degree_raw == 7 and c.type_raw == 5 for c in best))

    def test_inversion_prefers_degree_i_not_iii(self):
        best = self._best([0, 4, 7], bass_pc=4, main_mode="major")
        self.assertTrue(any(c.root_degree_raw == 0 and c.inversion_raw == 1 for c in best))
        self.assertFalse(any(c.root_degree_raw == 2 for c in best))

    def test_add_note(self):
        best = self._best([0, 2, 4, 7], bass_pc=0, main_mode="major")
        self.assertTrue(any(c.root_degree_raw == 0 and c.type_raw == 5 and 9 in c.add_degrees for c in best))

    def test_sus_chord(self):
        best = self._best([0, 5, 7], bass_pc=0, main_mode="major")
        self.assertTrue(any(c.root_degree_raw == 0 and 4 in c.suspension_degrees for c in best))

    def test_borrowed_candidate(self):
        best = self._best([3, 6, 8], bass_pc=3, main_mode="major")
        self.assertTrue(all(c.borrowed and c.mode_name != "major" for c in best))

    def test_tie_case_returns_all(self):
        c1 = ChordCandidate("major", False, 0, 5, 0, [0, 4, 7], [], [], [], [], [0, 4, 7], [], [], score=10)
        c2 = ChordCandidate("minor", True, 0, 5, 0, [0, 3, 7], [], [], [], [], [0, 3, 7], [], [], score=10)
        best = select_best_candidates([c1, c2])
        self.assertEqual(len(best), 2)

    def _build_synthetic_midi(self):
        def note(pitch: int, start: float, end: float):
            return SimpleNamespace(pitch=pitch, start=start, end=end)

        melody = SimpleNamespace(
            name="melody",
            is_drum=False,
            notes=[note(72, 0.25, 0.75), note(74, 1.25, 1.75)],
        )
        chords = SimpleNamespace(
            name="chords",
            is_drum=False,
            notes=[
                note(60, 0.0, 1.0),
                note(64, 0.0, 1.0),
                note(67, 0.0, 1.0),
                note(62, 1.0, 2.0),
                note(65, 1.0, 2.0),
                note(69, 1.0, 2.0),
            ],
        )
        return SimpleNamespace(instruments=[melody, chords])

    def test_select_target_instrument(self):
        pm = self._build_synthetic_midi()
        instrument = select_target_instrument(pm, "chords")
        self.assertEqual(instrument.name, "chords")
        self.assertEqual(len(instrument.notes), 6)

    def test_extract_harmonic_onsets_only_selected_instrument(self):
        pm = self._build_synthetic_midi()
        chords = select_target_instrument(pm, "chords")
        onsets = extract_harmonic_onsets(chords)
        self.assertEqual(onsets, [0.0, 1.0])

    def test_build_sounding_sonority_only_selected_instrument(self):
        pm = self._build_synthetic_midi()
        chords = select_target_instrument(pm, "chords")
        sonority = build_sounding_sonority(chords, 0.25)
        self.assertEqual(sorted(sonority["observed_pitches"]), [60, 64, 67])
        self.assertEqual(sonority["observed_pcs"], [0, 4, 7])

    def test_debug_score_breakdown_consistent(self):
        candidates = generate_all_candidates([0, 4, 7], 0, "major", self.ctx)
        triad = next(c for c in candidates if c.mode_name == "major" and c.root_degree_raw == 0 and c.type_raw == 5)
        breakdown = explain_score_candidate(triad, [0, 4, 7], 0, "major", self.ctx)
        self.assertEqual(breakdown["total"], score_candidate(triad, [0, 4, 7], 0, "major", self.ctx))
        self.assertTrue(breakdown["positive_terms"])
        self.assertTrue(breakdown["negative_terms"])
        self.assertTrue(breakdown["human_readable_terms"])

    def test_all_candidates_sorted(self):
        pm = self._build_synthetic_midi()
        fake_pretty_midi = SimpleNamespace(PrettyMIDI=lambda _: pm)
        with patch.dict(sys.modules, {"pretty_midi": fake_pretty_midi}):
            predictions = predict_chords_for_midi(
                "unused.mid",
                tonic_pc=0,
                main_mode="major",
                instrument_name="chords",
                include_all_candidates=True,
            )
        first_onset = predictions[0]
        candidates = first_onset["candidates"]
        self.assertGreater(len(candidates), 1)
        restored = [ChordCandidate(**{k: v for k, v in c.items() if k != "score_breakdown"}) for c in candidates]
        sorted_copy = sorted(restored, key=lambda c: _candidate_sort_key(c, "major", self.ctx))
        self.assertEqual([c.score for c in restored], [c.score for c in sorted_copy])


if __name__ == "__main__":
    unittest.main()
