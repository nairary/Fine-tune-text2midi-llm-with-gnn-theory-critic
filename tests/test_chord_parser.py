from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataloader.theory_helpers import build_theory_context
from src.observer.chord_parser import ChordCandidate, generate_all_candidates, select_best_candidates


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


if __name__ == "__main__":
    unittest.main()
