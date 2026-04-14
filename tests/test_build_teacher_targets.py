from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.observer.build_teacher_targets import TeacherTargetBuildError, build_teacher_targets, load_jsonl_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class BuildTeacherTargetsTests(unittest.TestCase):
    def test_load_jsonl_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad.jsonl"
            path.write_text('{"song_id": "s1"\n', encoding="utf-8")
            with self.assertRaises(TeacherTargetBuildError):
                load_jsonl_rows(path)

    def test_build_success(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            song_path_1 = tmp / "s1.json"
            song_path_2 = tmp / "s2.json"
            song_path_1.write_text(json.dumps({"meta": {}, "melody": [], "chords": []}), encoding="utf-8")
            song_path_2.write_text(json.dumps({"meta": {}, "melody": [], "chords": []}), encoding="utf-8")
            cfg_path = tmp / "cfg.yaml"
            ckpt_path = tmp / "ckpt.pt"
            cfg_path.write_text("model: {}\n", encoding="utf-8")
            ckpt_path.write_text("x", encoding="utf-8")

            rows = [
                {"song_id": "s1", "encoded_song_path": str(song_path_1), "midi_path": "a.mid"},
                {"song_id": "s2", "encoded_song_path": str(song_path_2), "midi_path": "b.mid"},
            ]
            scores = [{"graph_score": 0.5}, {"graph_score": 1.5}]

            with patch("src.observer.build_teacher_targets.build_model_from_config", return_value=object()), patch(
                "src.observer.build_teacher_targets.score_song", side_effect=scores
            ), patch("omegaconf.OmegaConf.load", return_value=object()):
                out_rows = build_teacher_targets(
                    rows=rows,
                    teacher_checkpoint=ckpt_path,
                    teacher_config=cfg_path,
                    encoded_song_field="encoded_song_path",
                    encoded_song_root=None,
                    split="train",
                    device="cpu",
                )

        self.assertEqual(len(out_rows), 2)
        self.assertEqual(out_rows[0]["song_id"], "s1")
        self.assertAlmostEqual(out_rows[0]["teacher_score"], 0.5)
        self.assertEqual(out_rows[0]["split"], "train")

    def test_missing_song_id_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cfg_path = tmp / "cfg.yaml"
            ckpt_path = tmp / "ckpt.pt"
            cfg_path.write_text("model: {}\n", encoding="utf-8")
            ckpt_path.write_text("x", encoding="utf-8")

            with self.assertRaisesRegex(TeacherTargetBuildError, "Line 1: song_id"):
                build_teacher_targets(
                    rows=[{"encoded_song_path": "x.json"}],
                    teacher_checkpoint=ckpt_path,
                    teacher_config=cfg_path,
                    encoded_song_field="encoded_song_path",
                    encoded_song_root=None,
                    split=None,
                    device="cpu",
                )

    def test_duplicate_song_id_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            song_path = tmp / "s1.json"
            song_path.write_text(json.dumps({"meta": {}, "melody": [], "chords": []}), encoding="utf-8")
            cfg_path = tmp / "cfg.yaml"
            ckpt_path = tmp / "ckpt.pt"
            cfg_path.write_text("model: {}\n", encoding="utf-8")
            ckpt_path.write_text("x", encoding="utf-8")

            rows = [
                {"song_id": "s1", "encoded_song_path": str(song_path)},
                {"song_id": "s1", "encoded_song_path": str(song_path)},
            ]

            with patch("src.observer.build_teacher_targets.build_model_from_config", return_value=object()), patch(
                "src.observer.build_teacher_targets.score_song", return_value={"graph_score": 0.5}
            ), patch("omegaconf.OmegaConf.load", return_value=object()):
                with self.assertRaisesRegex(TeacherTargetBuildError, "Duplicate song_id"):
                    build_teacher_targets(
                        rows=rows,
                        teacher_checkpoint=ckpt_path,
                        teacher_config=cfg_path,
                        encoded_song_field="encoded_song_path",
                        encoded_song_root=None,
                        split=None,
                        device="cpu",
                    )

    def test_missing_encoded_song_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cfg_path = tmp / "cfg.yaml"
            ckpt_path = tmp / "ckpt.pt"
            cfg_path.write_text("model: {}\n", encoding="utf-8")
            ckpt_path.write_text("x", encoding="utf-8")
            rows = [{"song_id": "s1", "encoded_song_path": str(tmp / "missing.json")}]

            with self.assertRaisesRegex(TeacherTargetBuildError, "bootstrap teacher model"):
                build_teacher_targets(
                    rows=rows,
                    teacher_checkpoint=ckpt_path,
                    teacher_config=cfg_path,
                    encoded_song_field="encoded_song_path",
                    encoded_song_root=None,
                    split=None,
                    device="cpu",
                )

    def test_non_finite_teacher_score_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            song_path = tmp / "s1.json"
            song_path.write_text(json.dumps({"meta": {}, "melody": [], "chords": []}), encoding="utf-8")
            cfg_path = tmp / "cfg.yaml"
            ckpt_path = tmp / "ckpt.pt"
            cfg_path.write_text("model: {}\n", encoding="utf-8")
            ckpt_path.write_text("x", encoding="utf-8")
            rows = [{"song_id": "s1", "encoded_song_path": str(song_path)}]

            with patch("src.observer.build_teacher_targets.build_model_from_config", return_value=object()), patch(
                "src.observer.build_teacher_targets.score_song", return_value={"graph_score": math.nan}
            ), patch("omegaconf.OmegaConf.load", return_value=object()):
                with self.assertRaisesRegex(TeacherTargetBuildError, "not finite"):
                    build_teacher_targets(
                        rows=rows,
                        teacher_checkpoint=ckpt_path,
                        teacher_config=cfg_path,
                        encoded_song_field="encoded_song_path",
                        encoded_song_root=None,
                        split=None,
                        device="cpu",
                    )

    def test_encoded_song_root_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            (tmp / "train").mkdir(parents=True, exist_ok=True)
            song_path = tmp / "train" / "s1.json"
            song_path.write_text(json.dumps({"meta": {}, "melody": [], "chords": []}), encoding="utf-8")
            cfg_path = tmp / "cfg.yaml"
            ckpt_path = tmp / "ckpt.pt"
            cfg_path.write_text("model: {}\n", encoding="utf-8")
            ckpt_path.write_text("x", encoding="utf-8")
            rows = [{"song_id": "s1"}]

            with patch("src.observer.build_teacher_targets.build_model_from_config", return_value=object()), patch(
                "src.observer.build_teacher_targets.score_song", return_value={"graph_score": 0.1}
            ), patch("omegaconf.OmegaConf.load", return_value=object()):
                out_rows = build_teacher_targets(
                    rows=rows,
                    teacher_checkpoint=ckpt_path,
                    teacher_config=cfg_path,
                    encoded_song_field="encoded_song_path",
                    encoded_song_root=tmp,
                    split="train",
                    device="cpu",
                )

        self.assertEqual(out_rows[0]["song_id"], "s1")
        self.assertAlmostEqual(out_rows[0]["teacher_score"], 0.1)


if __name__ == "__main__":
    unittest.main()
