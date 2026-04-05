from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def test_rows_and_summary_tables_and_exports(tmp_path: Path):
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    import pandas as pd

    from scripts.eval_teacher_ood_corruptions import build_rows_dataframe, build_summary_dataframe, write_outputs

    rows_input = [
        {
            "song_id": "song_a",
            "split": "test",
            "mode": "out_of_key_note",
            "applied": True,
            "topology_changed": False,
            "note_corrupted_indices": [0],
            "chord_corrupted_indices": [],
            "onset_corrupted_indices": [],
            "score_real": 1.2,
            "score_corrupted": 0.8,
            "score_gap": 0.4,
            "rank_success": 1,
            "metadata_json": "{}",
        },
        {
            "song_id": "song_b",
            "split": "test",
            "mode": "out_of_key_note",
            "applied": False,
            "topology_changed": False,
            "note_corrupted_indices": [],
            "chord_corrupted_indices": [],
            "onset_corrupted_indices": [],
            "score_real": 0.7,
            "score_corrupted": None,
            "score_gap": None,
            "rank_success": None,
            "metadata_json": "{}",
        },
        {
            "song_id": "song_a",
            "split": "test",
            "mode": "octave_leap_violation",
            "applied": True,
            "topology_changed": False,
            "note_corrupted_indices": [2],
            "chord_corrupted_indices": [],
            "onset_corrupted_indices": [],
            "score_real": 1.2,
            "score_corrupted": 1.3,
            "score_gap": -0.1,
            "rank_success": 0,
            "metadata_json": "{}",
        },
    ]

    rows_df = build_rows_dataframe(rows_input)
    summary_df = build_summary_dataframe(rows_df)

    assert {"score_gap", "rank_success", "metadata_json"}.issubset(rows_df.columns)
    assert {"mode", "n_total", "n_applied", "success_rate", "mean_gap"}.issubset(summary_df.columns)

    outdir = tmp_path / "exports"
    write_outputs(rows_df, summary_df, outdir)

    rows_csv = outdir / "teacher_ood_eval_rows.csv"
    summary_csv = outdir / "teacher_ood_eval_summary.csv"
    workbook = outdir / "teacher_ood_eval.xlsx"

    assert rows_csv.exists()
    assert summary_csv.exists()
    assert workbook.exists()

    excel = pd.ExcelFile(workbook)
    assert {"summary", "rows"}.issubset(set(excel.sheet_names))
