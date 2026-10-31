"""Tidy CSV output for every experiment. Notes are `#` lines above the header."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
STUB_NOTE = "STUB PIPELINE: placeholder numbers from tests/stubs. Never cite these."


def write_results(df: pd.DataFrame, name: str, notes: tuple[str, ...] = (),
                  out_dir: Path | str = RESULTS_DIR) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        for note in notes:
            handle.write(f"# {note}\n")
        df.to_csv(handle, index=False)
    return path


def read_results(name: str, out_dir: Path | str = RESULTS_DIR) -> pd.DataFrame:
    return pd.read_csv(Path(out_dir) / f"{name}.csv", comment="#")


def read_notes(name: str, out_dir: Path | str = RESULTS_DIR) -> list[str]:
    notes = []
    with (Path(out_dir) / f"{name}.csv").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("# "):
                break
            notes.append(line[2:].strip())
    return notes
