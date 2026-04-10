from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})


def make_notes_dataset(
    *,
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
) -> pd.DataFrame:
    temp_dir = _resolve(data_temp_dir)
    blank_panel_path = temp_dir / "blank_panel.dta"
    if not blank_panel_path.exists():
        raise FileNotFoundError(
            f"Missing blank panel at {blank_panel_path}. Run make_blank_panel() before creating notes.dta."
        )

    panel = pd.read_dta(blank_panel_path, convert_categoricals=False)
    notes = panel[[col for col in ["ISO3", "year"] if col in panel.columns]].copy()
    _save_dta(notes, temp_dir / "notes.dta")
    return notes


__all__ = ["make_notes_dataset"]
