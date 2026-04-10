from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def make_sources_dataset(
    *,
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
) -> pd.DataFrame:
    temp_dir = _resolve(data_temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{"source": "", "note": "", "variable": ""}])
    _save_dta(df, temp_dir / "notes_sources.dta")
    return df
__all__ = ["make_sources_dataset"]
