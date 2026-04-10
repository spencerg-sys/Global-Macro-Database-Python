from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_mitchell(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    pure_python_partial: bool = True,
) -> pd.DataFrame:
    out = _mitchell_partial_final_assembly(
        data_raw_dir=data_raw_dir,
        data_clean_dir=data_clean_dir,
        data_helper_dir=data_helper_dir,
        data_temp_dir=data_temp_dir,
    )
    out = _apply_clean_overrides(out, source_name="MITCHELL", data_helper_dir=data_helper_dir)
    clean_dir = _resolve(data_clean_dir)
    out_path = clean_dir / "aggregators" / "MITCHELL" / "MITCHELL.dta"
    _save_dta(out, out_path)
    return out
__all__ = ["clean_mitchell"]
