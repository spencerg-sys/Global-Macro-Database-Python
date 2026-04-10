from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_homer_sylla(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    path = raw_dir / "aggregators" / "Homer_Sylla" / "Homer_Sylla.xlsx"
    df = _read_excel_compat(path, usecols="A:E", nrows=2221)
    df = df.rename(columns={col: f"Homer_Sylla_{col}" for col in df.columns if col not in {"ISO3", "year"}})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    for col in ["Homer_Sylla_strate", "Homer_Sylla_cbrate", "Homer_Sylla_ltrate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df = df[["ISO3", "year", "Homer_Sylla_strate", "Homer_Sylla_cbrate", "Homer_Sylla_ltrate"]].copy()
    df = _sort_keys(df)
    out_path = clean_dir / "aggregators" / "Homer_Sylla" / "Homer_Sylla.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_homer_sylla"]
