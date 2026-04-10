from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_bra_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = _read_excel_compat(raw_dir / "country_level" / "BRA_1.xls", sheet_name="Processed")
    df["pop"] = pd.to_numeric(df["pop"], errors="coerce") / 1_000_000
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.iloc[:-1].copy()
    df = df.rename(columns={col: f"CS1_{col}" for col in df.columns if col != "year"})
    df["ISO3"] = "BRA"
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    for col in ["CS1_USDfx", "CS1_rGDP", "CS1_pop"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df = df[["ISO3", "year", "CS1_USDfx", "CS1_rGDP", "CS1_pop"]].copy()
    df = _sort_keys(df)
    out_path = clean_dir / "country_level" / "BRA_1.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_bra_1"]
