from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_arg_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = pd.read_excel(raw_dir / "country_level" / "ARG_1.xlsx", sheet_name="CS1_M3_GDP")
    df["CS1_M3_GDP"] = df["CS1_M3_GDP"].astype(str).str.replace("%", "", regex=False)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    df["CS1_M3_GDP"] = pd.to_numeric(df["CS1_M3_GDP"], errors="coerce").astype("float64")
    df["ISO3"] = "ARG"
    df = df[["ISO3", "year", "CS1_M3_GDP"]].copy()
    df = _sort_keys(df)
    out_path = clean_dir / "country_level" / "ARG_1.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_arg_1"]
