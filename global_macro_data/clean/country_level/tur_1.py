from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_tur_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = _load_dta(raw_dir / "country_level" / "TUR_1.dta")
    df = df.loc[df["series_code"].astype(str) == "cpiree", ["period", "value"]].copy()
    df["year"] = pd.to_numeric(df["period"].astype(str).str.slice(0, 4), errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.sort_values(["year", "period"], kind="mergesort").groupby("year", sort=False).tail(1).copy()
    df = df.rename(columns={"value": "CS1_REER"})
    df["ISO3"] = "TUR"
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    df["CS1_REER"] = pd.to_numeric(df["CS1_REER"], errors="coerce").astype("float64")
    df = df[["ISO3", "year", "CS1_REER"]].copy()
    df = _sort_keys(df)
    out_path = clean_dir / "country_level" / "TUR_1.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_tur_1"]
