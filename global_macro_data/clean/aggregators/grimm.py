from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_grimm(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = _load_dta(raw_dir / "aggregators" / "Grimm" / "Grimm.dta")
    df = df.sort_values(["iso3", "year", "month"], kind="mergesort").groupby(["iso3", "year"], sort=False).tail(1).copy()
    df = df.loc[df["iso3"].astype(str) != "XXK", ["iso3", "year", "R_Policy"]].copy()
    df = df.rename(columns={"iso3": "ISO3", "R_Policy": "Grimm_cbrate"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("float64")
    df["Grimm_cbrate"] = pd.to_numeric(df["Grimm_cbrate"], errors="coerce").astype("float64")
    df = _sort_keys(df)
    out_path = clean_dir / "aggregators" / "Grimm" / "Grimm.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_grimm"]
