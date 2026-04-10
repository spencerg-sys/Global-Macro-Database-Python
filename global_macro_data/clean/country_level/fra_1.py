from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_fra_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = _load_dta(raw_dir / "country_level" / "FRA_1.dta")
    df = df[["period", "value", "BS_ITEM"]].copy()
    df["BS_ITEM"] = df["BS_ITEM"].astype(str).str.slice(0, 2)
    wide = df.pivot(index="period", columns="BS_ITEM", values="value").reset_index()
    wide.columns.name = None
    wide["year"] = wide["period"].astype(str).str.slice(0, 4)
    wide["month"] = wide["period"].astype(str).str.slice(-2)
    wide = wide.drop(columns=["period"])
    for col in [c for c in wide.columns if c != "year"]:
        wide[col] = wide[col].astype(str).replace("NA", "")
    for col in wide.columns:
        wide[col] = pd.to_numeric(wide[col], errors="coerce")
    wide = wide.sort_values(["year", "month"]).groupby("year", sort=False).tail(1).copy()
    wide = wide.drop(columns=["month"])
    wide["ISO3"] = "FRA"
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int16")
    if "M1" in wide.columns:
        wide["M1"] = pd.to_numeric(wide["M1"], errors="coerce").astype("float64")
    if "M2" in wide.columns:
        wide["M2"] = pd.to_numeric(wide["M2"], errors="coerce").astype("float64")
    if "M3" in wide.columns:
        wide["M3"] = pd.to_numeric(wide["M3"], errors="coerce").astype("int32")
    wide = wide.rename(columns={col: f"CS1_{col}" for col in wide.columns if col not in {"ISO3", "year"}})
    wide = wide[["ISO3", "year", "CS1_M1", "CS1_M2", "CS1_M3"]].copy()
    wide = _sort_keys(wide)
    out_path = clean_dir / "country_level" / "FRA_1.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_fra_1"]
