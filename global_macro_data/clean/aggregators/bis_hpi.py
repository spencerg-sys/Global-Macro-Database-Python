from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_bis_hpi(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    df = _load_dta(raw_dir / "aggregators" / "BIS" / "BIS_HPI.dta")
    df = df.loc[~df["ref_area"].astype(str).isin(["4T", "5R", "XM", "XW"])].copy()
    df = df.drop(columns=["dataset_name", "freq", "series_code", "series_name"], errors="ignore")
    df = df.rename(columns={"ref_area": "ISO2"})

    countrylist = _load_dta(helper_dir / "countrylist.dta")[["ISO2", "ISO3"]].copy()
    df = df.merge(countrylist, on="ISO2", how="left")
    df = df.loc[df["ISO3"].notna()].copy()
    df = df.drop(columns=["ISO2"])
    df["value"] = pd.to_numeric(df["value"].replace("NA", ""), errors="coerce")
    df["year"] = pd.to_numeric(df["period"].astype(str).str.slice(0, 4), errors="coerce").astype("Int64")
    df = df.sort_values(["ISO3", "year", "period"]).groupby(["ISO3", "year"], as_index=False).tail(1).copy()
    df = df.drop(columns=["period"])
    df = df.rename(columns={"value": "BIS_HPI"})
    df = df[["ISO3", "year", "BIS_HPI"]].copy()

    df = _sort_keys(df)
    out_path = clean_dir / "aggregators" / "BIS" / "BIS_HPI.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_bis_hpi"]
