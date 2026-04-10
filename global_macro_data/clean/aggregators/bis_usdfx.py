from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_bis_usdfx(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    df = _load_dta(raw_dir / "aggregators" / "BIS" / "BIS_USDfx.dta")
    df = df.loc[
        ~df["series_name"].astype(str).str.contains("Waemu|World", regex=True, na=False)
        & (df["ref_area"].astype(str) != "XM")
    ].copy()
    df = df[["period", "value", "ref_area"]].copy()
    df["value"] = pd.to_numeric(df["value"].replace("NA", ""), errors="coerce")
    df = df.rename(columns={"ref_area": "ISO2", "period": "year", "value": "BIS_USDfx"})

    countrylist = _load_dta(helper_dir / "countrylist.dta")[["ISO2", "ISO3"]].copy()
    df = df.merge(countrylist, on="ISO2", how="left")
    df = df.loc[df["ISO3"].notna()].copy()
    df = df.drop(columns=["ISO2"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df[["ISO3", "year", "BIS_USDfx"]].copy()

    df = sh.gmdfixunits(df, "BIS_USDfx", if_mask=df["ISO3"].astype(str) == "MRT", divide=10, data_temp_dir=temp_dir)
    df = sh.gmdfixunits(df, "BIS_USDfx", if_mask=df["ISO3"].astype(str) == "SLE", multiply=1000, data_temp_dir=temp_dir)

    df = _sort_keys(df)
    out_path = clean_dir / "aggregators" / "BIS" / "BIS_USDfx.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_bis_usdfx"]
