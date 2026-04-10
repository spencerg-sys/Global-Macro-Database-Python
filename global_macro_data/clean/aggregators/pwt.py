from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_pwt(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = _load_dta(raw_dir / "aggregators" / "PWT" / "pwt1001.dta")
    df = df.rename(columns={"countrycode": "ISO3", "pop": "PWT_pop", "rgdpna": "PWT_rGDP_USD", "xr": "PWT_USDfx"})
    df = df.loc[df["ISO3"].astype(str) != "CH2", ["ISO3", "year", "PWT_pop", "PWT_rGDP_USD", "PWT_USDfx"]].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    df["PWT_pop"] = pd.to_numeric(df["PWT_pop"], errors="coerce").astype("float64")
    df["PWT_rGDP_USD"] = pd.to_numeric(df["PWT_rGDP_USD"], errors="coerce").astype("float32")
    df["PWT_USDfx"] = pd.to_numeric(df["PWT_USDfx"], errors="coerce").astype("float64")

    irn_mask = df["ISO3"].astype(str).eq("IRN") & df["year"].between(1958, 1959)
    df.loc[irn_mask, "PWT_USDfx"] = 75.75
    df.loc[df["ISO3"].astype(str).isin(["LBR", "ZWE"]), "PWT_USDfx"] = np.nan
    df.loc[df["ISO3"].astype(str).eq("SLE"), "PWT_USDfx"] = pd.to_numeric(df.loc[df["ISO3"].astype(str).eq("SLE"), "PWT_USDfx"], errors="coerce") * 1000
    df.loc[df["ISO3"].astype(str).eq("VEN") & df["year"].ge(2014), "PWT_USDfx"] = np.nan
    df.loc[df["ISO3"].astype(str).eq("IDN") & df["year"].le(1965), "PWT_USDfx"] = np.nan

    df = _sort_keys(df)
    out_path = clean_dir / "aggregators" / "PWT" / "PWT.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_pwt"]
