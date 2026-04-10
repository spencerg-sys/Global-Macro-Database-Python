from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_fao(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    path = raw_dir / "aggregators" / "FAO" / "FAO_macro.xls"

    countrylist = _load_dta(helper_dir / "countrylist.dta")[["ISOnum", "ISO3"]].copy()
    countrylist["ISOnum"] = pd.to_numeric(countrylist["ISOnum"], errors="coerce")
    df = _read_excel_compat(path)
    df = df[["Area", "Element Code", "Year", "Value", "Area Code (M49)", "Item"]].copy()
    df = df.rename(columns={"Area": "countryname", "Element Code": "ElementCode", "Year": "year", "Area Code (M49)": "ISOnum"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["ISOnum"] = pd.to_numeric(df["ISOnum"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    code = pd.to_numeric(df["ElementCode"], errors="coerce")
    df["indicator"] = ""
    df.loc[code.eq(6224) & df["Item"].eq("Gross Domestic Product"), "indicator"] = "nGDP"
    df.loc[code.eq(6225) & df["Item"].eq("Gross Domestic Product"), "indicator"] = "rGDP"
    df.loc[code.eq(6224) & df["Item"].eq("Gross Fixed Capital Formation"), "indicator"] = "finv"
    df = df.loc[df["indicator"] != ""].copy()
    df = df.merge(countrylist, on="ISOnum", how="left")
    manual_iso = {
        "Yemen Dem": "YMD",
        "Yemen Ar Rp": "YEM",
        "Sudan (former)": "SDN",
        "Ethiopia PDR": "ETH",
    }
    for country, iso3 in manual_iso.items():
        df.loc[df["countryname"] == country, "ISO3"] = iso3
    df = df.loc[df["ISO3"].notna(), ["ISO3", "year", "Value", "indicator"]].copy()

    wide = df.pivot(index=["year", "indicator"], columns="ISO3", values="Value").reset_index()
    if "YMD" in wide.columns:
        wide["YMD"] = pd.to_numeric(wide["YMD"], errors="coerce") * 26
        if "YEM" not in wide.columns:
            wide["YEM"] = np.nan
        mask = pd.to_numeric(wide["YMD"], errors="coerce").notna()
        wide.loc[mask, "YEM"] = pd.to_numeric(wide.loc[mask, "YEM"], errors="coerce") + pd.to_numeric(wide.loc[mask, "YMD"], errors="coerce")
        wide = wide.drop(columns=["YMD"], errors="ignore")

    iso_cols = [col for col in wide.columns if col not in {"year", "indicator"}]
    wide = wide.rename(columns={col: f"FAO_{col}" for col in iso_cols})
    long = wide.melt(id_vars=["year", "indicator"], var_name="ISO3", value_name="value")
    long["ISO3"] = long["ISO3"].astype("string").str.removeprefix("FAO_")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    out = long.pivot(index=["ISO3", "year"], columns="indicator", values="value").reset_index()
    out.columns.name = None
    out = out.rename(columns={col: f"FAO_{col}" for col in out.columns if col not in {"ISO3", "year"}})
    out["FAO_finv_GDP"] = (pd.to_numeric(out["FAO_finv"], errors="coerce") / pd.to_numeric(out["FAO_nGDP"], errors="coerce") * 100).astype("float32")
    out["ISO3"] = out["ISO3"].astype(str)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int16")
    for col in ["FAO_finv", "FAO_nGDP", "FAO_rGDP"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    out = out[["ISO3", "year", "FAO_finv", "FAO_nGDP", "FAO_rGDP", "FAO_finv_GDP"]].copy()
    out = _sort_keys(out)
    out_path = clean_dir / "aggregators" / "FAO" / "FAO.dta"
    _save_dta(out, out_path)
    return out
__all__ = ["clean_fao"]
