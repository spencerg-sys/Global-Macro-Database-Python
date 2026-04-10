from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_afdb(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    df = _load_dta(raw_dir / "aggregators" / "AFDB" / "AFDB.dta")
    df = df[["period", "country", "indicator", "value"]].copy()

    indicator_map = {
        "FM.LBL.MONY.CN": "M1",
        "FM.LBL.MQMY.CN": "M2",
        "GC.BAL.CASH.GD.ZS": "govdef_GDP",
        "GC.REV.TOTL.GD.ZS": "govrev_GDP",
        "GC.XPN.TOTL.GD.ZS": "govexp_GDP",
        "SL.TLF.15UP.UEM": "unemp",
        "LM.POP.EPP.TOT": "emp",
        "NY.GDP.MKTP.CN": "nGDP",
    }
    df["indicator"] = df["indicator"].astype(str).map(lambda x: indicator_map.get(x, x))

    keys = df[["country", "period"]].drop_duplicates().reset_index(drop=True)
    wide_values = df.pivot_table(index=["country", "period"], columns="indicator", values="value", aggfunc="first").reset_index()
    wide_values.columns.name = None
    wide = keys.merge(wide_values, on=["country", "period"], how="left")
    wide = wide.rename(columns={"country": "ISO3", "period": "year"})
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int16")

    for col in ["M1", "M2", "emp"]:
        if col in wide.columns:
            divisor = 1_000_000 if col in {"M1", "M2"} else 1_000
            wide[col] = pd.to_numeric(wide[col], errors="coerce") / divisor

    valid_iso3 = _load_dta(helper_dir / "countrylist.dta")[["ISO3"]].drop_duplicates()
    wide = wide.merge(valid_iso3, on="ISO3", how="inner")

    value_cols = [col for col in wide.columns if col not in {"ISO3", "year"}]
    if value_cols:
        wide = wide.loc[wide[value_cols].notna().any(axis=1)].copy()

    for col in ["M1", "M2", "nGDP"]:
        if col in wide.columns:
            wide.loc[wide["ISO3"].astype(str) == "MRT", col] = pd.to_numeric(wide.loc[wide["ISO3"].astype(str) == "MRT", col], errors="coerce") / 10
            wide.loc[wide["ISO3"].astype(str) == "STP", col] = pd.to_numeric(wide.loc[wide["ISO3"].astype(str) == "STP", col], errors="coerce") / 1000
    for col in ["M1", "M2"]:
        if col in wide.columns:
            zmb_mask = (wide["ISO3"].astype(str) == "ZMB") & (pd.to_numeric(wide["year"], errors="coerce") > 2000)
            wide.loc[zmb_mask, col] = pd.to_numeric(wide.loc[zmb_mask, col], errors="coerce") * 100
    if "nGDP" in wide.columns:
        wide.loc[(wide["ISO3"].astype(str) == "ZMB") & (pd.to_numeric(wide["year"], errors="coerce") <= 1999), "nGDP"] = pd.to_numeric(
            wide.loc[(wide["ISO3"].astype(str) == "ZMB") & (pd.to_numeric(wide["year"], errors="coerce") <= 1999), "nGDP"],
            errors="coerce",
        ) / 1000
        tun_mask = (wide["ISO3"].astype(str) == "TUN") & pd.to_numeric(wide["year"], errors="coerce").between(2000, 2005)
        wide.loc[tun_mask, "nGDP"] = pd.to_numeric(wide.loc[tun_mask, "nGDP"], errors="coerce") / 1000
        wide.loc[(wide["ISO3"].astype(str) == "TUN") & pd.to_numeric(wide["year"], errors="coerce").eq(2000), "nGDP"] = pd.to_numeric(
            wide.loc[(wide["ISO3"].astype(str) == "TUN") & pd.to_numeric(wide["year"], errors="coerce").eq(2000), "nGDP"],
            errors="coerce",
        ) * 10
        wide.loc[(wide["ISO3"].astype(str) == "TGO") & pd.to_numeric(wide["year"], errors="coerce").between(2000, 2005), "nGDP"] = pd.to_numeric(
            wide.loc[(wide["ISO3"].astype(str) == "TGO") & pd.to_numeric(wide["year"], errors="coerce").between(2000, 2005), "nGDP"],
            errors="coerce",
        ) * 10
        wide.loc[(wide["ISO3"].astype(str) == "SWZ") & pd.to_numeric(wide["year"], errors="coerce").between(2000, 2005), "nGDP"] = pd.to_numeric(
            wide.loc[(wide["ISO3"].astype(str) == "SWZ") & pd.to_numeric(wide["year"], errors="coerce").between(2000, 2005), "nGDP"],
            errors="coerce",
        ) / 1000
        wide.loc[(wide["ISO3"].astype(str) == "MOZ") & (pd.to_numeric(wide["year"], errors="coerce") >= 2006), "nGDP"] = pd.to_numeric(
            wide.loc[(wide["ISO3"].astype(str) == "MOZ") & (pd.to_numeric(wide["year"], errors="coerce") >= 2006), "nGDP"],
            errors="coerce",
        ) / 1000

    for col in ["M1", "M2"]:
        if col in wide.columns:
            wide.loc[(wide["ISO3"].astype(str) == "COG") & (pd.to_numeric(wide["year"], errors="coerce") < 1995), col] = pd.NA
    if "M1" in wide.columns:
        wide.loc[(wide["ISO3"].astype(str) == "LBR") & (pd.to_numeric(wide["year"], errors="coerce") > 2000), "M1"] = pd.NA
    if "M2" in wide.columns:
        wide.loc[(wide["ISO3"].astype(str) == "LBR") & (pd.to_numeric(wide["year"], errors="coerce") >= 2000), "M2"] = pd.NA

    wide = wide.rename(columns={col: f"AFDB_{col}" for col in wide.columns if col not in {"ISO3", "year"}})
    if "AFDB_nGDP" in wide.columns:
        wide["AFDB_nGDP"] = pd.to_numeric(wide["AFDB_nGDP"], errors="coerce") / 1_000_000
    for result, ratio in [("AFDB_govdef", "AFDB_govdef_GDP"), ("AFDB_govrev", "AFDB_govrev_GDP"), ("AFDB_govexp", "AFDB_govexp_GDP")]:
        if ratio in wide.columns and "AFDB_nGDP" in wide.columns:
            wide[result] = pd.to_numeric(wide[ratio], errors="coerce") * pd.to_numeric(wide["AFDB_nGDP"], errors="coerce") / 100
            wide[result] = pd.to_numeric(wide[result], errors="coerce").astype("float32")

    ordered_cols = [
        "ISO3",
        "year",
        "AFDB_M1",
        "AFDB_M2",
        "AFDB_emp",
        "AFDB_govdef_GDP",
        "AFDB_govexp_GDP",
        "AFDB_govrev_GDP",
        "AFDB_nGDP",
        "AFDB_unemp",
        "AFDB_govdef",
        "AFDB_govrev",
        "AFDB_govexp",
    ]
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int32")
    wide = _sort_keys(wide)
    wide = wide[[col for col in ordered_cols if col in wide.columns]]
    out_path = clean_dir / "aggregators" / "AFDB" / "AFDB.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_afdb"]
