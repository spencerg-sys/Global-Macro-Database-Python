from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_imf_gfs(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    df = _load_dta(raw_dir / "aggregators" / "IMF" / "IMF_GFS.dta")
    df = df[["period", "value", "REF_AREA", "CLASSIFICATION", "REF_SECTOR", "UNIT_MEASURE"]].copy()
    df["value"] = df["value"].replace("NA", "")
    df["period"] = pd.to_numeric(df["period"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["CLASSIFICATION"] = (
        df["CLASSIFICATION"].astype(str) + "_" + df["REF_SECTOR"].astype(str) + "_" + df["UNIT_MEASURE"].astype(str)
    )
    df = df.drop(columns=["REF_SECTOR", "UNIT_MEASURE"])
    keys = df[["REF_AREA", "period"]].drop_duplicates().reset_index(drop=True)
    wide_values = df.pivot(index=["REF_AREA", "period"], columns="CLASSIFICATION", values="value").reset_index()
    wide = keys.merge(wide_values, on=["REF_AREA", "period"], how="left")
    wide.columns.name = None
    wide = wide.rename(columns={"REF_AREA": "ISO2", "period": "year"})

    def fill_primary(primary: str, fallback: str, target: str) -> None:
        wide[target] = pd.to_numeric(wide.get(primary), errors="coerce")
        mask = wide[target].isna()
        wide.loc[mask, target] = pd.to_numeric(wide.loc[mask, fallback], errors="coerce")

    fill_primary("G11__Z_S1311_XDC", "G11__Z_S1311B_XDC", "govtax")
    fill_primary("G1__Z_S1311_XDC", "G1__Z_S1311B_XDC", "govrev")
    fill_primary("G2M__Z_S1311_XDC", "G2M__Z_S1311B_XDC", "govexp")
    fill_primary("GNLB__Z_S1311_XDC_R_B1GQ", "GNLB__Z_S1311B_XDC_R_B1GQ", "govdef_GDP")

    wide = wide[["ISO2", "year", "govtax", "govrev", "govexp", "govdef_GDP"]].copy()
    countrylist = _load_dta(helper_dir / "countrylist.dta")[["ISO2", "ISO3"]].copy()
    wide = wide.merge(countrylist, on="ISO2", how="left")
    wide = wide.loc[wide["ISO3"].notna()].copy()
    wide = wide.drop(columns=["ISO2"])

    value_cols = [col for col in wide.columns if col not in {"ISO3", "year"}]
    wide = wide.rename(columns={col: f"IMF_GFS_{col}" for col in value_cols})

    mult_countries = "AGO ARM AUS BDI BEN BFA BGD BOL BRA CHL CHN CIV CAN CMR COD COG COL CRI CZE DEU DNK DOM DZA EGY ESP FRA GAB GBR GNQ HUN IND IDN IDN IRN IRN IRQ ISL ITA JAM JPN KAZ KEN KHM KOR LAO LBN LKA MDG MLI MMR MNG MWI NER NGA NOR PAK PHL PRY RUS RWA SAU SEN SLE SRB SWE TGO THA TUR TZA UGA USA UZB VNM YEM ZAF ZMB".split()
    div_countries = "FSM KIR LBR MHL PLW WSM BOL AIA MSR".split()
    for country in mult_countries:
        mask = wide["ISO3"].astype(str) == country
        for col in [c for c in ["IMF_GFS_govexp", "IMF_GFS_govrev", "IMF_GFS_govtax"] if c in wide.columns]:
            wide.loc[mask, col] = pd.to_numeric(wide.loc[mask, col], errors="coerce") * 1000
    for country in div_countries:
        mask = wide["ISO3"].astype(str) == country
        for col in [c for c in ["IMF_GFS_govexp", "IMF_GFS_govrev", "IMF_GFS_govtax"] if c in wide.columns]:
            wide.loc[mask, col] = pd.to_numeric(wide.loc[mask, col], errors="coerce") / 1000

    slv_mask = (wide["ISO3"].astype(str) == "SLV") & (pd.to_numeric(wide["year"], errors="coerce") <= 2000)
    ecu_mask = (wide["ISO3"].astype(str) == "ECU") & (pd.to_numeric(wide["year"], errors="coerce") <= 2000)
    sdn_mask = (wide["ISO3"].astype(str) == "SDN") & pd.to_numeric(wide["year"], errors="coerce").between(1998, 1999)
    kor_mask = wide["ISO3"].astype(str) == "KOR"
    cod97_mask = (wide["ISO3"].astype(str) == "COD") & (pd.to_numeric(wide["year"], errors="coerce") <= 1997)
    cod93_mask = (wide["ISO3"].astype(str) == "COD") & (pd.to_numeric(wide["year"], errors="coerce") <= 1993)

    for col in [c for c in ["IMF_GFS_govexp", "IMF_GFS_govrev", "IMF_GFS_govtax"] if c in wide.columns]:
        wide.loc[slv_mask, col] = pd.to_numeric(wide.loc[slv_mask, col], errors="coerce") / 8.75
        wide.loc[ecu_mask, col] = pd.to_numeric(wide.loc[ecu_mask, col], errors="coerce") / 2500
        if col == "IMF_GFS_govrev":
            wide.loc[sdn_mask, col] = pd.to_numeric(wide.loc[sdn_mask, col], errors="coerce") / 10
        wide.loc[kor_mask, col] = pd.to_numeric(wide.loc[kor_mask, col], errors="coerce") * 1000
        wide.loc[cod97_mask, col] = pd.to_numeric(wide.loc[cod97_mask, col], errors="coerce") / 100
        wide.loc[cod93_mask, col] = pd.to_numeric(wide.loc[cod93_mask, col], errors="coerce") / 1000

    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int16")
    wide = _sort_keys(wide)
    wide = wide[["ISO3", "year"] + [col for col in wide.columns if col not in {"ISO3", "year"}]]
    out_path = clean_dir / "aggregators" / "IMF" / "IMF_GFS.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_imf_gfs"]
