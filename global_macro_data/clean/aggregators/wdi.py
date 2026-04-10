from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_wdi(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    df = _load_dta(raw_dir / "aggregators" / "WB" / "WDI.dta")
    df = df.loc[df["incomelevel"].astype(str) != "NA"].copy()
    df = df.loc[df["region"].fillna("").astype(str) != ""].copy()
    df = df.rename(columns={"countrycode": "ISO3"})
    df = df.loc[df["ISO3"].astype(str) != "CHI"].copy()

    drop_cols = [
        "region",
        "adminregion",
        "adminregionname",
        "incomelevel",
        "lendingtype",
        "lendingtypename",
        "countryname",
        "regionname",
        "incomelevelname",
        "indicatorname",
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    year_cols = [col for col in df.columns if re.fullmatch(r"yr\d{4}", str(col))]
    long_df = df.melt(
        id_vars=["ISO3", "indicatorcode"],
        value_vars=year_cols,
        var_name="year_col",
        value_name="yr",
    )
    long_df["year"] = long_df["year_col"].str.extract(r"(\d{4})").astype("Int64")
    long_df["indicatorcode_clean"] = long_df["indicatorcode"].astype(str).map(_sanitize_identifier_name)
    if long_df.duplicated(["ISO3", "year", "indicatorcode_clean"]).any():
        dupes = long_df.loc[long_df.duplicated(["ISO3", "year", "indicatorcode_clean"], keep=False), ["ISO3", "year", "indicatorcode_clean"]]
        raise ValueError(f"WDI reshape keys are not unique: {dupes.head().to_dict('records')}")
    wide = long_df.pivot(index=["ISO3", "year"], columns="indicatorcode_clean", values="yr").reset_index()
    wide.columns.name = None

    if {"GC_DOD_TOTL_CN", "NY_GDP_MKTP_CN"}.issubset(wide.columns):
        wide["GC_DOD_TOTL_CN"] = (
            pd.to_numeric(wide["GC_DOD_TOTL_CN"], errors="coerce")
            / pd.to_numeric(wide["NY_GDP_MKTP_CN"], errors="coerce")
        ) * 100

    rename_map = {
        "SP_POP_TOTL": "pop",
        "NY_GDP_MKTP_CN": "nGDP",
        "NY_GDP_MKTP_KN": "rGDP",
        "NY_GDP_MKTP_CD": "nGDP_USD",
        "NY_GDP_MKTP_KD": "rGDP_USD",
        "NY_GDP_PCAP_KN": "rGDP_pc",
        "FP_CPI_TOTL": "CPI",
        "FP_CPI_TOTL_ZG": "infl",
        "NE_GDI_TOTL_CN": "inv",
        "NE_GDI_FTOT_CN": "finv",
        "NY_GNS_ICTR_CN": "sav",
        "NE_CON_TOTL_CN": "cons",
        "NE_CON_TOTL_KN": "rcons",
        "NE_IMP_GNFS_CN": "imports",
        "NE_IMP_GNFS_CD": "imports_USD",
        "NE_EXP_GNFS_CN": "exports",
        "NE_EXP_GNFS_CD": "exports_USD",
        "PA_NUS_FCRF": "USDfx",
        "PX_REX_REER": "REER",
        "BN_CAB_XOKA_GD_ZS": "CA_GDP",
        "GC_TAX_TOTL_CN": "govtax",
        "GC_DOD_TOTL_CN": "govdebt_GDP",
        "GC_XPN_TOTL_CN": "govexp",
        "GC_REV_XGRT_GD_ZS": "govrev_GDP",
        "GC_TAX_TOTL_GD_ZS": "govtax_GDP",
    }
    wide = wide.rename(columns={k: v for k, v in rename_map.items() if k in wide.columns})

    million_vars = [
        "nGDP",
        "rGDP",
        "rGDP_USD",
        "nGDP_USD",
        "inv",
        "finv",
        "sav",
        "cons",
        "rcons",
        "imports",
        "imports_USD",
        "exports",
        "exports_USD",
        "pop",
        "govexp",
        "govtax",
    ]
    for col in [c for c in million_vars if c in wide.columns]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce") / 1_000_000

    for col in [c for c in ["finv", "inv", "cons"] if c in wide.columns]:
        wide.loc[pd.to_numeric(wide[col], errors="coerce").eq(0), col] = pd.NA

    ven_mask = wide["ISO3"].astype(str) == "VEN"
    for col in [c for c in million_vars if c in wide.columns and c != "pop"]:
        wide.loc[ven_mask, col] = pd.to_numeric(wide.loc[ven_mask, col], errors="coerce") / (10**11)

    sle_mask = wide["ISO3"].astype(str) == "SLE"
    for col in [c for c in ["USDfx"] + [v for v in million_vars if v != "pop"] if c in wide.columns]:
        wide.loc[sle_mask, col] = pd.to_numeric(wide.loc[sle_mask, col], errors="coerce") * 1000

    afg_mask = (wide["ISO3"].astype(str) == "AFG") & (pd.to_numeric(wide["year"], errors="coerce") <= 1978)
    for col in [c for c in [v for v in million_vars if v != "pop"] if c in wide.columns]:
        wide.loc[afg_mask, col] = pd.to_numeric(wide.loc[afg_mask, col], errors="coerce") / 1000

    if "USDfx" in wide.columns:
        wide.loc[wide["ISO3"].astype(str) == "STP", "USDfx"] = pd.to_numeric(
            wide.loc[wide["ISO3"].astype(str) == "STP", "USDfx"], errors="coerce"
        ) * 1000

    if "govdebt_GDP" in wide.columns:
        esp_mask = (wide["ISO3"].astype(str) == "ESP") & pd.to_numeric(wide["year"], errors="coerce").between(1970, 1971)
        wide.loc[esp_mask, "govdebt_GDP"] = pd.to_numeric(wide.loc[esp_mask, "govdebt_GDP"], errors="coerce") / 100

    if "rGDP" in wide.columns:
        omn_mask = (wide["ISO3"].astype(str) == "OMN") & (pd.to_numeric(wide["year"], errors="coerce") <= 1964)
        wide.loc[omn_mask, "rGDP"] = pd.to_numeric(wide.loc[omn_mask, "rGDP"], errors="coerce") * (10**6)

    if {"govrev_GDP", "nGDP"}.issubset(wide.columns):
        wide["govrev"] = (pd.to_numeric(wide["govrev_GDP"], errors="coerce") * pd.to_numeric(wide["nGDP"], errors="coerce")) / 100
    if {"CA_GDP", "nGDP"}.issubset(wide.columns):
        wide["CA"] = (pd.to_numeric(wide["CA_GDP"], errors="coerce") * pd.to_numeric(wide["nGDP"], errors="coerce")) / 100

    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")
    wide = wide.merge(eur_fx[["ISO3", "EUR_irrevocable_FX"]], on="ISO3", how="left")
    fx_mask = wide["EUR_irrevocable_FX"].notna() & (pd.to_numeric(wide["year"], errors="coerce") <= 1998)
    if "USDfx" in wide.columns:
        wide.loc[fx_mask, "USDfx"] = pd.to_numeric(wide.loc[fx_mask, "USDfx"], errors="coerce") / pd.to_numeric(
            wide.loc[fx_mask, "EUR_irrevocable_FX"], errors="coerce"
        )
    wide = wide.drop(columns=["EUR_irrevocable_FX"], errors="ignore")

    if {"rGDP_USD", "pop"}.issubset(wide.columns):
        wide["rGDP_pc_USD"] = pd.to_numeric(wide["rGDP_USD"], errors="coerce") / pd.to_numeric(wide["pop"], errors="coerce")

    ratio_exprs = {
        "cons_GDP": ("cons", "nGDP"),
        "imports_GDP": ("imports", "nGDP"),
        "exports_GDP": ("exports", "nGDP"),
        "govexp_GDP": ("govexp", "nGDP"),
        "finv_GDP": ("finv", "nGDP"),
        "inv_GDP": ("inv", "nGDP"),
    }
    for result, (num, den) in ratio_exprs.items():
        if {num, den}.issubset(wide.columns):
            wide[result] = (pd.to_numeric(wide[num], errors="coerce") / pd.to_numeric(wide[den], errors="coerce")) * 100

    usa_mask = (wide["ISO3"].astype(str) == "USA") & (pd.to_numeric(wide["year"], errors="coerce") <= 1971)
    if "finv_GDP" in wide.columns:
        wide.loc[usa_mask, "finv_GDP"] = pd.NA

    value_cols = [col for col in wide.columns if col not in {"ISO3", "year"}]
    wide = wide.rename(columns={col: f"WDI_{col}" for col in value_cols})
    wide = wide.drop(columns=["WDI_FM_LBL_BMNY_CN"], errors="ignore")
    wide = _coerce_numeric_dtypes(wide, WDI_DTYPE_MAP)
    wide = _sort_keys(wide)

    out_path = clean_dir / "aggregators" / "WB" / "WDI.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_wdi"]
