from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_wdi_arc(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    df = _read_excel_compat(raw_dir / "aggregators" / "WB" / "WDI_1999.xlsx", dtype=str)
    keep_cols = [col for col in df.columns if re.fullmatch(r"\d{4} \[YR\d{4}\]", str(col))]
    df = df[["Country Code", "Series Name", "Series Code", "Version Code"] + keep_cols].copy()
    df = df.rename(columns={"Country Code": "ISO3", "Series Name": "SeriesName", "Series Code": "SeriesCode", "Version Code": "VersionCode"})
    for col in keep_cols:
        values = df[col].astype("string").str.strip().replace({"..": "", "nan": ""})
        bad = values.notna() & values.ne("") & pd.to_numeric(values, errors="coerce").isna()
        if bool(bad.any()):
            raise ValueError(f"Variable not numeric: {col}")
        df[col] = _excel_numeric_series(values, mode="float")

    year_cols = [col for col in keep_cols if df[col].notna().any()]
    df = df.loc[df[year_cols].notna().any(axis=1)].copy()
    long = df.melt(id_vars=["ISO3", "SeriesName", "SeriesCode", "VersionCode"], value_vars=year_cols, var_name="year_token", value_name="YR")
    long["SeriesCode"] = long["SeriesCode"].astype("string").str.replace(".", "_", regex=False)
    long["year"] = pd.to_numeric(long["year_token"].astype("string").str.extract(r"(\d{4})")[0], errors="coerce")
    long = long.sort_values(["ISO3", "SeriesCode", "year", "VersionCode"], kind="mergesort")
    long = long.loc[long["YR"].notna()].copy()
    long = long.groupby(["ISO3", "SeriesCode", "year"], sort=False, as_index=False).tail(1).copy()
    long = long.drop(columns=["SeriesName", "VersionCode", "year_token"], errors="ignore")
    wide = long.pivot(index=["ISO3", "year"], columns="SeriesCode", values="YR").reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={
            "BN_CAB_XOKA_GD_ZS": "CA_GDP",
            "FP_CPI_TOTL": "CPI",
            "FP_CPI_TOTL_ZG": "infl",
            "GB_DOD_TOTL_GD_ZS": "govdebt_GDP",
            "GB_TAX_TOTL_CN": "govtax",
            "NE_CON_TOTL_CN": "cons",
            "NE_EXP_GNFS_CD": "exports_USD",
            "NE_EXP_GNFS_CN": "exports",
            "NE_GDI_FTOT_CN": "finv",
            "NE_GDI_TOTL_CN": "inv",
            "NE_IMP_GNFS_CD": "imports_USD",
            "NE_IMP_GNFS_CN": "imports",
            "NY_GDP_MKTP_CN": "nGDP",
            "NY_GDP_MKTP_KN": "rGDP",
            "PX_REX_REER": "REER",
            "SP_POP_TOTL": "pop",
        }
    )

    nonzero_exempt = {"ISO3", "year", "infl"}
    for col in [c for c in wide.columns if c not in nonzero_exempt]:
        wide.loc[pd.to_numeric(wide[col], errors="coerce").eq(0), col] = pd.NA

    for col in [c for c in ["nGDP", "rGDP", "inv", "finv", "cons", "imports", "exports", "pop", "govtax"] if c in wide.columns]:
        wide[col] = _apply_scale_chain(
            wide[col],
            ops=[("div", 1_000_000.0)],
            storage="double",
        )

    valid_iso3 = _load_dta(helper_dir / "countrylist.dta")[["ISO3"]].dropna().drop_duplicates().copy()
    wide = wide.merge(valid_iso3, on="ISO3", how="inner")

    scale_ops_by_iso: dict[str, list[tuple[str, float]]] = {
        "VEN": [("mul", _pow10_literal(-14))],
        "ROU": [("div", 10000.0)],
        "STP": [("div", 1000.0)],
        "AFG": [("div", 1000.0)],
        "TUR": [("div", 1_000_000.0)],
        "AGO": [("div", 1_000_000.0)],
        "ZMB": [("div", 1000.0)],
        "SUR": [("div", 1000.0)],
        "MOZ": [("div", 1000.0)],
        "BGR": [("div", 1000.0)],
        "GHA": [("div", 10000.0)],
        "SLV": [("div", 8.75)],
        "ECU": [("div", 25.0)],
        "TKM": [("div", 5000.0)],
        "TJK": [("div", 1000.0)],
        "SDN": [("div", 1000.0)],
        "AZE": [("div", 10000.0)],
        "BLR": [("div", 1_000_000.0)],
        "COD": [("div", 100000.0)],
        "MRT": [("div", 10.0)],
    }
    for iso3, ops in scale_ops_by_iso.items():
        mask = wide["ISO3"].eq(iso3)
        for col in [c for c in ["nGDP", "rGDP", "inv", "finv", "cons", "imports", "exports", "govtax"] if c in wide.columns]:
            wide.loc[mask, col] = _apply_scale_chain(
                wide.loc[mask, col],
                ops=ops,
                storage="double",
            )

    if "rGDP" in wide.columns:
        wide.loc[wide["ISO3"].eq("ZWE"), "rGDP"] = pd.NA

    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].copy()
    wide = wide.merge(eur_fx, on="ISO3", how="left")
    fx_mask = wide["EUR_irrevocable_FX"].notna()
    for col in [c for c in ["nGDP", "rGDP", "inv", "finv", "cons", "imports", "exports", "govtax"] if c in wide.columns]:
        wide.loc[fx_mask, col] = pd.to_numeric(wide.loc[fx_mask, col], errors="coerce") / pd.to_numeric(wide.loc[fx_mask, "EUR_irrevocable_FX"], errors="coerce")
    wide = wide.drop(columns=["EUR_irrevocable_FX"], errors="ignore")

    if {"cons", "nGDP"}.issubset(wide.columns):
        wide["cons_GDP"] = pd.to_numeric(wide["cons"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    if {"imports", "nGDP"}.issubset(wide.columns):
        wide["imports_GDP"] = pd.to_numeric(wide["imports"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    if {"exports", "nGDP"}.issubset(wide.columns):
        wide["exports_GDP"] = pd.to_numeric(wide["exports"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    if {"govtax", "nGDP"}.issubset(wide.columns):
        wide["govtax_GDP"] = pd.to_numeric(wide["govtax"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    if {"finv", "nGDP"}.issubset(wide.columns):
        wide["finv_GDP"] = pd.to_numeric(wide["finv"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    if {"inv", "nGDP"}.issubset(wide.columns):
        wide["inv_GDP"] = pd.to_numeric(wide["inv"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100

    for col in [c for c in wide.columns if c not in {"ISO3", "year"}]:
        wide.loc[pd.to_numeric(wide[col], errors="coerce").eq(0), col] = pd.NA

    wide = wide.rename(columns={col: f"WDI_ARC_{col}" for col in wide.columns if col not in {"ISO3", "year"}})
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int32")
    float64_cols = [
        "WDI_ARC_CA_GDP",
        "WDI_ARC_CPI",
        "WDI_ARC_infl",
        "WDI_ARC_govdebt_GDP",
        "WDI_ARC_govtax",
        "WDI_ARC_cons",
        "WDI_ARC_exports_USD",
        "WDI_ARC_exports",
        "WDI_ARC_finv",
        "WDI_ARC_inv",
        "WDI_ARC_imports_USD",
        "WDI_ARC_imports",
        "WDI_ARC_nGDP",
        "WDI_ARC_rGDP",
        "WDI_ARC_REER",
        "WDI_ARC_pop",
    ]
    float32_cols = [
        "WDI_ARC_cons_GDP",
        "WDI_ARC_imports_GDP",
        "WDI_ARC_exports_GDP",
        "WDI_ARC_govtax_GDP",
        "WDI_ARC_finv_GDP",
        "WDI_ARC_inv_GDP",
    ]
    for col in [c for c in float64_cols if c in wide.columns]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float64")
    for col in [c for c in float32_cols if c in wide.columns]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float32")
    ordered = [
        "ISO3",
        "year",
        "WDI_ARC_CA_GDP",
        "WDI_ARC_CPI",
        "WDI_ARC_infl",
        "WDI_ARC_govdebt_GDP",
        "WDI_ARC_govtax",
        "WDI_ARC_cons",
        "WDI_ARC_exports_USD",
        "WDI_ARC_exports",
        "WDI_ARC_finv",
        "WDI_ARC_inv",
        "WDI_ARC_imports_USD",
        "WDI_ARC_imports",
        "WDI_ARC_nGDP",
        "WDI_ARC_rGDP",
        "WDI_ARC_REER",
        "WDI_ARC_pop",
        "WDI_ARC_cons_GDP",
        "WDI_ARC_imports_GDP",
        "WDI_ARC_exports_GDP",
        "WDI_ARC_govtax_GDP",
        "WDI_ARC_finv_GDP",
        "WDI_ARC_inv_GDP",
    ]
    wide = wide[[col for col in ordered if col in wide.columns]].copy()
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int32")
    for col in [c for c in float64_cols if c in wide.columns]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float64")
    for col in [c for c in float32_cols if c in wide.columns]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float32")
    wide = _sort_keys(wide)
    out_path = clean_dir / "aggregators" / "WB" / "WDI_ARC.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_wdi_arc"]
