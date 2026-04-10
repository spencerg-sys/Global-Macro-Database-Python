from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_imf_weo(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    from .wdi import clean_wdi

    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    dataset_code = _parse_weo_dataset_code()

    wdi_path = clean_dir / "aggregators" / "WB" / "WDI.dta"
    if not wdi_path.exists():
        clean_wdi(data_raw_dir=raw_dir, data_clean_dir=clean_dir, data_helper_dir=data_helper_dir)
    wdi = _load_dta(wdi_path)
    panel = wdi[["ISO3", "year"]].copy()

    df = _load_dta(raw_dir / "aggregators" / "IMF" / "IMF_WEO.dta")
    df = df[["period", "value", "weo_country", "weo_subject", "dataset_code"]].copy()
    df = df.loc[df["dataset_code"].astype(str) == dataset_code].copy()
    df["weo_country"] = df["weo_country"].replace({"WBG": "PSE", "UVK": "XKX"})
    panel = pd.concat(
        [
            panel,
            df.loc[df["value"].astype(str) != "NA", ["weo_country", "period"]].rename(columns={"weo_country": "ISO3", "period": "year"}),
        ],
        ignore_index=True,
    ).drop_duplicates(["ISO3", "year"])
    df = df.loc[df["value"].astype(str) != "NA"].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if df.duplicated(["period", "weo_country", "weo_subject"]).any():
        dupes = df.loc[df.duplicated(["period", "weo_country", "weo_subject"], keep=False), ["period", "weo_country", "weo_subject"]]
        raise ValueError(f"IMF_WEO reshape keys are not unique: {dupes.head().to_dict('records')}")
    wide = df.pivot(index=["period", "weo_country"], columns="weo_subject", values="value").reset_index()
    wide.columns.name = None
    rename_map = {
        "period": "year",
        "weo_country": "ISO3",
        "BCA_NGDPD": "CA_GDP",
        "GGXWDG_NGDP": "govdebt_GDP",
        "GGXCNL_NGDP": "govdef_GDP",
        "GGX_NGDP": "govexp_GDP",
        "GGR_NGDP": "govrev_GDP",
        "LP": "pop",
        "PCPI": "CPI",
        "LUR": "unemp",
        "NID_NGDP": "inv_GDP",
        "NGDPRPC": "rGDP_pc",
        "NGDP_R": "rGDP",
        "NGDP": "nGDP",
        "TM_RPCH": "imports_yoy",
        "TX_RPCH": "exports_yoy",
    }
    wide = wide.rename(columns={k: v for k, v in rename_map.items() if k in wide.columns})
    wide = wide.drop(columns=["GGSB_NPGDP"], errors="ignore")
    wide = panel.merge(wide, on=["ISO3", "year"], how="outer", sort=False)
    wide["nGDP"] = pd.to_numeric(wide["nGDP"], errors="coerce") * 1000
    wide["rGDP"] = pd.to_numeric(wide["rGDP"], errors="coerce") * 1000

    sen_mask = (wide["ISO3"].astype(str) == "SEN") & (pd.to_numeric(wide["year"], errors="coerce") == 1971) & pd.to_numeric(
        wide["govdebt_GDP"], errors="coerce"
    ).eq(0.71)
    wide.loc[sen_mask, "govdebt_GDP"] = pd.NA
    cog_mask = (wide["ISO3"].astype(str) == "COG") & pd.to_numeric(wide["govdebt_GDP"], errors="coerce").eq(0)
    wide.loc[cog_mask, "govdebt_GDP"] = pd.NA

    nominal_from_ratio = {
        "govexp": "govexp_GDP",
        "CA": "CA_GDP",
        "inv": "inv_GDP",
        "govdef": "govdef_GDP",
        "govrev": "govrev_GDP",
    }
    for result, ratio in nominal_from_ratio.items():
        if {ratio, "nGDP"}.issubset(wide.columns):
            wide[result] = (pd.to_numeric(wide[ratio], errors="coerce") * pd.to_numeric(wide["nGDP"], errors="coerce")) / 100

    mac_mask = (wide["ISO3"].astype(str) == "MAC") & pd.to_numeric(wide["govdebt_GDP"], errors="coerce").eq(0)
    wide.loc[mac_mask, "govdebt_GDP"] = pd.NA

    wide = wide.merge(wdi[["ISO3", "year", "WDI_imports", "WDI_exports"]], on=["ISO3", "year"], how="left")
    wide["exports"] = pd.to_numeric(wide["WDI_exports"], errors="coerce").astype("float32")
    wide["imports"] = pd.to_numeric(wide["WDI_imports"], errors="coerce").astype("float32")
    wide = _sort_keys(wide)

    for iso3, idx in wide.groupby("ISO3").groups.items():
        ordered_idx = list(idx)
        for i in range(1, len(ordered_idx)):
            prev = ordered_idx[i - 1]
            cur = ordered_idx[i]
            prev_year = pd.to_numeric(wide.at[prev, "year"], errors="coerce")
            cur_year = pd.to_numeric(wide.at[cur, "year"], errors="coerce")
            if pd.isna(prev_year) or pd.isna(cur_year) or cur_year <= prev_year:
                continue
            if pd.isna(wide.at[cur, "WDI_exports"]) and pd.notna(wide.at[prev, "exports"]):
                growth = pd.to_numeric(wide.at[cur, "exports_yoy"], errors="coerce")
                prev_exports = np.float32(pd.to_numeric(wide.at[prev, "exports"], errors="coerce"))
                wide.at[cur, "exports"] = np.float32(np.float64(prev_exports) * (1 + np.float64(growth) / 100))
            if pd.isna(wide.at[cur, "WDI_imports"]) and pd.notna(wide.at[prev, "imports"]):
                growth = pd.to_numeric(wide.at[cur, "imports_yoy"], errors="coerce")
                prev_imports = np.float32(pd.to_numeric(wide.at[prev, "imports"], errors="coerce"))
                wide.at[cur, "imports"] = np.float32(np.float64(prev_imports) * (1 + np.float64(growth) / 100))

    wide = wide.drop(columns=[col for col in wide.columns if col.endswith("yoy") or col.startswith("WDI_")], errors="ignore")
    wide["infl"] = (
        pd.to_numeric(wide["CPI"], errors="coerce") / wide.groupby("ISO3")["CPI"].shift(1).pipe(pd.to_numeric, errors="coerce") - 1
    ) * 100

    protect_patterns = ["ISO3", "year", "*_GDP", "*infl", "*pop", "*CPI", "*unemp"]
    som_mask = wide["ISO3"].astype(str) == "SOM"
    for col in [c for c in wide.columns if c not in {"ISO3", "year"}]:
        if any(Path(col).match(pattern) for pattern in protect_patterns):
            continue
        wide.loc[som_mask, col] = pd.NA

    if {"imports", "nGDP"}.issubset(wide.columns):
        wide["imports_GDP"] = (pd.to_numeric(wide["imports"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce")) * 100
    if {"exports", "nGDP"}.issubset(wide.columns):
        wide["exports_GDP"] = (pd.to_numeric(wide["exports"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce")) * 100
    wide.loc[wide["ISO3"].astype(str) == "ZWE", ["imports_GDP", "exports_GDP"]] = pd.NA

    value_cols = [col for col in wide.columns if col not in {"ISO3", "year"}]
    wide = wide.rename(columns={col: f"IMF_WEO_{col}" for col in value_cols})

    zwe_mask = (wide["ISO3"].astype(str) == "ZWE") & (pd.to_numeric(wide["year"], errors="coerce") <= 2009)
    for col in [c for c in wide.columns if c not in {"ISO3", "year"}]:
        if any(Path(col).match(pattern) for pattern in ["*infl", "*CPI", "*unemp"]):
            continue
        wide.loc[zwe_mask, col] = pd.NA
    wide = wide.loc[~(wide["ISO3"].astype(str).eq("ZWE") & (pd.to_numeric(wide["year"], errors="coerce") <= 2009))].copy()

    wide = _sort_keys(wide)
    ordered_cols = ["ISO3", "year"] + [col for col in wide.columns if col not in {"ISO3", "year"}]
    wide = wide[ordered_cols]
    wide = _coerce_numeric_dtypes(wide, IMF_WEO_DTYPE_MAP)
    out_path = clean_dir / "aggregators" / "IMF" / "IMF_WEO.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_imf_weo"]
