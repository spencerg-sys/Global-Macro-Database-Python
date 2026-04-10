from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_imf_mfs(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    df = _load_dta(raw_dir / "aggregators" / "IMF" / "IMF_MFS.dta")
    df = df[["period", "value", "indicator", "ref_area"]].copy()
    wide = df.pivot_table(index=["ref_area", "period"], columns="indicator", values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={
            "ref_area": "ISO2",
            "period": "year",
            "14____XDC": "historical_M0",
            "34____XDC": "historical_M1",
            "35L___XDC": "historical_M2",
            "FIGB_PA": "ltrate",
            "FITB_PA": "strate",
            "FPOLM_PA": "cbrate",
            "FMB_XDC": "modern_M2",
            "FMA_XDC": "modern_M0",
            "FMBCC_XDC": "modern_M1",
        }
    )

    countrylist = _load_dta(helper_dir / "countrylist.dta")[["ISO2", "ISO3"]].copy()
    wide = wide.merge(countrylist, on="ISO2", how="left")
    wide = wide.loc[wide["ISO3"].notna()].copy()

    for col in [c for c in wide.columns if c.startswith("historical_") or c.startswith("modern_")]:
        wide.loc[pd.to_numeric(wide[col], errors="coerce").eq(0), col] = pd.NA

    irq_mask = wide["ISO3"].astype(str) == "IRQ"
    first_mean = pd.to_numeric(wide.loc[irq_mask & pd.to_numeric(wide["year"], errors="coerce").eq(2008), "historical_M2"], errors="coerce").mean()
    second_mean = pd.to_numeric(wide.loc[irq_mask & pd.to_numeric(wide["year"], errors="coerce").eq(2008), "modern_M2"], errors="coerce").mean()
    ratio = second_mean / first_mean if pd.notna(first_mean) and first_mean != 0 else pd.NA
    if pd.notna(ratio):
        wide.loc[irq_mask & (pd.to_numeric(wide["year"], errors="coerce") <= 2008), "historical_M2"] = (
            pd.to_numeric(wide.loc[irq_mask & (pd.to_numeric(wide["year"], errors="coerce") <= 2008), "historical_M2"], errors="coerce") * ratio
        )
    fill_mask = irq_mask & wide["historical_M2"].isna()
    wide.loc[fill_mask, "historical_M2"] = wide.loc[fill_mask, "modern_M2"]
    wide.loc[irq_mask, "modern_M2"] = pd.NA

    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int16")
    result = sh.splice(wide, priority="modern historical", generate="M0", varname="M0", base_year=2016, method="chainlink", save="NO")
    result = sh.splice(result, priority="modern historical", generate="M1", varname="M1", base_year=2016, method="chainlink", save="NO")
    result = sh.splice(result, priority="modern historical", generate="M2", varname="M2", base_year=2016, method="chainlink", save="NO")

    result = result.drop(columns=["ISO2"] + [c for c in result.columns if c.startswith("historical_") or c.startswith("modern_")], errors="ignore")
    value_cols = [col for col in result.columns if col not in {"ISO3", "year"}]
    result = result.rename(columns={col: f"IMF_MFS_{col}" for col in value_cols})

    bra_mask = (result["ISO3"].astype(str) == "BRA") & (pd.to_numeric(result["year"], errors="coerce") <= 1993)
    for col in [c for c in ["IMF_MFS_M1", "IMF_MFS_M2"] if c in result.columns]:
        result.loc[bra_mask, col] = pd.to_numeric(result.loc[bra_mask, col], errors="coerce") / 2750

    mrt_mask = result["ISO3"].astype(str) == "MRT"
    for col in [c for c in ["IMF_MFS_M1", "IMF_MFS_M2"] if c in result.columns]:
        result.loc[mrt_mask, col] = pd.to_numeric(result.loc[mrt_mask, col], errors="coerce") / 10

    stp_mask = result["ISO3"].astype(str) == "STP"
    for col in [c for c in ["IMF_MFS_M1", "IMF_MFS_M2"] if c in result.columns]:
        result.loc[stp_mask, col] = pd.to_numeric(result.loc[stp_mask, col], errors="coerce") / 1000

    ven_mask = result["ISO3"].astype(str) == "VEN"
    for col in [c for c in ["IMF_MFS_M0", "IMF_MFS_M1", "IMF_MFS_M2"] if c in result.columns]:
        result.loc[ven_mask, col] = pd.to_numeric(result.loc[ven_mask, col], errors="coerce") / (10**14)

    result = _sort_keys(result)
    result = result[["ISO3", "year"] + [col for col in result.columns if col not in {"ISO3", "year"}]]
    out_path = clean_dir / "aggregators" / "IMF" / "IMF_MFS.dta"
    _save_dta(result, out_path)
    return result
__all__ = ["clean_imf_mfs"]
