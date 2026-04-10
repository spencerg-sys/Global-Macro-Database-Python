from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_gbr_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)

    headline = _read_excel_compat(raw_dir / "country_level" / "GBR_1.xlsx", sheet_name="A1. Headline series", header=None)
    headline = headline.iloc[
        7:938,
        [_excel_column_to_index(col) for col in ["A", "B", "H", "V", "W", "Y", "AB", "AO", "AP", "AS", "AT", "AU", "BA", "BE", "BL", "BM", "BN", "BX", "CB", "CC", "T", "U"]],
    ].copy()
    headline.columns = [
        "year",
        "rGDP",
        "rGDP_ENG",
        "nGDP_ENG",
        "nGDP",
        "pop",
        "unemp",
        "CPI",
        "infl",
        "cbrate",
        "strate",
        "ltrate",
        "USDfx",
        "HPI",
        "M0",
        "M1",
        "M2",
        "govexp",
        "CA",
        "CA_GDP",
        "exports",
        "imports",
    ]
    for col in headline.columns:
        headline[col] = pd.to_numeric(headline[col], errors="coerce")
    headline["ISO3"] = "GBR"

    nGDP_1 = sh._sum_mean_only(pd.to_numeric(headline.loc[headline["year"] == 1700, "nGDP_ENG"], errors="coerce"))[1]
    nGDP_2 = sh._sum_mean_only(pd.to_numeric(headline.loc[headline["year"] == 1700, "nGDP"], errors="coerce"))[1]
    if pd.notna(nGDP_1) and pd.notna(nGDP_2):
        ratio = sh._safe_ratio(nGDP_2, nGDP_1)
        mask = pd.to_numeric(headline["year"], errors="coerce") <= 1699
        headline.loc[mask, "nGDP"] = pd.to_numeric(headline.loc[mask, "nGDP_ENG"], errors="coerce") * ratio
    headline = headline.drop(columns=["nGDP_ENG"], errors="ignore")

    rGDP_1 = sh._sum_mean_only(pd.to_numeric(headline.loc[headline["year"] == 1700, "rGDP_ENG"], errors="coerce"))[1]
    rGDP_2 = sh._sum_mean_only(pd.to_numeric(headline.loc[headline["year"] == 1700, "rGDP"], errors="coerce"))[1]
    if pd.notna(rGDP_1) and pd.notna(rGDP_2):
        ratio = sh._safe_ratio(rGDP_2, rGDP_1)
        mask = pd.to_numeric(headline["year"], errors="coerce") <= 1699
        headline.loc[mask, "rGDP"] = pd.to_numeric(headline.loc[mask, "rGDP_ENG"], errors="coerce") * ratio
    headline = headline.drop(columns=["rGDP_ENG"], errors="ignore")
    headline["deflator"] = (
        pd.to_numeric(headline["nGDP"], errors="coerce") / pd.to_numeric(headline["rGDP"], errors="coerce")
    ).astype("float32").astype("float64")

    trade = _read_excel_compat(raw_dir / "country_level" / "GBR_1.xlsx", sheet_name="A35. Trade volumes and prices", header=None, dtype=str)
    trade = trade.iloc[:, [_excel_column_to_index(col) for col in ["A", "V", "Z"]]].copy()
    trade.columns = ["year", "exports_deflator", "imports_deflator"]
    trade = trade.iloc[5:].copy()
    trade["year"] = pd.to_numeric(trade["year"], errors="coerce")
    trade["exports_deflator"] = _excel_numeric_series_sig(trade["exports_deflator"], significant_digits=15)
    trade["imports_deflator"] = _excel_numeric_series_sig(trade["imports_deflator"], significant_digits=15)
    trade = trade.loc[trade["year"].notna()].copy()

    merged = trade.merge(headline, on="year", how="outer")
    deflator_1 = sh._sum_mean_only(pd.to_numeric(merged.loc[merged["year"] == 1772, "deflator"], errors="coerce"))[1]
    deflator_2 = sh._sum_mean_only(pd.to_numeric(merged.loc[merged["year"] == 1772, "exports_deflator"], errors="coerce"))[1]
    deflator_3 = sh._sum_mean_only(pd.to_numeric(merged.loc[merged["year"] == 1772, "exports_deflator"], errors="coerce"))[1]
    if pd.notna(deflator_1) and pd.notna(deflator_2):
        ratio = sh._safe_ratio(deflator_2, deflator_1)
        mask = pd.to_numeric(merged["year"], errors="coerce") <= 1772
        merged.loc[mask, "exports_deflator"] = pd.to_numeric(merged.loc[mask, "deflator"], errors="coerce") * ratio
    merged["exports"] = pd.to_numeric(merged["exports"], errors="coerce") * pd.to_numeric(merged["exports_deflator"], errors="coerce")
    if pd.notna(deflator_1) and pd.notna(deflator_3):
        ratio = sh._safe_ratio(deflator_3, deflator_1)
        mask = pd.to_numeric(merged["year"], errors="coerce") <= 1772
        merged.loc[mask, "imports_deflator"] = pd.to_numeric(merged.loc[mask, "deflator"], errors="coerce") * ratio
    merged["imports"] = pd.to_numeric(merged["imports"], errors="coerce") * pd.to_numeric(merged["imports_deflator"], errors="coerce")
    merged = merged.drop(columns=["exports_deflator", "imports_deflator"], errors="ignore")

    debt = _read_excel_compat(raw_dir / "country_level" / "GBR_1.xlsx", sheet_name="A29. The National Debt", header=None, dtype=str)
    debt = debt.iloc[:, [_excel_column_to_index(col) for col in ["A", "AR"]]].copy()
    debt.columns = ["year", "govdebt_GDP"]
    debt = debt.iloc[17:334].copy()
    debt["year"] = pd.to_numeric(debt["year"].astype(str).str.slice(0, 4), errors="coerce")
    debt["govdebt_GDP"] = _excel_numeric_series(debt["govdebt_GDP"], mode="g16")

    merged = debt.merge(merged, on="year", how="outer")
    merged["imports_GDP"] = pd.to_numeric(merged["imports"], errors="coerce") / pd.to_numeric(merged["nGDP"], errors="coerce") * 100
    merged["exports_GDP"] = pd.to_numeric(merged["exports"], errors="coerce") / pd.to_numeric(merged["nGDP"], errors="coerce") * 100
    merged["govexp_GDP"] = pd.to_numeric(merged["govexp"], errors="coerce") / pd.to_numeric(merged["nGDP"], errors="coerce") * 100

    merged["pop"] = pd.to_numeric(merged["pop"], errors="coerce") / 1000
    merged["USDfx"] = 1 / pd.to_numeric(merged["USDfx"], errors="coerce")
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce").astype("int16")
    value_cols = [col for col in merged.columns if col not in {"ISO3", "year"}]
    merged = merged.rename(columns={col: f"CS1_{col}" for col in value_cols})
    for col in [c for c in merged.columns if c.startswith("CS1_")]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    for col in ["CS1_deflator", "CS1_imports_GDP", "CS1_exports_GDP", "CS1_govexp_GDP"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float32")
    for col in [c for c in merged.columns if c.startswith("CS1_") and c not in {"CS1_deflator", "CS1_imports_GDP", "CS1_exports_GDP", "CS1_govexp_GDP"}]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float64")
    merged = merged[
        [
            "ISO3",
            "year",
            "CS1_govdebt_GDP",
            "CS1_rGDP",
            "CS1_exports",
            "CS1_imports",
            "CS1_nGDP",
            "CS1_pop",
            "CS1_unemp",
            "CS1_CPI",
            "CS1_infl",
            "CS1_cbrate",
            "CS1_strate",
            "CS1_ltrate",
            "CS1_USDfx",
            "CS1_HPI",
            "CS1_M0",
            "CS1_M1",
            "CS1_M2",
            "CS1_govexp",
            "CS1_CA",
            "CS1_CA_GDP",
            "CS1_deflator",
            "CS1_imports_GDP",
            "CS1_exports_GDP",
            "CS1_govexp_GDP",
        ]
    ].copy()
    for col in ["CS1_deflator", "CS1_imports_GDP", "CS1_exports_GDP", "CS1_govexp_GDP"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float32")
    for col in [c for c in merged.columns if c.startswith("CS1_") and c not in {"CS1_deflator", "CS1_imports_GDP", "CS1_exports_GDP", "CS1_govexp_GDP"}]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float64")
    merged = _sort_keys(merged)
    out_path = clean_dir / "country_level" / "GBR_1.dta"
    _save_dta(merged, out_path)
    return merged
__all__ = ["clean_gbr_1"]
