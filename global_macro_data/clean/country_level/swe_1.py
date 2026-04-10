from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_swe_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    path = raw_dir / "country_level" / "SWE_1.xlsx"

    def _swe1_value(value: object) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if text in {"", "<NA>", "nan", "None"}:
            return np.nan
        return float(text)

    def _read_numeric_sheet(sheet: str) -> pd.DataFrame:
        df = pd.read_excel(path, sheet_name=sheet, dtype=str)
        for col in df.columns:
            if col == "year":
                continue
            df[col] = df[col].map(_swe1_value)
        return df

    gov = _read_numeric_sheet("gov")
    year_str = gov["year"].astype("string")
    gov["year"] = pd.to_numeric(year_str.str.slice(0, 2) + year_str.str.slice(-2), errors="coerce")
    gov.loc[pd.to_numeric(gov["year"], errors="coerce").eq(1753) & pd.to_numeric(gov["govexp"], errors="coerce").eq(1620), "year"] = 1754
    gov["ISO3"] = "SWE"
    master = gov[["ISO3", "year", "govexp", "REVENUE", "gov_debt", "DEFICIT"]].copy()

    for sheet in ["fx", "CPI", "money", "national_accounts", "GDP"]:
        current = _read_numeric_sheet(sheet)
        current["year"] = pd.to_numeric(current["year"], errors="coerce")
        current["ISO3"] = "SWE"
        keep_cols = ["ISO3", "year"] + [col for col in current.columns if col not in {"ISO3", "year"}]
        master = current[keep_cols].merge(master, on=["ISO3", "year"], how="outer")

    master["govexp"] = pd.to_numeric(master["govexp"], errors="coerce") / 1000
    master["gov_debt"] = pd.to_numeric(master["gov_debt"], errors="coerce") / 1000
    master["DEFICIT"] = pd.to_numeric(master["DEFICIT"], errors="coerce") / 1000
    master["REVENUE"] = pd.to_numeric(master["REVENUE"], errors="coerce") / 1000
    master["pop"] = pd.to_numeric(master["pop"], errors="coerce") / 1_000_000

    master = master.rename(columns={"DEFICIT": "govdef", "REVENUE": "govrev", "gov_debt": "govdebt"})
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")

    prev_cpi = pd.to_numeric(master["CPI"], errors="coerce").groupby(master["ISO3"]).shift(1)
    prev_year = pd.to_numeric(master["year"], errors="coerce").groupby(master["ISO3"]).shift(1)
    year_num = pd.to_numeric(master["year"], errors="coerce")
    infl = (pd.to_numeric(master["CPI"], errors="coerce") - prev_cpi) / prev_cpi * 100
    master["infl"] = infl.where(prev_cpi.notna() & year_num.eq(prev_year + 1))

    n_gdp = pd.to_numeric(master["nGDP"], errors="coerce")
    master["cons_GDP"] = pd.to_numeric(master["cons"], errors="coerce") / n_gdp * 100
    master["imports_GDP"] = pd.to_numeric(master["imports"], errors="coerce") / n_gdp * 100
    master["exports_GDP"] = pd.to_numeric(master["exports"], errors="coerce") / n_gdp * 100
    master["finv_GDP"] = pd.to_numeric(master["finv"], errors="coerce") / n_gdp * 100
    master["govrev_GDP"] = pd.to_numeric(master["govrev"], errors="coerce") / n_gdp * 100
    master["govexp_GDP"] = pd.to_numeric(master["govexp"], errors="coerce") / n_gdp * 100
    master["govdebt_GDP"] = pd.to_numeric(master["govdebt"], errors="coerce") / n_gdp * 100

    master = master.rename(columns={col: f"CS1_{col}" for col in master.columns if col not in {"ISO3", "year"}})
    master = master.drop(columns=["CS1_DMKfx", "CS1_GBPfx", "CS1_FRFfx"], errors="ignore")

    for col in [
        "CS1_nGDP",
        "CS1_rGDP",
        "CS1_rGDP_pc",
        "CS1_rGDP_pc_index",
        "CS1_pop",
        "CS1_cons",
        "CS1_finv",
        "CS1_exports",
        "CS1_imports",
        "CS1_M3",
        "CS1_M0",
        "CS1_CPI",
        "CS1_govexp",
        "CS1_govrev",
        "CS1_govdebt",
        "CS1_govdef",
    ]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
    for col in [
        "CS1_infl",
        "CS1_cons_GDP",
        "CS1_imports_GDP",
        "CS1_exports_GDP",
        "CS1_finv_GDP",
        "CS1_govrev_GDP",
        "CS1_govexp_GDP",
        "CS1_govdebt_GDP",
    ]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float32")

    master = master[
        [
            "ISO3",
            "year",
            "CS1_nGDP",
            "CS1_rGDP",
            "CS1_rGDP_pc",
            "CS1_rGDP_pc_index",
            "CS1_pop",
            "CS1_cons",
            "CS1_finv",
            "CS1_exports",
            "CS1_imports",
            "CS1_M3",
            "CS1_M0",
            "CS1_CPI",
            "CS1_govexp",
            "CS1_govrev",
            "CS1_govdebt",
            "CS1_govdef",
            "CS1_infl",
            "CS1_cons_GDP",
            "CS1_imports_GDP",
            "CS1_exports_GDP",
            "CS1_finv_GDP",
            "CS1_govrev_GDP",
            "CS1_govexp_GDP",
            "CS1_govdebt_GDP",
        ]
    ].copy()
    for col in [
        "CS1_nGDP",
        "CS1_rGDP",
        "CS1_rGDP_pc",
        "CS1_rGDP_pc_index",
        "CS1_pop",
        "CS1_cons",
        "CS1_finv",
        "CS1_exports",
        "CS1_imports",
        "CS1_M3",
        "CS1_M0",
        "CS1_CPI",
        "CS1_govexp",
        "CS1_govrev",
        "CS1_govdebt",
        "CS1_govdef",
    ]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
    for col in [
        "CS1_infl",
        "CS1_cons_GDP",
        "CS1_imports_GDP",
        "CS1_exports_GDP",
        "CS1_finv_GDP",
        "CS1_govrev_GDP",
        "CS1_govexp_GDP",
        "CS1_govdebt_GDP",
    ]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float32")
    master = _sort_keys(master)
    out_path = clean_dir / "country_level" / "SWE_1.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_swe_1"]
