from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_can_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    path = raw_dir / "country_level" / "CAN_1.xlsx"

    master = pd.read_excel(path, sheet_name="gov")
    master = master.loc[pd.to_numeric(master["year"], errors="coerce").notna()].copy()
    master["year"] = pd.to_numeric(master["year"], errors="coerce")

    for sheet in ["pop", "Finance", "trade", "national_accounts"]:
        current = pd.read_excel(path, sheet_name=sheet)
        current = current.loc[pd.to_numeric(current["year"], errors="coerce").notna()].copy()
        current["year"] = pd.to_numeric(current["year"], errors="coerce")
        master = current.merge(master, on="year", how="outer")

    value_cols = [col for col in master.columns if col != "year"]
    master = master.rename(columns={col: f"CS1_{col}" for col in value_cols})
    master["ISO3"] = "CAN"
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")

    prev_cpi = pd.to_numeric(master["CS1_CPI"], errors="coerce").groupby(master["ISO3"]).shift(1)
    prev_year = pd.to_numeric(master["year"], errors="coerce").groupby(master["ISO3"]).shift(1)
    year_num = pd.to_numeric(master["year"], errors="coerce")
    infl = (pd.to_numeric(master["CS1_CPI"], errors="coerce") - prev_cpi) / prev_cpi * 100
    master["CS1_infl"] = infl.where(prev_cpi.notna() & year_num.eq(prev_year + 1))

    master["CS1_cons"] = pd.to_numeric(master["CS1_cons_gov"], errors="coerce") + pd.to_numeric(master["CS1_cons_HH"], errors="coerce")
    master["CS1_inv"] = pd.to_numeric(master["CS1_inventory_change"], errors="coerce") + pd.to_numeric(master["CS1_finv"], errors="coerce")
    master = master.drop(columns=["CS1_inventory_change", "CS1_cons_HH", "CS1_cons_gov"], errors="ignore")

    n_gdp = pd.to_numeric(master["CS1_nGDP"], errors="coerce")
    master["CS1_cons_GDP"] = pd.to_numeric(master["CS1_cons"], errors="coerce") / n_gdp * 100
    master["CS1_imports_GDP"] = pd.to_numeric(master["CS1_imports"], errors="coerce") / n_gdp * 100
    master["CS1_exports_GDP"] = pd.to_numeric(master["CS1_exports"], errors="coerce") / n_gdp * 100
    master["CS1_finv_GDP"] = pd.to_numeric(master["CS1_finv"], errors="coerce") / n_gdp * 100
    master["CS1_inv_GDP"] = pd.to_numeric(master["CS1_inv"], errors="coerce") / n_gdp * 100
    master["CS1_govrev_GDP"] = pd.to_numeric(master["CS1_govrev"], errors="coerce") / n_gdp * 100
    master["CS1_govexp_GDP"] = pd.to_numeric(master["CS1_govexp"], errors="coerce") / n_gdp * 100
    master["CS1_govtax_GDP"] = pd.to_numeric(master["CS1_govtax"], errors="coerce") / n_gdp * 100
    master["CS1_govdebt_GDP"] = pd.to_numeric(master["CS1_govdebt"], errors="coerce") / n_gdp * 100

    for col in [
        "CS1_finv",
        "CS1_nGDP",
        "CS1_rGDP",
        "CS1_exports",
        "CS1_imports",
        "CS1_USDfx",
        "CS1_strate",
        "CS1_ltrate",
        "CS1_M0",
        "CS1_M1",
        "CS1_M2",
        "CS1_M3",
        "CS1_CPI",
        "CS1_pop",
        "CS1_govtax",
        "CS1_govrev",
        "CS1_govexp",
        "CS1_govdebt",
    ]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
    for col in [
        "CS1_infl",
        "CS1_cons",
        "CS1_inv",
        "CS1_cons_GDP",
        "CS1_imports_GDP",
        "CS1_exports_GDP",
        "CS1_finv_GDP",
        "CS1_inv_GDP",
        "CS1_govrev_GDP",
        "CS1_govexp_GDP",
        "CS1_govtax_GDP",
        "CS1_govdebt_GDP",
    ]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float32")

    master = master[
        [
            "ISO3",
            "year",
            "CS1_finv",
            "CS1_nGDP",
            "CS1_rGDP",
            "CS1_exports",
            "CS1_imports",
            "CS1_USDfx",
            "CS1_strate",
            "CS1_ltrate",
            "CS1_M0",
            "CS1_M1",
            "CS1_M2",
            "CS1_M3",
            "CS1_CPI",
            "CS1_pop",
            "CS1_govtax",
            "CS1_govrev",
            "CS1_govexp",
            "CS1_govdebt",
            "CS1_infl",
            "CS1_cons",
            "CS1_inv",
            "CS1_cons_GDP",
            "CS1_imports_GDP",
            "CS1_exports_GDP",
            "CS1_finv_GDP",
            "CS1_inv_GDP",
            "CS1_govrev_GDP",
            "CS1_govexp_GDP",
            "CS1_govtax_GDP",
            "CS1_govdebt_GDP",
        ]
    ].copy()
    master = _sort_keys(master)
    out_path = clean_dir / "country_level" / "CAN_1.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_can_1"]
