from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_usa_2(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    path = raw_dir / "country_level" / "USA_2.xlsx"

    gov = pd.read_excel(path, sheet_name="gov", dtype=str)
    for col in gov.columns:
        gov[col] = pd.to_numeric(gov[col], errors="coerce")
    master = gov.copy()

    for sheet in ["trade", "GDP", "ltrate", "CPI", "sav"]:
        current = pd.read_excel(path, sheet_name=sheet, dtype=str)
        for col in current.columns:
            current[col] = pd.to_numeric(current[col], errors="coerce")
        master = current.merge(master, on="year", how="outer")

    master["pop"] = pd.to_numeric(master["pop"], errors="coerce") / 1000
    master = master.rename(columns={"gov_debt": "govdebt", "DEFICIT": "govdef", "REVENUE": "govrev"})

    n_gdp = pd.to_numeric(master["nGDP"], errors="coerce")
    master["govdebt_GDP"] = (
        pd.to_numeric(master["govdebt"], errors="coerce") / n_gdp
    ).astype("float32").astype("float64")
    master["govdef_GDP"] = pd.to_numeric(master["govdef"], errors="coerce") / n_gdp
    master["govrev_GDP"] = pd.to_numeric(master["govrev"], errors="coerce") / n_gdp
    master["govexp_GDP"] = pd.to_numeric(master["govexp"], errors="coerce") / n_gdp

    master["govexp"] = pd.to_numeric(master["govexp"], errors="coerce") / 1000
    master["govrev"] = pd.to_numeric(master["govrev"], errors="coerce") / 1000
    master["govdef"] = pd.to_numeric(master["govdef"], errors="coerce") / 1000
    master["inv"] = pd.to_numeric(master["inv"], errors="coerce") * 1000
    master["sav"] = pd.to_numeric(master["sav"], errors="coerce") * 1000
    master["govdebt_GDP"] = pd.to_numeric(master["govdebt_GDP"], errors="coerce") / 10

    master["ISO3"] = "USA"
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    prev_cpi = pd.to_numeric(master["CPI"], errors="coerce").groupby(master["ISO3"]).shift(1)
    prev_year = pd.to_numeric(master["year"], errors="coerce").groupby(master["ISO3"]).shift(1)
    year_num = pd.to_numeric(master["year"], errors="coerce")
    infl = (pd.to_numeric(master["CPI"], errors="coerce") - prev_cpi) / prev_cpi * 100
    master["infl"] = infl.where(prev_cpi.notna() & year_num.eq(prev_year + 1))
    master["inv_GDP"] = pd.to_numeric(master["inv"], errors="coerce") / n_gdp * 100

    master = master.rename(columns={col: f"CS2_{col}" for col in master.columns if col not in {"ISO3", "year"}})
    master = master.drop(columns=["CS2_nGDP_pc", "CS2_GBPfx"], errors="ignore")
    for col in [
        "CS2_sav",
        "CS2_inv",
        "CS2_CPI",
        "CS2_ltrate",
        "CS2_strate",
        "CS2_rGDP",
        "CS2_nGDP",
        "CS2_rGDP_pc",
        "CS2_deflator",
        "CS2_CA",
        "CS2_govrev",
        "CS2_govexp",
        "CS2_govdef",
        "CS2_govdebt",
        "CS2_pop",
    ]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
    for col in [
        "CS2_govdebt_GDP",
        "CS2_govdef_GDP",
        "CS2_govrev_GDP",
        "CS2_govexp_GDP",
        "CS2_infl",
        "CS2_inv_GDP",
    ]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float32")

    master = master[
        [
            "ISO3",
            "year",
            "CS2_sav",
            "CS2_inv",
            "CS2_CPI",
            "CS2_ltrate",
            "CS2_strate",
            "CS2_rGDP",
            "CS2_nGDP",
            "CS2_rGDP_pc",
            "CS2_deflator",
            "CS2_CA",
            "CS2_govrev",
            "CS2_govexp",
            "CS2_govdef",
            "CS2_govdebt",
            "CS2_pop",
            "CS2_govdebt_GDP",
            "CS2_govdef_GDP",
            "CS2_govrev_GDP",
            "CS2_govexp_GDP",
            "CS2_infl",
            "CS2_inv_GDP",
        ]
    ].copy()
    for col in [
        "CS2_sav",
        "CS2_inv",
        "CS2_CPI",
        "CS2_ltrate",
        "CS2_strate",
        "CS2_rGDP",
        "CS2_nGDP",
        "CS2_rGDP_pc",
        "CS2_deflator",
        "CS2_CA",
        "CS2_govrev",
        "CS2_govexp",
        "CS2_govdef",
        "CS2_govdebt",
        "CS2_pop",
    ]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
    for col in [
        "CS2_govdebt_GDP",
        "CS2_govdef_GDP",
        "CS2_govrev_GDP",
        "CS2_govexp_GDP",
        "CS2_infl",
        "CS2_inv_GDP",
    ]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float32")
    master = _sort_keys(master)
    out_path = clean_dir / "country_level" / "USA_2.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_usa_2"]
