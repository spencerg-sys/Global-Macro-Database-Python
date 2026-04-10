from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def merge_final_data(
    *,
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
) -> pd.DataFrame:
    final_dir = _resolve(data_final_dir)
    helper_dir = _resolve(data_helper_dir)
    blank_panel_path = _require_blank_panel(data_temp_dir)
    master = pd.read_dta(blank_panel_path, convert_categoricals=False)
    files = sorted(
        p
        for p in final_dir.glob("*.dta")
        if p.name.startswith("chainlinked_") or p.name in {"SovDebtCrisis.dta", "CurrencyCrisis.dta", "BankingCrisis.dta"}
    )

    varnames: list[str] = []
    for file in files:
        match = re.search(r"chainlinked_(.+)\.dta$", file.name)
        if match:
            varnames.append(match.group(1))
        printname = file.name
        sh._emit(f"Merging file {printname}")
        using = pd.read_dta(file, convert_categoricals=False)
        master = _merge_update_1to1(master, using, keys=["ISO3", "year"], error_label=printname)

    keep_cols = ["ISO3", "year"] + [v for v in varnames if v in master.columns] + [c for c in ["SovDebtCrisis", "CurrencyCrisis", "BankingCrisis"] if c in master.columns]
    master = master[keep_cols].copy()

    if {"rGDP", "pop"}.issubset(master.columns):
        master["rGDP_pc"] = pd.to_numeric(master["rGDP"], errors="coerce") / pd.to_numeric(master["pop"], errors="coerce")
    if {"nGDP", "rGDP"}.issubset(master.columns):
        master["deflator"] = (pd.to_numeric(master["nGDP"], errors="coerce") / pd.to_numeric(master["rGDP"], errors="coerce")) * 100
    if {"govdebt_GDP", "nGDP"}.issubset(master.columns):
        master["govdebt"] = (pd.to_numeric(master["govdebt_GDP"], errors="coerce") * pd.to_numeric(master["nGDP"], errors="coerce")) / 100
    if {"govdef_GDP", "nGDP"}.issubset(master.columns):
        master["govdef"] = (pd.to_numeric(master["govdef_GDP"], errors="coerce") * pd.to_numeric(master["nGDP"], errors="coerce")) / 100
    if {"CA_GDP", "nGDP"}.issubset(master.columns):
        master["CA"] = (pd.to_numeric(master["CA_GDP"], errors="coerce") * pd.to_numeric(master["nGDP"], errors="coerce")) / 100

    vars_only = [col for col in master.columns if col not in {"ISO3", "year"}]
    all_missing = master[vars_only].isna().all(axis=1)
    master["first_year"] = master["year"].where(~all_missing)
    master["first_year_final"] = master.groupby("ISO3")["first_year"].transform("min")
    master = master.loc[master["year"] >= master["first_year_final"]].copy()
    master = master.drop(columns=["first_year", "first_year_final"])

    vars_only = [col for col in master.columns if col not in {"ISO3", "year"}]
    all_missing = master[vars_only].isna().all(axis=1)
    master["last_year"] = master["year"].where(~all_missing)
    master["last_year_final"] = master.groupby("ISO3")["last_year"].transform("max")
    master = master.loc[master["year"] <= master["last_year_final"]].copy()
    master = master.drop(columns=["last_year", "last_year_final"])

    countrylist = pd.read_dta(helper_dir / "countrylist.dta", convert_categoricals=False)[["ISO3", "countryname"]].copy()
    master = master.merge(countrylist, on="ISO3", how="left")
    order = [
        "countryname", "ISO3", "year", "nGDP", "rGDP", "rGDP_pc", "rGDP_USD", "deflator",
        "cons", "rcons", "cons_GDP", "inv", "inv_GDP", "finv", "finv_GDP", "exports", "exports_GDP",
        "imports", "imports_GDP", "CA", "CA_GDP", "USDfx", "REER", "govexp", "govexp_GDP", "govrev",
        "govrev_GDP", "govtax", "govtax_GDP", "govdef", "govdef_GDP", "govdebt", "govdebt_GDP",
        "HPI", "CPI", "infl", "pop", "unemp", "strate", "ltrate", "cbrate", "M0", "M1", "M2", "M3", "M4",
        "SovDebtCrisis", "CurrencyCrisis", "BankingCrisis",
    ]
    ordered = [col for col in order if col in master.columns] + [col for col in master.columns if col not in order]
    master = master[ordered].copy()

    replacements = {
        "Saint Helena, Ascension and Tristan da Cunha": "St-Helena",
        "United States Minor Outlying Islands": "USA Minor Outlying Islands",
        "Democratic Republic of the Congo": "Congo DR",
        "Micronesia (Federated States of)": "Micronesia",
        "Saint Vincent and the Grenadines": "St-Vincent",
        "Bonaire, Sint Eustatius and Saba": "Bonaire",
        "German Democratic Republic": "East Germany",
        "Saint Pierre and Miquelon": "St-Pierre",
        "Turks and Caicos Islands": "Turks and Caicos",
    }
    master["countryname"] = master["countryname"].replace(replacements)
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("float64")
    for col in master.columns:
        if col in {"countryname", "ISO3", "year"}:
            continue
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float32")
    master = _key_sort(master, ["ISO3", "year"])
    _save_dta(master, final_dir / "data_final.dta")
    return master
__all__ = ["merge_final_data"]
