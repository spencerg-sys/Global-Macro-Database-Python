from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_adb(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    def _year_columns(df: pd.DataFrame) -> list[str]:
        return [col for col in df.columns if str(col).isdigit()]

    def _reshape_codes(df: pd.DataFrame, value_name: str, *, code_col: str = "code") -> pd.DataFrame:
        years = _year_columns(df)
        long_df = df.melt(id_vars=[code_col, "countryname"], value_vars=years, var_name="year", value_name=value_name)
        wide = long_df.pivot_table(index=["countryname", "year"], columns=code_col, values=value_name, aggfunc="first").reset_index()
        wide.columns.name = None
        return wide

    pop = pd.read_excel(raw_dir / "aggregators" / "ADB" / "ADB_pop.xlsx", sheet_name="Data", dtype=str)
    pop["code"] = pop["Indicator"].replace({"Total population": "pop", "Unemployment rate": "unemp"})
    pop = pop.loc[pop["code"].isin(["pop", "unemp"])].copy()
    pop = pop.rename(columns={"Economy": "countryname"})
    pop = pop.drop(columns=["Indicator", "Unit of Measure"], errors="ignore")
    master = _reshape_codes(pop, "ADB_").rename(columns={"pop": "ADB_pop", "unemp": "ADB_unemp"})
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    for col in [c for c in ["ADB_pop", "ADB_unemp"] if c in master.columns]:
        master[col] = pd.to_numeric(master[col], errors="coerce")
    if "ADB_pop" in master.columns:
        master["ADB_pop"] = pd.to_numeric(master["ADB_pop"], errors="coerce") / 1_000_000

    cpi = pd.read_excel(raw_dir / "aggregators" / "ADB" / "ADB_CPI.xlsx", sheet_name="Data", dtype=str)
    cpi = cpi.loc[cpi["Unit of Measure"].fillna("").astype(str) != ""].copy()
    cpi = cpi.rename(columns={"Economy": "countryname"})
    cpi = cpi.drop(columns=["Indicator", "Unit of Measure"], errors="ignore")
    years = _year_columns(cpi)
    cpi = cpi.melt(id_vars=["countryname"], value_vars=years, var_name="year", value_name="ADB_CPI")
    cpi["ADB_CPI"] = cpi["ADB_CPI"].astype(str).str.replace("...", "", regex=False)
    cpi["year"] = pd.to_numeric(cpi["year"], errors="coerce").astype("int16")
    cpi["ADB_CPI"] = pd.to_numeric(cpi["ADB_CPI"], errors="coerce")
    master = master.merge(cpi, on=["countryname", "year"], how="outer", sort=False)

    gov = pd.read_excel(raw_dir / "aggregators" / "ADB" / "ADB_gov.xlsx", sheet_name="Data", dtype=str)
    gov = gov.loc[gov["Unit of Measure"].fillna("").astype(str) != ""].copy()
    gov["code"] = gov["Indicator"].replace(
        {
            "Government Revenue (% of GDP)": "govrev_GDP",
            "Government Net Lending/Net Borrowing (% of GDP)": "govdef_GDP",
            "Government Taxes (% of GDP)": "govtax_GDP",
            "Government Expenditure (% of GDP)": "govexp_GDP",
        }
    )
    gov = gov.rename(columns={"Economy": "countryname"})
    gov = gov.drop(columns=["Indicator", "Unit of Measure"], errors="ignore")
    gov = _reshape_codes(gov, "ADB_")
    gov = gov.rename(columns={col: f"ADB_{col}" for col in ["govrev_GDP", "govdef_GDP", "govtax_GDP", "govexp_GDP"] if col in gov.columns})
    gov["year"] = pd.to_numeric(gov["year"], errors="coerce").astype("int16")
    for col in [c for c in gov.columns if c.startswith("ADB_")]:
        gov[col] = pd.to_numeric(gov[col].astype(str).str.replace("...", "", regex=False), errors="coerce")
    master = master.merge(gov, on=["countryname", "year"], how="outer", sort=False)

    macro = pd.read_excel(raw_dir / "aggregators" / "ADB" / "ADB_macro.xlsx", sheet_name="Data", dtype=str)
    macro = macro.loc[~macro["Unit of Measure"].fillna("").astype(str).isin(["US Dollar", ""])].copy()
    macro = macro.loc[~macro["Indicator"].astype(str).isin(["Demand deposits (excluding government deposits)", "Overall balance"])].copy()
    macro["code"] = ""
    indicator = macro["Indicator"].astype(str)
    unit = macro["Unit of Measure"].fillna("").astype(str)
    macro.loc[indicator.eq("Yield on Short-Term Treasury Bills (% per annum, period averages)"), "code"] = "strate"
    macro.loc[indicator.eq("External trade—Imports, cif"), "code"] = "imports"
    macro.loc[indicator.eq("External trade—Exports, fob"), "code"] = "exports"
    macro.loc[indicator.eq("Money supply (M1)"), "code"] = "M1"
    macro.loc[indicator.eq("BOP—Overall balance (% of GDP)"), "code"] = "CA_GDP"
    macro.loc[indicator.eq("GDP at constant prices"), "code"] = "rGDP"
    macro.loc[indicator.eq("Average of period"), "code"] = "USDfx"
    macro.loc[indicator.eq("Gross domestic saving at current prices"), "code"] = "sav"
    macro.loc[indicator.eq("Currency in circulation"), "code"] = "M0"
    macro.loc[indicator.eq("CPI (national)—Food and nonalcoholic beverages price index (% annual change)"), "code"] = "infl"
    macro.loc[indicator.eq("Gross capital formation at current prices"), "code"] = "inv"
    macro.loc[indicator.eq("Money supply (M2)") & unit.ne("percent"), "code"] = "M2"
    macro.loc[indicator.eq("GDP at current prices"), "code"] = "nGDP"
    macro.loc[indicator.eq("Money supply (M3)"), "code"] = "M3"
    macro.loc[indicator.eq("Money supply (M4)"), "code"] = "M4"
    macro = macro.loc[macro["code"].ne("")].copy()
    macro = macro.rename(columns={"Economy": "countryname"})
    macro = macro.drop(columns=["Indicator", "Unit of Measure"], errors="ignore")
    macro = _reshape_codes(macro, "ADB_")
    macro = macro.rename(columns={col: f"ADB_{col}" for col in ["strate", "imports", "exports", "M1", "CA_GDP", "rGDP", "USDfx", "sav", "M0", "infl", "inv", "M2", "nGDP", "M3", "M4"] if col in macro.columns})
    macro["year"] = pd.to_numeric(macro["year"], errors="coerce").astype("int16")
    for col in [c for c in macro.columns if c.startswith("ADB_")]:
        macro[col] = pd.to_numeric(macro[col].astype(str).str.replace("...", "", regex=False), errors="coerce")
    for col in [c for c in macro.columns if re.fullmatch(r"ADB_M\d|ADB_exports|ADB_imports|ADB_inv|ADB_nGDP|ADB_rGDP|ADB_sav", c)]:
        macro[col] = pd.to_numeric(macro[col], errors="coerce") / 1_000_000
    master = master.merge(macro, on=["countryname", "year"], how="outer", sort=False)

    if {"ADB_govexp_GDP", "ADB_nGDP"}.issubset(master.columns):
        master["ADB_govexp"] = pd.to_numeric(master["ADB_govexp_GDP"], errors="coerce") * pd.to_numeric(master["ADB_nGDP"], errors="coerce") / 100
    if {"ADB_govrev_GDP", "ADB_nGDP"}.issubset(master.columns):
        master["ADB_govrev"] = pd.to_numeric(master["ADB_govrev_GDP"], errors="coerce") * pd.to_numeric(master["ADB_nGDP"], errors="coerce") / 100
    if {"ADB_govdef_GDP", "ADB_nGDP"}.issubset(master.columns):
        master["ADB_govdef"] = pd.to_numeric(master["ADB_govdef_GDP"], errors="coerce") * pd.to_numeric(master["ADB_nGDP"], errors="coerce") / 100
    if {"ADB_govtax_GDP", "ADB_nGDP"}.issubset(master.columns):
        master["ADB_govtax"] = pd.to_numeric(master["ADB_govtax_GDP"], errors="coerce") * pd.to_numeric(master["ADB_nGDP"], errors="coerce") / 100

    master["countryname"] = master["countryname"].replace(
        {
            "Brunei Darussalam": "Brunei",
            "China, People's Republic of": "China",
            "Hong Kong, China": "Hong Kong",
            "Korea, Republic of": "South Korea",
            "Kyrgyz Republic": "Kyrgyzstan",
            "Lao People's Democratic Republic": "Laos",
            "Taipei,China": "Taiwan",
            "Viet Nam": "Vietnam",
            "Micronesia, Federated States of": "Micronesia (Federated States of)",
        }
    )
    countrylist = _load_dta(helper_dir / "countrylist.dta")[["countryname", "ISO3"]].copy()
    master = master.merge(countrylist, on="countryname", how="left")
    master = master.loc[master["ISO3"].notna()].copy()
    master = master.drop(columns=["countryname"])

    if "ADB_M2" in master.columns:
        mask = (master["ISO3"].astype(str) == "CHN") & pd.to_numeric(master["year"], errors="coerce").eq(2020) & (pd.to_numeric(master["ADB_M2"], errors="coerce") < 1)
        master.loc[mask, "ADB_M2"] = pd.to_numeric(master.loc[mask, "ADB_M2"], errors="coerce") * (10**9)
        mask = (master["ISO3"].astype(str) == "HKG") & pd.to_numeric(master["year"], errors="coerce").eq(2020) & (pd.to_numeric(master["ADB_M2"], errors="coerce") < 1)
        master.loc[mask, "ADB_M2"] = pd.to_numeric(master.loc[mask, "ADB_M2"], errors="coerce") * (10**9)
    if "ADB_govtax" in master.columns:
        master.loc[master["ISO3"].astype(str) == "TUV", "ADB_govtax"] = pd.NA
    if "ADB_rGDP" in master.columns:
        mask = (master["ISO3"].astype(str) == "MMR") & pd.to_numeric(master["year"], errors="coerce").eq(2000)
        master.loc[mask, "ADB_rGDP"] = pd.to_numeric(master.loc[mask, "ADB_rGDP"], errors="coerce") * 10

    for col in [c for c in master.columns if c not in {"ISO3", "year"}]:
        master[col] = pd.to_numeric(master[col], errors="coerce")
        master.loc[master[col].eq(0), col] = pd.NA

    if {"ADB_imports", "ADB_nGDP"}.issubset(master.columns):
        master["ADB_imports_GDP"] = pd.to_numeric(master["ADB_imports"], errors="coerce") / pd.to_numeric(master["ADB_nGDP"], errors="coerce").replace(0, np.nan) * 100
    if {"ADB_exports", "ADB_nGDP"}.issubset(master.columns):
        master["ADB_exports_GDP"] = pd.to_numeric(master["ADB_exports"], errors="coerce") / pd.to_numeric(master["ADB_nGDP"], errors="coerce").replace(0, np.nan) * 100
    if {"ADB_inv", "ADB_nGDP"}.issubset(master.columns):
        master["ADB_inv_GDP"] = pd.to_numeric(master["ADB_inv"], errors="coerce") / pd.to_numeric(master["ADB_nGDP"], errors="coerce").replace(0, np.nan) * 100

    for col in ["ADB_govexp", "ADB_govrev", "ADB_govdef", "ADB_govtax", "ADB_imports_GDP", "ADB_exports_GDP", "ADB_inv_GDP"]:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce").astype("float32")

    ordered_cols = [
        "ISO3",
        "year",
        "ADB_CA_GDP",
        "ADB_M0",
        "ADB_M1",
        "ADB_M2",
        "ADB_M3",
        "ADB_M4",
        "ADB_USDfx",
        "ADB_exports",
        "ADB_imports",
        "ADB_infl",
        "ADB_inv",
        "ADB_nGDP",
        "ADB_rGDP",
        "ADB_sav",
        "ADB_strate",
        "ADB_govdef_GDP",
        "ADB_govexp_GDP",
        "ADB_govrev_GDP",
        "ADB_govtax_GDP",
        "ADB_CPI",
        "ADB_pop",
        "ADB_unemp",
        "ADB_govexp",
        "ADB_govrev",
        "ADB_govdef",
        "ADB_govtax",
        "ADB_imports_GDP",
        "ADB_exports_GDP",
        "ADB_inv_GDP",
    ]
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    master = _sort_keys(master)
    if master.duplicated(["ISO3", "year"]).any():
        raise sh.PipelineRuntimeError("ISO3 year do not uniquely identify observations", code=459)
    master = master[[col for col in ordered_cols if col in master.columns]]

    out_path = clean_dir / "aggregators" / "ADB" / "ADB.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_adb"]
