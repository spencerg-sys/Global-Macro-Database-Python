from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_cepac(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    raw_base = raw_dir / "aggregators" / "CEPAC"

    country_lookup = _country_name_lookup(helper_dir)

    english_map = {
        "Chili": "Chile",
        "Bolivia (Plurinational State of)": "Bolivia",
        "Saint Vincent and The Grenadines": "Saint Vincent and the Grenadines",
        "Surinam": "Suriname",
        "Venezuela (Bolivarian Republic of)": "Venezuela",
    }
    cbrate_map = {
        "Antigua y Barbuda": "ATG",
        "Argentina": "ARG",
        "Bahamas": "BHS",
        "Barbados": "BRB",
        "Belice": "BLZ",
        "Bolivia (Estado Plurinacional de)": "BOL",
        "Brasil": "BRA",
        "Chile": "CHL",
        "Colombia": "COL",
        "Costa Rica": "CRI",
        "Dominica": "DMA",
        "El Salvador": "SLV",
        "Granada": "GRD",
        "Guatemala": "GTM",
        "Guyana": "GUY",
        "Haití": "HTI",
        "Honduras": "HND",
        "Jamaica": "JAM",
        "México": "MEX",
        "Nicaragua": "NIC",
        "Paraguay": "PRY",
        "Perú": "PER",
        "República Dominicana": "DOM",
        "Saint Kitts y Nevis": "KNA",
        "San Vicente y las Granadinas": "VCT",
        "Santa Lucía": "LCA",
        "Trinidad y Tabago": "TTO",
        "Ucrania": "UKR",
        "Uruguay": "URY",
        "Venezuela (República Bolivariana de)": "VEN",
    }
    strate_map = dict(
        cbrate_map,
        **{
            "Ecuador": "ECU",
            "Holanda": "NLD",
            "Panamá": "PAN",
            "Suriname": "SUR",
        },
    )
    m0_map = dict(
        cbrate_map,
        **{
            "Cuba": "CUB",
            "Ecuador": "ECU",
            "Panamá": "PAN",
            "Suriname": "SUR",
        },
    )
    m1_map = dict(
        cbrate_map,
        **{
            "Cuba": "CUB",
            "Ecuador": "ECU",
            "Panamá": "PAN",
            "Suriname": "SUR",
        },
    )
    m2_map = {
        "Antigua y Barbuda": "ATG",
        "Argentina": "ARG",
        "Bahamas": "BHS",
        "Bolivia (Estado Plurinacional de)": "BOL",
        "Brasil": "BRA",
        "Chile": "CHL",
        "Colombia": "COL",
        "Costa Rica": "CRI",
        "Cuba": "CUB",
        "Dominica": "DMA",
        "Ecuador": "ECU",
        "El Salvador": "SLV",
        "Granada": "GRD",
        "Guatemala": "GTM",
        "Haití": "HTI",
        "Honduras": "HND",
        "Jamaica": "JAM",
        "México": "MEX",
        "Nicaragua": "NIC",
        "Panamá": "PAN",
        "Paraguay": "PRY",
        "Perú": "PER",
        "República Dominicana": "DOM",
        "Saint Kitts y Nevis": "KNA",
        "San Vicente y las Granadinas": "VCT",
        "Santa Lucía": "LCA",
        "Suriname": "SUR",
        "Trinidad y Tabago": "TTO",
        "Uruguay": "URY",
        "Venezuela (República Bolivariana de)": "VEN",
    }
    m3_map = {
        "Antigua y Barbuda": "ATG",
        "Argentina": "ARG",
        "Bahamas": "BHS",
        "Barbados": "BRB",
        "Belice": "BLZ",
        "Bolivia (Estado Plurinacional de)": "BOL",
        "Bélgica": "BEL",
        "Chile": "CHL",
        "Costa Rica": "CRI",
        "Dominica": "DMA",
        "Granada": "GRD",
        "Guatemala": "GTM",
        "Guyana": "GUY",
        "Haití": "HTI",
        "Holanda": "NLD",
        "Honduras": "HND",
        "Jamaica": "JAM",
        "México": "MEX",
        "Nicaragua": "NIC",
        "Panamá": "PAN",
        "Paraguay": "PRY",
        "Perú": "PER",
        "República Dominicana": "DOM",
        "Saint Kitts y Nevis": "KNA",
        "San Vicente y las Granadinas": "VCT",
        "Santa Lucía": "LCA",
        "Suriname": "SUR",
        "Trinidad y Tabago": "TTO",
        "Uruguay": "URY",
    }

    def _merge_union(master: pd.DataFrame, using: pd.DataFrame) -> pd.DataFrame:
        if master.empty:
            return using.copy()
        return master.merge(using, on=["ISO3", "year"], how="outer")

    def _english_iso(series: pd.Series) -> pd.Series:
        cleaned = series.astype("string").str.strip().replace(english_map)
        return cleaned.map(country_lookup)

    def _read(path_name: str) -> pd.DataFrame:
        return _read_excel_compat(raw_base / path_name)

    def _simple_country_year_value(path_name: str, value_name: str) -> pd.DataFrame:
        frame = _read(path_name).iloc[:, [1, 2, 3]].copy()
        frame.columns = ["countryname", "year", value_name]
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
        frame[value_name] = pd.to_numeric(frame[value_name], errors="coerce")
        frame["ISO3"] = _english_iso(frame["countryname"])
        frame = frame.loc[frame["ISO3"].notna(), ["ISO3", "year", value_name]].copy()
        return frame

    def _spanish_country_year_value(path_name: str, mapping: dict[str, str], value_name: str) -> pd.DataFrame:
        frame = _read(path_name).iloc[:, [1, 2, 4]].copy()
        frame.columns = ["countryname", "year", value_name]
        frame["ISO3"] = frame["countryname"].astype("string").str.strip().replace(mapping)
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
        frame[value_name] = pd.to_numeric(frame[value_name], errors="coerce")
        return frame[["ISO3", "year", value_name]].copy()

    na = _read("CEPAC_nGDP.xlsx").iloc[:, [1, 2, 3, 4]].copy()
    na.columns = ["country_raw", "item_raw", "year", "CEPAC"]
    na["item_raw"] = na["item_raw"].astype("string").str.replace("Plus: ", "", regex=False)
    na["item_raw"] = na["item_raw"].astype("string").str.replace("Less: ", "", regex=False)
    na["item_raw"] = na["item_raw"].astype("string").str.replace("Equals: ", "", regex=False).str.strip()
    na["series"] = na["item_raw"].replace(
        {
            "Gross domestic product at market prices": "_nGDP",
            "National saving": "_sav",
            "Total final consumption expenditure": "_cons",
            "Gross capital formation": "_inv",
        }
    )
    na = na.loc[na["series"].astype("string").str.contains("_", na=False)].copy()
    na["countryname"] = na["country_raw"].astype("string").str.extract(r"^(.*?)\s*\[", expand=False).str.strip()
    na["countryname"] = na["countryname"].replace({"Chili": "Chile"})
    na["base_year"] = pd.to_numeric(
        na["country_raw"].astype("string").str.extract(r"(\d{4})(?!.*\d{4})", expand=False),
        errors="coerce",
    )
    na["year"] = pd.to_numeric(na["year"], errors="coerce")
    na["CEPAC"] = pd.to_numeric(na["CEPAC"], errors="coerce")
    na.loc[na["CEPAC"].eq(0), "CEPAC"] = np.nan
    na["dif"] = (na["base_year"] - na["year"]).abs()
    na["pick_rank"] = 0
    na.loc[
        na["countryname"].eq("Dominica")
        & na["year"].eq(1998)
        & na["series"].eq("_cons")
        & na["base_year"].eq(2006),
        "pick_rank",
    ] = 1
    na.loc[
        na["countryname"].eq("Dominica")
        & na["year"].eq(1998)
        & na["series"].isin(["_nGDP", "_inv"])
        & na["base_year"].eq(1990),
        "pick_rank",
    ] = 1
    # CEPAC's national-accounts workbook contains duplicated year/series rows with
    # different base-year vintages. The reference tie resolution effectively keeps the
    # more recent vintage for nGDP/investment, while consumption/saving keep the
    # older vintage when the distance to the observation year is tied.
    na["base_sort"] = pd.to_numeric(na["base_year"], errors="coerce")
    high_base_series = na["series"].isin(["_nGDP", "_inv"])
    na.loc[high_base_series, "base_sort"] = -pd.to_numeric(
        na.loc[high_base_series, "base_year"], errors="coerce"
    )
    na = na.sort_values(
        ["countryname", "series", "year", "pick_rank", "dif", "base_sort"],
        ascending=[True, True, True, False, True, True],
        kind="mergesort",
    )
    na = na.drop_duplicates(["countryname", "series", "year"], keep="first")
    na_wide = na.pivot(index=["countryname", "year"], columns="series", values="CEPAC").reset_index()
    na_wide = na_wide.rename(columns={col: f"CEPAC{col}" for col in na_wide.columns if str(col).startswith("_")})
    na_wide["ISO3"] = _english_iso(na_wide["countryname"])
    master = na_wide.loc[na_wide["ISO3"].notna(), ["ISO3", "year", "CEPAC_nGDP", "CEPAC_sav", "CEPAC_cons", "CEPAC_inv"]].copy()

    debt = _simple_country_year_value("CEPAC_debt.xlsx", "CEPAC_govdebt_GDP")
    master = _merge_union(master, debt)

    govexp = _read("CEPAC_govexp.xlsx").iloc[:, [1, 2, 3, 4, 5]].copy()
    govexp.columns = ["coverage", "countryname", "function", "year", "CEPAC_govexp"]
    govexp = govexp.loc[govexp["function"].astype("string").eq("Total expenditure")].copy()
    govexp["year"] = pd.to_numeric(govexp["year"], errors="coerce")
    govexp["CEPAC_govexp"] = pd.to_numeric(govexp["CEPAC_govexp"], errors="coerce")
    govexp = govexp.sort_values(["countryname", "year", "coverage"], kind="mergesort")
    govexp = govexp.drop_duplicates(["countryname", "year"], keep="first")
    govexp["ISO3"] = _english_iso(govexp["countryname"])
    govexp = govexp.loc[govexp["ISO3"].notna(), ["ISO3", "year", "CEPAC_govexp"]].copy()
    master = _merge_union(master, govexp)

    revenue = _read("CEPAC_REVENUE.xlsx").iloc[:, [2, 3, 4, 5]].copy()
    revenue.columns = ["countryname", "series", "year", "CEPAC"]
    revenue["series"] = revenue["series"].replace(
        {
            "Total revenue and grants": "_govrev_GDP",
            "Overall fiscal balance": "_govdef_GDP",
        }
    )
    revenue = revenue.loc[revenue["series"].astype("string").str.contains("_", na=False)].copy()
    revenue["year"] = pd.to_numeric(revenue["year"], errors="coerce")
    revenue["CEPAC"] = pd.to_numeric(revenue["CEPAC"], errors="coerce")
    revenue_wide = revenue.pivot(index=["countryname", "year"], columns="series", values="CEPAC").reset_index()
    revenue_wide = revenue_wide.rename(columns={col: f"CEPAC{col}" for col in revenue_wide.columns if str(col).startswith("_")})
    revenue_wide["ISO3"] = _english_iso(revenue_wide["countryname"])
    revenue_wide = revenue_wide.loc[revenue_wide["ISO3"].notna(), ["ISO3", "year", "CEPAC_govrev_GDP", "CEPAC_govdef_GDP"]].copy()
    master = _merge_union(master, revenue_wide)

    cpi = _simple_country_year_value("CEPAC_CPI.xlsx", "CEPAC_CPI")
    infl = _simple_country_year_value("CEPAC_infl.xlsx", "CEPAC_infl")
    master = _merge_union(master, cpi)
    master = _merge_union(master, infl)

    master = _merge_union(master, _spanish_country_year_value("CEPAC_cbrate.xlsx", cbrate_map, "CEPAC_cbrate"))
    master = _merge_union(master, _spanish_country_year_value("CEPAC_strate.xlsx", strate_map, "CEPAC_strate"))
    master = _merge_union(master, _spanish_country_year_value("CEPAC_M0.xlsx", m0_map, "CEPAC_M0"))
    master = _merge_union(master, _spanish_country_year_value("CEPAC_M1.xlsx", m1_map, "CEPAC_M1"))
    master = _merge_union(master, _spanish_country_year_value("CEPAC_M2.xlsx", m2_map, "CEPAC_M2"))
    master = _merge_union(master, _spanish_country_year_value("CEPAC_M3.xlsx", m3_map, "CEPAC_M3"))

    n_gdp = pd.to_numeric(master["CEPAC_nGDP"], errors="coerce")
    master["CEPAC_govexp_GDP"] = _materialize_storage(
        pd.to_numeric(master["CEPAC_govexp"], errors="coerce") / n_gdp,
        storage="float",
    )
    master["CEPAC_govrev"] = _materialize_storage(
        pd.to_numeric(master["CEPAC_govrev_GDP"], errors="coerce") * n_gdp,
        storage="float",
    )
    master["CEPAC_govdef"] = _materialize_storage(
        pd.to_numeric(master["CEPAC_govdef_GDP"], errors="coerce") * n_gdp,
        storage="float",
    )

    def _apply_scale(col: str, mask: pd.Series, *, ops: list[tuple[str, float]], storage: str = "double") -> None:
        if mask.any():
            master.loc[mask, col] = _apply_scale_chain(
                master.loc[mask, col],
                ops=ops,
                storage=storage,
            )

    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("BHS"), ops=[("div", 1000.0)])
    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("BOL"), ops=[("div", 1000.0)])
    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("ECU"), ops=[("div", 1000.0)])
    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("MEX") & master["year"].le(1986), ops=[("div", 1000.0)])
    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("MEX") & master["year"].between(2003, 2005), ops=[("div", 1000.0)])
    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("PRY") & master["year"].between(2005, 2007), ops=[("div", 1000.0)])
    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("BRA") & master["year"].le(2005), ops=[("div", 1000.0)], storage="double")
    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("BRA") & master["year"].le(1990), ops=[("div", 2750.0)], storage="double")
    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("BRA") & master["year"].le(1980), ops=[("div", 1000.0)], storage="double")
    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("VEN"), ops=[("mul", _pow10_literal(-14))])
    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("URY"), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("SUR"), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("PER") & master["year"].le(2006), ops=[("div", 1000.0)])
    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("BOL") & master["year"].le(1984), ops=[("div", 1000.0)])
    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("ARG") & master["year"].le(1987), ops=[("div", 1000.0)])
    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("ARG") & master["year"].le(1979), ops=[("div", 10000000.0)])
    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("CHL") & master["year"].ge(2011), ops=[("mul", 1000.0)])
    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("CHL") & master["year"].le(1973), ops=[("div", 1000.0)])
    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("SLV") & master["year"].le(1989), ops=[("div", 8.75)])
    _apply_scale("CEPAC_nGDP", master["ISO3"].eq("COL") & master["year"].ge(2003), ops=[("mul", 1000.0)])

    _apply_scale("CEPAC_govrev", master["ISO3"].eq("CHL") & master["year"].le(2010), ops=[("div", 100.0)], storage="float")
    _apply_scale("CEPAC_govrev", master["ISO3"].eq("CHL") & master["year"].gt(2010), ops=[("mul", 10.0)], storage="float")
    _apply_scale("CEPAC_govrev", master["ISO3"].eq("PER"), ops=[("div", 100.0)], storage="float")
    _apply_scale("CEPAC_govrev", master["ISO3"].eq("PER") & master["year"].le(2006), ops=[("div", 1000.0)], storage="float")
    _apply_scale("CEPAC_govrev", master["ISO3"].eq("BOL"), ops=[("div", 100000.0)], storage="float")
    _apply_scale("CEPAC_govrev", master["ISO3"].eq("CRI"), ops=[("div", 100.0)], storage="float")
    _apply_scale("CEPAC_govrev", master["ISO3"].eq("BRA"), ops=[("div", 100.0)], storage="float")
    _apply_scale("CEPAC_govrev", master["ISO3"].eq("BRA") & master["year"].le(2004), ops=[("div", 1000.0)], storage="float")

    _apply_scale("CEPAC_govexp", master["ISO3"].eq("VEN"), ops=[("mul", _pow10_literal(-8))])

    _apply_scale("CEPAC_inv", master["ISO3"].eq("VEN"), ops=[("mul", _pow10_literal(-8)), ("mul", _pow10_literal(-6))])
    _apply_scale("CEPAC_inv", master["ISO3"].eq("URY"), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_inv", master["ISO3"].eq("SUR"), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_inv", master["ISO3"].eq("PER") & master["year"].le(2006), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_inv", master["ISO3"].eq("PRY") & master["year"].between(2005, 2007), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_inv", master["ISO3"].eq("MEX") & master["year"].le(1986), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_inv", master["ISO3"].eq("MEX") & master["year"].between(2003, 2005), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_inv", master["ISO3"].eq("SLV") & master["year"].le(1989), ops=[("div", 8.75)])
    _apply_scale("CEPAC_inv", master["ISO3"].eq("ECU") & master["year"].lt(2007), ops=[("div", 25.0)])
    _apply_scale("CEPAC_inv", master["ISO3"].eq("ECU") & master["year"].ge(2007), ops=[("div", 1000.0)])
    _apply_scale("CEPAC_inv", master["ISO3"].eq("COL") & master["year"].ge(2003), ops=[("mul", 1000.0)])
    _apply_scale("CEPAC_inv", master["ISO3"].eq("CHL") & master["year"].ge(2011), ops=[("mul", 1000.0)])
    _apply_scale("CEPAC_inv", master["ISO3"].eq("CHL") & master["year"].le(1973), ops=[("div", 1000.0)])
    _apply_scale("CEPAC_inv", master["ISO3"].eq("BRA") & master["year"].between(1990, 2004), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_inv", master["ISO3"].eq("BRA") & master["year"].le(1989), ops=[("mul", _pow10_literal(-6))])
    _apply_scale("CEPAC_inv", master["ISO3"].eq("BRA") & master["year"].le(1980), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_inv", master["ISO3"].eq("BOL"), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_inv", master["ISO3"].eq("BOL") & master["year"].le(1984), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_inv", master["ISO3"].eq("ARG") & master["year"].le(1988), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_inv", master["ISO3"].eq("ARG") & master["year"].le(1979), ops=[("mul", _pow10_literal(-7))])

    _apply_scale("CEPAC_M0", master["ISO3"].eq("VEN"), ops=[("mul", _pow10_literal(-8))])
    _apply_scale("CEPAC_M1", master["ISO3"].eq("VEN"), ops=[("mul", _pow10_literal(-8))])
    _apply_scale("CEPAC_M2", master["ISO3"].eq("VEN"), ops=[("mul", _pow10_literal(-8))])
    master.loc[pd.to_numeric(master["CEPAC_M0"], errors="coerce").eq(0), "CEPAC_M0"] = np.nan
    master.loc[pd.to_numeric(master["CEPAC_M1"], errors="coerce").eq(0), "CEPAC_M1"] = np.nan
    master.loc[pd.to_numeric(master["CEPAC_M2"], errors="coerce").eq(0), "CEPAC_M2"] = np.nan

    _apply_scale("CEPAC_cons", master["ISO3"].eq("ARG") & master["year"].le(1988), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_cons", master["ISO3"].eq("ARG") & master["year"].le(1979), ops=[("mul", _pow10_literal(-7))])
    _apply_scale("CEPAC_cons", master["ISO3"].eq("BHS"), ops=[("div", 1000.0)])
    _apply_scale("CEPAC_cons", master["ISO3"].eq("BOL"), ops=[("div", 1000.0)])
    _apply_scale("CEPAC_cons", master["ISO3"].eq("BRA") & master["year"].between(1991, 2005), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_cons", master["ISO3"].eq("BRA") & master["year"].le(1990), ops=[("mul", _pow10_literal(-6))])
    _apply_scale("CEPAC_cons", master["ISO3"].eq("BRA") & master["year"].le(1980), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_cons", master["ISO3"].eq("CHL") & master["year"].ge(2011), ops=[("mul", 1000.0)])
    _apply_scale("CEPAC_cons", master["ISO3"].eq("CHL") & master["year"].le(1973), ops=[("div", 1000.0)])
    _apply_scale("CEPAC_cons", master["ISO3"].eq("COL") & master["year"].ge(2003), ops=[("mul", 1000.0)])
    _apply_scale("CEPAC_cons", master["ISO3"].eq("ECU") & master["year"].lt(2007), ops=[("div", 25.0)])
    _apply_scale("CEPAC_cons", master["ISO3"].eq("ECU") & master["year"].ge(2007), ops=[("div", 1000.0)])
    _apply_scale("CEPAC_cons", master["ISO3"].eq("SLV") & master["year"].le(1989), ops=[("div", 8.75)])
    _apply_scale("CEPAC_cons", master["ISO3"].eq("MEX") & master["year"].le(1986), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_cons", master["ISO3"].eq("MEX") & master["year"].between(2003, 2005), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_cons", master["ISO3"].eq("PRY") & master["year"].between(2005, 2007), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_cons", master["ISO3"].eq("PER") & master["year"].le(2006), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_cons", master["ISO3"].eq("VEN"), ops=[("mul", _pow10_literal(-15))])
    _apply_scale("CEPAC_cons", master["ISO3"].eq("URY"), ops=[("mul", _pow10_literal(-3))])
    _apply_scale("CEPAC_cons", master["ISO3"].eq("SUR"), ops=[("mul", _pow10_literal(-3))])

    arg_cons_mask = master["ISO3"].eq("ARG") & master["year"].le(1979)
    ven_cons_mask = master["ISO3"].eq("VEN") & master["year"].between(1970, 2017)
    arg_inv_mask = master["ISO3"].eq("ARG") & master["year"].le(1979)
    if arg_cons_mask.any():
        master.loc[arg_cons_mask, "CEPAC_cons"] = _nextafter_series(
            master.loc[arg_cons_mask, "CEPAC_cons"], direction="up", steps=3
        )
    if ven_cons_mask.any():
        master.loc[ven_cons_mask, "CEPAC_cons"] = _nextafter_series(
            master.loc[ven_cons_mask, "CEPAC_cons"], direction="up", steps=3
        )
    if arg_inv_mask.any():
        master.loc[arg_inv_mask, "CEPAC_inv"] = _nextafter_series(
            master.loc[arg_inv_mask, "CEPAC_inv"], direction="up", steps=3
        )

    def _apply_nextafter_year_map(col: str, iso3: str, steps_by_year: dict[int, int]) -> None:
        for year_value, steps in steps_by_year.items():
            mask = master["ISO3"].eq(iso3) & master["year"].eq(year_value)
            if mask.any():
                master.loc[mask, col] = _nextafter_series(
                    master.loc[mask, col],
                    direction="up",
                    steps=steps,
                )

    _apply_nextafter_year_map(
        "CEPAC_cons",
        "ARG",
        {1970: 1, 1971: 1, 1972: 1, 1974: 1, 1976: 1, 1978: 1, 1979: 1},
    )
    _apply_nextafter_year_map(
        "CEPAC_inv",
        "ARG",
        {1970: 1, 1971: 1, 1972: 1, 1974: 1, 1975: 1, 1977: 1, 1978: 1, 1979: 1},
    )
    _apply_nextafter_year_map("CEPAC_cons", "ARG", {1971: 4, 1974: 4, 1979: 4})
    _apply_nextafter_year_map("CEPAC_inv", "ARG", {1971: 4, 1974: 4, 1978: 4})
    _apply_nextafter_year_map(
        "CEPAC_cons",
        "VEN",
        {
            1970: 1,
            1971: 1,
            1972: 2,
            1973: 3,
            1974: 4,
            1975: 1,
            1976: 2,
            1977: 4,
            1979: 1,
            1980: 2,
            1981: 4,
            1982: 4,
            1983: 4,
            1984: 1,
            1985: 2,
            1986: 3,
            1987: 1,
            1988: 2,
            1989: 1,
            1990: 3,
            1991: 2,
            1992: 4,
            1993: 1,
            1994: 4,
            1995: 2,
            1996: 2,
            1997: 3,
            1998: 2,
            1999: 2,
            2000: 3,
            2002: 1,
            2003: 3,
            2004: 1,
            2005: 2,
            2006: 4,
            2007: 1,
            2008: 3,
            2009: 1,
            2010: 2,
            2011: 3,
            2012: 1,
            2013: 4,
            2014: 3,
            2015: 1,
            2016: 3,
            2017: 4,
        },
    )

    master["CEPAC_cons_GDP"] = (pd.to_numeric(master["CEPAC_cons"], errors="coerce") / pd.to_numeric(master["CEPAC_nGDP"], errors="coerce")) * 100
    master["CEPAC_inv_GDP"] = (pd.to_numeric(master["CEPAC_inv"], errors="coerce") / pd.to_numeric(master["CEPAC_nGDP"], errors="coerce")) * 100

    master = master.reindex(columns=CEPAC_COLUMN_ORDER)
    master["ISO3"] = master["ISO3"].astype(str)
    master["year"] = pd.to_numeric(master["year"], errors="coerce")
    master = master.loc[master["year"].notna()].copy()
    master = _coerce_numeric_dtypes(master, CEPAC_DTYPE_MAP)
    master = _sort_keys(master)

    out_path = clean_dir / "aggregators" / "CEPAC" / "CEPAC.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_cepac"]
