from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_hfs(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    raw_path = raw_dir / "aggregators" / "HFS" / "General_tables.xlsx"

    country_lookup = _country_name_lookup(helper_dir)
    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].drop_duplicates().copy()

    raw = _read_excel_compat(raw_path, sheet_name="Annual data 1800 onward", header=None, dtype=str)
    raw = raw.dropna(axis=1, how="all").iloc[2:].reset_index(drop=True)
    fixed = ["countryname", "category", "series", "unit", "scale", "time", "source", "start", "end", "notes"]
    raw.columns = fixed + [f"y_{str(raw.iloc[0, idx]).strip()}" for idx in range(10, raw.shape[1])]
    raw = raw.iloc[1:].reset_index(drop=True)
    year_cols = [col for col in raw.columns if str(col).startswith("y_")]

    def _merge_master(master: pd.DataFrame | None, part: pd.DataFrame | None) -> pd.DataFrame:
        def _collapse(frame: pd.DataFrame) -> pd.DataFrame:
            if not frame.duplicated(["ISO3", "year"]).any():
                return frame.copy()
            grouped = (
                frame.groupby(["ISO3", "year"], as_index=False, sort=False)
                .agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
            )
            return grouped

        if part is None or part.empty:
            return master.copy() if master is not None else pd.DataFrame(columns=["ISO3", "year"])
        part = _collapse(part.copy())
        part["year"] = pd.to_numeric(part["year"], errors="coerce")
        part = part.loc[part["year"].notna()].copy()
        if master is None or master.empty:
            return _sort_keys(part)
        master = _collapse(master.copy())
        merge_keys = ["ISO3", "year"]
        merged = part.merge(master, on=merge_keys, how="outer", suffixes=("", "_using"), indicator=True, sort=False)
        overlap = [col for col in part.columns if col not in merge_keys and col in master.columns]
        right_only = merged["_merge"].eq("right_only")
        for col in overlap:
            using_col = f"{col}_using"
            if using_col in merged.columns:
                merged.loc[right_only, col] = merged.loc[right_only, using_col]
                merged = merged.drop(columns=[using_col], errors="ignore")
        merged = merged.drop(columns=["_merge"], errors="ignore")
        return _sort_keys(merged)

    def _parse_numeric(series: pd.Series, *, mode: str = "g16") -> pd.Series:
        text = series.astype("string")
        text = text.str.replace("−", "-", regex=False)
        text = text.str.strip().replace({"": pd.NA})
        if mode != "g16":
            return _excel_numeric_series(text, mode=mode)

        # The reference import-excel allstring path materializes scientific-notation
        # strings with a shorter mantissa than pandas/openpyxl.
        def _parse_text_numeric(value: object) -> float:
            if pd.isna(value):
                return np.nan
            raw = str(value)
            if raw == "-":
                return np.nan
            try:
                numeric = float(raw)
            except ValueError:
                return np.nan
            if "e" in raw.lower():
                return float(format(numeric, ".12g"))
            return float(format(numeric, ".16g"))

        return text.map(_parse_text_numeric)

    def _long(
        frame: pd.DataFrame,
        id_vars: list[str],
        value_name: str = "value",
        missing: Iterable[str] = (),
        *,
        parse_mode: str = "g16",
    ) -> pd.DataFrame:
        work = frame[id_vars + year_cols].melt(id_vars=id_vars, value_vars=year_cols, var_name="year", value_name=value_name)
        work["year"] = pd.to_numeric(work["year"].astype("string").str.replace("y_", "", regex=False), errors="coerce")
        value = work[value_name].astype("string").str.strip()
        for marker in missing:
            value = value.mask(value.eq(str(marker)))
        work[value_name] = _parse_numeric(value, mode=parse_mode)
        return work

    def _wide(frame: pd.DataFrame, id_cols: list[str], value_name: str = "value") -> pd.DataFrame:
        work = frame.copy()
        if work.empty:
            return pd.DataFrame(columns=id_cols)
        work = work.drop_duplicates(id_cols + ["series"], keep="first")
        wide = work.set_index(id_cols + ["series"])[value_name].unstack("series").reset_index()
        wide.columns = [str(col) for col in wide.columns]
        return wide

    def _map_iso(frame: pd.DataFrame, country_col: str = "countryname", keep_unmatched: bool = False) -> pd.DataFrame:
        out = frame.copy()
        out[country_col] = out[country_col].astype("string").str.strip()
        out["ISO3"] = out[country_col].map(country_lookup)
        if not keep_unmatched:
            out = out.loc[out["ISO3"].notna()].copy()
        return out

    def _strip_c_prefix(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.rename(columns={col: str(col)[2:] for col in frame.columns if str(col).startswith("c_")})

    def _convert_eur(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        out = frame.merge(eur_fx, on="ISO3", how="left")
        mask = out["EUR_irrevocable_FX"].notna()
        for col in columns:
            if col in out.columns:
                out.loc[mask, col] = (
                    pd.to_numeric(out.loc[mask, col], errors="coerce")
                    / pd.to_numeric(out.loc[mask, "EUR_irrevocable_FX"], errors="coerce")
                )
        return out.drop(columns=["EUR_irrevocable_FX"], errors="ignore")

    def _apply_scale_ops(
        frame: pd.DataFrame,
        *,
        mask: pd.Series,
        col: str,
        ops: list[tuple[str, float]],
        storage: str = "double",
    ) -> None:
        if col not in frame.columns or not mask.any():
            return
        frame.loc[mask, col] = _apply_scale_chain(
            pd.to_numeric(frame.loc[mask, col], errors="coerce"),
            ops=ops,
            storage=storage,
        )

    master: pd.DataFrame | None = None

    nunes = raw.loc[raw["source"].fillna("").str.contains("Nunes", na=False, regex=False)].copy()
    nunes = nunes.loc[nunes["series"].eq("Monetary base (M0)")].copy()
    nunes["countryname"] = nunes["countryname"].replace({"Timor-Leste (East Timor)": "Timor-Leste"})
    nunes = nunes.loc[~nunes["countryname"].eq("Goa"), ["countryname"] + year_cols].copy()
    nunes = _long(nunes, ["countryname"], value_name="M0")
    nunes = _map_iso(nunes)
    nunes = nunes[["ISO3", "year", "M0"]].copy()
    nunes["M0"] = pd.to_numeric(nunes["M0"], errors="coerce") / 1000
    master = _merge_master(master, nunes)

    tjip = raw.loc[raw["source"].fillna("").str.contains("Tjipilica", na=False, regex=False)].copy()
    tjip.loc[tjip["series"].eq("Price index"), "series"] = "CPI"
    tjip.loc[tjip["series"].fillna("").str.contains("Imports", na=False), "series"] = "imports"
    tjip.loc[tjip["series"].fillna("").str.contains("Exports", na=False), "series"] = "exports"
    tjip.loc[tjip["series"].eq("Government spending"), "series"] = "govexp"
    tjip.loc[tjip["series"].eq("Government revenue"), "series"] = "govrev"
    tjip.loc[tjip["series"].eq("Nominal GDP"), "series"] = "nGDP_index_LCU"
    tjip.loc[tjip["series"].eq("Real GDP"), "series"] = "rGDP_index_LCU"
    tjip.loc[tjip["series"].isin(["Real GDP per person", "GDP per person"]), "series"] = "rGDP_pc_index_LCU"
    tjip.loc[tjip["series"].eq("Population"), "series"] = "pop"
    tjip.loc[tjip["series"].eq("Monetary base (M0)"), "series"] = "M0"
    tjip["countryname"] = tjip["countryname"].replace(
        {"Timor-Leste (East Timor)": "Timor-Leste", "Macao": "Macau"}
    )
    tjip = tjip.loc[~tjip["countryname"].eq("Goa"), ["countryname", "series"] + year_cols].copy()
    tjip = _long(tjip, ["countryname", "series"], missing=["?"])
    tjip = _wide(tjip, ["countryname", "year"])
    tjip = _map_iso(tjip)
    tjip = tjip.drop(columns=["countryname"], errors="ignore")
    if "pop" in tjip.columns:
        tjip["pop"] = pd.to_numeric(tjip["pop"], errors="coerce") / 1000
    for col in ["govexp", "govrev", "M0"]:
        if col in tjip.columns:
            tjip[col] = pd.to_numeric(tjip[col], errors="coerce") / 1000
    for col in ["exports", "imports"]:
        if col in tjip.columns:
            mask = tjip["ISO3"].ne("MAC")
            tjip.loc[mask, col] = pd.to_numeric(tjip.loc[mask, col], errors="coerce") / 1000
    if "M2" in tjip.columns:
        mask = tjip["ISO3"].eq("AGO")
        tjip.loc[mask, "M2"] = pd.to_numeric(tjip.loc[mask, "M2"], errors="coerce") / 1000
    for col in ["govexp", "imports", "exports", "govrev", "M0", "M2"]:
        if col in tjip.columns:
            mask = tjip["ISO3"].eq("AGO")
            tjip.loc[mask, col] = pd.to_numeric(tjip.loc[mask, col], errors="coerce") / 1_000_000
            mask = tjip["ISO3"].eq("MAC")
            tjip.loc[mask, col] = pd.to_numeric(tjip.loc[mask, col], errors="coerce") / 5
    master = _merge_master(master, tjip)

    ifs = raw.loc[raw["source"].fillna("").str.contains("IFS", na=False, regex=False)].copy()
    ifs = ifs.loc[~ifs["countryname"].fillna("").str.contains("Egypt", na=False)].copy()
    ifs.loc[ifs["series"].eq("Exports") & ifs["source"].eq("IFS (April 1948)"), "series"] = "c_exports"
    ifs.loc[ifs["series"].eq("Imports") & ifs["source"].eq("IFS (April 1948)"), "series"] = "c_imports"
    ifs = ifs.loc[ifs["series"].fillna("").str.contains("c_", na=False)].copy()
    to_convert = {str(value).strip() for value in ifs.loc[ifs["scale"].eq("Billions"), "countryname"].dropna().tolist()}
    ifs = _long(ifs[["countryname", "series"] + year_cols].copy(), ["countryname", "series"], missing=["n.a.", "0"])
    ifs = _wide(ifs, ["countryname", "year"])
    ifs = _strip_c_prefix(ifs)
    ifs["countryname"] = ifs["countryname"].astype("string").str.strip()
    for country in to_convert:
        mask = ifs["countryname"].eq(country)
        for col in ["imports", "exports"]:
            if col in ifs.columns:
                ifs.loc[mask, col] = pd.to_numeric(ifs.loc[mask, col], errors="coerce") * 1000
    for country in ["Czech Republic", "Iran", "Belgium"]:
        mask = ifs["countryname"].eq(country)
        for col in ["imports", "exports"]:
            if col in ifs.columns:
                ifs.loc[mask, col] = pd.to_numeric(ifs.loc[mask, col], errors="coerce") * 1000
    for country, ops in [
        ("Uruguay", [("div", 1_000_000.0)]),
        ("Peru", [("mul", _pow10_literal(-9))]),
        ("Brazil", [("mul", _pow10_literal(-15, adjust=1)), ("div", 2750.0)]),
        ("Bolivia", [("mul", _pow10_literal(-9))]),
        ("Venezuela", [("mul", _pow10_literal(-14))]),
    ]:
        mask = ifs["countryname"].eq(country)
        for col in ["imports", "exports"]:
            _apply_scale_ops(ifs, mask=mask, col=col, ops=ops)
    for country in ["Czech Republic", "Iran"]:
        mask = ifs["countryname"].eq(country)
        for col in ["imports", "exports"]:
            if col in ifs.columns:
                ifs.loc[mask, col] = pd.to_numeric(ifs.loc[mask, col], errors="coerce") / 1000
    ifs = _map_iso(ifs)
    ifs = ifs[["ISO3", "year", "exports", "imports"]].copy()
    ifs = _drop_rows_with_all_missing(ifs)
    ifs = _convert_eur(ifs, ["imports", "exports"])
    master = _merge_master(master, ifs)

    inegi = raw.loc[raw["source"].fillna("").str.contains("INEGI", na=False, regex=False)].copy()
    inegi.loc[inegi["series"].eq("Market exchange rate, US dollar"), "series"] = "c_USDfx"
    inegi.loc[inegi["series"].eq("M1: old methodology"), "series"] = "c_M1"
    inegi.loc[inegi["series"].eq("M2: old methodology"), "series"] = "c_M2"
    inegi.loc[inegi["series"].eq("M3: old methodology"), "series"] = "c_M3"
    inegi.loc[inegi["series"].eq("M4: old methodology"), "series"] = "c_M4"
    inegi.loc[inegi["series"].eq("Imports"), "series"] = "c_imports_USD"
    inegi.loc[inegi["series"].eq("Exports"), "series"] = "c_exports_USD"
    inegi.loc[inegi["series"].eq("Federal debt: total") & inegi["scale"].eq("Thousands"), "series"] = "c_cgov_debt"
    inegi.loc[inegi["series"].eq("Federal revenue") & inegi["scale"].eq("Millions"), "series"] = "c_govrev"
    inegi.loc[inegi["series"].eq("Federal spending") & inegi["scale"].eq("Millions"), "series"] = "c_govexp"
    inegi.loc[inegi["series"].eq("Gross domestic product") & inegi["unit"].eq("Constant 1970 Mexican pesos"), "series"] = "c_rGDP_LCU"
    inegi.loc[inegi["series"].eq("Monetary base (M0)"), "series"] = "c_M0"
    inegi = inegi.loc[inegi["series"].fillna("").str.contains("c_", na=False)].copy()
    inegi = _long(inegi[["countryname", "series"] + year_cols].copy(), ["countryname", "series"])
    inegi = _wide(inegi, ["countryname", "year"])
    inegi = _strip_c_prefix(inegi)
    inegi["ISO3"] = "MEX"
    inegi = inegi.drop(columns=["countryname"], errors="ignore")
    if {"imports_USD", "USDfx"}.issubset(inegi.columns):
        inegi["imports"] = _materialize_storage(
            pd.to_numeric(inegi["imports_USD"], errors="coerce")
            * pd.to_numeric(inegi["USDfx"], errors="coerce"),
            storage="float",
        )
    if {"exports_USD", "USDfx"}.issubset(inegi.columns):
        inegi["exports"] = _materialize_storage(
            pd.to_numeric(inegi["exports_USD"], errors="coerce")
            * pd.to_numeric(inegi["USDfx"], errors="coerce"),
            storage="float",
        )
    redenom_cols = ["imports", "exports", "cgov_debt", "govrev", "govexp"]
    redenom_mask = pd.to_numeric(inegi["year"], errors="coerce").lt(1993)
    for col in redenom_cols:
        if col in inegi.columns:
            inegi.loc[redenom_mask, col] = pd.to_numeric(inegi.loc[redenom_mask, col], errors="coerce") / 1000
            if col in {"imports", "exports"}:
                inegi[col] = _materialize_storage(inegi[col], storage="float")
    if "USDfx" in inegi.columns:
        mask = pd.to_numeric(inegi["year"], errors="coerce").le(1992)
        inegi.loc[mask, "USDfx"] = pd.to_numeric(inegi.loc[mask, "USDfx"], errors="coerce") / 1000
    if "cgov_debt" in inegi.columns:
        inegi["cgov_debt"] = pd.to_numeric(inegi["cgov_debt"], errors="coerce") / 1000
    if "rGDP_LCU" in inegi.columns:
        inegi.loc[pd.to_numeric(inegi["year"], errors="coerce").lt(1895), "rGDP_LCU"] = np.nan
    master = _merge_master(master, inegi)

    sarbi = raw.loc[raw["source"].fillna("").str.contains("SARBI", na=False, regex=False)].copy()
    sarbi.loc[
        sarbi["series"].eq("Market exchange rate: rate obtained by British Secretary of State for India for transfers on India"),
        "series",
    ] = "c_GBPfx"
    sarbi.loc[sarbi["series"].eq("Revenue, Indian government--series 1"), "series"] = "c_REVENUE1"
    sarbi.loc[sarbi["series"].eq("Revenue, Indian government--series 2"), "series"] = "c_REVENUE2"
    sarbi.loc[sarbi["series"].eq("Spending, Indian government--series 1"), "series"] = "c_govexp1"
    sarbi.loc[sarbi["series"].eq("Spending, Indian government--series 2"), "series"] = "c_govexp2"
    sarbi.loc[sarbi["series"].eq("Indian government debt: total"), "series"] = "c_cgov_debt"
    sarbi = sarbi.loc[sarbi["series"].fillna("").str.contains("c_", na=False)].copy()
    sarbi = _long(sarbi[["countryname", "series"] + year_cols].copy(), ["countryname", "series"])
    sarbi = _wide(sarbi, ["countryname", "year"])
    sarbi = _strip_c_prefix(sarbi)
    sarbi["ISO3"] = "IND"
    sarbi = sarbi.drop(columns=["countryname"], errors="ignore")
    if {"REVENUE1", "REVENUE2"}.issubset(sarbi.columns):
        sarbi["REVENUE1"] = sarbi["REVENUE1"].combine_first(sarbi["REVENUE2"])
    if {"govexp1", "govexp2"}.issubset(sarbi.columns):
        sarbi["govexp1"] = sarbi["govexp1"].combine_first(sarbi["govexp2"])
    sarbi = sarbi.rename(columns={"govexp1": "govexp", "REVENUE1": "govrev"})
    sarbi = sarbi.drop(columns=["govexp2", "REVENUE2"], errors="ignore")
    for col in ["cgov_debt", "govrev", "govexp"]:
        if col in sarbi.columns:
            sarbi[col] = pd.to_numeric(sarbi[col], errors="coerce") / 1_000_000
    master = _merge_master(master, sarbi)

    colombia = raw.loc[raw["countryname"].eq("Colombia")].copy()
    colombia.loc[colombia["series"].eq("Central government deficit"), "series"] = "c_DEFICIT"
    colombia.loc[colombia["series"].eq("Final consumption"), "series"] = "c_cons"
    colombia.loc[colombia["series"].eq("Government revenue"), "series"] = "c_govrev"
    colombia.loc[colombia["series"].eq("Government spending: total"), "series"] = "c_govexp"
    colombia.loc[colombia["series"].eq("Gross domestic capital formation: total") & colombia["start"].eq("1925"), "series"] = "c_inv"
    colombia.loc[
        colombia["series"].eq("Gross domestic capital formation: gross fixed domestic capital formation"),
        "series",
    ] = "c_finv"
    colombia.loc[colombia["series"].eq("Saving: total, series 1"), "series"] = "c_sav"
    colombia = colombia.loc[colombia["series"].fillna("").str.contains("c_", na=False)].copy()
    colombia = _long(colombia[["countryname", "series"] + year_cols].copy(), ["countryname", "series"])
    colombia = _wide(colombia, ["countryname", "year"])
    colombia = _strip_c_prefix(colombia)
    colombia["ISO3"] = "COL"
    colombia = colombia.drop(columns=["countryname"], errors="ignore")
    master = _merge_master(master, colombia)

    chile = raw.loc[raw["countryname"].eq("Chile")].copy()
    chile.loc[chile["series"].eq("Consumer prices"), "series"] = "c_CPI"
    chile.loc[chile["series"].eq("GDP deflator"), "series"] = "c_deflator"
    chile.loc[chile["series"].eq("Government revenue, nominal"), "series"] = "c_govrev"
    chile.loc[chile["series"].eq("Government spending, nominal"), "series"] = "c_govexp"
    chile.loc[chile["series"].eq("Gross domestic product (GDP), real"), "series"] = "c_rGDP_LCU"
    chile.loc[chile["series"].eq("M1 (Estimate A: coins valued at face value)"), "series"] = "c_M1"
    chile.loc[chile["series"].eq("M2 (Estimate A: coins valued at face value)"), "series"] = "c_M2"
    chile.loc[chile["series"].eq("Monetary base (M0) (Estimate A: coins valued at face value)"), "series"] = "c_M0"
    chile.loc[chile["series"].eq("Short-run interest rate"), "series"] = "c_strate"
    chile = chile.loc[chile["series"].fillna("").str.contains("c_", na=False)].copy()
    chile = _long(chile[["countryname", "series"] + year_cols].copy(), ["countryname", "series"])
    chile = _wide(chile, ["countryname", "year"])
    chile = _strip_c_prefix(chile)
    chile["ISO3"] = "CHL"
    chile = chile.drop(columns=["countryname"], errors="ignore")
    for col in ["govrev", "govexp", "rGDP_LCU"]:
        _apply_scale_ops(
            chile,
            mask=chile["ISO3"].eq("CHL"),
            col=col,
            ops=[("mul", _pow10_literal(-6))],
        )
    master = _merge_master(master, chile)

    brazil = raw.loc[raw["countryname"].eq("Brazil")].copy()
    brazil.loc[brazil["series"].eq("M1: total"), "series"] = "c_M1"
    brazil.loc[brazil["series"].eq("M2: total"), "series"] = "c_M2"
    brazil = brazil.loc[brazil["series"].fillna("").str.contains("c_", na=False)].copy()
    brazil = _long(brazil[["countryname", "series"] + year_cols].copy(), ["countryname", "series"])
    brazil = _wide(brazil, ["countryname", "year"])
    brazil = _strip_c_prefix(brazil)
    brazil["ISO3"] = "BRA"
    brazil = brazil.drop(columns=["countryname"], errors="ignore")
    for col in ["M1", "M2"]:
        _apply_scale_ops(
            brazil,
            mask=brazil["ISO3"].eq("BRA"),
            col=col,
            ops=[("mul", _pow10_literal(-15, adjust=1)), ("div", 2750.0), ("div", 2750.0)],
        )
    brazil["year"] = pd.to_numeric(brazil["year"], errors="coerce") + 1
    master = _merge_master(master, brazil)

    cuba = raw.loc[raw["countryname"].eq("Cuba")].copy()
    cuba.loc[cuba["series"].eq("Balance"), "series"] = "c_DEFICIT"
    cuba.loc[cuba["series"].eq("Revenue"), "series"] = "c_govrev"
    cuba.loc[cuba["series"].eq("Spending"), "series"] = "c_govexp"
    cuba = cuba.loc[cuba["series"].fillna("").str.contains("c_", na=False)].copy()
    cuba = _long(cuba[["countryname", "series"] + year_cols].copy(), ["countryname", "series"])
    cuba = _wide(cuba, ["countryname", "year"])
    cuba = _strip_c_prefix(cuba)
    cuba["ISO3"] = "CUB"
    cuba = cuba.drop(columns=["countryname"], errors="ignore")
    for col in ["DEFICIT", "govrev", "govexp"]:
        if col in cuba.columns:
            cuba[col] = pd.to_numeric(cuba[col], errors="coerce") / 1_000_000
    cuba = _drop_rows_with_all_missing(cuba)
    master = _merge_master(master, cuba)

    egypt = raw.loc[raw["countryname"].fillna("").str.contains("Egypt", na=False)].copy()
    egypt.loc[egypt["series"].eq("Exports") & egypt["source"].eq("IFS (April 1948)"), "series"] = "c_exports"
    egypt.loc[egypt["series"].eq("Imports") & egypt["source"].eq("IFS (April 1948)"), "series"] = "c_imports"
    egypt = egypt.loc[egypt["series"].fillna("").str.contains("c_", na=False)].copy()
    egypt = _long(egypt[["countryname", "series"] + year_cols].copy(), ["countryname", "series"])
    egypt = _wide(egypt, ["countryname", "year"])
    egypt = _strip_c_prefix(egypt)
    egypt["ISO3"] = "EGY"
    egypt = egypt.drop(columns=["countryname"], errors="ignore")
    egypt = _drop_rows_with_all_missing(egypt)
    master = _merge_master(master, egypt)

    indonesia = raw.loc[raw["countryname"].eq("Indonesia")].copy()
    indonesia.loc[indonesia["series"].eq("Government revenue, according to Civil Statements"), "series"] = "c_govrev"
    indonesia.loc[indonesia["series"].eq("Government spending, according to Civil Statements"), "series"] = "c_govexp"
    indonesia.loc[indonesia["series"].eq("GDP"), "series"] = "c_rGDP_LCU"
    indonesia = indonesia.loc[indonesia["series"].fillna("").str.contains("c_", na=False)].copy()
    indonesia = _long(indonesia[["countryname", "series"] + year_cols].copy(), ["countryname", "series"])
    indonesia = _wide(indonesia, ["countryname", "year"])
    indonesia = _strip_c_prefix(indonesia)
    indonesia["ISO3"] = "IDN"
    indonesia = indonesia.drop(columns=["countryname"], errors="ignore")
    indonesia = _drop_rows_with_all_missing(indonesia)
    if "rGDP_LCU" in indonesia.columns:
        indonesia["rGDP_LCU"] = pd.to_numeric(indonesia["rGDP_LCU"], errors="coerce") * 1000
    for col in ["govexp", "govrev"]:
        if col in indonesia.columns:
            indonesia[col] = pd.to_numeric(indonesia[col], errors="coerce") / 100_000
    master = _merge_master(master, indonesia)

    spain = raw.loc[raw["countryname"].fillna("").str.contains("Spain", na=False)].copy()
    spain.loc[spain["series"].eq("Monetary base"), "series"] = "c_M0"
    spain.loc[spain["series"].eq("M1"), "series"] = "c_M1"
    spain.loc[spain["series"].eq("M2"), "series"] = "c_M2"
    spain.loc[spain["series"].eq("M3"), "series"] = "c_M3"
    spain.loc[spain["series"].eq("Main official exchange rate, pound sterling") & spain["start"].eq("1821"), "series"] = "c_GBPfx"
    spain = spain.loc[spain["series"].fillna("").str.contains("c_", na=False)].copy()
    spain = _long(spain[["countryname", "series"] + year_cols].copy(), ["countryname", "series"])
    spain = _wide(spain, ["countryname", "year"])
    spain = _strip_c_prefix(spain)
    spain["ISO3"] = "ESP"
    spain = spain.drop(columns=["countryname"], errors="ignore")
    spain = _convert_eur(spain, ["M1", "M2", "M3"])
    spain = _drop_rows_with_all_missing(spain)
    master = _merge_master(master, spain)

    japan = raw.loc[raw["countryname"].fillna("").str.contains("Japan", na=False)].copy()
    japan.loc[japan["series"].eq("National government debt") & japan["start"].eq("1805"), "series"] = "c_cgov_debt1"
    japan.loc[japan["series"].eq("National government debt") & japan["start"].eq("1846"), "series"] = "c_cgov_debt2"
    japan.loc[japan["series"].eq("National government revenue, net"), "series"] = "c_govrev"
    japan.loc[japan["series"].eq("National government spending, net"), "series"] = "c_govexp"
    japan.loc[japan["series"].eq("Unemployment rate"), "series"] = "c_unemp"
    japan = japan.loc[japan["series"].fillna("").str.contains("c_", na=False)].copy()
    japan = _long(japan[["countryname", "series"] + year_cols].copy(), ["countryname", "series"], missing=["..."])
    japan = _wide(japan, ["countryname", "year"])
    japan = _strip_c_prefix(japan)
    japan["ISO3"] = "JPN"
    japan = japan.drop(columns=["countryname"], errors="ignore")
    if {"cgov_debt1", "cgov_debt2"}.issubset(japan.columns):
        japan["cgov_debt1"] = japan["cgov_debt1"].combine_first(japan["cgov_debt2"])
    japan = japan.rename(columns={"cgov_debt1": "cgov_debt"})
    japan = japan.drop(columns=["cgov_debt2"], errors="ignore")
    for col in ["govrev", "govexp"]:
        if col in japan.columns:
            japan[col] = pd.to_numeric(japan[col], errors="coerce") / (10 ** 5)
    japan = _drop_rows_with_all_missing(japan)
    master = _merge_master(master, japan)

    ireland = raw.loc[raw["countryname"].fillna("").str.contains("Ireland", na=False)].copy()
    ireland = ireland.loc[ireland["series"].eq("Nominal GDP: Republic of Ireland"), ["countryname"] + year_cols].copy()
    ireland = _long(ireland, ["countryname"], value_name="nGDP_LCU")
    ireland["ISO3"] = "IRL"
    ireland = ireland.drop(columns=["countryname"], errors="ignore")
    ireland = ireland.loc[ireland["nGDP_LCU"].notna(), ["ISO3", "year", "nGDP_LCU"]].copy()
    master = _merge_master(master, ireland)

    league = raw.loc[raw["source"].fillna("").str.contains("League", na=False, regex=False)].copy()
    league.loc[league["series"].fillna("").str.contains("Retail prices", na=False) & league["unit"].eq("Index, 1929 = 100"), "series"] = "c_CPI_1929"
    league.loc[league["series"].fillna("").str.contains("Retail prices", na=False) & league["unit"].ne("Index, 1929 = 100"), "series"] = "c_CPI_n1929"
    league = league.loc[league["series"].fillna("").str.contains("c_", na=False)].copy()
    counts = league.groupby("countryname")["countryname"].transform("size")
    league = league.loc[counts.eq(1), ["countryname", "series"] + year_cols].copy()
    league = _long(league, ["countryname", "series"])
    league = _wide(league, ["countryname", "year"])
    league["countryname"] = league["countryname"].replace({"Myanmar (Burma)": "Myanmar"})
    league = _map_iso(league)
    league = league.drop(columns=["countryname"], errors="ignore")
    league = _strip_c_prefix(league)
    league = league.rename(columns={"CPI_1929": "CPI"})
    league = league.drop(columns=["CPI_n1929"], errors="ignore")
    league = _drop_rows_with_all_missing(league)
    master = _merge_master(master, league)

    if master is None:
        master = pd.DataFrame(columns=["ISO3", "year"])
    for iso3, ops in [
        ("FIN", [("mul", 10.0)]),
        ("FRA", [("div", 100.0)]),
        ("POL", [("div", 1000.0)]),
        ("GNB", [("mul", 650.0)]),
        ("NIC", [("div", 500_000_000.0)]),
        ("ZAF", [("mul", 2.0)]),
    ]:
        mask = master["ISO3"].eq(iso3)
        for col in ["imports", "exports"]:
            _apply_scale_ops(master, mask=mask, col=col, ops=ops)
    for iso3 in ["POL", "SRB"]:
        mask = master["ISO3"].eq(iso3)
        for col in ["imports", "exports"]:
            if col in master.columns:
                master.loc[mask, col] = np.nan

    master = _sort_keys(master)
    if "CPI" in master.columns:
        lag = _lag_if_consecutive_year(master, "CPI")
        master["infl"] = np.where(lag.notna(), ((pd.to_numeric(master["CPI"], errors="coerce") - lag) / lag) * 100, np.nan)
    else:
        master["infl"] = np.nan

    master = master.rename(columns={col: f"HFS_{col}" for col in master.columns if col not in {"ISO3", "year"}})
    master = _drop_rows_with_all_missing(master)
    master = master.rename(columns={col: str(col).replace("_LCU", "") for col in master.columns if col.endswith("_LCU")})
    master = master.rename(columns={"HFS_cgov_debt": "HFS_govdebt", "HFS_DEFICIT": "HFS_govdef"})
    master = master.drop(columns=["HFS_GBPfx"], errors="ignore")
    master = master.reindex(columns=HFS_COLUMN_ORDER)
    master["ISO3"] = master["ISO3"].astype(str)
    master = _coerce_numeric_dtypes(master, HFS_DTYPE_MAP)
    master = _sort_keys(master)

    out_path = clean_dir / "aggregators" / "HFS" / "HFS.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_hfs"]
