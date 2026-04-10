from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_prt_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    path = raw_dir / "country_level" / "PRT_1.xlsx"

    def _clean_numeric(series: pd.Series, replacements: list[tuple[str, str]] | None = None) -> pd.Series:
        out = series.astype("string")
        if replacements is not None:
            for old, new in replacements:
                out = out.str.replace(old, new, regex=False)
        return pd.to_numeric(out, errors="coerce")

    def _extract_last_year(series: pd.Series) -> pd.Series:
        values = []
        for value in series.astype("string"):
            matches = re.findall(r"(\d{4})", "" if pd.isna(value) else str(value))
            values.append(matches[-1] if matches else None)
        return pd.to_numeric(pd.Series(values, index=series.index), errors="coerce")

    pop = pd.read_excel(path, sheet_name="pop", dtype=str)
    pop["year"] = pd.to_numeric(pop["year"], errors="coerce")
    pop["pop"] = _clean_numeric(pop["pop"], [(" ", "")]) / 1_000_000
    master = pop[["year", "pop"]].copy()

    gdp = pd.read_excel(path, sheet_name="GDP", dtype=str)
    gdp["year"] = pd.to_numeric(gdp["year"], errors="coerce")
    gdp["nGDP"] = _clean_numeric(gdp["nGDP"], [(" ", ""), ("?", "")])
    master = gdp[["year", "nGDP"]].merge(master, on="year", how="outer")

    money = pd.read_excel(path, sheet_name="money_supply", dtype=str)
    money["year"] = pd.to_numeric(money["year"], errors="coerce")
    for col in ["M0", "M1", "M2"]:
        money[col] = _clean_numeric(money[col], [(" ", "")])
    master = money[["year", "M0", "M1", "M2"]].merge(master, on="year", how="outer")

    cbrate = pd.read_excel(path, sheet_name="cbrate", dtype=str)
    cbrate["year"] = pd.to_numeric(pd.to_datetime(cbrate["date"], errors="coerce").dt.year, errors="coerce").astype("float64")
    missing_year = cbrate["year"].isna()
    cbrate.loc[missing_year, "year"] = _extract_last_year(cbrate.loc[missing_year, "date"])
    cbrate["cbrate"] = pd.to_numeric(cbrate["cbrate"], errors="coerce")
    cbrate["year"] = pd.to_numeric(cbrate["year"], errors="coerce")
    cbrate = cbrate.loc[cbrate["year"].notna(), ["year", "cbrate"]].copy()
    cbrate = cbrate.sort_values("year", kind="mergesort").groupby("year", sort=False).tail(1).copy()
    master = cbrate.merge(master, on="year", how="outer")

    cpi = pd.read_excel(path, sheet_name="CPI", dtype=str)
    cpi["year"] = pd.to_numeric(cpi["year"], errors="coerce")
    cpi["CPI"] = _clean_numeric(cpi["CPI"], [(" ", ""), ("?", "")])
    master = cpi[["year", "CPI"]].merge(master, on="year", how="outer")

    govdebt = pd.read_excel(path, sheet_name="govdebt", dtype=str)
    govdebt["year"] = pd.to_numeric(govdebt["year"], errors="coerce")
    govdebt["govdebt"] = _clean_numeric(govdebt["govdebt"], [(" ", "")])
    master = govdebt[["year", "govdebt"]].merge(master, on="year", how="outer")

    govtax = pd.read_excel(path, sheet_name="govtax", dtype=str)
    govtax["year"] = _extract_last_year(govtax["date"])
    govtax["govtax"] = _clean_numeric(govtax["govtax"], [(" ", "")])
    govtax = govtax.loc[govtax["year"].notna(), ["year", "govtax"]].copy()
    master = govtax.merge(master, on="year", how="outer")

    trade = pd.read_excel(path, sheet_name="trade", dtype=str)
    trade["year"] = _clean_numeric(trade["year"].astype("string").str.replace(r"[^0-9]", "", regex=True))
    trade["exports"] = _clean_numeric(trade["exports"], [(" ", ""), ("-", "")])
    trade["imports"] = _clean_numeric(trade["imports"], [(" ", ""), ("-", "")])
    trade = trade.loc[trade["year"].notna(), ["year", "exports", "imports"]].copy()
    master = trade.merge(master, on="year", how="outer")

    usdfx = pd.read_excel(path, sheet_name="USDfx", dtype=str)
    usdfx["year"] = pd.to_numeric(usdfx["year"], errors="coerce")
    usdfx["USDfx"] = pd.to_numeric(usdfx["USDfx"], errors="coerce")
    master = usdfx[["year", "USDfx"]].merge(master, on="year", how="outer")

    master["ISO3"] = "PRT"
    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].copy()
    master = master.merge(eur_fx, on="ISO3", how="left")
    fx_mask = master["EUR_irrevocable_FX"].notna()
    for col in ["USDfx", "exports", "imports", "govtax", "govdebt", "M0", "M1", "M2", "nGDP"]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
        master.loc[fx_mask, col] = master.loc[fx_mask, col] / pd.to_numeric(
            master.loc[fx_mask, "EUR_irrevocable_FX"], errors="coerce"
        )
    master = master.drop(columns=["EUR_irrevocable_FX"], errors="ignore")

    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    prev_cpi = pd.to_numeric(master["CPI"], errors="coerce").groupby(master["ISO3"]).shift(1)
    prev_year = pd.to_numeric(master["year"], errors="coerce").groupby(master["ISO3"]).shift(1)
    year_num = pd.to_numeric(master["year"], errors="coerce")
    infl = (pd.to_numeric(master["CPI"], errors="coerce") - prev_cpi) / prev_cpi * 100
    master["infl"] = infl.where(prev_cpi.notna() & year_num.eq(prev_year + 1))

    n_gdp = pd.to_numeric(master["nGDP"], errors="coerce")
    master["imports_GDP"] = pd.to_numeric(master["imports"], errors="coerce") / n_gdp * 100
    master["exports_GDP"] = pd.to_numeric(master["exports"], errors="coerce") / n_gdp * 100
    master["govdebt_GDP"] = pd.to_numeric(master["govdebt"], errors="coerce") / n_gdp * 100
    master["govtax_GDP"] = pd.to_numeric(master["govtax"], errors="coerce") / n_gdp * 100

    master = master.rename(columns={col: f"CS1_{col}" for col in master.columns if col not in {"ISO3", "year"}})
    for col in [
        "CS1_USDfx",
        "CS1_exports",
        "CS1_imports",
        "CS1_govtax",
        "CS1_govdebt",
        "CS1_CPI",
        "CS1_cbrate",
        "CS1_M0",
        "CS1_M1",
        "CS1_M2",
        "CS1_nGDP",
        "CS1_pop",
    ]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
    for col in ["CS1_infl", "CS1_imports_GDP", "CS1_exports_GDP", "CS1_govdebt_GDP", "CS1_govtax_GDP"]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float32")

    master = master[
        [
            "ISO3",
            "year",
            "CS1_USDfx",
            "CS1_exports",
            "CS1_imports",
            "CS1_govtax",
            "CS1_govdebt",
            "CS1_CPI",
            "CS1_cbrate",
            "CS1_M0",
            "CS1_M1",
            "CS1_M2",
            "CS1_nGDP",
            "CS1_pop",
            "CS1_infl",
            "CS1_imports_GDP",
            "CS1_exports_GDP",
            "CS1_govdebt_GDP",
            "CS1_govtax_GDP",
        ]
    ].copy()
    master = _sort_keys(master)
    out_path = clean_dir / "country_level" / "PRT_1.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_prt_1"]
