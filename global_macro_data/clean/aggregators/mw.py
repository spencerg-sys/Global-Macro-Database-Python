from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_mw(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    path = raw_dir / "aggregators" / "MW" / "Measuring_worth.xlsx"

    def _load_sheet(sheet_name: str) -> pd.DataFrame:
        return _read_excel_compat(path, sheet_name=sheet_name).copy()

    def _merge_master(master: pd.DataFrame | None, part: pd.DataFrame) -> pd.DataFrame:
        if master is None:
            return part.copy()
        keys = ["ISO3", "year"]
        master_idx = master.set_index(keys)
        part_idx = part.set_index(keys)
        union_index = master_idx.index.union(part_idx.index)
        master_idx = master_idx.reindex(union_index)
        part_idx = part_idx.reindex(union_index)
        combined = master_idx.copy()
        for col in part_idx.columns:
            if col in combined.columns:
                combined[col] = combined[col].combine_first(part_idx[col])
            else:
                combined[col] = part_idx[col]
        return combined.reset_index()

    us_gdp = _load_sheet("US_GDP")
    us_gdp["pop"] = pd.to_numeric(us_gdp["pop"], errors="coerce") / 1000
    us_gdp["nGDP"] = pd.to_numeric(us_gdp["nGDP"], errors="coerce") * 1000
    us_gdp["rGDP"] = pd.to_numeric(us_gdp["rGDP"], errors="coerce") * 1000
    master: pd.DataFrame | None = us_gdp.copy()

    master = _merge_master(master, _load_sheet("US_prices"))

    uk_gdp = _load_sheet("UK_GDP")
    uk_gdp["pop"] = pd.to_numeric(uk_gdp["pop"], errors="coerce") / 1000
    master = _merge_master(master, uk_gdp)

    master = _merge_master(master, _load_sheet("UK_prices"))
    master = _merge_master(master, _load_sheet("Rates"))

    esp = _load_sheet("ESP")
    esp["pop"] = pd.to_numeric(esp["pop"], errors="coerce") / 1000
    master = _merge_master(master, esp)

    aus = _load_sheet("AUS")
    aus["pop"] = pd.to_numeric(aus["pop"], errors="coerce") / 1000000
    master = _merge_master(master, aus)

    exchange = _load_sheet("exchange_rates")
    exchange["USDfx"] = exchange["USDfx"].astype("string").str.extract(r"(^[0-9.]+)", expand=False)
    exchange["USDfx"] = pd.to_numeric(exchange["USDfx"], errors="coerce")
    exchange = exchange.sort_values(["ISO3", "year"]).reset_index(drop=True)
    exchange["_dup_has_zero"] = exchange.groupby(["ISO3", "year"])["USDfx"].transform(lambda s: s.eq(0).any())
    zero_mask = exchange["_dup_has_zero"] & exchange["USDfx"].eq(0)
    zero_groups = exchange.loc[zero_mask, ["ISO3", "year"]].drop_duplicates()
    exchange = exchange.drop_duplicates(["ISO3", "year"], keep="last")
    if not zero_groups.empty:
        exchange = exchange.merge(zero_groups.assign(_force_zero=True), on=["ISO3", "year"], how="left")
        exchange = exchange.sort_values(["ISO3", "year"]).reset_index(drop=True)
        exchange.loc[exchange["_force_zero"].eq(True), "USDfx"] = 0.0
        exchange = exchange.drop(columns=["_force_zero"], errors="ignore")
    exchange = exchange.drop(columns=["_dup_has_zero"], errors="ignore").reset_index(drop=True)
    master = _merge_master(master, exchange)

    assert master is not None
    master = _sort_keys(master)
    prev_cpi = _lag_if_consecutive_year(master, "CPI")
    master["infl"] = (
        (pd.to_numeric(master["CPI"], errors="coerce") - pd.to_numeric(prev_cpi, errors="coerce"))
        / pd.to_numeric(prev_cpi, errors="coerce")
        * 100
    )
    master.loc[prev_cpi.isna(), "infl"] = np.nan
    master.loc[pd.to_numeric(prev_cpi, errors="coerce").eq(0), "infl"] = np.nan

    rename_map = {col: f"MW_{col}" for col in master.columns if col not in {"ISO3", "year"}}
    master = master.rename(columns=rename_map)

    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].drop_duplicates().copy()
    master = master.merge(eur_fx, on="ISO3", how="left", indicator=True)
    merge_mask = master["_merge"].eq("both")
    master.loc[merge_mask, "MW_USDfx"] = pd.to_numeric(master.loc[merge_mask, "MW_USDfx"], errors="coerce") / pd.to_numeric(
        master.loc[merge_mask, "EUR_irrevocable_FX"], errors="coerce"
    )
    master = master.drop(columns=["EUR_irrevocable_FX", "_merge"], errors="ignore")

    master.loc[pd.to_numeric(master["MW_USDfx"], errors="coerce").eq(0), "MW_USDfx"] = np.nan

    arg_mask = master["ISO3"].eq("ARG")
    for cutoff, scale in [(1991, 10000), (1984, 1000), (1982, 10000), (1969, 100)]:
        mask = arg_mask & pd.to_numeric(master["year"], errors="coerce").le(cutoff)
        master.loc[mask, "MW_USDfx"] = pd.to_numeric(master.loc[mask, "MW_USDfx"], errors="coerce") / scale

    master.loc[master["ISO3"].eq("AUS") & pd.to_numeric(master["year"], errors="coerce").le(1965), "MW_USDfx"] = pd.to_numeric(
        master.loc[master["ISO3"].eq("AUS") & pd.to_numeric(master["year"], errors="coerce").le(1965), "MW_USDfx"], errors="coerce"
    ) * 2

    chl_mask = master["ISO3"].eq("CHL")
    for cutoff in [1975, 1959]:
        mask = chl_mask & pd.to_numeric(master["year"], errors="coerce").le(cutoff)
        master.loc[mask, "MW_USDfx"] = pd.to_numeric(master.loc[mask, "MW_USDfx"], errors="coerce") / 1000

    master.loc[master["ISO3"].eq("MEX") & pd.to_numeric(master["year"], errors="coerce").le(1992), "MW_USDfx"] = pd.to_numeric(
        master.loc[master["ISO3"].eq("MEX") & pd.to_numeric(master["year"], errors="coerce").le(1992), "MW_USDfx"], errors="coerce"
    ) / 1000
    master.loc[master["ISO3"].eq("NZL") & pd.to_numeric(master["year"], errors="coerce").le(1967), "MW_USDfx"] = pd.to_numeric(
        master.loc[master["ISO3"].eq("NZL") & pd.to_numeric(master["year"], errors="coerce").le(1967), "MW_USDfx"], errors="coerce"
    ) * 2

    per_mask = master["ISO3"].eq("PER")
    master.loc[per_mask & pd.to_numeric(master["year"], errors="coerce").le(1990), "MW_USDfx"] = pd.to_numeric(
        master.loc[per_mask & pd.to_numeric(master["year"], errors="coerce").le(1990), "MW_USDfx"], errors="coerce"
    ) / 1000000
    master.loc[per_mask & pd.to_numeric(master["year"], errors="coerce").le(1984), "MW_USDfx"] = pd.to_numeric(
        master.loc[per_mask & pd.to_numeric(master["year"], errors="coerce").le(1984), "MW_USDfx"], errors="coerce"
    ) / 1000

    master.loc[master["ISO3"].eq("ZAF") & pd.to_numeric(master["year"], errors="coerce").le(1960), "MW_USDfx"] = pd.to_numeric(
        master.loc[master["ISO3"].eq("ZAF") & pd.to_numeric(master["year"], errors="coerce").le(1960), "MW_USDfx"], errors="coerce"
    ) * 2

    bra_mask = master["ISO3"].eq("BRA")
    for cutoff, scale in [(1994, 2750), (1993, 1000), (1988, 1000), (1985, 1000), (1966, 1000)]:
        mask = bra_mask & pd.to_numeric(master["year"], errors="coerce").le(cutoff)
        master.loc[mask, "MW_USDfx"] = pd.to_numeric(master.loc[mask, "MW_USDfx"], errors="coerce") / scale

    master.loc[master["ISO3"].eq("VEN"), "MW_USDfx"] = pd.to_numeric(master.loc[master["ISO3"].eq("VEN"), "MW_USDfx"], errors="coerce") / 1000
    master.loc[master["ISO3"].eq("ISR") & pd.to_numeric(master["year"], errors="coerce").le(1985), "MW_USDfx"] = pd.to_numeric(
        master.loc[master["ISO3"].eq("ISR") & pd.to_numeric(master["year"], errors="coerce").le(1985), "MW_USDfx"], errors="coerce"
    ) / 1000
    master.loc[master["ISO3"].eq("FRA") & pd.to_numeric(master["year"], errors="coerce").le(1959), "MW_USDfx"] = pd.to_numeric(
        master.loc[master["ISO3"].eq("FRA") & pd.to_numeric(master["year"], errors="coerce").le(1959), "MW_USDfx"], errors="coerce"
    ) / 100
    master.loc[master["ISO3"].eq("FIN") & pd.to_numeric(master["year"], errors="coerce").le(1941), "MW_USDfx"] = pd.to_numeric(
        master.loc[master["ISO3"].eq("FIN") & pd.to_numeric(master["year"], errors="coerce").le(1941), "MW_USDfx"], errors="coerce"
    ) / 100
    master.loc[master["ISO3"].eq("BEL") & pd.to_numeric(master["year"], errors="coerce").between(1927, 1940), "MW_USDfx"] = pd.to_numeric(
        master.loc[master["ISO3"].eq("BEL") & pd.to_numeric(master["year"], errors="coerce").between(1927, 1940), "MW_USDfx"], errors="coerce"
    ) / 5
    master.loc[master["ISO3"].eq("DEU") & pd.to_numeric(master["year"], errors="coerce").le(1924), "MW_USDfx"] = np.nan

    master = master.drop(columns=["MW_nGDP_GBP", "MW_nGDP_pc"], errors="ignore")
    expected = ["ISO3", "year", "MW_USDfx", "MW_nGDP", "MW_rGDP", "MW_deflator", "MW_rGDP_pc", "MW_pop", "MW_CPI", "MW_strate", "MW_ltrate", "MW_infl"]
    for col in expected:
        if col not in master.columns:
            master[col] = np.nan
    master = master[expected].copy()

    master["ISO3"] = master["ISO3"].astype("object")
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    for col in ["MW_USDfx", "MW_nGDP", "MW_rGDP", "MW_deflator", "MW_rGDP_pc", "MW_pop", "MW_CPI", "MW_strate", "MW_ltrate"]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
    master["MW_infl"] = pd.to_numeric(master["MW_infl"], errors="coerce").astype("float32")
    master = _sort_keys(master)
    out_path = clean_dir / "aggregators" / "MW" / "MW.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_mw"]
