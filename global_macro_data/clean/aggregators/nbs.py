from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_nbs(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    path = raw_dir / "aggregators" / "NBS" / "NBS.xlsx"
    sheet_cache: dict[str, pd.DataFrame] = {}

    def _sheet(sheet_name: str) -> pd.DataFrame:
        if sheet_name not in sheet_cache:
            sheet_cache[sheet_name] = _read_excel_compat(path, sheet_name=sheet_name, header=None, dtype=str)
        return sheet_cache[sheet_name].copy()

    def _select(sheet_name: str, columns: dict[str, str]) -> pd.DataFrame:
        frame = _sheet(sheet_name)
        idx = [_excel_column_to_index(col) for col in columns]
        out = frame.iloc[:, idx].copy()
        out.columns = [columns[col] for col in columns]
        return out

    def _replace_missing_strings(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in out.columns:
            out[col] = out[col].replace({"..": pd.NA, ".": pd.NA, "": pd.NA})
        return out

    def _drop_all_missing(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[frame.notna().any(axis=1)].reset_index(drop=True)

    def _to_numeric(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in out.columns:
            out[col] = _excel_numeric_series(out[col], mode="float")
        return out

    def _drop_one_based(frame: pd.DataFrame, row_number: int) -> pd.DataFrame:
        if 1 <= row_number <= len(frame):
            frame = frame.drop(frame.index[row_number - 1]).reset_index(drop=True)
        return frame

    def _year_end(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.groupby("year", sort=False).tail(1).reset_index(drop=True)

    def _finalize(frame: pd.DataFrame, iso3: str) -> pd.DataFrame:
        out = frame.copy()
        out["ISO3"] = iso3
        out["year"] = pd.to_numeric(out["year"], errors="coerce")
        out = out.loc[out["year"].notna()].copy()
        return out

    def _merge_master(master: pd.DataFrame | None, part: pd.DataFrame) -> pd.DataFrame:
        part = part.copy()
        part["year"] = pd.to_numeric(part["year"], errors="coerce")
        if master is None:
            return part
        master = master.copy()
        master["year"] = pd.to_numeric(master["year"], errors="coerce")
        master_isos = set(master["ISO3"].dropna().astype(str))
        part_isos = set(part["ISO3"].dropna().astype(str))
        if master_isos.isdisjoint(part_isos):
            return pd.concat([master, part], ignore_index=True, sort=False)
        keys = ["ISO3", "year"]
        merged = part.merge(master, on=keys, how="outer", suffixes=("", "_using"), indicator=True, sort=False)
        overlap = [col for col in part.columns if col not in keys and col in master.columns]
        right_only = merged["_merge"].eq("right_only")
        for col in overlap:
            using_col = f"{col}_using"
            if using_col in merged.columns:
                merged.loc[right_only, col] = merged.loc[right_only, using_col]
                merged = merged.drop(columns=[using_col], errors="ignore")
        merged = merged.drop(columns=["_merge"], errors="ignore")
        return merged

    master: pd.DataFrame | None = None

    gr_national = _select(
        "GR data tables A",
        {"DK": "year", "DL": "nGDP", "DM": "rGDP", "DN": "deflator", "DO": "rGDP_pc", "DP": "imports", "DQ": "exports", "DR": "pop"},
    )
    gr_national = gr_national.iloc[4:].reset_index(drop=True)
    gr_national = _drop_all_missing(gr_national)
    gr_national = _replace_missing_strings(gr_national)
    gr_national = _to_numeric(gr_national)
    for col in ["nGDP", "rGDP", "imports", "exports"]:
        gr_national[col] = pd.to_numeric(gr_national[col], errors="coerce") / 1000
    master = _merge_master(master, _finalize(gr_national, "GRC"))

    gr_money = _select("GR data tables A", {"L": "year", "M": "M3", "N": "M0", "BE": "cbrate", "BF": "cbrate_bis", "BG": "strate"})
    gr_money = gr_money.iloc[4:].reset_index(drop=True)
    gr_money = gr_money.iloc[:98].reset_index(drop=True)
    gr_money = gr_money.where(gr_money.ne(""), pd.NA)
    gr_money["cbrate"] = gr_money["cbrate"].where(gr_money["cbrate"].notna(), gr_money["cbrate_bis"])
    gr_money = gr_money.drop(columns=["cbrate_bis"], errors="ignore")
    gr_money["cbrate"] = gr_money["cbrate"].replace({"9*": "9"})
    gr_money = _replace_missing_strings(gr_money)
    gr_money = _to_numeric(gr_money)
    for col in ["M3", "M0"]:
        gr_money[col] = pd.to_numeric(gr_money[col], errors="coerce") / 1000
    master = _merge_master(master, _finalize(gr_money, "GRC"))

    gr_fx = _select("GR data tables A", {"CG": "year", "CJ": "USDfx", "CH": "GBPfx", "CI": "FRFfx"})
    gr_fx = gr_fx.iloc[42:].reset_index(drop=True)
    gr_fx = gr_fx.iloc[:27].reset_index(drop=True)
    gr_fx = _replace_missing_strings(gr_fx)
    gr_fx = _to_numeric(gr_fx)
    master = _merge_master(master, _finalize(gr_fx, "GRC"))

    gr_fiscal = _select("GR data tables A", {"CM": "year", "CN": "govrev", "CO": "govtax", "CR": "govexp"})
    gr_fiscal = gr_fiscal.iloc[4:].reset_index(drop=True)
    gr_fiscal = gr_fiscal.loc[~gr_fiscal["year"].astype("string").str.contains("Note", na=False)].reset_index(drop=True)
    gr_fiscal = _drop_all_missing(gr_fiscal)
    gr_fiscal = _replace_missing_strings(gr_fiscal)
    gr_fiscal = _to_numeric(gr_fiscal)
    for col in ["govrev", "govtax", "govexp"]:
        gr_fiscal[col] = pd.to_numeric(gr_fiscal[col], errors="coerce") / 1000
    master = _merge_master(master, _finalize(gr_fiscal, "GRC"))

    gr_cpi = _select("GR data tables A", {"CX": "year", "CY": "CPI"})
    gr_cpi = gr_cpi.iloc[4:].reset_index(drop=True)
    gr_cpi = _drop_all_missing(gr_cpi)
    gr_cpi = _replace_missing_strings(gr_cpi)
    gr_cpi = _to_numeric(gr_cpi)
    master = _merge_master(master, _finalize(gr_cpi, "GRC"))

    ro_main = _select(
        "RO data tables A",
        {"BG": "year", "BH": "nGDP", "BI": "imports", "BK": "exports", "BM": "pop", "AQ": "govrev", "AR": "govtax", "AS": "govtax_bis", "AT": "govrev_bis", "AU": "govexp", "AV": "govexp_bis", "AX": "govdebt", "AK": "USDfx", "AF": "FRFfx", "AH": "GBPfx"},
    )
    ro_main = ro_main.iloc[6:].reset_index(drop=True)
    ro_main = _drop_all_missing(ro_main)
    ro_main = _replace_missing_strings(ro_main)
    ro_main = _to_numeric(ro_main)
    ro_main["govexp"] = pd.to_numeric(ro_main["govexp"], errors="coerce").where(
        pd.to_numeric(ro_main["govexp_bis"], errors="coerce").isna(),
        pd.to_numeric(ro_main["govexp"], errors="coerce") + pd.to_numeric(ro_main["govexp_bis"], errors="coerce"),
    )
    ro_main["govrev"] = pd.to_numeric(ro_main["govrev"], errors="coerce").where(
        pd.to_numeric(ro_main["govrev_bis"], errors="coerce").isna(),
        pd.to_numeric(ro_main["govrev"], errors="coerce") + pd.to_numeric(ro_main["govrev_bis"], errors="coerce"),
    )
    ro_main["govtax"] = pd.to_numeric(ro_main["govtax"], errors="coerce").where(
        pd.to_numeric(ro_main["govtax_bis"], errors="coerce").isna(),
        pd.to_numeric(ro_main["govtax"], errors="coerce") + pd.to_numeric(ro_main["govtax_bis"], errors="coerce"),
    )
    ro_main = ro_main.drop(columns=["govexp_bis", "govrev_bis", "govtax_bis"], errors="ignore")
    for col in ["imports", "exports", "govrev", "govtax", "govdebt", "govexp"]:
        ro_main[col] = pd.to_numeric(ro_main[col], errors="coerce") / 1000
    ro_main["govrev_GDP"] = pd.to_numeric(ro_main["govrev"], errors="coerce") / pd.to_numeric(ro_main["nGDP"], errors="coerce") * 100
    ro_main["govtax_GDP"] = pd.to_numeric(ro_main["govtax"], errors="coerce") / pd.to_numeric(ro_main["nGDP"], errors="coerce") * 100
    ro_main["govdebt_GDP"] = pd.to_numeric(ro_main["govdebt"], errors="coerce") / pd.to_numeric(ro_main["nGDP"], errors="coerce") * 100
    ro_main = ro_main.drop(columns=["govdebt", "govrev", "govtax"], errors="ignore")
    master = _merge_master(master, _finalize(ro_main, "ROU"))

    ro_money = _select("RO data tables A", {"L": "year", "Q": "M0", "R": "M3"})
    ro_money = ro_money.iloc[6:].reset_index(drop=True)
    ro_money = _drop_all_missing(ro_money)
    ro_money = _replace_missing_strings(ro_money)
    ro_money = _to_numeric(ro_money)
    ro_money = ro_money.loc[ro_money[["M0", "M3"]].notna().any(axis=1)].reset_index(drop=True)
    for col in ["M0", "M3"]:
        ro_money[col] = pd.to_numeric(ro_money[col], errors="coerce") / 1000
    master = _merge_master(master, _finalize(ro_money, "ROU"))

    ro_cpi = _select("RO data tables A", {"BB": "year", "BC": "CPI"})
    ro_cpi = ro_cpi.iloc[6:].reset_index(drop=True)
    ro_cpi = _drop_all_missing(ro_cpi)
    ro_cpi = _replace_missing_strings(ro_cpi)
    ro_cpi = _to_numeric(ro_cpi)
    master = _merge_master(master, _finalize(ro_cpi, "ROU"))

    ro_cbrate = _select("RO data tables A", {"T": "year", "U": "month", "W": "cbrate"})
    ro_cbrate = ro_cbrate.iloc[6:].reset_index(drop=True)
    ro_cbrate = _drop_all_missing(ro_cbrate)
    ro_cbrate = _replace_missing_strings(ro_cbrate)
    ro_cbrate = _to_numeric(ro_cbrate)
    ro_cbrate = _year_end(ro_cbrate)
    ro_cbrate = ro_cbrate.drop(columns=["month"], errors="ignore")
    master = _merge_master(master, _finalize(ro_cbrate, "ROU"))

    ro_strate = _select("RO data tables A", {"Y": "year", "Z": "month", "AB": "strate"})
    ro_strate = ro_strate.iloc[6:].reset_index(drop=True)
    ro_strate = _drop_all_missing(ro_strate)
    ro_strate = _replace_missing_strings(ro_strate)
    ro_strate = _to_numeric(ro_strate)
    ro_strate = _year_end(ro_strate)
    ro_strate = ro_strate.drop(columns=["month"], errors="ignore")
    ro_strate = _drop_one_based(ro_strate, 21)
    ro_strate = _drop_one_based(ro_strate, 6)
    master = _merge_master(master, _finalize(ro_strate, "ROU"))

    al_trade = _select("AL data tables A", {"AX": "year", "AY": "exports", "AZ": "imports", "BA": "pop"})
    al_trade = al_trade.iloc[4:].reset_index(drop=True)
    al_trade = _drop_all_missing(al_trade)
    al_trade = _replace_missing_strings(al_trade)
    al_trade = _to_numeric(al_trade)
    for col in ["imports", "exports"]:
        al_trade[col] = pd.to_numeric(al_trade[col], errors="coerce") / 1000
    master = _merge_master(master, _finalize(al_trade, "ALB"))

    al_fiscal = _select("AL data tables A", {"AJ": "year", "AK": "govrev", "AL": "govexp", "AM": "govtax"})
    al_fiscal = al_fiscal.iloc[4:].reset_index(drop=True)
    al_fiscal = _drop_all_missing(al_fiscal)
    al_fiscal["year"] = al_fiscal["year"].astype("string").str[-4:]
    al_fiscal.loc[al_fiscal["year"].eq("1922"), "year"] = "1923"
    al_fiscal.loc[al_fiscal["year"].eq("1921"), "year"] = "1922"
    al_fiscal = _replace_missing_strings(al_fiscal)
    al_fiscal = _to_numeric(al_fiscal)
    for col in ["govrev", "govtax", "govexp"]:
        al_fiscal[col] = pd.to_numeric(al_fiscal[col], errors="coerce") / 1000
    master = _merge_master(master, _finalize(al_fiscal, "ALB"))

    al_money = _select("AL data tables A", {"L": "year", "T": "M0", "U": "M3"})
    al_money = al_money.iloc[4:].reset_index(drop=True)
    al_money = _drop_all_missing(al_money)
    al_money = _drop_one_based(al_money, 13)
    al_money = _replace_missing_strings(al_money)
    al_money = _to_numeric(al_money)
    for col in ["M0", "M3"]:
        al_money[col] = pd.to_numeric(al_money[col], errors="coerce") / 1000
    master = _merge_master(master, _finalize(al_money, "ALB"))

    al_cbrate = _select("AL data tables A", {"W": "year", "Y": "cbrate"})
    al_cbrate = al_cbrate.iloc[4:].reset_index(drop=True)
    al_cbrate = _drop_all_missing(al_cbrate)
    al_cbrate = _replace_missing_strings(al_cbrate)
    al_cbrate = _to_numeric(al_cbrate)
    master = _merge_master(master, _finalize(al_cbrate, "ALB"))

    bg_main = _select(
        "BG data tables A",
        {"BI": "year", "BJ": "nGDP", "BK": "rGDP", "BM": "imports", "BL": "exports", "BN": "pop", "AW": "govrev", "AX": "govexp", "AZ": "govdebt", "BA": "govdebt_d"},
    )
    bg_main = bg_main.iloc[5:].reset_index(drop=True)
    bg_main = _drop_all_missing(bg_main)
    bg_main = _replace_missing_strings(bg_main)
    bg_main = _to_numeric(bg_main)
    bg_main["govdebt"] = pd.to_numeric(bg_main["govdebt"], errors="coerce").where(
        pd.to_numeric(bg_main["govdebt_d"], errors="coerce").isna(),
        pd.to_numeric(bg_main["govdebt"], errors="coerce") + pd.to_numeric(bg_main["govdebt_d"], errors="coerce"),
    )
    for col in ["nGDP", "rGDP", "imports", "exports", "govrev", "govdebt", "govexp"]:
        bg_main[col] = pd.to_numeric(bg_main[col], errors="coerce") / 1000
    bg_main["govdebt_GDP"] = pd.to_numeric(bg_main["govdebt"], errors="coerce") / pd.to_numeric(bg_main["nGDP"], errors="coerce") * 100
    bg_main["govrev_GDP"] = pd.to_numeric(bg_main["govrev"], errors="coerce") / pd.to_numeric(bg_main["nGDP"], errors="coerce") * 100
    bg_main = bg_main.drop(columns=["govrev", "govdebt", "govdebt_d"], errors="ignore")
    master = _merge_master(master, _finalize(bg_main, "BGR"))

    bg_fx = _select("BG data tables A", {"AO": "year", "AS": "USDfx", "AP": "GBPfx", "AQ": "FRFfx"})
    bg_fx = bg_fx.iloc[5:].reset_index(drop=True)
    bg_fx = _drop_all_missing(bg_fx)
    bg_fx = _replace_missing_strings(bg_fx)
    bg_fx = _to_numeric(bg_fx)
    master = _merge_master(master, _finalize(bg_fx, "BGR"))

    bg_cpi = _select("BG data tables A", {"BC": "year", "BD": "CPI", "BE": "wholesale_price_index", "BF": "retail_price_index", "BG": "gen_market_price_index"})
    bg_cpi = bg_cpi.iloc[5:].reset_index(drop=True)
    bg_cpi = _drop_all_missing(bg_cpi)
    bg_cpi = _replace_missing_strings(bg_cpi)
    bg_cpi = _to_numeric(bg_cpi)
    bg_cpi["CPI"] = pd.to_numeric(bg_cpi["CPI"], errors="coerce").where(
        pd.to_numeric(bg_cpi["CPI"], errors="coerce").notna(),
        pd.to_numeric(bg_cpi["retail_price_index"], errors="coerce"),
    )
    bg_cpi["CPI"] = pd.to_numeric(bg_cpi["CPI"], errors="coerce").where(
        pd.to_numeric(bg_cpi["CPI"], errors="coerce").notna(),
        pd.to_numeric(bg_cpi["gen_market_price_index"], errors="coerce"),
    )
    bg_cpi["CPI"] = pd.to_numeric(bg_cpi["CPI"], errors="coerce").where(
        pd.to_numeric(bg_cpi["CPI"], errors="coerce").notna(),
        pd.to_numeric(bg_cpi["wholesale_price_index"], errors="coerce"),
    )
    bg_cpi = bg_cpi[["year", "CPI"]].copy()
    master = _merge_master(master, _finalize(bg_cpi, "BGR"))

    bg_cbrate = _select("BG data tables A", {"X": "year", "Z": "month", "AA": "cbrate"})
    bg_cbrate = bg_cbrate.iloc[5:].reset_index(drop=True)
    bg_cbrate = _drop_all_missing(bg_cbrate)
    bg_cbrate = bg_cbrate.copy()
    bg_cbrate["year"] = pd.to_numeric(bg_cbrate["year"], errors="coerce")
    bg_cbrate["cbrate"] = pd.to_numeric(bg_cbrate["cbrate"], errors="coerce")
    bg_cbrate = bg_cbrate.loc[bg_cbrate["year"].notna()].reset_index(drop=True)
    bg_cbrate = _year_end(bg_cbrate)
    bg_cbrate = bg_cbrate.drop(columns=["month"], errors="ignore")
    master = _merge_master(master, _finalize(bg_cbrate, "BGR"))

    bg_money = _select("BG data tables A", {"Q": "year", "R": "M0", "V": "M3"})
    bg_money = bg_money.iloc[5:].reset_index(drop=True)
    bg_money = _drop_all_missing(bg_money)
    bg_money = _replace_missing_strings(bg_money)
    bg_money = _to_numeric(bg_money)
    for col in ["M0", "M3"]:
        bg_money[col] = pd.to_numeric(bg_money[col], errors="coerce") / 1000
    master = _merge_master(master, _finalize(bg_money, "BGR"))

    tr_main = _select(
        "TR data tables A",
        {"AN": "year", "AO": "nGDP", "AP": "rGDP", "AQ": "exports", "AR": "imports", "AS": "pop", "AI": "CPI", "AC": "govtax", "AD": "govrev", "AE": "govdebt", "AF": "govdebt_d", "Z": "USDfx", "G": "M0", "M": "M1", "N": "M2", "O": "M3"},
    )
    tr_main = tr_main.iloc[6:].reset_index(drop=True)
    tr_main = _drop_all_missing(tr_main)
    tr_main = _replace_missing_strings(tr_main)
    tr_main = _to_numeric(tr_main)
    tr_main["govdebt"] = pd.to_numeric(tr_main["govdebt"], errors="coerce").where(
        pd.to_numeric(tr_main["govdebt_d"], errors="coerce").isna(),
        pd.to_numeric(tr_main["govdebt"], errors="coerce") + pd.to_numeric(tr_main["govdebt_d"], errors="coerce"),
    )
    tr_main = tr_main.drop(columns=["govdebt_d"], errors="ignore")
    for col in ["nGDP", "rGDP", "imports", "exports", "govrev", "govtax", "govdebt", "M0", "M1", "M2", "M3"]:
        tr_main[col] = pd.to_numeric(tr_main[col], errors="coerce") / 1000
    tr_main["govdebt_GDP"] = pd.to_numeric(tr_main["govdebt"], errors="coerce") / pd.to_numeric(tr_main["nGDP"], errors="coerce") * 100
    tr_main["govrev_GDP"] = pd.to_numeric(tr_main["govrev"], errors="coerce") / pd.to_numeric(tr_main["nGDP"], errors="coerce") * 100
    tr_main["govtax_GDP"] = pd.to_numeric(tr_main["govtax"], errors="coerce") / pd.to_numeric(tr_main["nGDP"], errors="coerce") * 100
    tr_main = tr_main.drop(columns=["govrev", "govdebt", "govtax"], errors="ignore")
    master = _merge_master(master, _finalize(tr_main, "TUR"))

    tr_rates = _select("TR data tables A", {"Q": "year", "R": "cbrate", "S": "strate", "T": "ltrate"})
    tr_rates = tr_rates.iloc[6:].reset_index(drop=True)
    tr_rates = _drop_all_missing(tr_rates)
    tr_rates = _replace_missing_strings(tr_rates)
    tr_rates = _to_numeric(tr_rates)
    master = _merge_master(master, _finalize(tr_rates, "TUR"))

    rs_pop = _select("SE data tables A", {"BY": "year", "CC": "pop"})
    rs_pop = rs_pop.iloc[4:].reset_index(drop=True)
    rs_pop = _drop_all_missing(rs_pop)
    rs_pop = _replace_missing_strings(rs_pop)
    rs_pop = _to_numeric(rs_pop)
    rs_pop["pop"] = pd.to_numeric(rs_pop["pop"], errors="coerce") / 1000
    master = _merge_master(master, _finalize(rs_pop, "SRB"))

    rs_rates = _select("SE data tables A", {"A": "year", "AI": "cbrate", "AK": "strate", "I": "M0"})
    rs_rates = rs_rates.iloc[4:].reset_index(drop=True)
    rs_rates = _drop_all_missing(rs_rates)
    rs_rates = _replace_missing_strings(rs_rates)
    rs_rates = _to_numeric(rs_rates)
    master = _merge_master(master, _finalize(rs_rates, "SRB"))

    at_main = _select("AH data tables A", {"BV": "year", "Z": "cbrate", "AA": "strate", "BR": "CPI", "BW": "nGDP", "BY": "rGDP", "CA": "rGDP_pc", "CG": "pop"})
    at_main = at_main.iloc[4:].reset_index(drop=True)
    at_main = _drop_all_missing(at_main)
    at_main = _replace_missing_strings(at_main)
    at_main = _to_numeric(at_main)
    at_main["pop"] = pd.to_numeric(at_main["pop"], errors="coerce") / 10
    master = _merge_master(master, _finalize(at_main, "AUT"))

    hu_main = _select("AH data tables A", {"BV": "year", "Z": "cbrate", "AA": "strate", "BX": "nGDP", "BZ": "rGDP", "CB": "rGDP_pc", "CH": "pop"})
    hu_main = hu_main.iloc[4:].reset_index(drop=True)
    hu_main = _drop_all_missing(hu_main)
    hu_main = _replace_missing_strings(hu_main)
    hu_main = _to_numeric(hu_main)
    hu_main["pop"] = pd.to_numeric(hu_main["pop"], errors="coerce") / 10
    master = _merge_master(master, _finalize(hu_main, "HUN"))

    assert master is not None
    master = _sort_keys(master)
    mask = master["ISO3"].astype(str).eq("GRC")
    if "govdebt" in master.columns:
        master.loc[mask, "govdebt_GDP"] = pd.to_numeric(master.loc[mask, "govdebt"], errors="coerce") / pd.to_numeric(master.loc[mask, "nGDP"], errors="coerce") * 100
    if "govrev" in master.columns:
        master.loc[mask, "govrev_GDP"] = pd.to_numeric(master.loc[mask, "govrev"], errors="coerce") / pd.to_numeric(master.loc[mask, "nGDP"], errors="coerce") * 100

    master["imports_GDP"] = pd.to_numeric(master["imports"], errors="coerce") / pd.to_numeric(master["nGDP"], errors="coerce") * 100
    master["exports_GDP"] = pd.to_numeric(master["exports"], errors="coerce") / pd.to_numeric(master["nGDP"], errors="coerce") * 100
    master["govexp_GDP"] = pd.to_numeric(master["govexp"], errors="coerce") / pd.to_numeric(master["nGDP"], errors="coerce") * 100
    master["govrev_GDP"] = pd.to_numeric(master["govrev"], errors="coerce") / pd.to_numeric(master["nGDP"], errors="coerce") * 100
    master["govtax_GDP"] = pd.to_numeric(master["govtax"], errors="coerce") / pd.to_numeric(master["nGDP"], errors="coerce") * 100

    master = master.rename(columns={col: f"NBS_{col}" for col in master.columns if col not in {"ISO3", "year"}})
    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].drop_duplicates().copy()
    master = master.merge(eur_fx, on="ISO3", how="left")
    fx_mask = master["EUR_irrevocable_FX"].notna()
    for col in ["NBS_nGDP", "NBS_rGDP", "NBS_M0", "NBS_M1", "NBS_M2", "NBS_M3", "NBS_govexp", "NBS_govrev", "NBS_govtax"]:
        if col in master.columns:
            master.loc[fx_mask, col] = (
                pd.to_numeric(master.loc[fx_mask, col], errors="coerce")
                / pd.to_numeric(master.loc[fx_mask, "EUR_irrevocable_FX"], errors="coerce")
            )
    master = master.drop(columns=["EUR_irrevocable_FX"], errors="ignore")

    tur_mask = master["ISO3"].astype(str).eq("TUR")
    master.loc[tur_mask, "NBS_USDfx"] = pd.to_numeric(master.loc[tur_mask, "NBS_USDfx"], errors="coerce") / 100000
    for col in ["NBS_nGDP", "NBS_rGDP", "NBS_M0", "NBS_M1", "NBS_M2", "NBS_M3", "NBS_govexp", "NBS_govrev", "NBS_govtax", "NBS_exports", "NBS_imports"]:
        if col in master.columns:
            master.loc[tur_mask, col] = pd.to_numeric(master.loc[tur_mask, col], errors="coerce") / 1000000

    rou_mask = master["ISO3"].astype(str).eq("ROU")
    for col in ["NBS_nGDP", "NBS_rGDP", "NBS_M0", "NBS_M1", "NBS_M2", "NBS_M3", "NBS_govexp", "NBS_govrev", "NBS_govtax", "NBS_exports", "NBS_imports", "NBS_USDfx"]:
        if col in master.columns:
            master.loc[rou_mask, col] = pd.to_numeric(master.loc[rou_mask, col], errors="coerce") * (10 ** -8)

    grc_mask = master["ISO3"].astype(str).eq("GRC")
    for col in ["NBS_rGDP", "NBS_M0", "NBS_M1", "NBS_M2", "NBS_M3", "NBS_govexp", "NBS_govrev", "NBS_govtax", "NBS_exports", "NBS_imports"]:
        if col in master.columns:
            master.loc[grc_mask, col] = pd.to_numeric(master.loc[grc_mask, col], errors="coerce") * (10 ** -6)
            master.loc[grc_mask, col] = pd.to_numeric(master.loc[grc_mask, col], errors="coerce") / 5

    bgr_mask = master["ISO3"].astype(str).eq("BGR")
    for col in ["NBS_nGDP", "NBS_USDfx", "NBS_rGDP", "NBS_M0", "NBS_M1", "NBS_M2", "NBS_M3", "NBS_govexp", "NBS_govrev", "NBS_govtax", "NBS_exports", "NBS_imports"]:
        if col in master.columns:
            master.loc[bgr_mask, col] = pd.to_numeric(master.loc[bgr_mask, col], errors="coerce") / 1000000

    master.loc[grc_mask, "NBS_USDfx"] = pd.to_numeric(master.loc[grc_mask, "NBS_USDfx"], errors="coerce") / 500

    prev_cpi = _lag_if_consecutive_year(master, "NBS_CPI")
    master["NBS_infl"] = (
        (
            pd.to_numeric(master["NBS_CPI"], errors="coerce") - pd.to_numeric(prev_cpi, errors="coerce")
        )
        / pd.to_numeric(prev_cpi, errors="coerce")
        * 100
    )
    master.loc[prev_cpi.isna(), "NBS_infl"] = np.nan
    master = master.drop(columns=["NBS_GBPfx", "NBS_FRFfx"], errors="ignore")

    expected = [
        "ISO3",
        "year",
        "NBS_cbrate",
        "NBS_strate",
        "NBS_nGDP",
        "NBS_rGDP",
        "NBS_rGDP_pc",
        "NBS_pop",
        "NBS_CPI",
        "NBS_M0",
        "NBS_ltrate",
        "NBS_M1",
        "NBS_M2",
        "NBS_M3",
        "NBS_USDfx",
        "NBS_exports",
        "NBS_imports",
        "NBS_govdebt_GDP",
        "NBS_govrev_GDP",
        "NBS_govtax_GDP",
        "NBS_govexp",
        "NBS_govrev",
        "NBS_govtax",
        "NBS_deflator",
        "NBS_imports_GDP",
        "NBS_exports_GDP",
        "NBS_govexp_GDP",
        "NBS_infl",
    ]
    for col in expected:
        if col not in master.columns:
            master[col] = np.nan
    master = master[expected].copy()
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    for col in [
        "NBS_cbrate",
        "NBS_strate",
        "NBS_nGDP",
        "NBS_rGDP",
        "NBS_rGDP_pc",
        "NBS_pop",
        "NBS_CPI",
        "NBS_M0",
        "NBS_ltrate",
        "NBS_M1",
        "NBS_M2",
        "NBS_M3",
        "NBS_USDfx",
        "NBS_exports",
        "NBS_imports",
        "NBS_govexp",
        "NBS_govrev",
        "NBS_govtax",
        "NBS_deflator",
    ]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
    for col in ["NBS_govdebt_GDP", "NBS_govrev_GDP", "NBS_govtax_GDP", "NBS_imports_GDP", "NBS_exports_GDP", "NBS_govexp_GDP", "NBS_infl"]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float32")

    master = _sort_keys(master)
    out_path = clean_dir / "aggregators" / "NBS" / "NBS.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_nbs"]
