from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_isl_2(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    path = raw_dir / "country_level" / "ISL_2.xlsx"

    def _numeric_string(series: pd.Series) -> pd.Series:
        return _excel_numeric_series(series.astype("string").str.strip(), mode="float")

    def _dedupe_names(names: list[str]) -> list[str]:
        seen: dict[str, int] = {}
        deduped: list[str] = []
        for name in names:
            stem = _sanitize_identifier_name(name).strip("_")
            stem = stem or "var"
            count = seen.get(stem, 0)
            deduped_name = stem if count == 0 else f"{stem}_{count + 1}"
            seen[stem] = count + 1
            deduped.append(deduped_name)
        return deduped

    cbrate_raw = _read_excel_compat(path, sheet_name="cbrate", header=None, dtype=str)
    cbrate_raw = cbrate_raw.iloc[4:, :2].copy()
    cbrate_raw.columns = ["A", "B"]
    date_parts = cbrate_raw["A"].astype("string").str.strip().str.extract(r"^(?P<day>\d+)\.(?P<month>\d+)\.(?P<year>\d{4})$")
    cbrate_raw["year"] = pd.to_numeric(date_parts["year"], errors="coerce")
    cbrate_raw["month"] = pd.to_numeric(date_parts["month"], errors="coerce")
    cbrate_raw["cbrate"] = pd.to_numeric(cbrate_raw["B"], errors="coerce")
    cbrate = cbrate_raw.loc[cbrate_raw["year"].notna(), ["year", "month", "cbrate"]].copy()
    cbrate = cbrate.sort_values(["year", "month"], kind="mergesort").groupby("year", sort=False).tail(1).copy()
    master = cbrate[["year", "cbrate"]].copy()

    for sheet in ["Gov", "gov_debt", "CPI", "rates", "USDfx", "national_accounts", "Trade", "Money"]:
        current = _read_excel_compat(path, sheet_name=sheet, header=None, dtype=str)
        first_col = current.iloc[:, 0].astype("string").str.strip()
        year_rows = first_col.eq("year")
        if not year_rows.any():
            raise ValueError(f"Could not find year header in sheet {sheet}")
        current = current.iloc[int(year_rows[year_rows].index[0]) :].reset_index(drop=True).copy()
        current = current.loc[
            :,
            [
                col
                for col in current.columns
                if current[col].astype("string").str.strip().replace({"": pd.NA, "<NA>": pd.NA}).notna().any()
            ],
        ].copy()
        headers = _dedupe_names(current.iloc[0].astype("string").fillna("").tolist())
        current = current.iloc[1:].reset_index(drop=True).copy()
        current.columns = headers
        for col in current.columns:
            current[col] = _numeric_string(current[col])
        if current.select_dtypes(include=["object", "string"]).columns.tolist():
            raise ValueError(f"Not all variables in the {sheet} sheet are numeric.")
        current = current.loc[current["year"].notna()].copy()
        overlap = [col for col in current.columns if col != "year" and col in master.columns]
        master = current.merge(master, on="year", how="outer", suffixes=("", "_using"), indicator=True)
        right_only = master["_merge"].eq("right_only")
        for col in overlap:
            using_col = f"{col}_using"
            if using_col in master.columns:
                master.loc[right_only, col] = pd.to_numeric(master.loc[right_only, using_col], errors="coerce")
                master = master.drop(columns=[using_col], errors="ignore")
        master = master.drop(columns=["_merge"], errors="ignore")

    year_num = pd.to_numeric(master["year"], errors="coerce")
    for col in ["nGDP", "rGDP"]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
        master.loc[year_num.le(1945), col] = pd.to_numeric(master.loc[year_num.le(1945), col], errors="coerce") / 100

    for col in ["M0", "M1", "M2", "M3", "USDfx", "gov_debt"]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
        master.loc[year_num.le(1980), col] = pd.to_numeric(master.loc[year_num.le(1980), col], errors="coerce") / 100

    for col in ["exports", "imports"]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
        master.loc[year_num.lt(1945), col] = pd.to_numeric(master.loc[year_num.lt(1945), col], errors="coerce") / 100000

    government_cols = [col for col in master.columns if col.startswith("central") or col.startswith("local")]
    for col in government_cols:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
        master.loc[year_num.le(1945), col] = pd.to_numeric(master.loc[year_num.le(1945), col], errors="coerce") / 100000

    revenue = pd.to_numeric(master.get("REVENUE"), errors="coerce")
    master["REVENUE"] = revenue.where(
        revenue.notna(),
        pd.to_numeric(master.get("central_gov_REVENUE"), errors="coerce")
        + pd.to_numeric(master.get("local_gov_revenue"), errors="coerce"),
    )
    master["REVENUE"] = pd.to_numeric(master["REVENUE"], errors="coerce").where(
        pd.to_numeric(master["REVENUE"], errors="coerce").notna(),
        pd.to_numeric(master.get("central_gov_REVENUE"), errors="coerce"),
    )
    govexp = pd.to_numeric(master.get("govexp"), errors="coerce")
    master["govexp"] = govexp.where(
        govexp.notna(),
        pd.to_numeric(master.get("central_govexp"), errors="coerce")
        + pd.to_numeric(master.get("local_gov_exp"), errors="coerce"),
    )
    master["govexp"] = pd.to_numeric(master["govexp"], errors="coerce").where(
        pd.to_numeric(master["govexp"], errors="coerce").notna(),
        pd.to_numeric(master.get("central_govexp"), errors="coerce"),
    )
    deficit = pd.to_numeric(master.get("DEFICIT"), errors="coerce")
    master["DEFICIT"] = deficit.where(
        deficit.notna(),
        pd.to_numeric(master.get("central_gov_DEFICIT"), errors="coerce")
        + pd.to_numeric(master.get("local_gov_balance"), errors="coerce"),
    )
    master["govtax"] = pd.to_numeric(master.get("direct_taxes"), errors="coerce") + pd.to_numeric(master.get("indirect_taxes"), errors="coerce")
    master["govtax"] = pd.to_numeric(master["govtax"], errors="coerce").where(
        pd.to_numeric(master["govtax"], errors="coerce").notna(),
        pd.to_numeric(master.get("central_gov_direct_taxes"), errors="coerce")
        + pd.to_numeric(master.get("central_gov_ftrade_taxes"), errors="coerce")
        + pd.to_numeric(master.get("central_gov_other_taxes"), errors="coerce"),
    )
    master = master.rename(columns={"REVENUE": "govrev", "DEFICIT": "govdef", "gov_debt": "govdebt"})
    n_gdp = pd.to_numeric(master["nGDP"], errors="coerce")
    for col in [c for c in master.columns if c.startswith("gov")]:
        master[f"{col}_GDP"] = pd.to_numeric(master[col], errors="coerce") / n_gdp * 100

    master = master.drop(
        columns=[
            col
            for col in master.columns
            if col.startswith("central")
            or col.startswith("local")
            or col in {"direct_taxes", "indirect_taxes"}
        ],
        errors="ignore",
    )
    master["inv"] = pd.to_numeric(master["finv"], errors="coerce") + pd.to_numeric(master["temp_inv"], errors="coerce")
    master = master.drop(columns=["temp_inv"], errors="ignore")
    master["ISO3"] = "ISL"
    master["cons_GDP"] = pd.to_numeric(master["cons"], errors="coerce") / n_gdp * 100
    master["imports_GDP"] = pd.to_numeric(master["imports"], errors="coerce") / n_gdp * 100
    master["exports_GDP"] = pd.to_numeric(master["exports"], errors="coerce") / n_gdp * 100
    master["finv_GDP"] = pd.to_numeric(master["finv"], errors="coerce") / n_gdp * 100
    master["inv_GDP"] = pd.to_numeric(master["inv"], errors="coerce") / n_gdp * 100
    master = master.rename(columns={col: f"CS2_{col}" for col in master.columns if col not in {"ISO3", "year"}})

    late_mask = pd.to_numeric(master["year"], errors="coerce").ge(1980)
    for col in ["CS2_govexp", "CS2_govrev", "CS2_govtax"]:
        master[col] = pd.to_numeric(master[col], errors="coerce")
        master.loc[late_mask, col] = pd.to_numeric(master.loc[late_mask, col], errors="coerce") * 1000

    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    for col in [
        "CS2_M0",
        "CS2_M1",
        "CS2_M2",
        "CS2_M3",
        "CS2_exports",
        "CS2_imports",
        "CS2_nGDP",
        "CS2_rGDP",
        "CS2_finv",
        "CS2_sav",
        "CS2_cons",
        "CS2_USDfx",
        "CS2_strate",
        "CS2_ltrate",
        "CS2_CPI",
        "CS2_infl",
        "CS2_govdebt",
        "CS2_govrev",
        "CS2_govexp",
        "CS2_govdef",
        "CS2_cbrate",
    ]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
    for col in [
        "CS2_govtax",
        "CS2_govdebt_GDP",
        "CS2_govrev_GDP",
        "CS2_govexp_GDP",
        "CS2_govdef_GDP",
        "CS2_govtax_GDP",
        "CS2_inv",
        "CS2_cons_GDP",
        "CS2_imports_GDP",
        "CS2_exports_GDP",
        "CS2_finv_GDP",
        "CS2_inv_GDP",
    ]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float32")

    master = master[
        [
            "ISO3",
            "year",
            "CS2_M0",
            "CS2_M1",
            "CS2_M2",
            "CS2_M3",
            "CS2_exports",
            "CS2_imports",
            "CS2_nGDP",
            "CS2_rGDP",
            "CS2_finv",
            "CS2_sav",
            "CS2_cons",
            "CS2_USDfx",
            "CS2_strate",
            "CS2_ltrate",
            "CS2_CPI",
            "CS2_infl",
            "CS2_govdebt",
            "CS2_govrev",
            "CS2_govexp",
            "CS2_govdef",
            "CS2_cbrate",
            "CS2_govtax",
            "CS2_govdebt_GDP",
            "CS2_govrev_GDP",
            "CS2_govexp_GDP",
            "CS2_govdef_GDP",
            "CS2_govtax_GDP",
            "CS2_inv",
            "CS2_cons_GDP",
            "CS2_imports_GDP",
            "CS2_exports_GDP",
            "CS2_finv_GDP",
            "CS2_inv_GDP",
        ]
    ].copy()
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    for col in [
        "CS2_M0",
        "CS2_M1",
        "CS2_M2",
        "CS2_M3",
        "CS2_exports",
        "CS2_imports",
        "CS2_nGDP",
        "CS2_rGDP",
        "CS2_finv",
        "CS2_sav",
        "CS2_cons",
        "CS2_USDfx",
        "CS2_strate",
        "CS2_ltrate",
        "CS2_CPI",
        "CS2_infl",
        "CS2_govdebt",
        "CS2_govrev",
        "CS2_govexp",
        "CS2_govdef",
        "CS2_cbrate",
    ]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
    for col in [
        "CS2_govtax",
        "CS2_govdebt_GDP",
        "CS2_govrev_GDP",
        "CS2_govexp_GDP",
        "CS2_govdef_GDP",
        "CS2_govtax_GDP",
        "CS2_inv",
        "CS2_cons_GDP",
        "CS2_imports_GDP",
        "CS2_exports_GDP",
        "CS2_finv_GDP",
        "CS2_inv_GDP",
    ]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float32")
    master = _sort_keys(master)
    out_path = clean_dir / "country_level" / "ISL_2.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_isl_2"]
