from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_amf(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    countrylist = _load_dta(helper_dir / "countrylist.dta")[["countryname", "ISO3"]].copy()

    def _load_amf_table(path: Path) -> pd.DataFrame:
        raw = pd.read_excel(path, header=None, dtype=str)
        year_cols = list(raw.columns[3:])
        keep_mask = ~raw[year_cols].isna().all(axis=1)
        raw = raw.loc[keep_mask].reset_index(drop=True)
        raw.columns = ["A", "B", "C"] + [str(col) for col in raw.iloc[0, 3:].tolist()]
        return raw

    def _attach_iso3(df: pd.DataFrame) -> pd.DataFrame:
        out = df.merge(countrylist, left_on="countryname", right_on="countryname", how="left")
        out.loc[out["countryname"].astype(str) == "Emirates", "ISO3"] = "ARE"
        out = out.loc[out["ISO3"].notna()].copy()
        return out.drop(columns=["countryname"])

    na = _load_amf_table(raw_dir / "aggregators" / "AMF" / "AMF_national_accounts.xlsx")
    na["A"] = na["A"].replace(
        {
            "|==>GDP": "nGDP",
            "|==>Total Consumption": "cons",
            "|===>Exports of Goods and Services": "exports",
            "|===>Imports of Goods and Services": "imports",
            "|==>Total Investment": "inv",
            "|==>First: the percentage change in the consumer price index": "infl",
        }
    )
    na.loc[na["A"].astype(str).str.contains(r"\|==>GDP at base price", regex=True, na=False), "A"] = "rGDP"
    na = na.loc[~na["A"].astype(str).str.contains("=", regex=False, na=False)].copy()
    na = na.drop(columns=["C"], errors="ignore").rename(columns={"A": "series", "B": "countryname"})
    na = na.iloc[1:].reset_index(drop=True)
    na_long = na.melt(id_vars=["countryname", "series"], var_name="year", value_name="AMF_")
    na_long["year"] = pd.to_numeric(na_long["year"], errors="coerce")
    na_long["AMF_"] = pd.to_numeric(na_long["AMF_"], errors="coerce")
    if na_long.duplicated(["countryname", "series", "year"]).any():
        raise sh.PipelineRuntimeError("countryname series year do not uniquely identify observations", code=459)
    na_wide = na_long.pivot(index=["countryname", "year"], columns="series", values="AMF_").reset_index()
    na_wide.columns.name = None
    na_wide = _attach_iso3(na_wide)
    na_wide = na_wide.rename(columns={col: f"AMF_{col}" for col in na_wide.columns if col not in {"ISO3", "year"}})
    na_wide = _sort_keys(na_wide, keys=("ISO3", "year"))

    fx = _load_amf_table(raw_dir / "aggregators" / "AMF" / "AMF_USDfx.xlsx")
    fx = fx.loc[fx["A"].astype(str) != "|=>Domestic Currency Per U.S. Dollar(Period Average)"].copy()
    fx = fx.drop(columns=["A", "C", "1970"], errors="ignore").rename(columns={"B": "countryname"})
    fx = fx.iloc[1:].reset_index(drop=True)
    fx_long = fx.melt(id_vars=["countryname"], var_name="year", value_name="AMF_USDfx")
    fx_long["year"] = pd.to_numeric(fx_long["year"], errors="coerce")
    fx_long["AMF_USDfx"] = pd.to_numeric(fx_long["AMF_USDfx"], errors="coerce")
    fx_long = _attach_iso3(fx_long)
    fx_long = fx_long[["ISO3", "year", "AMF_USDfx"]]
    master = fx_long.merge(na_wide, on=["ISO3", "year"], how="outer", sort=False)

    ca = _load_amf_table(raw_dir / "aggregators" / "AMF" / "AMF_CA_balance.xlsx")
    ca = ca.loc[ca["A"].astype(str).isin(["|=>Current Account Balance", "Item"])].copy()
    ca = ca.drop(columns=["A", "C", "1970"], errors="ignore").rename(columns={"B": "countryname"})
    ca = ca.iloc[1:].reset_index(drop=True)
    ca_long = ca.melt(id_vars=["countryname"], var_name="year", value_name="AMF_CA")
    ca_long["year"] = pd.to_numeric(ca_long["year"], errors="coerce")
    ca_long["AMF_CA"] = pd.to_numeric(ca_long["AMF_CA"], errors="coerce")
    ca_long = _attach_iso3(ca_long)
    ca_long = ca_long[["ISO3", "year", "AMF_CA"]]
    master = ca_long.merge(master, on=["ISO3", "year"], how="outer", sort=False)

    gov = _load_amf_table(raw_dir / "aggregators" / "AMF" / "AMF_gov_finance.xlsx")
    gov["A"] = gov["A"].replace(
        {
            "|=>Total Public Revenues": "govrev",
            "|===> Taxes Revenue": "govtax",
            "|=>Total Public Expenditure": "govexp",
            "|>Overall  Surplus (+) / Deficit (-)": "govdef",
        }
    )
    gov = gov.loc[~gov["A"].astype(str).str.contains(r"\|", regex=True, na=False)].copy()
    gov = gov.drop(columns=["C"], errors="ignore").rename(columns={"A": "series", "B": "countryname"})
    gov = gov.iloc[1:].reset_index(drop=True)
    gov_long = gov.melt(id_vars=["countryname", "series"], var_name="year", value_name="AMF_")
    gov_long["year"] = pd.to_numeric(gov_long["year"], errors="coerce")
    gov_long["AMF_"] = pd.to_numeric(gov_long["AMF_"], errors="coerce")
    if gov_long.duplicated(["countryname", "series", "year"]).any():
        raise sh.PipelineRuntimeError("countryname series year do not uniquely identify observations", code=459)
    gov_wide = gov_long.pivot(index=["countryname", "year"], columns="series", values="AMF_").reset_index()
    gov_wide.columns.name = None
    gov_wide = _attach_iso3(gov_wide)
    gov_wide = gov_wide.rename(columns={col: f"AMF_{col}" for col in gov_wide.columns if col not in {"ISO3", "year"}})
    master = gov_wide.merge(master, on=["ISO3", "year"], how="outer", sort=False)

    fix_specs = [
        ("AMF_imports", "MRT", lambda year: year <= 2012, 10),
        ("AMF_exports", "MRT", lambda year: year <= 2012, 10),
        ("AMF_govexp", "MRT", lambda year: year <= 2012, 10),
        ("AMF_govrev", "MRT", lambda year: year <= 2012, 10),
        ("AMF_govdef", "MRT", lambda year: year <= 2012, 10),
        ("AMF_govtax", "MRT", lambda year: year <= 2012, 10),
        ("AMF_cons", "MRT", lambda year: year <= 2012, 10),
        ("AMF_USDfx", "MRT", lambda year: year <= 2009, 10),
        ("AMF_imports", "SDN", lambda year: year <= 1996, 100),
        ("AMF_exports", "SDN", lambda year: year <= 1996, 100),
        ("AMF_inv", "SDN", lambda year: year <= 1996, 100),
    ]
    year_num = pd.to_numeric(master["year"], errors="coerce")
    iso3 = master["ISO3"].astype(str)
    for column, country, predicate, divisor in fix_specs:
        if column in master.columns:
            mask = iso3.eq(country) & predicate(year_num)
            master.loc[mask, column] = pd.to_numeric(master.loc[mask, column], errors="coerce") / divisor

    ratio_specs = {
        "AMF_govexp_GDP": ("AMF_govexp", "AMF_nGDP"),
        "AMF_govdef_GDP": ("AMF_govdef", "AMF_nGDP"),
        "AMF_govrev_GDP": ("AMF_govrev", "AMF_nGDP"),
        "AMF_govtax_GDP": ("AMF_govtax", "AMF_nGDP"),
        "AMF_CA_GDP": ("AMF_CA", "AMF_nGDP"),
        "AMF_cons_GDP": ("AMF_cons", "AMF_nGDP"),
        "AMF_imports_GDP": ("AMF_imports", "AMF_nGDP"),
        "AMF_exports_GDP": ("AMF_exports", "AMF_nGDP"),
        "AMF_inv_GDP": ("AMF_inv", "AMF_nGDP"),
    }
    for result, (num, den) in ratio_specs.items():
        if {num, den}.issubset(master.columns):
            denominator = pd.to_numeric(master[den], errors="coerce").replace(0, np.nan)
            master[result] = pd.to_numeric(master[num], errors="coerce") / denominator * 100
            master[result] = pd.to_numeric(master[result], errors="coerce").astype("float32")

    ordered_cols = [
        "ISO3",
        "year",
        "AMF_govdef",
        "AMF_govexp",
        "AMF_govrev",
        "AMF_govtax",
        "AMF_CA",
        "AMF_USDfx",
        "AMF_cons",
        "AMF_exports",
        "AMF_imports",
        "AMF_infl",
        "AMF_inv",
        "AMF_nGDP",
        "AMF_rGDP",
        "AMF_govexp_GDP",
        "AMF_govdef_GDP",
        "AMF_govrev_GDP",
        "AMF_govtax_GDP",
        "AMF_CA_GDP",
        "AMF_cons_GDP",
        "AMF_imports_GDP",
        "AMF_exports_GDP",
        "AMF_inv_GDP",
    ]
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    master = _sort_keys(master)
    if master.duplicated(["ISO3", "year"]).any():
        raise sh.PipelineRuntimeError("ISO3 year do not uniquely identify observations", code=459)
    master = master[[col for col in ordered_cols if col in master.columns]]
    out_path = clean_dir / "aggregators" / "AMF" / "AMF.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_amf"]
