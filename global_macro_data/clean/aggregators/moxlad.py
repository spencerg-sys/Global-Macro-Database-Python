from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_moxlad(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    path = raw_dir / "aggregators" / "MOXLAD" / "MOxLAD-2023-10-21_04-43.xls"

    def _moxlad_value(value: object) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if text in {"", "<NA>", "nan", "None"}:
            return np.nan
        try:
            return float(format(float(text), ".16g"))
        except (TypeError, ValueError):
            return np.nan

    def _moxlad_pow10(exp: int) -> float:
        return _pow10_literal(exp)

    raw = _read_excel_compat(path, header=None, dtype=str)
    if len(raw) >= 145:
        raw = raw.drop(index=144).reset_index(drop=True)
    raw = raw.rename(columns={raw.columns[0]: "year"}).copy()

    long = raw.melt(id_vars=["year"], var_name="_key", value_name="temp")
    long["desc"] = long.groupby("_key", sort=False)["temp"].transform("first")
    long = long.loc[~long["year"].isin(["Years", "Title", "Unit"])].copy()
    long = long.drop(columns=["_key"])
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long["temp"] = long["temp"].map(_moxlad_value)

    desc = long["desc"].astype("string")
    split = desc.str.split("-", n=1, expand=True)
    long["countryname"] = split[0].astype("string").str.strip()
    long["varname"] = split[1].astype("string").str.strip()
    long["varname"] = long["varname"].replace(
        {
            "IPC_IPC": "MOXLAD_CPI",
            "CCN_DEF_1970": "MOXLAD_deflator",
            "CCN_PBI_C": "MOXLAD_nGDP",
            "POB_POB": "MOXLAD_pop",
            "CCN_TCN_LCU": "MOXLAD_USDfx",
        }
    )
    long = long.loc[~long["varname"].isin(["CCN_DEF_1970", "CCN_DEF_1970_UML"])].copy()

    lookup = _country_name_lookup(helper_dir)
    long["ISO3"] = long["countryname"].map(lookup)
    if long["ISO3"].isna().any():
        missing = sorted(long.loc[long["ISO3"].isna(), "countryname"].dropna().astype(str).unique())
        raise ValueError(f"Unmapped MOXLAD countries: {missing}")

    wide = (
        long.pivot(index=["ISO3", "year"], columns="varname", values="temp")
        .reset_index()
        .rename_axis(columns=None)
    )

    if "MOXLAD_pop" in wide.columns:
        wide["MOXLAD_pop"] = pd.to_numeric(wide["MOXLAD_pop"], errors="coerce") / 1000

    arg_mask = wide["ISO3"].eq("ARG")
    wide.loc[arg_mask & pd.to_numeric(wide["year"], errors="coerce").le(1991), "MOXLAD_USDfx"] = pd.to_numeric(
        wide.loc[arg_mask & pd.to_numeric(wide["year"], errors="coerce").le(1991), "MOXLAD_USDfx"], errors="coerce"
    ) / 10000
    wide.loc[arg_mask & pd.to_numeric(wide["year"], errors="coerce").le(1984), "MOXLAD_USDfx"] = pd.to_numeric(
        wide.loc[arg_mask & pd.to_numeric(wide["year"], errors="coerce").le(1984), "MOXLAD_USDfx"], errors="coerce"
    ) / 1000
    wide.loc[arg_mask & pd.to_numeric(wide["year"], errors="coerce").le(1982), "MOXLAD_USDfx"] = pd.to_numeric(
        wide.loc[arg_mask & pd.to_numeric(wide["year"], errors="coerce").le(1982), "MOXLAD_USDfx"], errors="coerce"
    ) / 10000
    wide.loc[arg_mask & pd.to_numeric(wide["year"], errors="coerce").le(1969), "MOXLAD_USDfx"] = pd.to_numeric(
        wide.loc[arg_mask & pd.to_numeric(wide["year"], errors="coerce").le(1969), "MOXLAD_USDfx"], errors="coerce"
    ) / 100
    wide.loc[arg_mask & pd.to_numeric(wide["year"], errors="coerce").le(1991), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[arg_mask & pd.to_numeric(wide["year"], errors="coerce").le(1991), "MOXLAD_nGDP"], errors="coerce"
    ) / 10000
    wide.loc[arg_mask & pd.to_numeric(wide["year"], errors="coerce").le(1984), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[arg_mask & pd.to_numeric(wide["year"], errors="coerce").le(1984), "MOXLAD_nGDP"], errors="coerce"
    ) / 1000
    wide.loc[arg_mask & pd.to_numeric(wide["year"], errors="coerce").le(1982), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[arg_mask & pd.to_numeric(wide["year"], errors="coerce").le(1982), "MOXLAD_nGDP"], errors="coerce"
    ) / 10000
    wide.loc[arg_mask & pd.to_numeric(wide["year"], errors="coerce").le(1969), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[arg_mask & pd.to_numeric(wide["year"], errors="coerce").le(1969), "MOXLAD_nGDP"], errors="coerce"
    ) / 100

    bra_mask = wide["ISO3"].eq("BRA")
    for cutoff, scale in [(1994, 2750), (1993, 1000), (1988, 1000), (1985, 1000), (1966, 1000)]:
        year_mask = bra_mask & pd.to_numeric(wide["year"], errors="coerce").le(cutoff)
        wide.loc[year_mask, "MOXLAD_USDfx"] = pd.to_numeric(wide.loc[year_mask, "MOXLAD_USDfx"], errors="coerce") / scale
        wide.loc[year_mask, "MOXLAD_nGDP"] = pd.to_numeric(wide.loc[year_mask, "MOXLAD_nGDP"], errors="coerce") / scale

    chl_mask = wide["ISO3"].eq("CHL")
    wide.loc[chl_mask & pd.to_numeric(wide["year"], errors="coerce").le(1975), "MOXLAD_USDfx"] = pd.to_numeric(
        wide.loc[chl_mask & pd.to_numeric(wide["year"], errors="coerce").le(1975), "MOXLAD_USDfx"], errors="coerce"
    ) / 1000
    wide.loc[chl_mask & pd.to_numeric(wide["year"], errors="coerce").le(1975), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[chl_mask & pd.to_numeric(wide["year"], errors="coerce").le(1975), "MOXLAD_nGDP"], errors="coerce"
    ) / 1000
    wide.loc[chl_mask & pd.to_numeric(wide["year"], errors="coerce").le(1959), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[chl_mask & pd.to_numeric(wide["year"], errors="coerce").le(1959), "MOXLAD_nGDP"], errors="coerce"
    ) / 1000
    wide.loc[chl_mask & pd.to_numeric(wide["year"], errors="coerce").le(1959), "MOXLAD_USDfx"] = pd.to_numeric(
        wide.loc[chl_mask & pd.to_numeric(wide["year"], errors="coerce").le(1959), "MOXLAD_USDfx"], errors="coerce"
    ) / 1000

    mex_mask = wide["ISO3"].eq("MEX")
    wide.loc[mex_mask & pd.to_numeric(wide["year"], errors="coerce").le(1992), "MOXLAD_USDfx"] = pd.to_numeric(
        wide.loc[mex_mask & pd.to_numeric(wide["year"], errors="coerce").le(1992), "MOXLAD_USDfx"], errors="coerce"
    ) / 1000
    wide.loc[mex_mask & pd.to_numeric(wide["year"], errors="coerce").lt(1993), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[mex_mask & pd.to_numeric(wide["year"], errors="coerce").lt(1993), "MOXLAD_nGDP"], errors="coerce"
    ) / 1000

    per_mask = wide["ISO3"].eq("PER")
    wide.loc[per_mask & pd.to_numeric(wide["year"], errors="coerce").le(1990), "MOXLAD_USDfx"] = pd.to_numeric(
        wide.loc[per_mask & pd.to_numeric(wide["year"], errors="coerce").le(1990), "MOXLAD_USDfx"], errors="coerce"
    ) / 1000000
    wide.loc[per_mask & pd.to_numeric(wide["year"], errors="coerce").le(1984), "MOXLAD_USDfx"] = pd.to_numeric(
        wide.loc[per_mask & pd.to_numeric(wide["year"], errors="coerce").le(1984), "MOXLAD_USDfx"], errors="coerce"
    ) / 1000
    wide.loc[per_mask & pd.to_numeric(wide["year"], errors="coerce").le(1990), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[per_mask & pd.to_numeric(wide["year"], errors="coerce").le(1990), "MOXLAD_nGDP"], errors="coerce"
    ) * _moxlad_pow10(-6)
    wide.loc[per_mask & pd.to_numeric(wide["year"], errors="coerce").le(1984), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[per_mask & pd.to_numeric(wide["year"], errors="coerce").le(1984), "MOXLAD_nGDP"], errors="coerce"
    ) * _moxlad_pow10(-3)

    bol_mask = wide["ISO3"].eq("BOL")
    wide.loc[bol_mask & pd.to_numeric(wide["year"], errors="coerce").le(1984), "MOXLAD_USDfx"] = pd.to_numeric(
        wide.loc[bol_mask & pd.to_numeric(wide["year"], errors="coerce").le(1984), "MOXLAD_USDfx"], errors="coerce"
    ) / 1000000
    wide.loc[bol_mask & pd.to_numeric(wide["year"], errors="coerce").le(1962), "MOXLAD_USDfx"] = pd.to_numeric(
        wide.loc[bol_mask & pd.to_numeric(wide["year"], errors="coerce").le(1962), "MOXLAD_USDfx"], errors="coerce"
    ) / 1000
    wide.loc[bol_mask & pd.to_numeric(wide["year"], errors="coerce").le(1984), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[bol_mask & pd.to_numeric(wide["year"], errors="coerce").le(1984), "MOXLAD_nGDP"], errors="coerce"
    ) / 1000000
    wide.loc[bol_mask & pd.to_numeric(wide["year"], errors="coerce").le(1962), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[bol_mask & pd.to_numeric(wide["year"], errors="coerce").le(1962), "MOXLAD_nGDP"], errors="coerce"
    ) / 1000

    ven_mask = wide["ISO3"].eq("VEN")
    wide.loc[ven_mask, "MOXLAD_USDfx"] = pd.to_numeric(wide.loc[ven_mask, "MOXLAD_USDfx"], errors="coerce") / 1000
    wide.loc[ven_mask, "MOXLAD_nGDP"] = pd.to_numeric(wide.loc[ven_mask, "MOXLAD_nGDP"], errors="coerce") * _moxlad_pow10(-14)

    ury_mask = wide["ISO3"].eq("URY")
    wide.loc[ury_mask & pd.to_numeric(wide["year"], errors="coerce").le(1993), "MOXLAD_USDfx"] = pd.to_numeric(
        wide.loc[ury_mask & pd.to_numeric(wide["year"], errors="coerce").le(1993), "MOXLAD_USDfx"], errors="coerce"
    ) / 1000
    wide.loc[ury_mask & pd.to_numeric(wide["year"], errors="coerce").le(1974), "MOXLAD_USDfx"] = pd.to_numeric(
        wide.loc[ury_mask & pd.to_numeric(wide["year"], errors="coerce").le(1974), "MOXLAD_USDfx"], errors="coerce"
    ) / 1000
    wide.loc[ury_mask & pd.to_numeric(wide["year"], errors="coerce").le(1993), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[ury_mask & pd.to_numeric(wide["year"], errors="coerce").le(1993), "MOXLAD_nGDP"], errors="coerce"
    ) * _moxlad_pow10(-3)
    wide.loc[ury_mask & pd.to_numeric(wide["year"], errors="coerce").le(1974), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[ury_mask & pd.to_numeric(wide["year"], errors="coerce").le(1974), "MOXLAD_nGDP"], errors="coerce"
    ) * _moxlad_pow10(-3)
    wide.loc[ury_mask & pd.to_numeric(wide["year"], errors="coerce").eq(1959), "MOXLAD_USDfx"] = np.nan

    nic_mask = wide["ISO3"].eq("NIC")
    wide.loc[nic_mask & pd.to_numeric(wide["year"], errors="coerce").le(1990), "MOXLAD_USDfx"] = pd.to_numeric(
        wide.loc[nic_mask & pd.to_numeric(wide["year"], errors="coerce").le(1990), "MOXLAD_USDfx"], errors="coerce"
    ) / 5000000
    wide.loc[nic_mask & pd.to_numeric(wide["year"], errors="coerce").le(1988), "MOXLAD_USDfx"] = pd.to_numeric(
        wide.loc[nic_mask & pd.to_numeric(wide["year"], errors="coerce").le(1988), "MOXLAD_USDfx"], errors="coerce"
    ) / 1000
    wide.loc[nic_mask & pd.to_numeric(wide["year"], errors="coerce").le(1912), "MOXLAD_USDfx"] = pd.to_numeric(
        wide.loc[nic_mask & pd.to_numeric(wide["year"], errors="coerce").le(1912), "MOXLAD_USDfx"], errors="coerce"
    ) / 12.5
    wide.loc[nic_mask & pd.to_numeric(wide["year"], errors="coerce").le(1990), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[nic_mask & pd.to_numeric(wide["year"], errors="coerce").le(1990), "MOXLAD_nGDP"], errors="coerce"
    ) / 5000
    wide.loc[nic_mask & pd.to_numeric(wide["year"], errors="coerce").eq(1989), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[nic_mask & pd.to_numeric(wide["year"], errors="coerce").eq(1989), "MOXLAD_nGDP"], errors="coerce"
    ) / 1000
    wide.loc[nic_mask & pd.to_numeric(wide["year"], errors="coerce").le(1988), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[nic_mask & pd.to_numeric(wide["year"], errors="coerce").le(1988), "MOXLAD_nGDP"], errors="coerce"
    ) / 1000
    wide.loc[nic_mask & pd.to_numeric(wide["year"], errors="coerce").le(1912), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[nic_mask & pd.to_numeric(wide["year"], errors="coerce").le(1912), "MOXLAD_nGDP"], errors="coerce"
    ) / 12.5
    wide.loc[nic_mask & pd.to_numeric(wide["year"], errors="coerce").le(1986), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[nic_mask & pd.to_numeric(wide["year"], errors="coerce").le(1986), "MOXLAD_nGDP"], errors="coerce"
    ) / 100
    wide.loc[nic_mask & pd.to_numeric(wide["year"], errors="coerce").between(1969, 1972), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[nic_mask & pd.to_numeric(wide["year"], errors="coerce").between(1969, 1972), "MOXLAD_nGDP"], errors="coerce"
    ) / 10

    pry_mask = wide["ISO3"].eq("PRY")
    wide.loc[pry_mask & pd.to_numeric(wide["year"], errors="coerce").le(1919), "MOXLAD_USDfx"] = pd.to_numeric(
        wide.loc[pry_mask & pd.to_numeric(wide["year"], errors="coerce").le(1919), "MOXLAD_USDfx"], errors="coerce"
    ) / 0.0175
    wide.loc[pry_mask & pd.to_numeric(wide["year"], errors="coerce").le(1942), "MOXLAD_USDfx"] = pd.to_numeric(
        wide.loc[pry_mask & pd.to_numeric(wide["year"], errors="coerce").le(1942), "MOXLAD_USDfx"], errors="coerce"
    ) / 100

    wide.loc[wide["ISO3"].eq("SLV"), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[wide["ISO3"].eq("SLV"), "MOXLAD_nGDP"], errors="coerce"
    ) / 8.75
    wide.loc[wide["ISO3"].eq("ECU"), "MOXLAD_nGDP"] = pd.to_numeric(
        wide.loc[wide["ISO3"].eq("ECU"), "MOXLAD_nGDP"], errors="coerce"
    ) * _moxlad_pow10(-3)
    wide.loc[wide["ISO3"].eq("HTI") & pd.to_numeric(wide["year"], errors="coerce").le(1921), "MOXLAD_USDfx"] = np.nan

    expected = ["ISO3", "year", "MOXLAD_CPI", "MOXLAD_USDfx", "MOXLAD_deflator", "MOXLAD_nGDP", "MOXLAD_pop"]
    for col in expected:
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide[expected].copy()

    wide = _sort_keys(wide)
    wide["CPI_2000"] = wide["MOXLAD_CPI"].where(pd.to_numeric(wide["year"], errors="coerce").eq(2000))
    wide["CPI_2000_all"] = wide.groupby("ISO3")["CPI_2000"].transform("mean")
    wide["CPI_2000_all"] = pd.to_numeric(wide["CPI_2000_all"], errors="coerce").astype("float32").astype("float64")
    wide["MOXLAD_CPI"] = pd.to_numeric(wide["MOXLAD_CPI"], errors="coerce") * 100 / pd.to_numeric(
        wide["CPI_2000_all"], errors="coerce"
    )
    wide = wide.drop(columns=["CPI_2000", "CPI_2000_all"])

    prev_cpi = _lag_if_consecutive_year(wide, "MOXLAD_CPI")
    wide["MOXLAD_infl"] = (
        (
            pd.to_numeric(wide["MOXLAD_CPI"], errors="coerce") - pd.to_numeric(prev_cpi, errors="coerce")
        )
        / pd.to_numeric(prev_cpi, errors="coerce")
        * 100
    )
    wide.loc[prev_cpi.isna(), "MOXLAD_infl"] = np.nan

    wide["ISO3"] = wide["ISO3"].astype("object")
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int16")
    for col in ["MOXLAD_CPI", "MOXLAD_USDfx", "MOXLAD_deflator", "MOXLAD_nGDP", "MOXLAD_pop"]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float64")
    wide["MOXLAD_infl"] = pd.to_numeric(wide["MOXLAD_infl"], errors="coerce").astype("float32")
    wide.columns = pd.Index([str(col) for col in wide.columns], dtype=object)
    wide = _sort_keys(wide)
    out_path = clean_dir / "aggregators" / "MOXLAD" / "MOXLAD.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_moxlad"]
