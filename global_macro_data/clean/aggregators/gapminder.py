from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_gapminder(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    path = raw_dir / "aggregators" / "Gapminder" / "Gapminder.xlsx"

    lookup = _country_name_lookup(helper_dir)
    df = _read_excel_compat(path, header=None, dtype=str)
    headers = df.iloc[0].astype("string").fillna("").tolist()
    df = df.iloc[1:].reset_index(drop=True).copy()
    df.columns = ["countryname"] + [f"pop{header}" for header in headers[1:]]
    long = df.melt(id_vars=["countryname"], var_name="year_token", value_name="pop")
    long["year"] = pd.to_numeric(long["year_token"].astype("string").str.removeprefix("pop"), errors="coerce")

    factor = long["pop"].astype("string").str[-1]
    pop_1 = long["pop"].astype("string")
    unit_mask = factor.isin(["B", "M", "k"])
    pop_1 = pop_1.where(~unit_mask, pop_1.str.slice(0, -1))
    pop_1 = pd.to_numeric(pop_1, errors="coerce")
    long["Gapminder_pop"] = pd.Series(np.nan, index=long.index, dtype="float64")
    long.loc[factor.eq("B"), "Gapminder_pop"] = pd.to_numeric(pop_1[factor.eq("B")] * 1000, errors="coerce").astype("float64")
    long.loc[factor.eq("M"), "Gapminder_pop"] = pd.to_numeric(pop_1[factor.eq("M")], errors="coerce").astype("float64")
    long.loc[factor.eq("k"), "Gapminder_pop"] = pd.to_numeric(pop_1[factor.eq("k")] / 1000, errors="coerce").astype("float64")
    long.loc[~factor.isin(["B", "M", "k"]), "Gapminder_pop"] = pd.to_numeric(pop_1[~factor.isin(["B", "M", "k"])] / 1_000_000, errors="coerce").astype("float64")

    long["countryname"] = long["countryname"].astype("string").str.strip()
    long["ISO3"] = long["countryname"].map(lookup)
    manual_iso = {
        "Congo, Dem. Rep.": "COD",
        "Congo, Rep.": "COG",
        "Cote d'Ivoire": "CIV",
        "Hong Kong, China": "HKG",
        "Kyrgyz Republic": "KGZ",
        "Lao": "LAO",
        "Micronesia, Fed. Sts.": "FSM",
        "North Macedonia": "MKD",
        "Russia": "RUS",
        "Slovak Republic": "SVK",
        "St. Kitts and Nevis": "KNA",
        "St. Lucia": "LCA",
        "St. Vincent and the Grenadines": "VCT",
        "UAE": "ARE",
        "UK": "GBR",
        "USA": "USA",
    }
    for country, iso3 in manual_iso.items():
        long.loc[long["countryname"] == country, "ISO3"] = iso3

    long = long.loc[long["ISO3"].notna() & long["year"].notna() & long["year"].le(2030), ["ISO3", "year", "Gapminder_pop"]].copy()
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("int16")
    long["Gapminder_pop"] = pd.to_numeric(long["Gapminder_pop"], errors="coerce").astype("float32")
    long = _sort_keys(long)
    out_path = clean_dir / "aggregators" / "Gapminder" / "Gapminder.dta"
    _save_dta(long, out_path)
    return long
__all__ = ["clean_gapminder"]
