from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_md(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)

    def _md_f32(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").astype("float32").astype("float64")

    md1 = _load_dta(raw_dir / "aggregators" / "MD" / "MD_1.dta")
    md1 = md1[["countryname", "year", "pop", "logexppercap", "logrevpercap", "yield", "loggdppercap"]].copy()
    keep_countries = {"france", "netherlands", "portugal", "spain", "sweden"}
    md1 = md1.loc[md1["countryname"].astype(str).isin(keep_countries)].copy()
    md1["pop"] = pd.to_numeric(md1["pop"], errors="coerce") / 1_000_000
    md1["exppercap"] = _md_f32(np.exp(pd.to_numeric(md1["logexppercap"], errors="coerce").astype("float64")))
    md1["revercap"] = _md_f32(np.exp(pd.to_numeric(md1["logrevpercap"], errors="coerce").astype("float64")))
    md1["gdppercap"] = _md_f32(np.exp(pd.to_numeric(md1["loggdppercap"], errors="coerce").astype("float64")))
    md1["expenditure"] = _md_f32(pd.to_numeric(md1["pop"], errors="coerce") * pd.to_numeric(md1["exppercap"], errors="coerce"))
    md1["gdp"] = _md_f32(pd.to_numeric(md1["pop"], errors="coerce") * pd.to_numeric(md1["gdppercap"], errors="coerce"))
    md1["revenue"] = _md_f32(pd.to_numeric(md1["pop"], errors="coerce") * pd.to_numeric(md1["revercap"], errors="coerce"))
    md1["MD_govexp_GDP"] = _md_f32(
        pd.to_numeric(md1["expenditure"], errors="coerce") / pd.to_numeric(md1["gdp"], errors="coerce") * 100
    )
    md1["MD_govrev_GDP"] = _md_f32(
        pd.to_numeric(md1["revenue"], errors="coerce") / pd.to_numeric(md1["gdp"], errors="coerce") * 100
    )
    md1["MD_pop"] = pd.to_numeric(md1["pop"], errors="coerce")
    md1["MD_rGDP"] = pd.to_numeric(md1["gdp"], errors="coerce")
    md1["MD_ltrate"] = pd.to_numeric(md1["yield"], errors="coerce") / 100
    md1["ISO3"] = md1["countryname"].replace(
        {"france": "FRA", "netherlands": "NLD", "portugal": "PRT", "spain": "ESP", "sweden": "SWE"}
    )
    md1 = md1[["ISO3", "year", "MD_pop", "MD_rGDP", "MD_govexp_GDP", "MD_govrev_GDP", "MD_ltrate"]].copy()

    md2 = _load_dta(raw_dir / "aggregators" / "MD" / "MD_2.dta")
    md2 = md2[["year", "ctycode", "taxgdp"]].copy()
    md2["ctycode"] = md2["ctycode"].replace(
        {
            "BUL": "BGR",
            "DEN": "DNK",
            "GER": "DEU",
            "GRE": "GRC",
            "JAP": "JPN",
            "NET": "NLD",
            "NZD": "NZL",
            "POR": "PRT",
            "ROM": "ROU",
            "SPA": "ESP",
            "SWI": "CHE",
            "UK": "GBR",
            "URU": "URY",
        }
    )
    md2 = md2.rename(columns={"ctycode": "ISO3", "taxgdp": "MD_govtax_GDP"})
    merged = md2.merge(md1, on=["ISO3", "year"], how="outer")

    merged["MD_govtax_GDP"] = pd.to_numeric(merged["MD_govtax_GDP"], errors="coerce") * 100
    merged["MD_govrev_GDP"] = _md_f32(pd.to_numeric(merged["MD_govrev_GDP"], errors="coerce") * 10)
    merged["MD_govexp_GDP"] = pd.to_numeric(merged["MD_govexp_GDP"], errors="coerce") * 10
    merged["MD_govrev_GDP"] = _md_f32(pd.to_numeric(merged["MD_govrev_GDP"], errors="coerce") * 10)

    merged["year"] = pd.to_numeric(merged["year"], errors="coerce").astype("float64")
    merged["MD_govtax_GDP"] = pd.to_numeric(merged["MD_govtax_GDP"], errors="coerce").astype("float32")
    merged["MD_pop"] = pd.to_numeric(merged["MD_pop"], errors="coerce").astype("float64")
    merged["MD_rGDP"] = pd.to_numeric(merged["MD_rGDP"], errors="coerce").astype("float32")
    merged["MD_govexp_GDP"] = pd.to_numeric(merged["MD_govexp_GDP"], errors="coerce").astype("float32")
    merged["MD_govrev_GDP"] = pd.to_numeric(merged["MD_govrev_GDP"], errors="coerce").astype("float32")
    merged["MD_ltrate"] = pd.to_numeric(merged["MD_ltrate"], errors="coerce").astype("float32")
    merged = merged[["ISO3", "year", "MD_govtax_GDP", "MD_pop", "MD_rGDP", "MD_govexp_GDP", "MD_govrev_GDP", "MD_ltrate"]].copy()
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce").astype("float64")
    merged["MD_govtax_GDP"] = pd.to_numeric(merged["MD_govtax_GDP"], errors="coerce").astype("float32")
    merged["MD_pop"] = pd.to_numeric(merged["MD_pop"], errors="coerce").astype("float64")
    merged["MD_rGDP"] = pd.to_numeric(merged["MD_rGDP"], errors="coerce").astype("float32")
    merged["MD_govexp_GDP"] = pd.to_numeric(merged["MD_govexp_GDP"], errors="coerce").astype("float32")
    merged["MD_govrev_GDP"] = pd.to_numeric(merged["MD_govrev_GDP"], errors="coerce").astype("float32")
    merged["MD_ltrate"] = pd.to_numeric(merged["MD_ltrate"], errors="coerce").astype("float32")
    merged = _sort_keys(merged)
    out_path = clean_dir / "aggregators" / "MD" / "MD.dta"
    _save_dta(merged, out_path)
    return merged
__all__ = ["clean_md"]
