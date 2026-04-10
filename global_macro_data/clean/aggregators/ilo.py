from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_ilo(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    path = raw_dir / "aggregators" / "ILO" / "ILO.dta"

    df = _load_dta(path)
    df = df.loc[
        df["sex_label"].astype(str).eq("Sex: Total")
        & df["classif1_label"].astype(str).eq("Age (Youth, adults): 15+")
    ].copy()
    df["indicator"] = df["source_label"].astype(str).str.split("-", n=1).str[0].str.strip()

    country_col = "ref_area" if "ref_area" in df.columns else "ref_area_label"
    df = df[[country_col, "indicator", "obs_value", "time"]].copy()
    wide = df.pivot(index=["time", country_col], columns="indicator", values="obs_value").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={country_col: "countryname"})

    wide["countryname"] = wide["countryname"].astype(str).replace(
        {
            "Cabo Verde": "Cape Verde",
            "Bolivia (Plurinational State of)": "Bolivia",
            "Brunei Darussalam": "Brunei",
            "Congo": "Republic of the Congo",
            "Congo, Democratic Republic of the": "Democratic Republic of the Congo",
            "Czechia": "Czech Republic",
            "C\u00f4te d'Ivoire": "Ivory Coast",
            "Hong Kong, China": "Hong Kong",
            "Iran (Islamic Republic of)": "Iran",
            "Lao People's Democratic Republic": "Laos",
            "Macao, China": "Macau",
            "North Macedonia": "Macedonia",
            "Syrian Arab Republic": "Syria",
            "Taiwan, China": "Taiwan",
            "Tanzania, United Republic of": "Tanzania",
            "T\u00fcrkiye": "Turkey",
            "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
            "United States of America": "United States",
            "Venezuela (Bolivarian Republic of)": "Venezuela",
            "Viet Nam": "Vietnam",
        }
    )

    lookup = _country_name_lookup(helper_dir)
    wide["ISO3"] = wide["countryname"].map(lookup)
    wide = wide.loc[wide["ISO3"].notna()].copy()

    rowmax_cols = [col for col in ["HIES", "HS", "ILO", "LFS", "OE", "PC"] if col in wide.columns]
    wide["ILO_unemp"] = wide[rowmax_cols].apply(pd.to_numeric, errors="coerce").max(axis=1, skipna=True)

    out = wide[["ISO3", "time", "ILO_unemp"]].rename(columns={"time": "year"}).copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int16")
    out["ILO_unemp"] = pd.to_numeric(out["ILO_unemp"], errors="coerce").astype("float32")
    out = _sort_keys(out[["ISO3", "year", "ILO_unemp"]])
    out_path = clean_dir / "aggregators" / "ILO" / "ILO.dta"
    _save_dta(out, out_path)
    return out
__all__ = ["clean_ilo"]
