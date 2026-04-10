from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_imf_hdd(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    df = _read_excel_compat(raw_dir / "aggregators" / "IMF" / "IMF_HDD.xls", sheet_name="DEBT1", header=None, dtype=str)
    df = df.drop(index=1).reset_index(drop=True)
    df = df.replace("no data", "")
    first_row = df.iloc[0].tolist()
    df.columns = ["A"] + [f"yeardate_{item}" for item in first_row[1:]]
    df = df.iloc[1:189].reset_index(drop=True)
    df = df.melt(id_vars=["A"], var_name="year_col", value_name="GOVDEBT_GDP")
    df = df.rename(columns={"A": "countryname"})
    df["year"] = pd.to_numeric(df["year_col"].astype(str).str.removeprefix("yeardate_"), errors="coerce").astype("int16")
    df["GOVDEBT_GDP"] = pd.to_numeric(df["GOVDEBT_GDP"], errors="coerce")
    df = df.drop(columns=["year_col"])

    df["countryname"] = df["countryname"].replace(
        {
            "Bahamas, The": "Bahamas",
            "Brunei Darussalam": "Brunei",
            "Cabo Verde": "Cape Verde",
            "China, People's Republic of": "China",
            "Congo, Dem. Rep. of the": "Democratic Republic of the Congo",
            "Congo, Republic of ": "Republic of the Congo",
            "Côte d'Ivoire": "Ivory Coast",
            "Gambia, The": "Gambia",
            "Hong Kong SAR": "Hong Kong",
            "Korea, Republic of": "South Korea",
            "Kyrgyz Republic": "Kyrgyzstan",
            "Lao P.D.R.": "Laos",
            "Micronesia, Fed. States of": "Micronesia (Federated States of)",
            "North Macedonia ": "Macedonia",
            "Slovak Republic": "Slovakia",
            "South Sudan, Republic of": "South Sudan",
            "São Tomé and Príncipe": "Sao Tome and Principe",
            "Taiwan Province of China": "Taiwan",
            "Türkiye, Republic of": "Turkey",
        }
    )
    countrylist = _load_dta(helper_dir / "countrylist.dta")[["countryname", "ISO3"]].copy()
    df = df.merge(countrylist, on="countryname", how="left")
    df = df.loc[df["year"].notna(), ["ISO3", "year", "GOVDEBT_GDP"]].copy()
    df = df.rename(columns={"GOVDEBT_GDP": "IMF_HDD_govdebt_GDP"})
    df = _sort_keys(df)
    if df.duplicated(["ISO3", "year"]).any():
        raise sh.PipelineRuntimeError("ISO3 year do not uniquely identify observations", code=459)
    out_path = clean_dir / "aggregators" / "IMF" / "IMF_HDD.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_imf_hdd"]
