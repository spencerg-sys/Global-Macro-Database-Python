from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_dallasfed_hpi(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    path = raw_dir / "aggregators" / "DallasFed" / "hp2304.xlsx"

    lookup = _country_name_lookup(helper_dir)

    def _reshape_sheet(sheet_name: str, value_name: str) -> pd.DataFrame:
        df = _read_excel_compat(path, sheet_name=sheet_name)
        df = df.iloc[1:].copy()
        keep_cols = [df.columns[0]]
        for col in df.columns[1:]:
            name = str(col)
            if name.startswith("Aggregate") or name.startswith("Unnamed"):
                continue
            keep_cols.append(col)
        df = df[keep_cols].copy()
        df = df.rename(columns={df.columns[0]: "date"})
        long = df.melt(id_vars="date", var_name="countryname", value_name=value_name)
        long["date"] = long["date"].astype("string").str.strip()
        long = long.loc[long["date"].notna() & long["date"].ne("")].copy()
        long["countryname"] = long["countryname"].astype("string").str.strip().replace(
            {
                "NewZealand": "New Zealand",
                "SAfrica": "South Africa",
                "S. Africa": "South Africa",
                "SKorea": "South Korea",
                "S. Korea": "South Korea",
                "UK": "United Kingdom",
                "US": "United States",
            }
        )
        long["ISO3"] = long["countryname"].map(lookup)
        long = long.loc[long["ISO3"].notna(), ["ISO3", "date", value_name]].copy()
        long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
        return long

    hpi = _reshape_sheet("HPI", "DALLASFED_HPI")
    rhpi = _reshape_sheet("RHPI", "DALLASFED_rHPI")
    merged = rhpi.merge(hpi, on=["ISO3", "date"], how="inner")
    merged["year"] = pd.to_numeric(merged["date"].astype("string").str.slice(0, 4), errors="coerce")
    merged["quarter"] = merged["date"].astype("string").str.slice(6, 7)
    merged = merged.loc[merged["quarter"].eq("4")].copy()

    for col in ["DALLASFED_HPI", "DALLASFED_rHPI"]:
        temp = merged[col].where(merged["year"].eq(2010))
        # The reference pipeline stores egen/max output as float, then uses it in a double expression.
        scaler = temp.groupby(merged["ISO3"]).transform("max").astype("float32")
        merged[col] = (
            pd.to_numeric(merged[col], errors="coerce")
            * (100 / pd.to_numeric(scaler, errors="coerce").astype("float64"))
        )

    out = merged[["ISO3", "year", "DALLASFED_rHPI", "DALLASFED_HPI"]].copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int16")
    out["DALLASFED_rHPI"] = pd.to_numeric(out["DALLASFED_rHPI"], errors="coerce").astype("float64")
    out["DALLASFED_HPI"] = pd.to_numeric(out["DALLASFED_HPI"], errors="coerce").astype("float64")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int16")
    out["DALLASFED_rHPI"] = pd.to_numeric(out["DALLASFED_rHPI"], errors="coerce").astype("float64")
    out["DALLASFED_HPI"] = pd.to_numeric(out["DALLASFED_HPI"], errors="coerce").astype("float64")
    out = _sort_keys(out)
    out_path = clean_dir / "aggregators" / "DallasFed" / "DALLASFED_HPI.dta"
    _save_dta(out, out_path)
    return out
__all__ = ["clean_dallasfed_hpi"]
