from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_clio(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    path = raw_dir / "aggregators" / "CLIO" / "CLIO.xlsx"

    lookup = _country_name_lookup(helper_dir)
    df = _read_excel_compat(path, sheet_name="Data Long Format")
    df = df.rename(columns={"country.name": "countryname"})
    df["countryname"] = df["countryname"].astype("string").str.strip()
    df["ISO3"] = df["countryname"].map(lookup)
    df.loc[df["countryname"] == "Russia", "ISO3"] = "RUS"
    df = df.loc[df["ISO3"].notna(), ["ISO3", "year", "value"]].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    df["CLIO_ltrate"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")
    df = df[["ISO3", "year", "CLIO_ltrate"]].copy()
    df = _sort_keys(df)
    out_path = clean_dir / "aggregators" / "CLIO" / "CLIO.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_clio"]
