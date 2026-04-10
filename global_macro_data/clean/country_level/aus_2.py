from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_aus_2(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    path = raw_dir / "country_level" / "AUS_2.xlsx"

    def _aus2_value(value: object) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if text in {"", "<NA>", "nan", "None"}:
            return np.nan
        return float(text)

    def _read_sheet(sheet: str) -> pd.DataFrame:
        df = pd.read_excel(path, sheet_name=sheet, dtype=str)
        df = df.dropna(how="all").dropna(axis=1, how="all").copy()
        for col in df.columns:
            if str(col) == "year":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = df[col].map(_aus2_value)
        return df

    master = _read_sheet("nGDP")
    master["ISO3"] = "AUS"
    master = master[["ISO3", "year", "nGDP"]].copy()

    for sheet in ["Money", "Gov"]:
        current = _read_sheet(sheet)
        current["ISO3"] = "AUS"
        keep_cols = ["ISO3", "year"] + [col for col in current.columns if col not in {"ISO3", "year"}]
        master = current[keep_cols].merge(master, on=["ISO3", "year"], how="outer")

    n_gdp = pd.to_numeric(master["nGDP"], errors="coerce")
    master["govtax_GDP"] = pd.to_numeric(master["TAXES"], errors="coerce") / n_gdp
    master["govrev_GDP"] = pd.to_numeric(master["REVENUE"], errors="coerce") / n_gdp
    master["govdef_GDP"] = pd.to_numeric(master["DEFICIT"], errors="coerce") / n_gdp

    master = master.rename(columns={"TAXES": "govtax", "REVENUE": "govrev", "DEFICIT": "govdef"})
    master["govexp"] = pd.to_numeric(master["govexp"], errors="coerce") / 100
    master["govrev"] = pd.to_numeric(master["govrev"], errors="coerce") / 100
    master["govdef"] = pd.to_numeric(master["govdef"], errors="coerce") / 100
    master["govtax"] = pd.to_numeric(master["govtax"], errors="coerce") / 100

    master = master.rename(columns={col: f"CS2_{col}" for col in master.columns if col not in {"ISO3", "year"}})
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    for col in [
        "CS2_govdebt_GDP",
        "CS2_govrev",
        "CS2_govexp",
        "CS2_govdef",
        "CS2_govtax",
        "CS2_M0",
        "CS2_M1",
        "CS2_M2",
        "CS2_M3",
        "CS2_nGDP",
    ]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
    for col in ["CS2_govtax_GDP", "CS2_govrev_GDP", "CS2_govdef_GDP"]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float32")

    master = master[
        [
            "ISO3",
            "year",
            "CS2_govdebt_GDP",
            "CS2_govrev",
            "CS2_govexp",
            "CS2_govdef",
            "CS2_govtax",
            "CS2_M0",
            "CS2_M1",
            "CS2_M2",
            "CS2_M3",
            "CS2_nGDP",
            "CS2_govtax_GDP",
            "CS2_govrev_GDP",
            "CS2_govdef_GDP",
        ]
    ].copy()
    for col in [
        "CS2_govdebt_GDP",
        "CS2_govrev",
        "CS2_govexp",
        "CS2_govdef",
        "CS2_govtax",
        "CS2_M0",
        "CS2_M1",
        "CS2_M2",
        "CS2_M3",
        "CS2_nGDP",
    ]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
    for col in ["CS2_govtax_GDP", "CS2_govrev_GDP", "CS2_govdef_GDP"]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float32")
    master = _sort_keys(master)
    out_path = clean_dir / "country_level" / "AUS_2.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_aus_2"]
