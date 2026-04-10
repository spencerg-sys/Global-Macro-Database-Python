from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_rr(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = _read_excel_compat(raw_dir / "aggregators" / "RR" / "Reinhart-Rogoff.xlsx")
    df = df.iloc[1:].copy()

    debt1 = next(col for col in df.columns if "SOVEREIGN EXTERNAL DEBT 1" in str(col))
    debt2 = next(col for col in df.columns if "SOVEREIGN EXTERNAL DEBT 2" in str(col))
    df = df.rename(
        columns={
            "CC3": "ISO3",
            "Country": "country_name",
            "Year": "year",
            "Banking Crisis ": "RR_crisisB",
            "Domestic_Debt_In_Default": "RR_crisisDD",
            debt1: "RR_crisisED1",
            debt2: "RR_crisisED2",
            "Currency Crises": "RR_crisisC",
        }
    )
    df = df[["ISO3", "country_name", "year", "RR_crisisB", "RR_crisisDD", "RR_crisisED1", "RR_crisisED2", "RR_crisisC"]].copy()
    df["ISO3"] = df["ISO3"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    df = _sort_keys(df)

    for col in ["RR_crisisB", "RR_crisisDD", "RR_crisisED1", "RR_crisisED2", "RR_crisisC"]:
        series = df[col].astype("string").str.strip()
        series = series.replace("n/a", pd.NA)
        series.loc[series.eq("2").fillna(False)] = "1"
        series.loc[series.str.contains("Hyperinflation", na=False)] = "1"
        numeric = pd.to_numeric(series, errors="coerce")
        begin = pd.Series(np.nan, index=df.index, dtype="float32")
        begin.loc[numeric.notna()] = np.float32(0)
        prev = numeric.groupby(df["ISO3"]).shift(1)
        first = df.groupby("ISO3").cumcount().eq(0)
        begin.loc[(numeric == 1) & ((numeric.ne(prev)) | first)] = np.float32(1)
        df[col] = begin.astype("float32")

    df = df.drop(columns=["country_name"], errors="ignore")
    df = df[["ISO3", "year", "RR_crisisB", "RR_crisisDD", "RR_crisisED1", "RR_crisisED2", "RR_crisisC"]]
    out_path = clean_dir / "aggregators" / "RR" / "RR.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_rr"]
