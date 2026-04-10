from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_aus_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = pd.read_csv(raw_dir / "country_level" / "AUS_1.csv", header=None)
    df = df.iloc[4:].reset_index(drop=True).copy()
    df.columns = [f"v{i}" for i in range(1, len(df.columns) + 1)]
    for col in ["v2", "v3", "v4", "v6", "v7", "v8", "v10"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["v3"] = pd.to_numeric(df["v3"], errors="coerce") * 2
    df["v2"] = pd.to_numeric(df["v2"], errors="coerce").where(pd.to_numeric(df["v2"], errors="coerce").notna(), pd.to_numeric(df["v3"], errors="coerce"))
    df = df.rename(
        columns={
            "v1": "year",
            "v2": "CS1_nGDP",
            "v4": "CS1_rGDP_USD",
            "v5": "CS1_deflator",
            "v7": "CS1_rGDP_pc_USD",
            "v8": "CS1_pop",
            "v9": "CS1_CPI",
        }
    )
    for col in ["CS1_CPI", "CS1_deflator"]:
        df.loc[pd.to_numeric(df[col], errors="coerce").eq(0), col] = pd.NA
    df["CS1_pop"] = pd.to_numeric(df["CS1_pop"], errors="coerce") / 1_000_000
    df["ISO3"] = "AUS"
    df = df[["ISO3", "year", "CS1_nGDP", "CS1_rGDP_USD", "CS1_deflator", "CS1_rGDP_pc_USD", "CS1_pop", "CS1_CPI"]].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    for col in ["CS1_nGDP", "CS1_rGDP_USD", "CS1_deflator", "CS1_rGDP_pc_USD", "CS1_pop", "CS1_CPI"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    prev_cpi = pd.to_numeric(df["CS1_CPI"], errors="coerce").groupby(df["ISO3"]).shift(1)
    prev_year = pd.to_numeric(df["year"], errors="coerce").groupby(df["ISO3"]).shift(1)
    year_num = pd.to_numeric(df["year"], errors="coerce")
    infl = (pd.to_numeric(df["CS1_CPI"], errors="coerce") - prev_cpi) / prev_cpi * 100
    df["CS1_infl"] = infl.where(prev_cpi.notna() & year_num.eq(prev_year + 1)).astype("float32")
    df = _sort_keys(df)
    out_path = clean_dir / "country_level" / "AUS_1.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_aus_1"]
