from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_madisson(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    path = raw_dir / "aggregators" / "MAD" / "maddison2023.dta"

    df = _load_dta(path).rename(
        columns={
            "countrycode": "ISO3",
            "gdppc": "MAD_rGDP_pc_USD",
            "pop": "MAD_pop",
        }
    )[["ISO3", "year", "MAD_rGDP_pc_USD", "MAD_pop"]].copy()

    df["MAD_pop"] = pd.to_numeric(df["MAD_pop"], errors="coerce") / 1000
    df = df.loc[pd.to_numeric(df["year"], errors="coerce").ge(1252)].copy()
    df.loc[
        df["ISO3"].eq("MEX") & pd.to_numeric(df["year"], errors="coerce").le(1895),
        "MAD_rGDP_pc_USD",
    ] = np.nan

    df = df.sort_values(["ISO3", "year"]).reset_index(drop=True)
    prev_gdp = df.groupby("ISO3")["MAD_rGDP_pc_USD"].shift(1)
    prev_year = df.groupby("ISO3")["year"].shift(1)
    gap = (
        pd.to_numeric(df["year"], errors="coerce") - pd.to_numeric(prev_year, errors="coerce")
    ).where(prev_gdp.notna() & pd.to_numeric(df["MAD_rGDP_pc_USD"], errors="coerce").notna())
    df.loc[gap.ne(1), "MAD_pop"] = np.nan

    prev_gdp = df.groupby("ISO3")["MAD_rGDP_pc_USD"].shift(1)
    prev_year = df.groupby("ISO3")["year"].shift(1)
    gap = (
        pd.to_numeric(df["year"], errors="coerce") - pd.to_numeric(prev_year, errors="coerce")
    ).where(prev_gdp.notna() & pd.to_numeric(df["MAD_rGDP_pc_USD"], errors="coerce").notna())
    df.loc[gap.ne(1), "MAD_rGDP_pc_USD"] = np.nan

    df["ISO3"] = df["ISO3"].astype("object")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    df["MAD_rGDP_pc_USD"] = pd.to_numeric(df["MAD_rGDP_pc_USD"], errors="coerce").astype("float64")
    df["MAD_pop"] = pd.to_numeric(df["MAD_pop"], errors="coerce").astype("float64")
    df = _sort_keys(df)
    out_path = clean_dir / "aggregators" / "MAD" / "Madisson.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_madisson"]
