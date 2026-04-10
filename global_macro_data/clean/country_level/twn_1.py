from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_twn_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = _read_excel_compat(raw_dir / "country_level" / "TWN_1.xls", header=None)
    df = df.iloc[4:321, :8].copy()
    df.columns = ["year", "CS1_pop", "CS1_USDfx", "drop_growth", "CS1_nGDP", "CS1_nGDP_USD", "CS1_nGDP_pc", "CS1_nGDP_pc_USD"]
    df["ISO3"] = "TWN"
    year_str = df["year"].astype("string").str.slice(0, 4)
    df = df.loc[year_str.str.slice(0, 2).isin(["19", "20"])].copy()
    df["year"] = pd.to_numeric(year_str.loc[df.index], errors="coerce")
    for col in ["CS1_pop", "CS1_USDfx", "CS1_nGDP", "CS1_nGDP_USD", "CS1_nGDP_pc", "CS1_nGDP_pc_USD", "drop_growth"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["CS1_pop"] = pd.to_numeric(df["CS1_pop"], errors="coerce") / 1_000_000
    df = df.drop(columns=["drop_growth", "CS1_nGDP_pc", "CS1_nGDP_pc_USD"], errors="ignore")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    df["CS1_pop"] = pd.to_numeric(df["CS1_pop"], errors="coerce").astype("float64")
    df["CS1_USDfx"] = pd.to_numeric(df["CS1_USDfx"], errors="coerce").astype("float64")
    df["CS1_nGDP"] = pd.to_numeric(df["CS1_nGDP"], errors="coerce").astype("int32")
    df["CS1_nGDP_USD"] = pd.to_numeric(df["CS1_nGDP_USD"], errors="coerce").astype("int32")
    df = df[["ISO3", "year", "CS1_pop", "CS1_USDfx", "CS1_nGDP", "CS1_nGDP_USD"]].copy()
    df = _sort_keys(df)
    out_path = clean_dir / "country_level" / "TWN_1.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_twn_1"]
