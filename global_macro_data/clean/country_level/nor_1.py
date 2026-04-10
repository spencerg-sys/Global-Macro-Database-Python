from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_nor_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = pd.read_excel(raw_dir / "country_level" / "NOR_1.xlsx", sheet_name="NOR")
    df["ISO3"] = "NOR"
    df = df.rename(columns={col: f"CS1_{col}" for col in df.columns if col not in {"ISO3", "year"}})
    df = df.drop(columns=["CS1_nGDP_pc"], errors="ignore")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    for col in ["CS1_nGDP", "CS1_rGDP", "CS1_rGDP_pc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df = df[["ISO3", "year", "CS1_nGDP", "CS1_rGDP", "CS1_rGDP_pc"]].copy()
    df = _sort_keys(df)
    out_path = clean_dir / "country_level" / "NOR_1.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_nor_1"]
