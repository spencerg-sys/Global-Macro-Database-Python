from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_aut_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = pd.read_excel(raw_dir / "country_level" / "AUT_1.xlsx", header=None)
    df = df.iloc[3:, [0, 4, 5]].copy()
    df.columns = ["year", "CS1_rGDP", "CS1_rGDP_pc"]
    df["ISO3"] = "AUT"
    df["CS1_rGDP_pc"] = df["CS1_rGDP_pc"].astype(str).replace("433-3", "433.3")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    df["CS1_rGDP"] = pd.to_numeric(df["CS1_rGDP"], errors="coerce").astype("float64")
    df["CS1_rGDP_pc"] = pd.to_numeric(df["CS1_rGDP_pc"], errors="coerce").astype("float64")
    df = df[["ISO3", "year", "CS1_rGDP", "CS1_rGDP_pc"]].copy()
    df = _sort_keys(df)
    out_path = clean_dir / "country_level" / "AUT_1.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_aut_1"]
