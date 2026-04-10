from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_fra_2(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = pd.read_excel(raw_dir / "country_level" / "FRA_2.xlsx", sheet_name="annuel", header=None)
    df = df.iloc[2:218, [0, 1, 3, 4]].copy()
    df.columns = ["year", "cbrate", "strate", "ltrate"]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for col in ["cbrate", "strate", "ltrate"]:
        df[col] = df[col].map(lambda value: np.nan if pd.isna(value) else float(format(float(value), ".16g")))
    df["ISO3"] = "FRA"
    df = df.rename(columns={col: f"CS2_{col}" for col in ["cbrate", "strate", "ltrate"]})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    for col in ["CS2_cbrate", "CS2_strate", "CS2_ltrate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df = df[["ISO3", "year", "CS2_cbrate", "CS2_strate", "CS2_ltrate"]].copy()
    for col in ["CS2_cbrate", "CS2_strate", "CS2_ltrate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df = _sort_keys(df)
    out_path = clean_dir / "country_level" / "FRA_2.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_fra_2"]
