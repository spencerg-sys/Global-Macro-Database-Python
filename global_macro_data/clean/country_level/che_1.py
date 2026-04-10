from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_che_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)

    money = _read_excel_compat(raw_dir / "country_level" / "CHE_1a.xls", sheet_name="T 2.2", header=None, dtype=str)
    money = money.iloc[3:48, [0, 1, 3]].copy()
    money.columns = ["year", "M1", "M3"]
    money["year"] = pd.to_numeric(money["year"].astype(str).str.extract(r"(\d{4})", expand=False), errors="coerce")
    for col in ["M1", "M3"]:
        money[col] = pd.to_numeric(money[col].astype(str).replace("-", pd.NA), errors="coerce")
    money = money.loc[money["year"].notna()].copy()

    base_df = _read_excel_compat(raw_dir / "country_level" / "CHE_1a.xls", sheet_name="T 1.3", header=None, dtype=str)
    base_df = base_df.iloc[3:48, [0, 3]].copy()
    base_df.columns = ["year", "M0"]
    base_df["year"] = pd.to_numeric(base_df["year"].astype(str).str.extract(r"(\d{4})", expand=False), errors="coerce")
    base_df["M0"] = pd.to_numeric(base_df["M0"].astype(str).replace("-", pd.NA), errors="coerce")
    base_df = base_df.loc[base_df["year"].notna()].copy()

    rates = _read_excel_compat(raw_dir / "country_level" / "CHE_1b.xls", sheet_name="1.1_A", header=None, dtype=str)
    rates = rates.iloc[6:106, [0, 1, 2]].copy()
    rates.columns = ["year", "cbrate", "strate"]
    rates["year"] = pd.to_numeric(rates["year"].astype(str).str.extract(r"(\d{4})", expand=False), errors="coerce")
    for col in ["cbrate", "strate"]:
        rates[col] = pd.to_numeric(rates[col].astype(str).replace("-", pd.NA), errors="coerce")
    rates = rates.loc[rates["year"].notna()].copy()

    merged = base_df.merge(money, on="year", how="outer")
    merged = rates.merge(merged, on="year", how="outer")
    merged["ISO3"] = "CHE"
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce").astype("int16")
    merged = merged.rename(columns={col: f"CS1_{col}" for col in ["cbrate", "strate", "M0", "M1", "M3"]})
    for col in ["CS1_cbrate", "CS1_strate", "CS1_M0", "CS1_M1", "CS1_M3"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float64")
    merged = merged[["ISO3", "year", "CS1_cbrate", "CS1_strate", "CS1_M0", "CS1_M1", "CS1_M3"]].copy()
    merged = _sort_keys(merged)
    out_path = clean_dir / "country_level" / "CHE_1.dta"
    _save_dta(merged, out_path)
    return merged
__all__ = ["clean_che_1"]
