from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_che_2(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    national = pd.read_excel(raw_dir / "country_level" / "CHE_2.xlsx", sheet_name="national_accounts")
    rates = pd.read_excel(raw_dir / "country_level" / "CHE_2.xlsx", sheet_name="rates")
    merged = rates.merge(national, on="year", how="outer")
    merged["ISO3"] = "CHE"
    merged["pop"] = pd.to_numeric(merged["pop"], errors="coerce") / 1000
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce").astype("int16")
    value_cols = [col for col in merged.columns if col not in {"ISO3", "year"}]
    merged = merged.rename(columns={col: f"CS2_{col}" for col in value_cols})
    for col in [c for c in merged.columns if c not in {"ISO3", "year"}]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float64")
    merged = merged[["ISO3", "year"] + [col for col in merged.columns if col not in {"ISO3", "year"}]].copy()
    merged = _sort_keys(merged)
    out_path = clean_dir / "country_level" / "CHE_2.dta"
    _save_dta(merged, out_path)
    return merged
__all__ = ["clean_che_2"]
