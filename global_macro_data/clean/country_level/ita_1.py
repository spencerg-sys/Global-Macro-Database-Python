from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_ita_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = pd.read_excel(raw_dir / "country_level" / "ITA_1.xlsx", sheet_name="ITA")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["rGDP_pc"] = pd.to_numeric(df["rGDP_pc"], errors="coerce") * 1000
    df["pop"] = pd.to_numeric(df["pop"], errors="coerce") / 1000
    n_gdp = pd.to_numeric(df["nGDP"], errors="coerce")
    df["cons_GDP"] = pd.to_numeric(df["cons"], errors="coerce") / n_gdp * 100
    df["exports_GDP"] = pd.to_numeric(df["exports"], errors="coerce") / n_gdp * 100
    df["finv_GDP"] = pd.to_numeric(df["finv"], errors="coerce") / n_gdp * 100
    df["inv_GDP"] = pd.to_numeric(df["inv"], errors="coerce") / n_gdp * 100
    df["ISO3"] = "ITA"
    df = df.rename(columns={col: f"CS1_{col}" for col in df.columns if col not in {"ISO3", "year"}})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    for col in [
        "CS1_nGDP",
        "CS1_deflator",
        "CS1_rGDP",
        "CS1_pop",
        "CS1_rGDP_pc",
        "CS1_cons",
        "CS1_exports",
        "CS1_finv",
        "CS1_inv",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in ["CS1_cons_GDP", "CS1_exports_GDP", "CS1_finv_GDP", "CS1_inv_GDP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df = df[
        [
            "ISO3",
            "year",
            "CS1_nGDP",
            "CS1_deflator",
            "CS1_rGDP",
            "CS1_pop",
            "CS1_rGDP_pc",
            "CS1_cons",
            "CS1_exports",
            "CS1_finv",
            "CS1_inv",
            "CS1_cons_GDP",
            "CS1_exports_GDP",
            "CS1_finv_GDP",
            "CS1_inv_GDP",
        ]
    ].copy()
    df = _sort_keys(df)
    out_path = clean_dir / "country_level" / "ITA_1.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_ita_1"]
