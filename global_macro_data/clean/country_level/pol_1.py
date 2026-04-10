from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_pol_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = _load_dta(raw_dir / "country_level" / "POL_1.dta")
    df["value"] = df["value"].replace("NA", "")
    df["period"] = pd.to_numeric(df["period"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["period", "value", "series_code", "ISO3"]].copy()
    mapping = {
        "APF-57": "CS1_CA_GDP",
        "ANA-31": "CS1_cons",
        "ANA-39": "CS1_exports",
        "ANA-40": "CS1_imports",
        "ANA-37": "CS1_finv",
        "ANA-35": "CS1_inv",
        "APF-9": "CS1_govdebt",
        "APF-8": "CS1_govdef",
        "APF-6": "CS1_govrev",
        "APF-7": "CS1_govexp",
        "AMON-6": "CS1_M3",
        "ANA-7": "CS1_nGDP",
        "ANA-45": "CS1_rGDP",
        "AMON-15": "CS1_USDfx",
        "AMON-11": "CS1_cbrate",
        "AMON-9": "CS1_M0",
        "APRI-16": "CS1_CPI",
        "APOP-6": "CS1_pop",
    }
    df["series_code"] = df["series_code"].astype(str).map(mapping).fillna(df["series_code"].astype(str))
    df = df.loc[df["series_code"].astype(str).str.startswith("CS1"), ["ISO3", "period", "series_code", "value"]].copy()
    wide = df.pivot(index=["ISO3", "period"], columns="series_code", values="value").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"period": "year"})
    wide["CS1_pop"] = pd.to_numeric(wide["CS1_pop"], errors="coerce") / 1000
    wide["CS1_USDfx"] = pd.to_numeric(wide["CS1_USDfx"], errors="coerce") / 100
    wide["CS1_cons_GDP"] = pd.to_numeric(wide["CS1_cons"], errors="coerce") / pd.to_numeric(wide["CS1_nGDP"], errors="coerce") * 100
    wide["CS1_imports_GDP"] = pd.to_numeric(wide["CS1_imports"], errors="coerce") / pd.to_numeric(wide["CS1_nGDP"], errors="coerce") * 100
    wide["CS1_exports_GDP"] = pd.to_numeric(wide["CS1_exports"], errors="coerce") / pd.to_numeric(wide["CS1_nGDP"], errors="coerce") * 100
    wide["CS1_finv_GDP"] = pd.to_numeric(wide["CS1_finv"], errors="coerce") / pd.to_numeric(wide["CS1_nGDP"], errors="coerce") * 100
    wide["CS1_inv_GDP"] = pd.to_numeric(wide["CS1_inv"], errors="coerce") / pd.to_numeric(wide["CS1_nGDP"], errors="coerce") * 100
    wide["CS1_govrev_GDP"] = pd.to_numeric(wide["CS1_govrev"], errors="coerce") / pd.to_numeric(wide["CS1_nGDP"], errors="coerce") * 100
    wide["CS1_govexp_GDP"] = pd.to_numeric(wide["CS1_govexp"], errors="coerce") / pd.to_numeric(wide["CS1_nGDP"], errors="coerce") * 100
    wide["CS1_govdebt_GDP"] = pd.to_numeric(wide["CS1_govdebt"], errors="coerce") / pd.to_numeric(wide["CS1_nGDP"], errors="coerce") * 100
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int16")
    ratio_cols = {"CS1_cons_GDP", "CS1_imports_GDP", "CS1_exports_GDP", "CS1_finv_GDP", "CS1_inv_GDP", "CS1_govrev_GDP", "CS1_govexp_GDP", "CS1_govdebt_GDP"}
    for col in [c for c in wide.columns if c.startswith("CS1_") and c not in ratio_cols]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float64")
    for col in sorted(ratio_cols):
        wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float32")
    wide = wide[
        [
            "ISO3",
            "year",
            "CS1_CA_GDP",
            "CS1_CPI",
            "CS1_M0",
            "CS1_M3",
            "CS1_USDfx",
            "CS1_cbrate",
            "CS1_cons",
            "CS1_exports",
            "CS1_finv",
            "CS1_govdebt",
            "CS1_govdef",
            "CS1_govexp",
            "CS1_govrev",
            "CS1_imports",
            "CS1_inv",
            "CS1_nGDP",
            "CS1_pop",
            "CS1_rGDP",
            "CS1_cons_GDP",
            "CS1_imports_GDP",
            "CS1_exports_GDP",
            "CS1_finv_GDP",
            "CS1_inv_GDP",
            "CS1_govrev_GDP",
            "CS1_govexp_GDP",
            "CS1_govdebt_GDP",
        ]
    ].copy()
    wide = _sort_keys(wide)
    out_path = clean_dir / "country_level" / "POL_1.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_pol_1"]
