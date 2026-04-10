from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_zaf_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = _load_dta(raw_dir / "country_level" / "ZAF_1.dta")
    mapping = {
        "KBP2000J": "strate",
        "KBP2003J": "ltrate",
        "KBP4420F": "govdef_GDP",
        "KBP4595J": "govtax",
        "KBP4597J": "govrev",
        "KBP4601J": "govexp",
        "KBP6006J": "nGDP",
        "KBP6006Y": "rGDP",
        "KBP6009J": "finv",
        "KBP6180J": "inv",
        "KBP6013J": "exports",
        "KBP6014J": "imports",
        "KBP6620J": "cons",
        "KBP1371J": "M1",
        "KBP1373J": "M2",
        "KBP1374J": "M3",
        "KBP1000J": "M0",
    }
    df["indicator"] = df["series_code"].astype(str).map(mapping).fillna("")
    df = df.loc[df["indicator"].astype(str) != "", ["period", "value", "indicator"]].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    wide = df.pivot(index="period", columns="indicator", values="value").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"period": "year"})
    wide["cons_GDP"] = pd.to_numeric(wide["cons"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide["imports_GDP"] = pd.to_numeric(wide["imports"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide["exports_GDP"] = pd.to_numeric(wide["exports"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide["finv_GDP"] = pd.to_numeric(wide["finv"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide["inv_GDP"] = pd.to_numeric(wide["inv"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide["govrev_GDP"] = pd.to_numeric(wide["govrev"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide["govexp_GDP"] = pd.to_numeric(wide["govexp"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide["govtax_GDP"] = pd.to_numeric(wide["govtax"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide = wide.rename(columns={col: f"CS1_{col}" for col in wide.columns if col != "year"})
    wide["ISO3"] = "ZAF"
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int16")
    ratio_cols = {"CS1_cons_GDP", "CS1_imports_GDP", "CS1_exports_GDP", "CS1_finv_GDP", "CS1_inv_GDP", "CS1_govrev_GDP", "CS1_govexp_GDP", "CS1_govtax_GDP"}
    for col in [c for c in wide.columns if c.startswith("CS1_") and c not in ratio_cols]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float64")
    for col in sorted(ratio_cols):
        wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float32")
    wide = wide[
        [
            "ISO3",
            "year",
            "CS1_M0",
            "CS1_M1",
            "CS1_M2",
            "CS1_M3",
            "CS1_cons",
            "CS1_exports",
            "CS1_finv",
            "CS1_govdef_GDP",
            "CS1_govexp",
            "CS1_govrev",
            "CS1_govtax",
            "CS1_imports",
            "CS1_inv",
            "CS1_ltrate",
            "CS1_nGDP",
            "CS1_rGDP",
            "CS1_strate",
            "CS1_cons_GDP",
            "CS1_imports_GDP",
            "CS1_exports_GDP",
            "CS1_finv_GDP",
            "CS1_inv_GDP",
            "CS1_govrev_GDP",
            "CS1_govexp_GDP",
            "CS1_govtax_GDP",
        ]
    ].copy()
    wide = _sort_keys(wide)
    out_path = clean_dir / "country_level" / "ZAF_1.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_zaf_1"]
