from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_sau_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = _load_dta(raw_dir / "country_level" / "SAU_1.dta")

    def _sau1_value(value: object) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if text in {"", "<NA>", "nan", "None"}:
            return np.nan
        try:
            return float(text)
        except ValueError:
            return np.nan

    df["indicator"] = ""
    name_cur = "Expenditure on Gross Domestic Product (at purchasers' values at current prices) (million riyals)"
    name_real = "Expenditure on Gross Domestic Product (at purchasers' values at constant prices (2010 = 100)) (million riyals)"
    code = df["series_code"].astype(str)
    dname = df["dataset_name"].astype(str)
    df.loc[(code == "expenditure-on-gross-domestic-product") & dname.eq(name_cur), "indicator"] = "nGDP"
    df.loc[(code == "exports-of-goods-services") & dname.eq(name_cur), "indicator"] = "exports"
    df.loc[(code == "imports-of-goods-services") & dname.eq(name_cur), "indicator"] = "imports"
    df.loc[(code == "total-final-consumption-expenditure") & dname.eq(name_cur), "indicator"] = "cons"
    df.loc[(code == "gross-fixed-capital-formation") & dname.eq(name_cur), "indicator"] = "finv"
    df.loc[(code == "change-in-stock") & dname.eq(name_cur), "indicator"] = "inv_1"
    df.loc[(code == "expenditure-on-gross-domestic-product") & dname.eq(name_real), "indicator"] = "rGDP"
    df = df.loc[df["indicator"].astype(str) != "", ["period", "value", "indicator"]].copy()
    df["value"] = df["value"].map(_sau1_value)
    wide = df.pivot(index="period", columns="indicator", values="value").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"period": "year"})
    wide["inv"] = (
        pd.to_numeric(wide["inv_1"], errors="coerce")
        + pd.to_numeric(wide["finv"], errors="coerce")
    ).astype("float32")
    wide = wide.drop(columns=["inv_1"], errors="ignore")
    wide["cons_GDP"] = pd.to_numeric(wide["cons"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide["imports_GDP"] = pd.to_numeric(wide["imports"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide["exports_GDP"] = pd.to_numeric(wide["exports"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide["finv_GDP"] = pd.to_numeric(wide["finv"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide["inv_GDP"] = pd.to_numeric(wide["inv"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide = wide.rename(columns={col: f"CS1_{col}" for col in wide.columns if col != "year"})
    wide["ISO3"] = "SAU"
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int16")
    for col in ["CS1_cons", "CS1_exports", "CS1_finv", "CS1_imports", "CS1_nGDP", "CS1_rGDP"]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float64")
    wide["CS1_inv"] = pd.to_numeric(wide["CS1_inv"], errors="coerce").astype("float32")
    for col in ["CS1_cons_GDP", "CS1_imports_GDP", "CS1_exports_GDP", "CS1_finv_GDP", "CS1_inv_GDP"]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float32")
    wide = wide[["ISO3", "year", "CS1_cons", "CS1_exports", "CS1_finv", "CS1_imports", "CS1_nGDP", "CS1_rGDP", "CS1_inv", "CS1_cons_GDP", "CS1_imports_GDP", "CS1_exports_GDP", "CS1_finv_GDP", "CS1_inv_GDP"]].copy()
    for col in ["CS1_cons", "CS1_exports", "CS1_finv", "CS1_imports", "CS1_nGDP", "CS1_rGDP"]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float64")
    wide["CS1_inv"] = pd.to_numeric(wide["CS1_inv"], errors="coerce").astype("float32")
    for col in ["CS1_cons_GDP", "CS1_imports_GDP", "CS1_exports_GDP", "CS1_finv_GDP", "CS1_inv_GDP"]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float32")
    wide = _sort_keys(wide)
    out_path = clean_dir / "country_level" / "SAU_1.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_sau_1"]
