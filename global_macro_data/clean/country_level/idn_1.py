from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_idn_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = _load_dta(raw_dir / "country_level" / "IDN_1.dta")
    df = df[["period", "value", "dataset_code", "series_name", "series_code", "REF_AREA"]].copy()
    mapping = {
        ("TABEL1_1", "1.A"): "M2",
        ("TABEL1_1", "2.A"): "M1",
        ("TABEL1_2", "1.A"): "M0",
        ("TABEL4_1", "2.A"): "govrev",
        ("TABEL4_2", "2.A"): "govexp",
        ("TABEL4_3", "2.A"): "govdef",
        ("TABEL4_4", "2.A"): "govdebt",
        ("TABEL5_1", "1.A"): "CA",
        ("TABEL5_40", "10.A"): "USDfx",
        ("TABEL7_3", "5.A"): "finv",
        ("TABEL7_3", "8.A"): "exports_goods",
        ("TABEL7_3", "9.A"): "exports_services",
        ("TABEL7_1", "64.A"): "nGDP",
        ("TABEL7_3", "10.A"): "imports_goods",
        ("TABEL7_3", "11.A"): "imports_services",
        ("TABEL8_1", "23.A"): "CPI",
    }
    key = list(zip(df["dataset_code"].astype(str), df["series_code"].astype(str)))
    df["dataset_code"] = pd.Series(key, index=df.index).map(mapping).fillna(df["dataset_code"].astype(str))
    df = df.rename(columns={"period": "year", "REF_AREA": "ISO3"})
    df["value"] = df["value"].replace("NA", "")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.drop(columns=["series_name", "series_code"])
    df = df.loc[~df["dataset_code"].astype(str).str.contains("TABEL", regex=False, na=False)].copy()
    wide = df.pivot(index=["ISO3", "year"], columns="dataset_code", values="value").reset_index()
    wide.columns.name = None
    wide["exports"] = pd.to_numeric(wide.get("exports_goods"), errors="coerce") + pd.to_numeric(wide.get("exports_services"), errors="coerce")
    wide["imports"] = pd.to_numeric(wide.get("imports_goods"), errors="coerce") + pd.to_numeric(wide.get("imports_services"), errors="coerce")
    wide = wide.drop(columns=["imports_goods", "exports_goods", "exports_services", "imports_services"], errors="ignore")
    for col in ["M0", "M1", "M2", "govrev", "govexp", "govdef", "govdebt", "nGDP", "finv"]:
        if col in wide.columns:
            wide[col] = pd.to_numeric(wide[col], errors="coerce") * 1000
    for col in ["exports", "imports"]:
        if col in wide.columns:
            # The reference pipeline materializes the scaled trade totals as float at the
            # `replace x = x * 1000` step rather than only at final storage.
            wide[col] = (
                pd.to_numeric(wide[col], errors="coerce")
                .astype("float32")
                .mul(np.float32(1000))
            )
    wide["imports_GDP"] = pd.to_numeric(wide["imports"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide["exports_GDP"] = pd.to_numeric(wide["exports"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide["finv_GDP"] = pd.to_numeric(wide["finv"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide["govdef_GDP"] = pd.to_numeric(wide["govdef"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide["govdebt_GDP"] = pd.to_numeric(wide["govdebt"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide["CA_GDP"] = pd.to_numeric(wide["CA"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide["govrev_GDP"] = pd.to_numeric(wide["govrev"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide["govexp_GDP"] = pd.to_numeric(wide["govexp"], errors="coerce") / pd.to_numeric(wide["nGDP"], errors="coerce") * 100
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int16")
    wide = wide.rename(columns={col: f"CS1_{col}" for col in wide.columns if col not in {"ISO3", "year"}})
    for col in [
        "CS1_CA",
        "CS1_CPI",
        "CS1_M0",
        "CS1_M1",
        "CS1_M2",
        "CS1_USDfx",
        "CS1_finv",
        "CS1_govdebt",
        "CS1_govdef",
        "CS1_govexp",
        "CS1_govrev",
        "CS1_nGDP",
    ]:
        if col in wide.columns:
            wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float64")
    for col in [
        "CS1_exports",
        "CS1_imports",
        "CS1_imports_GDP",
        "CS1_exports_GDP",
        "CS1_finv_GDP",
        "CS1_govdef_GDP",
        "CS1_govdebt_GDP",
        "CS1_CA_GDP",
        "CS1_govrev_GDP",
        "CS1_govexp_GDP",
    ]:
        if col in wide.columns:
            wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float32")
    wide = wide[
        [
            "ISO3",
            "year",
            "CS1_CA",
            "CS1_CPI",
            "CS1_M0",
            "CS1_M1",
            "CS1_M2",
            "CS1_USDfx",
            "CS1_finv",
            "CS1_govdebt",
            "CS1_govdef",
            "CS1_govexp",
            "CS1_govrev",
            "CS1_nGDP",
            "CS1_exports",
            "CS1_imports",
            "CS1_imports_GDP",
            "CS1_exports_GDP",
            "CS1_finv_GDP",
            "CS1_govdef_GDP",
            "CS1_govdebt_GDP",
            "CS1_CA_GDP",
            "CS1_govrev_GDP",
            "CS1_govexp_GDP",
        ]
    ].copy()
    for col in [
        "CS1_CA",
        "CS1_CPI",
        "CS1_M0",
        "CS1_M1",
        "CS1_M2",
        "CS1_USDfx",
        "CS1_finv",
        "CS1_govdebt",
        "CS1_govdef",
        "CS1_govexp",
        "CS1_govrev",
        "CS1_nGDP",
    ]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float64")
    for col in [
        "CS1_exports",
        "CS1_imports",
        "CS1_imports_GDP",
        "CS1_exports_GDP",
        "CS1_finv_GDP",
        "CS1_govdef_GDP",
        "CS1_govdebt_GDP",
        "CS1_CA_GDP",
        "CS1_govrev_GDP",
        "CS1_govexp_GDP",
    ]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float32")
    wide = _sort_keys(wide)
    out_path = clean_dir / "country_level" / "IDN_1.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_idn_1"]
