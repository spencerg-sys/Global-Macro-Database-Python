from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_ameco(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)

    df = _load_dta(raw_dir / "aggregators" / "AMECO" / "AMECO.dta")
    df = df[["period", "geo", "dataset_code", "series_code", "value", "frequency", "dataset_name", "unit"]].copy()
    df = df.loc[df["value"].astype(str) != "NA"].copy()
    df["period"] = pd.to_numeric(df["period"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    geo = df["geo"].astype(str)
    df = df.loc[~geo.isin(["ca12", "da12", "du15", "ea12", "ea19", "ea20", "eu15", "eu27", "cu15", "d-w"])].copy()

    df["indicator"] = ""
    unit = df["unit"].astype(str)
    dataset = df["dataset_code"].astype(str)
    df.loc[(dataset == "UCNT") & ~unit.isin(["mrd-pps", "mrd-ecu-eur"]), "indicator"] = "cons"
    df.loc[dataset == "NPTD", "indicator"] = "pop"
    df.loc[dataset == "OVGD", "indicator"] = "rGDP"
    df.loc[(dataset == "UIGT") & ~unit.isin(["mrd-pps", "mrd-ecu-eur"]), "indicator"] = "finv"
    df.loc[(dataset == "UITT") & ~unit.isin(["mrd-pps", "mrd-ecu-eur"]), "indicator"] = "inv"
    df.loc[(dataset == "UVGD") & ~unit.isin(["mrd-pps", "mrd-ecu-eur", "pps-eu-15-100", "eur-eu-15-100", "pps-eu-27-100", "eur-eu-27-100"]), "indicator"] = "nGDP"
    df.loc[dataset == "NUTN", "indicator"] = "unemp"
    df.loc[(dataset == "UMGS") & ~unit.isin(["mrd-pps", "mrd-ecu-eur"]), "indicator"] = "imports"
    df.loc[dataset == "XUNRQ-1", "indicator"] = "REER"
    df.loc[(dataset == "UXGS") & ~unit.isin(["mrd-pps", "mrd-ecu-eur"]), "indicator"] = "exports"
    df.loc[(dataset == "ILN") & unit.eq("-"), "indicator"] = "ltrate"
    df.loc[(dataset == "ISN") & unit.eq("-"), "indicator"] = "strate"
    df.loc[dataset == "ZCPIN", "indicator"] = "CPI"
    df = df.loc[df["indicator"] != ""].copy()

    df = df[["geo", "value", "period", "indicator"]].copy()
    wide = df.pivot(index=["geo", "period"], columns="indicator", values="value").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={col: f"AMECO_{col}" for col in wide.columns if col not in {"geo", "period"}})
    wide["ISO3"] = wide["geo"].astype(str).str.upper()
    wide["year"] = pd.to_numeric(wide["period"], errors="coerce").astype("int16")
    wide = wide.drop(columns=["geo", "period"])

    if "AMECO_unemp" in wide.columns:
        wide["AMECO_unemp"] = pd.to_numeric(wide["AMECO_unemp"], errors="coerce") / 1000
    if "AMECO_pop" in wide.columns:
        wide["AMECO_pop"] = pd.to_numeric(wide["AMECO_pop"], errors="coerce") / 1000
    for col in [c for c in ["AMECO_cons", "AMECO_exports", "AMECO_finv", "AMECO_imports", "AMECO_nGDP", "AMECO_rGDP", "AMECO_inv"] if c in wide.columns]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce") * 1000

    if {"AMECO_unemp", "AMECO_pop"}.issubset(wide.columns):
        wide["AMECO_unemp"] = pd.to_numeric(wide["AMECO_unemp"], errors="coerce") / pd.to_numeric(wide["AMECO_pop"], errors="coerce") * 100

    wide.loc[wide["ISO3"].astype(str) == "ROM", "ISO3"] = "ROU"

    for result, num in [
        ("AMECO_cons_GDP", "AMECO_cons"),
        ("AMECO_imports_GDP", "AMECO_imports"),
        ("AMECO_exports_GDP", "AMECO_exports"),
        ("AMECO_finv_GDP", "AMECO_finv"),
        ("AMECO_inv_GDP", "AMECO_inv"),
    ]:
        if {num, "AMECO_nGDP"}.issubset(wide.columns):
            wide[result] = pd.to_numeric(wide[num], errors="coerce") / pd.to_numeric(wide["AMECO_nGDP"], errors="coerce") * 100

    for col in [c for c in ["AMECO_cons_GDP", "AMECO_imports_GDP", "AMECO_exports_GDP", "AMECO_finv_GDP", "AMECO_inv_GDP"] if c in wide.columns]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float32")

    wide = _sort_keys(wide)
    wide = wide[["ISO3", "year"] + [col for col in wide.columns if col not in {"ISO3", "year"}]]
    out_path = clean_dir / "aggregators" / "AMECO" / "AMECO.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_ameco"]
