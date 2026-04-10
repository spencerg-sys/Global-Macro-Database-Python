from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_afristat(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    from .wdi import clean_wdi

    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    wdi_path = clean_dir / "aggregators" / "WB" / "WDI.dta"
    if not wdi_path.exists():
        clean_wdi(data_raw_dir=raw_dir, data_clean_dir=clean_dir, data_helper_dir=helper_dir)
    wdi = _load_dta(wdi_path)[["ISO3", "year", "WDI_nGDP"]].copy()

    def _load_sheet(sheet: str, value_name: str) -> pd.DataFrame:
        df = pd.read_excel(raw_dir / "aggregators" / "AFRISTAT" / "AFRISTAT.xlsx", sheet_name=sheet, header=None, dtype=str)
        df = df.drop(columns=[1], errors="ignore")
        header = df.iloc[0].copy()
        df.columns = ["countryname"] + [str(item) for item in header.iloc[1:]]
        df = df.iloc[1:].reset_index(drop=True)
        return df.melt(id_vars=["countryname"], var_name="year", value_name=value_name)

    master = _load_sheet("M1", "M1")
    for variable in ["M2", "M0_1", "M0_2", "DEBT_1", "DEBT_2", "govexp", "TAXES", "REVENUE"]:
        master = master.merge(_load_sheet(variable, variable), on=["countryname", "year"], how="outer", sort=False)

    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int32")
    for col in [c for c in master.columns if c not in {"countryname", "year"}]:
        master[col] = pd.to_numeric(master[col], errors="coerce")
        mask = master["countryname"].astype(str) != "Sao Tomé-et-Principe"
        master.loc[mask, col] = pd.to_numeric(master.loc[mask, col], errors="coerce") * 1000

    master = master.rename(columns={"REVENUE": "govrev", "TAXES": "govtax"})
    if {"DEBT_1", "DEBT_2"}.issubset(master.columns):
        master["govdebt"] = pd.to_numeric(master["DEBT_1"], errors="coerce") + pd.to_numeric(master["DEBT_2"], errors="coerce")
        master["govdebt"] = pd.to_numeric(master["govdebt"], errors="coerce").astype("float32")
    if {"M0_1", "M0_2"}.issubset(master.columns):
        master["M0"] = pd.to_numeric(master["M0_1"], errors="coerce") + pd.to_numeric(master["M0_2"], errors="coerce")
        master["M0"] = pd.to_numeric(master["M0"], errors="coerce").astype("float32")
    master = master.drop(columns=["DEBT_1", "DEBT_2", "M0_1", "M0_2"], errors="ignore")

    master["ISO3"] = master["countryname"].map(
        {
            "Burkina Faso": "BFA",
            "Burundi": "BDI",
            "Bénin": "BEN",
            "Cabo Verde": "CPV",
            "Cameroun": "CMR",
            "Comores": "COM",
            "Congo": "COG",
            "Côte d'Ivoire": "CIV",
            "Djibouti": "DJI",
            "Gabon": "GAB",
            "Guinée": "GIN",
            "Guinée équatoriale": "GNQ",
            "Guinée-Bissau": "GNB",
            "Madagascar": "MDG",
            "Mali": "MLI",
            "Mauritanie": "MRT",
            "Niger": "NER",
            "République centrafricaine": "CAF",
            "Sao Tomé-et-Principe": "STP",
            "Sénégal": "SEN",
            "Tchad": "TCD",
            "Togo": "TGO",
        }
    )
    master = master.drop(columns=["countryname"])

    master = master.merge(wdi, on=["ISO3", "year"], how="left", sort=False)
    denominator = pd.to_numeric(master["WDI_nGDP"], errors="coerce").replace(0, np.nan)
    if "govdebt" in master.columns:
        master["govdebt_GDP"] = pd.to_numeric(master["govdebt"], errors="coerce") / denominator * 100
    if "govexp" in master.columns:
        master["govexp_GDP"] = pd.to_numeric(master["govexp"], errors="coerce") / denominator * 100
    if "govrev" in master.columns:
        master["govrev_GDP"] = pd.to_numeric(master["govrev"], errors="coerce") / denominator * 100
    if "govtax" in master.columns:
        master["govtax_GDP"] = pd.to_numeric(master["govtax"], errors="coerce") / denominator * 100
    master = master.drop(columns=["WDI_nGDP"], errors="ignore")

    master = master.rename(columns={col: f"AFRISTAT_{col}" for col in master.columns if col not in {"ISO3", "year"}})
    for col in [
        "AFRISTAT_govdebt",
        "AFRISTAT_M0",
        "AFRISTAT_govdebt_GDP",
        "AFRISTAT_govexp_GDP",
        "AFRISTAT_govrev_GDP",
        "AFRISTAT_govtax_GDP",
    ]:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce").astype("float32")

    ordered_cols = [
        "ISO3",
        "year",
        "AFRISTAT_govrev",
        "AFRISTAT_govtax",
        "AFRISTAT_govexp",
        "AFRISTAT_M2",
        "AFRISTAT_M1",
        "AFRISTAT_govdebt",
        "AFRISTAT_M0",
        "AFRISTAT_govdebt_GDP",
        "AFRISTAT_govexp_GDP",
        "AFRISTAT_govrev_GDP",
        "AFRISTAT_govtax_GDP",
    ]
    master = _sort_keys(master)
    if master.duplicated(["ISO3", "year"]).any():
        raise sh.PipelineRuntimeError("ISO3 year do not uniquely identify observations", code=459)
    master = master[[col for col in ordered_cols if col in master.columns]]

    out_path = clean_dir / "aggregators" / "AFRISTAT" / "AFRISTAT.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_afristat"]
