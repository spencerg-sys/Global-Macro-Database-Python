from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_bceao(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)

    df = _load_dta(raw_dir / "aggregators" / "BCEAO" / "BCEAO.dta")
    df = df.loc[~df["series_name"].astype(str).str.contains("UMOA", na=False)].copy()
    df = df.loc[df["value"].astype(str) != "NA"].copy()

    series_name = df["series_name"].astype(str)
    df["ISO3"] = ""
    iso_map = {
        "COTE D'IVOIRE": "CIV",
        "SENEGAL": "SEN",
        "MALI": "MLI",
        "NIGER": "NER",
        "BENIN": "BEN",
        "BURKINA FASO": "BFA",
        "GUINEE BISSAU": "GNB",
        "TOGO": "TGO",
    }
    for marker, iso3 in iso_map.items():
        df.loc[series_name.str.contains(marker, na=False), "ISO3"] = iso3

    right_side = series_name.str.partition("\u2013")[2].str.strip()
    df["series_name"] = np.where(right_side.ne(""), right_side, series_name)
    df["series_name"] = df["series_name"].astype(str).str.replace(".", "", regex=False).str.strip()

    label = df["label"].astype(str)
    dataset_code = df["dataset_code"].astype(str)
    df["indicator"] = ""
    df.loc[label.eq("SR1015A0BQ") & dataset_code.eq("PIBC"), "indicator"] = "rGDP"
    df.loc[label.eq("SR1037A0BP") & dataset_code.eq("IMECO"), "indicator"] = "inv"
    df = df.loc[~(df["indicator"].eq("") & dataset_code.isin(["PIBC", "IMECO"]))].copy()

    indicator_map = {
        "SR1015A0BP": "nGDP",
        "SR1016A0BP": "cons",
        "SR1019A0BP": "finv",
        "SR1023A0BP": "exports",
        "SR1024A0BP": "imports",
        "SR3017A0BP": "CPI",
        "FP1001A0AP": "govrev",
        "FP1004A0AP": "govtax",
        "FP1023A0AP": "govexp",
        "FP1042A0AP": "govdef",
        "SF1270A0AP": "M0_bis",
        "SF1400A0AP": "M0",
        "SF1271A0AP": "M1_2",
        "SF1272A0AP": "M1_3",
        "SF1284A0AP": "M1_4",
        "SF1285A0AP": "M2_1",
        "SF1408A0AP": "M1",
        "SF1412A0AP": "M2",
        "SE1007A0AP": "CA",
    }
    remaining_mask = df["indicator"].eq("")
    df.loc[remaining_mask, "indicator"] = df.loc[remaining_mask, "label"].astype(str).map(indicator_map).fillna("")

    df = df.loc[df["indicator"].ne(""), ["period", "value", "ISO3", "indicator"]].copy()
    keys = df[["ISO3", "period"]].drop_duplicates().reset_index(drop=True)
    wide_values = df.pivot_table(index=["ISO3", "period"], columns="indicator", values="value", aggfunc="first").reset_index()
    wide_values.columns.name = None
    wide = keys.merge(wide_values, on=["ISO3", "period"], how="left")
    wide = wide.rename(columns={"period": "year"})

    for col in [c for c in wide.columns if c not in {"ISO3", "year"}]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce")

    if "M1_3" in wide.columns:
        wide.loc[wide["M1_3"].isna(), "M1_3"] = 0
    if {"M0_bis", "M1_2", "M1_3", "M1_4"}.issubset(wide.columns):
        wide["M1_bis"] = (
            pd.to_numeric(wide["M0_bis"], errors="coerce")
            + pd.to_numeric(wide["M1_2"], errors="coerce")
            + pd.to_numeric(wide["M1_3"], errors="coerce")
            + pd.to_numeric(wide["M1_4"], errors="coerce")
        )
    if {"M1_bis", "M2_1"}.issubset(wide.columns):
        wide["M2_bis"] = pd.to_numeric(wide["M1_bis"], errors="coerce") + pd.to_numeric(wide["M2_1"], errors="coerce")

    wide = wide.drop(columns=[c for c in wide.columns if re.fullmatch(r"M[012]_.+", str(c))], errors="ignore")

    for col in [c for c in wide.columns if c not in {"ISO3", "year", "CPI", "CA"}]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce") * 1000

    ratio_pairs = {
        "CA_GDP": ("CA", "nGDP"),
        "govdef_GDP": ("govdef", "nGDP"),
        "govexp_GDP": ("govexp", "nGDP"),
        "govtax_GDP": ("govtax", "nGDP"),
        "govrev_GDP": ("govrev", "nGDP"),
    }
    for result, (num, den) in ratio_pairs.items():
        if {num, den}.issubset(wide.columns):
            denominator = pd.to_numeric(wide[den], errors="coerce").replace(0, np.nan)
            wide[result] = pd.to_numeric(wide[num], errors="coerce") / denominator * 100

    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int16")
    wide = _sort_keys(wide)
    if "CPI" in wide.columns:
        lag = wide.groupby("ISO3", sort=False)["CPI"].shift(1)
        valid = lag.notna() & pd.to_numeric(lag, errors="coerce").ne(0)
        wide["infl"] = pd.NA
        wide.loc[valid, "infl"] = (
            (pd.to_numeric(wide.loc[valid, "CPI"], errors="coerce") - pd.to_numeric(lag.loc[valid], errors="coerce"))
            / pd.to_numeric(lag.loc[valid], errors="coerce")
            * 100
        )

    temp_master = wide.copy()

    rates = pd.read_excel(raw_dir / "aggregators" / "BCEAO" / "rates.xlsx")
    for iso3 in ["BEN", "BFA", "CIV", "NER", "SEN", "MLI", "TGO", "GNB"]:
        rates[iso3] = rates["TESC"]

    drop_cols = [col for col in ["TPEN", "TESC", "TINB"] if col in rates.columns]
    drop_cols.extend([col for col in rates.columns if str(col).startswith("P_")])
    rates["year"] = rates["Périodes"].astype(str).str.slice(0, 4)
    rates["quarter"] = rates["Périodes"].astype(str).str.slice(-1)
    rates = rates.drop(columns=["Périodes"] + drop_cols, errors="ignore")
    for col in rates.columns:
        rates[col] = pd.to_numeric(rates[col], errors="coerce")

    value_vars = [col for col in rates.columns if col not in {"year", "quarter"}]
    rates = rates.rename(columns={col: f"cbrate{col}" for col in value_vars})
    rates_long = rates.melt(
        id_vars=["year", "quarter"],
        value_vars=[f"cbrate{col}" for col in value_vars],
        var_name="ISO3",
        value_name="cbrate",
    )
    rates_long["ISO3"] = rates_long["ISO3"].astype(str).str.removeprefix("cbrate")
    rates_long["year"] = pd.to_numeric(rates_long["year"], errors="coerce").astype("int16")
    rates_long["quarter"] = pd.to_numeric(rates_long["quarter"], errors="coerce").astype("int16")
    rates_long["cbrate"] = pd.to_numeric(rates_long["cbrate"], errors="coerce")
    rates_long = rates_long.sort_values(["ISO3", "year", "quarter"]).groupby(["ISO3", "year"], as_index=False).tail(1).copy()
    rates_long = rates_long.drop(columns=["quarter"])
    rates_long.loc[(rates_long["ISO3"].astype(str) == "GNB") & (pd.to_numeric(rates_long["year"], errors="coerce") < 1997), "cbrate"] = pd.NA

    wide = rates_long.merge(temp_master, on=["ISO3", "year"], how="outer", sort=False)
    wide = wide.rename(columns={"cbrate": "BCEAO_cbrate"})

    ratio_pairs = {
        "cons_GDP": ("cons", "nGDP"),
        "imports_GDP": ("imports", "nGDP"),
        "exports_GDP": ("exports", "nGDP"),
        "finv_GDP": ("finv", "nGDP"),
        "inv_GDP": ("inv", "nGDP"),
    }
    for result, (num, den) in ratio_pairs.items():
        if {num, den}.issubset(wide.columns):
            denominator = pd.to_numeric(wide[den], errors="coerce").replace(0, np.nan)
            wide[result] = pd.to_numeric(wide[num], errors="coerce") / denominator * 100

    rename_map = {col: f"BCEAO_{col}" for col in wide.columns if col not in {"ISO3", "year"} and col != "BCEAO_cbrate"}
    wide = wide.rename(columns=rename_map)

    for col in [
        "BCEAO_cbrate",
        "BCEAO_CA_GDP",
        "BCEAO_govdef_GDP",
        "BCEAO_govexp_GDP",
        "BCEAO_govtax_GDP",
        "BCEAO_govrev_GDP",
        "BCEAO_infl",
        "BCEAO_cons_GDP",
        "BCEAO_imports_GDP",
        "BCEAO_exports_GDP",
        "BCEAO_finv_GDP",
        "BCEAO_inv_GDP",
    ]:
        if col in wide.columns:
            wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float32")

    ordered_cols = [
        "ISO3",
        "year",
        "BCEAO_cbrate",
        "BCEAO_CA",
        "BCEAO_CPI",
        "BCEAO_M0",
        "BCEAO_M1",
        "BCEAO_M2",
        "BCEAO_cons",
        "BCEAO_exports",
        "BCEAO_finv",
        "BCEAO_govdef",
        "BCEAO_govexp",
        "BCEAO_govrev",
        "BCEAO_govtax",
        "BCEAO_imports",
        "BCEAO_inv",
        "BCEAO_nGDP",
        "BCEAO_rGDP",
        "BCEAO_CA_GDP",
        "BCEAO_govdef_GDP",
        "BCEAO_govexp_GDP",
        "BCEAO_govtax_GDP",
        "BCEAO_govrev_GDP",
        "BCEAO_infl",
        "BCEAO_cons_GDP",
        "BCEAO_imports_GDP",
        "BCEAO_exports_GDP",
        "BCEAO_finv_GDP",
        "BCEAO_inv_GDP",
    ]
    wide = _sort_keys(wide)
    if wide.duplicated(["ISO3", "year"]).any():
        raise sh.PipelineRuntimeError("ISO3 year do not uniquely identify observations", code=459)
    wide = wide[[col for col in ordered_cols if col in wide.columns]]
    out_path = clean_dir / "aggregators" / "BCEAO" / "BCEAO.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_bceao"]
