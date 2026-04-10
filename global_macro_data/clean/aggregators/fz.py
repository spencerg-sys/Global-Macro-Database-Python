from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_fz(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    path = raw_dir / "aggregators" / "FZ" / "Flandreau_Zumer.xlsx"

    rename_map = {
        "United\nKingdom": "GBR",
        "Sweden": "SWE",
        "Switzerland": "CHE",
        "Spain": "ESP",
        "Russia": "RUS",
        "Portugal": "PRT",
        "Norway": "NOR",
        "Netherlands": "NLD",
        "Italy": "ITA",
        "Greece": "GRC",
        "Germany": "DEU",
        "France": "FRA",
        "Denmark": "DNK",
        "Brazil": "BRA",
        "Belgium": "BEL",
        "Argentina": "ARG",
    }

    def _load_sheet(sheet_name: str, value_name: str) -> pd.DataFrame:
        frame = _read_excel_compat(path, sheet_name=sheet_name)
        frame = frame.rename(columns={"Year": "year", **rename_map}).copy()
        for col in frame.columns:
            if pd.api.types.is_object_dtype(frame[col]) or str(frame[col].dtype) == "string":
                frame[col] = frame[col].replace({"n.a.": "", "-": "", ".": ""})
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        long = frame.melt(id_vars="year", var_name="ISO3", value_name=value_name)
        return long[["ISO3", "year", value_name]].copy()

    master = _load_sheet("exports", "exports")
    for sheet_name in ["REVENUE", "DEFICITS", "govdebt", "cb_reserves", "cb_notes", "nGDP", "CPI", "pop", "ltrate", "cbrate"]:
        master = master.merge(_load_sheet(sheet_name, sheet_name), on=["ISO3", "year"], how="outer")

    master["M0"] = pd.to_numeric(master["cb_notes"], errors="coerce") + pd.to_numeric(master["cb_reserves"], errors="coerce")
    master = master.drop(columns=["cb_notes", "cb_reserves"])
    master["govrev_GDP"] = pd.to_numeric(master["REVENUE"], errors="coerce") / pd.to_numeric(master["nGDP"], errors="coerce") * 100
    master["govdef_GDP"] = pd.to_numeric(master["DEFICITS"], errors="coerce") / pd.to_numeric(master["nGDP"], errors="coerce") * 100
    master["govdebt_GDP"] = pd.to_numeric(master["govdebt"], errors="coerce") / pd.to_numeric(master["nGDP"], errors="coerce") * 100
    master = master.rename(columns={"REVENUE": "govrev", "DEFICITS": "govdef"})

    float_cols = ["govdef", "govrev", "exports", "M0"]
    double_cols = ["nGDP", "govdebt"]

    for col in double_cols:
        master.loc[master["ISO3"].eq("FRA"), col] = pd.to_numeric(master.loc[master["ISO3"].eq("FRA"), col], errors="coerce") / 100
    for col in float_cols:
        mask = master["ISO3"].eq("FRA")
        master.loc[mask, col] = _apply_scale_chain(
            master.loc[mask, col],
            ops=[("div", 100.0)],
            storage="float",
        )

    for col in double_cols:
        master.loc[master["ISO3"].eq("ARG"), col] = pd.to_numeric(master.loc[master["ISO3"].eq("ARG"), col], errors="coerce") * _pow10_literal(-13)
    for col in float_cols:
        mask = master["ISO3"].eq("ARG")
        master.loc[mask, col] = _apply_scale_chain(
            master.loc[mask, col],
            ops=[("mul", _pow10_literal(-13))],
            storage="float",
        )
    master.loc[master["ISO3"].eq("ARG"), "govdebt"] = (
        pd.to_numeric(master.loc[master["ISO3"].eq("ARG"), "govdebt"], errors="coerce") * _pow10_literal(-13)
    )

    for col in double_cols:
        master.loc[master["ISO3"].eq("BRA"), col] = pd.to_numeric(master.loc[master["ISO3"].eq("BRA"), col], errors="coerce") * _pow10_literal(-12)
    for col in float_cols:
        mask = master["ISO3"].eq("BRA")
        master.loc[mask, col] = _apply_scale_chain(
            master.loc[mask, col],
            ops=[("mul", _pow10_literal(-12))],
            storage="float",
        )
    master.loc[master["ISO3"].eq("BRA"), "govdebt"] = (
        pd.to_numeric(master.loc[master["ISO3"].eq("BRA"), "govdebt"], errors="coerce") * _pow10_literal(-12)
    )
    for col in double_cols:
        master.loc[master["ISO3"].eq("BRA"), col] = pd.to_numeric(master.loc[master["ISO3"].eq("BRA"), col], errors="coerce") / 2750
    for col in float_cols:
        mask = master["ISO3"].eq("BRA")
        master.loc[mask, col] = _apply_scale_chain(
            master.loc[mask, col],
            ops=[("div", 2750.0)],
            storage="float",
        )
    master.loc[master["ISO3"].eq("BRA"), "govdebt"] = (
        pd.to_numeric(master.loc[master["ISO3"].eq("BRA"), "govdebt"], errors="coerce") / 2750
    )

    for col in double_cols:
        master.loc[master["ISO3"].eq("DEU"), col] = pd.to_numeric(master.loc[master["ISO3"].eq("DEU"), col], errors="coerce") * _pow10_literal(-12)
    for col in float_cols:
        mask = master["ISO3"].eq("DEU")
        master.loc[mask, col] = _apply_scale_chain(
            master.loc[mask, col],
            ops=[("mul", _pow10_literal(-12))],
            storage="float",
        )
    master.loc[master["ISO3"].eq("DEU"), "govdebt"] = (
        pd.to_numeric(master.loc[master["ISO3"].eq("DEU"), "govdebt"], errors="coerce") * _pow10_literal(-12)
    )

    for col in double_cols:
        master.loc[master["ISO3"].eq("GRC"), col] = pd.to_numeric(master.loc[master["ISO3"].eq("GRC"), col], errors="coerce") * _pow10_literal(-3)
        master.loc[master["ISO3"].eq("GRC"), col] = pd.to_numeric(master.loc[master["ISO3"].eq("GRC"), col], errors="coerce") / 4
    for col in float_cols:
        mask = master["ISO3"].eq("GRC")
        master.loc[mask, col] = _apply_scale_chain(
            master.loc[mask, col],
            ops=[("mul", _pow10_literal(-3)), ("div", 4.0)],
            storage="float",
        )
    master.loc[master["ISO3"].eq("GRC"), "govdebt"] = (
        pd.to_numeric(master.loc[master["ISO3"].eq("GRC"), "govdebt"], errors="coerce") * _pow10_literal(-3) / 4
    )

    master["pop"] = pd.to_numeric(master["pop"], errors="coerce") / 1000

    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].drop_duplicates().copy()
    master = master.merge(eur_fx, on="ISO3", how="left")
    matched = master["EUR_irrevocable_FX"].notna()
    for col in double_cols:
        master.loc[matched, col] = (
            pd.to_numeric(master.loc[matched, col], errors="coerce")
            / pd.to_numeric(master.loc[matched, "EUR_irrevocable_FX"], errors="coerce")
        )
    for col in float_cols:
        master.loc[matched, col] = _apply_scale_chain(
            master.loc[matched, col],
            ops=[("div", pd.to_numeric(master.loc[matched, "EUR_irrevocable_FX"], errors="coerce"))],
            storage="float",
        )
    master.loc[matched, "govdebt"] = (
        pd.to_numeric(master.loc[matched, "govdebt"], errors="coerce")
        / pd.to_numeric(master.loc[matched, "EUR_irrevocable_FX"], errors="coerce")
    )
    master = master.drop(columns=["EUR_irrevocable_FX"]).sort_values(["ISO3", "year"]).reset_index(drop=True)

    prev_cpi = _lag_if_consecutive_year(master, "CPI")
    master["infl"] = np.where(
        prev_cpi.notna(),
        (
            pd.to_numeric(master["CPI"], errors="coerce")
            - pd.to_numeric(prev_cpi, errors="coerce")
        )
        / pd.to_numeric(prev_cpi, errors="coerce")
        * 100,
        np.nan,
    )

    out = master.rename(columns={col: f"FZ_{col}" for col in master.columns if col not in {"ISO3", "year"}})
    out = out[
        [
            "ISO3",
            "year",
            "FZ_cbrate",
            "FZ_ltrate",
            "FZ_pop",
            "FZ_CPI",
            "FZ_nGDP",
            "FZ_govdebt",
            "FZ_govdef",
            "FZ_govrev",
            "FZ_exports",
            "FZ_M0",
            "FZ_govrev_GDP",
            "FZ_govdef_GDP",
            "FZ_govdebt_GDP",
            "FZ_infl",
        ]
    ].copy()
    out["ISO3"] = out["ISO3"].astype("object")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int16")
    for col in ["FZ_cbrate", "FZ_ltrate", "FZ_pop", "FZ_CPI", "FZ_nGDP", "FZ_govdebt"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in ["FZ_govdef", "FZ_govrev", "FZ_exports", "FZ_M0", "FZ_govrev_GDP", "FZ_govdef_GDP", "FZ_govdebt_GDP", "FZ_infl"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    out["ISO3"] = out["ISO3"].astype("object")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int16")
    for col in ["FZ_cbrate", "FZ_ltrate", "FZ_pop", "FZ_CPI", "FZ_nGDP", "FZ_govdebt"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in ["FZ_govdef", "FZ_govrev", "FZ_exports", "FZ_M0", "FZ_govrev_GDP", "FZ_govdef_GDP", "FZ_govdebt_GDP", "FZ_infl"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    out = _sort_keys(out)
    out_path = clean_dir / "aggregators" / "FZ" / "FZ.dta"
    _save_dta(out, out_path)
    return out
__all__ = ["clean_fz"]
