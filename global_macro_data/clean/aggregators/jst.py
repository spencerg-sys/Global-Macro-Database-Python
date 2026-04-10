from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_jst(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    from .wdi import clean_wdi

    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    clean_wdi(data_raw_dir=raw_dir, data_clean_dir=clean_dir, data_helper_dir=helper_dir)

    df = _load_dta(raw_dir / "aggregators" / "JST" / "JSTdatasetR6.dta")
    df = df.rename(
        columns={
            "iso": "ISO3",
            "pop": "JST_pop",
            "gdp": "JST_nGDP",
            "iy": "JST_inv_GDP",
            "cpi": "JST_CPI",
            "xrusd": "JST_USDfx",
            "ca": "JST_CA_LCU",
            "imports": "JST_imports",
            "exports": "JST_exports",
            "stir": "JST_strate",
            "ltrate": "JST_ltrate",
            "unemp": "JST_unemp",
            "debtgdp": "JST_govdebt_GDP",
            "revenue": "JST_govrev",
            "expenditure": "JST_govexp",
            "hpnom": "JST_HPI",
            "money": "JST_BroadM",
            "narrowm": "JST_NarrowM",
            "crisisJST": "JST_crisisB",
            "rgdpbarro": "JST_rGDP_pc_index",
        }
    )
    keep_cols = ["ISO3", "year"] + [col for col in df.columns if col.startswith("JST_")]
    df = df[keep_cols].copy()
    df = _drop_rows_with_all_missing(df)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")

    for col in ["JST_M0", "JST_M1", "JST_M2", "JST_M3", "JST_M4"]:
        df[col] = pd.Series(np.nan, index=df.index, dtype="float32")

    for c in ["NOR", "GBR", "USA"]:
        df.loc[df["ISO3"].astype(str) == c, "JST_M0"] = pd.to_numeric(df.loc[df["ISO3"].astype(str) == c, "JST_NarrowM"], errors="coerce").astype("float32")
    for c in ["AUS", "BEL", "CAD", "DNK", "FIN", "FRA", "DEU", "IRL", "ITA", "JPN", "NLD", "PRT", "ESP", "SWE", "CHE"]:
        df.loc[df["ISO3"].astype(str) == c, "JST_M1"] = pd.to_numeric(df.loc[df["ISO3"].astype(str) == c, "JST_NarrowM"], errors="coerce").astype("float32")
    for c in ["DNK", "CAD", "FIN", "FRA", "DEU", "IRL", "ITA", "JPN", "NLD", "NOR", "PRT"]:
        df.loc[df["ISO3"].astype(str) == c, "JST_M2"] = pd.to_numeric(df.loc[df["ISO3"].astype(str) == c, "JST_BroadM"], errors="coerce").astype("float32")
    for c in ["AUS", "BEL", "ESP", "SWE", "CHE", "USA"]:
        df.loc[df["ISO3"].astype(str) == c, "JST_M3"] = pd.to_numeric(df.loc[df["ISO3"].astype(str) == c, "JST_BroadM"], errors="coerce").astype("float32")
    df.loc[df["ISO3"].astype(str) == "GBR", "JST_M4"] = pd.to_numeric(df.loc[df["ISO3"].astype(str) == "GBR", "JST_BroadM"], errors="coerce").astype("float32")
    df = df.drop(columns=["JST_NarrowM", "JST_BroadM"], errors="ignore")

    df["JST_CA_GDP"] = (100 * pd.to_numeric(df["JST_CA_LCU"], errors="coerce") / pd.to_numeric(df["JST_nGDP"], errors="coerce")).astype("float32")
    df["JST_govrev_GDP"] = (100 * pd.to_numeric(df["JST_govrev"], errors="coerce") / pd.to_numeric(df["JST_nGDP"], errors="coerce")).astype("float32")
    df["JST_govexp_GDP"] = (100 * pd.to_numeric(df["JST_govexp"], errors="coerce") / pd.to_numeric(df["JST_nGDP"], errors="coerce")).astype("float32")
    df["JST_govdebt_GDP"] = pd.to_numeric(df["JST_govdebt_GDP"], errors="coerce") * 100
    df = df.drop(columns=["JST_CA_LCU"], errors="ignore")

    for col in ["JST_CPI", "JST_HPI"]:
        temp = pd.to_numeric(df[col], errors="coerce").astype("float32").where(df["year"] == 2010)
        scaler = temp.groupby(df["ISO3"]).transform("max").astype("float32")
        ratio = 100.0 / scaler.astype("float64")
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64") * ratio

    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].copy()
    df = df.merge(eur_fx, on="ISO3", how="left")
    mask = df["EUR_irrevocable_FX"].notna()
    for var in ["JST_nGDP", "JST_exports", "JST_imports", "JST_M0", "JST_M1", "JST_M2", "JST_M3", "JST_M4", "JST_govrev", "JST_govexp", "JST_USDfx"]:
        values = pd.to_numeric(df.loc[mask, var], errors="coerce") / pd.to_numeric(df.loc[mask, "EUR_irrevocable_FX"], errors="coerce")
        if var in {"JST_M0", "JST_M1", "JST_M2", "JST_M3", "JST_M4"}:
            values = values.astype("float32")
        df.loc[mask, var] = values
    df = df.drop(columns=["EUR_irrevocable_FX"], errors="ignore")

    df["JST_inv"] = (pd.to_numeric(df["JST_inv_GDP"], errors="coerce") * pd.to_numeric(df["JST_nGDP"], errors="coerce")).astype("float32")
    df = df.drop(columns=["JST_inv_GDP"], errors="ignore")
    df["JST_pop"] = pd.to_numeric(df["JST_pop"], errors="coerce") / 1000

    for c in ["USA", "CAN", "DNK", "FRA", "DEU", "ITA", "GBR"]:
        mask = df["ISO3"].astype(str) == c
        for var in ["JST_nGDP", "JST_exports", "JST_imports", "JST_inv", "JST_M0", "JST_M1", "JST_M2", "JST_M3", "JST_M4"]:
            values = pd.to_numeric(df.loc[mask, var], errors="coerce") * 1000
            if var in {"JST_inv", "JST_M0", "JST_M1", "JST_M2", "JST_M3", "JST_M4"}:
                values = values.astype("float32")
            df.loc[mask, var] = values

    mask = df["ISO3"].astype(str) == "JPN"
    for var in ["JST_nGDP", "JST_exports", "JST_imports", "JST_inv", "JST_M1", "JST_M2", "JST_govexp", "JST_govrev"]:
        values = pd.to_numeric(df.loc[mask, var], errors="coerce") * 1_000_000
        if var in {"JST_inv", "JST_M1", "JST_M2"}:
            values = values.astype("float32")
        df.loc[mask, var] = values

    prev_cpi = pd.to_numeric(df["JST_CPI"], errors="coerce").groupby(df["ISO3"]).shift(1)
    prev_year = pd.to_numeric(df["year"], errors="coerce").groupby(df["ISO3"]).shift(1)
    year_num = pd.to_numeric(df["year"], errors="coerce")
    infl = (pd.to_numeric(df["JST_CPI"], errors="coerce") - prev_cpi) / prev_cpi * 100
    infl = infl.where(prev_cpi.notna() & year_num.eq(prev_year + 1))
    df["JST_infl"] = infl.astype("float32")

    df["JST_govexp"] = pd.to_numeric(df["JST_govexp"], errors="coerce") * 1000
    df["JST_govrev"] = pd.to_numeric(df["JST_govrev"], errors="coerce") * 1000
    for c in ["AUS", "BEL", "CHE", "ESP", "FIN", "IRL", "JPN", "NLD", "NOR", "PRT", "SWE"]:
        mask = df["ISO3"].astype(str) == c
        df.loc[mask, "JST_govexp"] = pd.to_numeric(df.loc[mask, "JST_govexp"], errors="coerce") / 1000
        df.loc[mask, "JST_govrev"] = pd.to_numeric(df.loc[mask, "JST_govrev"], errors="coerce") / 1000

    df["JST_imports_GDP"] = (pd.to_numeric(df["JST_imports"], errors="coerce") / pd.to_numeric(df["JST_nGDP"], errors="coerce") * 100).astype("float32")
    df["JST_exports_GDP"] = (pd.to_numeric(df["JST_exports"], errors="coerce") / pd.to_numeric(df["JST_nGDP"], errors="coerce") * 100).astype("float32")
    df["JST_inv_GDP"] = (pd.to_numeric(df["JST_inv"], errors="coerce") / pd.to_numeric(df["JST_nGDP"], errors="coerce") * 100).astype("float32")

    wdi = _load_dta(clean_dir / "aggregators" / "WB" / "WDI.dta")[["ISO3", "year", "WDI_rGDP_pc"]].copy()
    df = df.merge(wdi, on=["ISO3", "year"], how="left")
    df = df.rename(columns={"JST_rGDP_pc_index": "JST_rGDP_pc"})
    spliced = sh.splice(df, priority="WDI JST", generate="rGDP_pc", varname="rGDP_pc", method="chainlink", base_year=2006, save="NO")
    out = spliced[["ISO3", "year"] + [col for col in spliced.columns if col.startswith("JST_")] + ["rGDP_pc"]].copy()
    out = out.drop(columns=["JST_rGDP_pc"], errors="ignore").rename(columns={"rGDP_pc": "JST_rGDP_pc"})
    out["JST_rGDP"] = pd.to_numeric(out["JST_rGDP_pc"], errors="coerce") * pd.to_numeric(out["JST_pop"], errors="coerce")
    out = out[["ISO3", "year", "JST_pop", "JST_nGDP", "JST_CPI", "JST_imports", "JST_exports", "JST_strate", "JST_ltrate", "JST_HPI", "JST_unemp", "JST_govdebt_GDP", "JST_govrev", "JST_govexp", "JST_USDfx", "JST_crisisB", "JST_M0", "JST_M1", "JST_M2", "JST_M3", "JST_M4", "JST_CA_GDP", "JST_govrev_GDP", "JST_govexp_GDP", "JST_inv", "JST_infl", "JST_imports_GDP", "JST_exports_GDP", "JST_inv_GDP", "JST_rGDP_pc", "JST_rGDP"]].copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int16")
    for col in ["JST_pop", "JST_nGDP", "JST_CPI", "JST_imports", "JST_exports", "JST_strate", "JST_ltrate", "JST_HPI", "JST_unemp", "JST_govdebt_GDP", "JST_govrev", "JST_govexp", "JST_USDfx"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    out["JST_crisisB"] = pd.to_numeric(out["JST_crisisB"], errors="coerce").fillna(0).astype("int8")
    for col in ["JST_M0", "JST_M1", "JST_M2", "JST_M3", "JST_M4", "JST_CA_GDP", "JST_govrev_GDP", "JST_govexp_GDP", "JST_inv", "JST_infl", "JST_imports_GDP", "JST_exports_GDP", "JST_inv_GDP", "JST_rGDP_pc", "JST_rGDP"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    out = _sort_keys(out)
    out_path = clean_dir / "aggregators" / "JST" / "JST.dta"
    _save_dta(out, out_path)
    return out
__all__ = ["clean_jst"]
