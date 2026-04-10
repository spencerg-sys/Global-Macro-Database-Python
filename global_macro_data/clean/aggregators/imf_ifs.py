from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_imf_ifs(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    df = _load_dta(raw_dir / "aggregators" / "IMF" / "IMF_IFS.dta")
    df = df[["period", "ref_area", "indicator", "value"]].copy()
    wide = df.pivot_table(index=["ref_area", "period"], columns="indicator", values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={
            "period": "year",
            "ref_area": "ISO2",
            "NGDP_XDC": "nGDP",
            "NGDP_R_XDC": "rGDP",
            "ENDA_XDC_USD_RATE": "USDfx",
            "EREER_IX": "REER",
            "FPOLM_PA": "cbrate",
            "LP_PE_NUM": "pop",
            "LUR_PT": "unemp",
            "BCAXF_BP6_USD": "CA_USD",
            "NFI_XDC": "finv",
            "NI_XDC": "inv",
            "NM_XDC": "imports",
            "NX_XDC": "exports",
            "NC_XDC": "cons",
            "PCPI_IX": "CPI",
            "PCPI_PC_CP_A_PT": "infl",
            "NC_R_XDC": "rcons",
        }
    )
    wide["CA"] = pd.to_numeric(wide.get("CA_USD"), errors="coerce") * pd.to_numeric(wide.get("USDfx"), errors="coerce")
    wide.loc[wide["ISO2"].astype(str) == "DE2", "ISO2"] = "DD"
    wide.loc[wide["ISO2"].astype(str) == "YUC", "ISO2"] = "YU"
    wide.loc[wide["ISO2"].astype(str) == "CSH", "ISO2"] = "CS"
    wide.loc[wide["ISO2"].astype(str) == "SUH", "ISO2"] = "SU"
    wide = wide.loc[~wide["ISO2"].astype(str).str.contains(r"[0-9]", regex=True, na=False)].copy()

    countrylist = _load_dta(helper_dir / "countrylist.dta")[["ISO2", "ISO3"]].copy()
    wide = wide.merge(countrylist, on="ISO2", how="left")
    wide = wide.loc[wide["ISO3"].notna()].copy()
    wide = wide.drop(columns=["ISO2"])
    if "pop" in wide.columns:
        wide["pop"] = pd.to_numeric(wide["pop"], errors="coerce") / 1000

    value_cols = [col for col in wide.columns if col not in {"ISO3", "year"}]
    wide = wide.rename(columns={col: f"IMF_IFS_{col}" for col in value_cols})

    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")
    wide = wide.merge(eur_fx[["ISO3", "EUR_irrevocable_FX"]], on="ISO3", how="left")
    mask = wide["EUR_irrevocable_FX"].notna()
    wide.loc[mask, "IMF_IFS_USDfx"] = pd.to_numeric(wide.loc[mask, "IMF_IFS_USDfx"], errors="coerce") / pd.to_numeric(wide.loc[mask, "EUR_irrevocable_FX"], errors="coerce")
    wide = wide.drop(columns=["EUR_irrevocable_FX"], errors="ignore")

    for col in [c for c in ["IMF_IFS_exports", "IMF_IFS_imports"] if c in wide.columns]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce").abs()

    wide.loc[(wide["ISO3"].astype(str) == "VNM") & pd.to_numeric(wide["IMF_IFS_nGDP"], errors="coerce").eq(4.657e-07), "IMF_IFS_nGDP"] = pd.NA
    wide.loc[wide["ISO3"].astype(str) == "LBR", "IMF_IFS_nGDP"] = pd.to_numeric(wide.loc[wide["ISO3"].astype(str) == "LBR", "IMF_IFS_nGDP"], errors="coerce") / 100
    wide.loc[wide["ISO3"].astype(str) == "IND", "IMF_IFS_nGDP"] = pd.to_numeric(wide.loc[wide["ISO3"].astype(str) == "IND", "IMF_IFS_nGDP"], errors="coerce") * 1000
    wide.loc[wide["ISO3"].astype(str) == "IDN", "IMF_IFS_nGDP"] = pd.to_numeric(wide.loc[wide["ISO3"].astype(str) == "IDN", "IMF_IFS_nGDP"], errors="coerce") * 1000
    wide.loc[(wide["ISO3"].astype(str) == "LBY") & pd.to_numeric(wide["year"], errors="coerce").between(1981, 1985), "IMF_IFS_nGDP"] = pd.to_numeric(wide.loc[(wide["ISO3"].astype(str) == "LBY") & pd.to_numeric(wide["year"], errors="coerce").between(1981, 1985), "IMF_IFS_nGDP"], errors="coerce") * 1000
    wide.loc[wide["ISO3"].astype(str) == "BFA", "IMF_IFS_rGDP"] = pd.to_numeric(wide.loc[wide["ISO3"].astype(str) == "BFA", "IMF_IFS_rGDP"], errors="coerce") * (10**9)
    wide.loc[wide["ISO3"].astype(str) == "IND", "IMF_IFS_rGDP"] = pd.to_numeric(wide.loc[wide["ISO3"].astype(str) == "IND", "IMF_IFS_rGDP"], errors="coerce") * (10**6)
    wide.loc[(wide["ISO3"].astype(str) == "SLV") & (pd.to_numeric(wide["year"], errors="coerce") <= 2004), "IMF_IFS_rGDP"] = pd.NA
    for col in [c for c in ["IMF_IFS_imports", "IMF_IFS_exports"] if c in wide.columns]:
        wide.loc[wide["ISO3"].astype(str).isin(["IND", "IDN"]), col] = pd.to_numeric(wide.loc[wide["ISO3"].astype(str).isin(["IND", "IDN"]), col], errors="coerce") * 1000
        wide.loc[(wide["ISO3"].astype(str) == "BFA") & (pd.to_numeric(wide["year"], errors="coerce") <= 1998), col] = pd.NA

    wide.loc[(wide["ISO3"].astype(str) == "BOL") & pd.to_numeric(wide["year"], errors="coerce").eq(1958), "IMF_IFS_USDfx"] = pd.NA
    wide.loc[wide["ISO3"].astype(str) == "STP", "IMF_IFS_USDfx"] = pd.to_numeric(wide.loc[wide["ISO3"].astype(str) == "STP", "IMF_IFS_USDfx"], errors="coerce") * 1000
    wide.loc[wide["ISO3"].astype(str) == "SLE", "IMF_IFS_USDfx"] = pd.to_numeric(wide.loc[wide["ISO3"].astype(str) == "SLE", "IMF_IFS_USDfx"], errors="coerce") * 1000
    wide.loc[(wide["ISO3"].astype(str) == "TUR") & (pd.to_numeric(wide["year"], errors="coerce") <= 1956), "IMF_IFS_USDfx"] = pd.to_numeric(wide.loc[(wide["ISO3"].astype(str) == "TUR") & (pd.to_numeric(wide["year"], errors="coerce") <= 1956), "IMF_IFS_USDfx"], errors="coerce") / 1_000_000
    wide.loc[(wide["ISO3"].astype(str) == "TUR") & pd.to_numeric(wide["year"], errors="coerce").eq(1957), "IMF_IFS_USDfx"] = pd.to_numeric(wide.loc[(wide["ISO3"].astype(str) == "TUR") & pd.to_numeric(wide["year"], errors="coerce").eq(1957), "IMF_IFS_USDfx"], errors="coerce") / 100_000
    wide.loc[(wide["ISO3"].astype(str) == "VNM") & pd.to_numeric(wide["year"], errors="coerce").eq(2023), "IMF_IFS_nGDP"] = 1.022e10
    wide.loc[(wide["ISO3"].astype(str) == "VNM") & pd.to_numeric(wide["year"], errors="coerce").eq(2023), "IMF_IFS_rGDP"] = 5.831e09

    wide["IMF_IFS_CA_GDP"] = pd.to_numeric(wide.get("IMF_IFS_CA"), errors="coerce") / pd.to_numeric(wide.get("IMF_IFS_nGDP"), errors="coerce") * 100
    wide = wide.drop(columns=["IMF_IFS_BGS_BP6_USD", "IMF_IFS_CA", "IMF_IFS_CA_USD", "IMF_IFS_EDNE_USD_XDC_RATE"], errors="ignore")
    for result, num in [
        ("IMF_IFS_cons_GDP", "IMF_IFS_cons"),
        ("IMF_IFS_imports_GDP", "IMF_IFS_imports"),
        ("IMF_IFS_exports_GDP", "IMF_IFS_exports"),
        ("IMF_IFS_finv_GDP", "IMF_IFS_finv"),
        ("IMF_IFS_inv_GDP", "IMF_IFS_inv"),
    ]:
        wide[result] = pd.to_numeric(wide.get(num), errors="coerce") / pd.to_numeric(wide.get("IMF_IFS_nGDP"), errors="coerce") * 100

    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int16")
    wide = _sort_keys(wide)
    wide = wide[["ISO3", "year"] + [col for col in wide.columns if col not in {"ISO3", "year"}]]
    out_path = clean_dir / "aggregators" / "IMF" / "IMF_IFS.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_imf_ifs"]
