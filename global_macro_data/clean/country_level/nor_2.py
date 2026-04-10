from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_nor_2(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    path = raw_dir / "country_level" / "NOR_2.xlsx"

    cbrate = pd.read_excel(path, sheet_name="cbrate")
    cbrate = cbrate.loc[cbrate["month"].astype("string").str.contains(":12", regex=False, na=False)].copy()
    cbrate["year"] = pd.to_numeric(cbrate["month"].astype("string").str.slice(0, 4), errors="coerce")
    cbrate = cbrate.drop(columns=["month"], errors="ignore")
    cbrate["cbrate"] = pd.to_numeric(cbrate["cbrate"], errors="coerce")
    cbrate["ISO3"] = "NOR"
    master = cbrate[["ISO3", "year", "cbrate"]].copy()

    for sheet in ["fx", "CPI", "gov", "ltrate", "strate", "HPI"]:
        current = pd.read_excel(path, sheet_name=sheet)
        for col in current.columns:
            current[col] = pd.to_numeric(current[col], errors="coerce")
        current["ISO3"] = "NOR"
        master = current[["ISO3", "year"] + [c for c in current.columns if c not in {"ISO3", "year"}]].merge(master, on=["ISO3", "year"], how="outer")

    master["Hamburg"] = pd.to_numeric(master["Hamburg"], errors="coerce").where(
        pd.to_numeric(master["Hamburg"], errors="coerce").notna(),
        pd.to_numeric(master["Copenhagen"], errors="coerce"),
    )
    master["Hamburg"] = pd.to_numeric(master["Hamburg"], errors="coerce").where(
        pd.to_numeric(master["Hamburg"], errors="coerce").notna(),
        pd.to_numeric(master["London"], errors="coerce"),
    )
    master = master.rename(columns={"Hamburg": "ltrate"})
    master = master.drop(columns=["London", "Copenhagen", "Paris", "Berlin", "Oslo"], errors="ignore")
    master["pop"] = pd.to_numeric(master["pop"], errors="coerce") / 1_000_000

    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    prev_cpi = pd.to_numeric(master["CPI"], errors="coerce").groupby(master["ISO3"]).shift(1)
    prev_year = pd.to_numeric(master["year"], errors="coerce").groupby(master["ISO3"]).shift(1)
    year_num = pd.to_numeric(master["year"], errors="coerce")
    infl = (pd.to_numeric(master["CPI"], errors="coerce") - prev_cpi) / prev_cpi * 100
    master["infl"] = infl.where(prev_cpi.notna() & year_num.eq(prev_year + 1))

    n_gdp = pd.to_numeric(master["nGDP"], errors="coerce")
    master["imports_GDP"] = pd.to_numeric(master["imports"], errors="coerce") / n_gdp * 100
    master["exports_GDP"] = pd.to_numeric(master["exports"], errors="coerce") / n_gdp * 100
    master["govexp_GDP"] = pd.to_numeric(master["govexp"], errors="coerce") / n_gdp * 100

    master = master.rename(columns={col: f"CS2_{col}" for col in master.columns if col not in {"ISO3", "year"}})
    master = master.drop(columns=["CS2_GBPfx", "CS2_DEMfx", "CS2_SEKfx", "CS2_FRFfx"], errors="ignore")
    for col in [
        "CS2_HPI",
        "CS2_strate",
        "CS2_ltrate",
        "CS2_govexp",
        "CS2_inv",
        "CS2_exports",
        "CS2_imports",
        "CS2_nGDP",
        "CS2_nGDP_pc",
        "CS2_pop",
        "CS2_rGDP",
        "CS2_rGDP_pc",
        "CS2_CPI",
        "CS2_USDfx",
        "CS2_cbrate",
    ]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
    for col in ["CS2_infl", "CS2_imports_GDP", "CS2_exports_GDP", "CS2_govexp_GDP"]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float32")
    master = master[
        [
            "ISO3",
            "year",
            "CS2_HPI",
            "CS2_strate",
            "CS2_ltrate",
            "CS2_govexp",
            "CS2_inv",
            "CS2_exports",
            "CS2_imports",
            "CS2_nGDP",
            "CS2_nGDP_pc",
            "CS2_pop",
            "CS2_rGDP",
            "CS2_rGDP_pc",
            "CS2_CPI",
            "CS2_USDfx",
            "CS2_cbrate",
            "CS2_infl",
            "CS2_imports_GDP",
            "CS2_exports_GDP",
            "CS2_govexp_GDP",
        ]
    ].copy()
    master = _sort_keys(master)
    out_path = clean_dir / "country_level" / "NOR_2.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_nor_2"]
