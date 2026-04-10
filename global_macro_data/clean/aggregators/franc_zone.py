from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_franc_zone(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)

    df = _load_dta(raw_dir / "aggregators" / "FRANC_ZONE" / "FRANC_ZONE.dta")
    df["value"] = df["value"].astype(str).replace("NA", "")
    df = df[["period", "country", "series_code", "value"]].copy()
    df["series_code"] = df["series_code"].astype(str).str.split(".", n=1).str[0]

    keys = df[["country", "period"]].drop_duplicates().reset_index(drop=True)
    wide_values = df.pivot_table(index=["country", "period"], columns="series_code", values="value", aggfunc="first").reset_index()
    wide_values.columns.name = None
    wide = keys.merge(wide_values, on=["country", "period"], how="left")

    for col in [c for c in wide.columns if c not in {"country", "period"}]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce")

    if {"gdp_FCFA", "gdp_KMF"}.issubset(wide.columns):
        wide.loc[wide["country"].astype(str) == "COM", "gdp_FCFA"] = wide.loc[wide["country"].astype(str) == "COM", "gdp_KMF"]
    wide = wide.drop(columns=[c for c in wide.columns if c.endswith("KMF")], errors="ignore")

    wide = wide.rename(
        columns={
            "country": "ISO3",
            "period": "year",
            "price_index_percent": "infl",
            "money_FCFA": "M2",
            "investment": "inv_GDP",
            "gdp_FCFA": "nGDP",
            "budget_balance_percent": "govdef_GDP",
        }
    )
    if "nGDP" in wide.columns:
        wide["nGDP"] = pd.to_numeric(wide["nGDP"], errors="coerce") * 1000
    if "M2" in wide.columns:
        wide["M2"] = pd.to_numeric(wide["M2"], errors="coerce") * 1000
    if {"inv_GDP", "nGDP"}.issubset(wide.columns):
        wide["inv"] = pd.to_numeric(wide["inv_GDP"], errors="coerce") * pd.to_numeric(wide["nGDP"], errors="coerce") / 100
    if {"govdef_GDP", "nGDP"}.issubset(wide.columns):
        wide["govdef"] = pd.to_numeric(wide["govdef_GDP"], errors="coerce") * pd.to_numeric(wide["nGDP"], errors="coerce") / 100

    wide = wide.rename(columns={col: f"FRANC_ZONE_{col}" for col in wide.columns if col not in {"ISO3", "year"}})
    ordered_cols = [
        "ISO3",
        "year",
        "FRANC_ZONE_govdef_GDP",
        "FRANC_ZONE_nGDP",
        "FRANC_ZONE_inv_GDP",
        "FRANC_ZONE_M2",
        "FRANC_ZONE_infl",
        "FRANC_ZONE_inv",
        "FRANC_ZONE_govdef",
    ]
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int32")
    wide = _sort_keys(wide)
    wide = wide[[col for col in ordered_cols if col in wide.columns]]
    out_path = clean_dir / "aggregators" / "FRANC_ZONE" / "FRANC_ZONE.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_franc_zone"]
