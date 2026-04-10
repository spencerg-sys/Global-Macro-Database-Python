from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_th_id(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    def _th_id_f32(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").astype("float32")

    df = _load_dta(raw_dir / "aggregators" / "TH_ID" / "TH_ID.dta")
    df["year"] = pd.to_datetime(df["Time"], errors="coerce").dt.year
    df = df.rename(columns={"country": "countryname"})
    df["countryname"] = df["countryname"].replace(
        {
            "GreatBritain": "United Kingdom",
            "UnitedStates": "United States",
            "NewZealand": "New Zealand",
            "SouthAfrica": "South Africa",
        }
    )

    lookup = _country_name_lookup(helper_dir)
    iso_map = pd.DataFrame({"countryname": list(lookup.keys()), "ISO3": list(lookup.values())})
    df = df.merge(iso_map, on="countryname", how="left")
    df = df.loc[df["ISO3"].notna(), ["ISO3", "year", "TOTEX", "TOTIM", "Time"]].copy()
    df = df.sort_values(["ISO3", "year", "Time"], kind="mergesort").reset_index(drop=True)
    df["exports"] = pd.to_numeric(df["TOTEX"], errors="coerce").fillna(0).groupby([df["ISO3"], df["year"]], sort=False).cumsum()
    df["imports"] = pd.to_numeric(df["TOTIM"], errors="coerce").fillna(0).groupby([df["ISO3"], df["year"]], sort=False).cumsum()
    df["exports"] = _th_id_f32(df["exports"])
    df["imports"] = _th_id_f32(df["imports"])
    df["month"] = pd.to_datetime(df["Time"], errors="coerce").dt.month
    df = df.loc[df["month"].eq(12), ["ISO3", "year", "imports", "exports"]].copy()
    df = df.loc[~(pd.to_numeric(df["exports"], errors="coerce").eq(0) | pd.to_numeric(df["imports"], errors="coerce").eq(0))].copy()
    df = df.rename(columns={"imports": "TH_ID_imports", "exports": "TH_ID_exports"})

    for col in ["TH_ID_exports", "TH_ID_imports"]:
        df[col] = _th_id_f32(df[col])
        for iso3, factor in [
            ("AUS", 2 / 1000),
            ("BGR", 1 / 1_000_000),
            ("CHL", 1 / 1000),
            ("EST", 1 / 1000),
            ("POL", 1 / 1000),
            ("FIN", 1 / 100),
            ("FRA", 1 / 100),
            ("MEX", 1 / 1000),
            ("ROU", 1 / 200_000_000),
            ("ZAF", 2 / 1000),
        ]:
            mask = df["ISO3"].eq(iso3)
            if mask.any():
                scaled = pd.to_numeric(df.loc[mask, col], errors="coerce").astype("float64") * factor
                df.loc[mask, col] = scaled.astype("float32")
                df[col] = _th_id_f32(df[col])
        df.loc[df["ISO3"].eq("YUG"), col] = pd.NA
        df[col] = _th_id_f32(df[col])

    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].copy()
    df = df.merge(eur_fx, on="ISO3", how="left")
    fx_mask = df["EUR_irrevocable_FX"].notna()
    for col in ["TH_ID_imports", "TH_ID_exports"]:
        converted = (
            pd.to_numeric(df.loc[fx_mask, col], errors="coerce").astype("float64")
            / pd.to_numeric(df.loc[fx_mask, "EUR_irrevocable_FX"], errors="coerce").astype("float64")
        ).astype("float32")
        df.loc[fx_mask, col] = converted
        df[col] = _th_id_f32(df[col])
    df = df.drop(columns=["EUR_irrevocable_FX"], errors="ignore")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("float32")
    df["TH_ID_exports"] = pd.to_numeric(df["TH_ID_exports"], errors="coerce").astype("float32")
    df["TH_ID_imports"] = pd.to_numeric(df["TH_ID_imports"], errors="coerce").astype("float32")
    df = df[["ISO3", "year", "TH_ID_exports", "TH_ID_imports"]].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("float32")
    df["TH_ID_exports"] = pd.to_numeric(df["TH_ID_exports"], errors="coerce").astype("float32")
    df["TH_ID_imports"] = pd.to_numeric(df["TH_ID_imports"], errors="coerce").astype("float32")
    df = _sort_keys(df)
    out_path = clean_dir / "aggregators" / "TH_ID" / "TH_ID.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_th_id"]
