from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_tena_usdfx(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    df = _read_excel_compat(raw_dir / "aggregators" / "Tena" / "trade" / "Tena_USDfx.xlsx", sheet_name="Processed")
    df = df.rename(columns={"year": "year"}).copy()
    long = df.melt(id_vars=["year"], var_name="countryname", value_name="USDfx")
    long["countryname"] = long["countryname"].replace(
        {
            "Brasil": "Brazil",
            "Costa Rica": "Costa Rica",
            "Czechoslowakia": "Czechoslovakia",
            "Dominican Republic": "Dominican Republic",
            "El Salvador": "El Salvador",
            "Korea": "South Korea",
            "New Zeland": "New Zealand",
            "Ottoman Empire/Turkey": "Turkey",
            "Persia (Iran)": "Iran",
            "Russia/USSR": "Russian Federation",
            "Serbia/Yugoslavia": "Serbia",
            "Siam (Thailand)": "Thailand",
            "United Kingdom": "United Kingdom",
            "United states": "United States",
        }
    )
    lookup = _country_name_lookup(helper_dir)
    iso_map = pd.DataFrame({"countryname": list(lookup.keys()), "ISO3": list(lookup.values())})
    long = long.merge(iso_map, on="countryname", how="left")
    long = long.loc[long["ISO3"].notna(), ["ISO3", "year", "USDfx"]].copy()
    long["USDfx"] = pd.to_numeric(long["USDfx"], errors="coerce")
    long.loc[pd.to_numeric(long["USDfx"], errors="coerce").eq(0), "USDfx"] = pd.NA

    long.loc[long["ISO3"].eq("AUS"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("AUS"), "USDfx"], errors="coerce") * 2
    long.loc[long["ISO3"].eq("BOL"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("BOL"), "USDfx"], errors="coerce") * _pow10_literal(-9)
    long.loc[long["ISO3"].eq("NZL"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("NZL"), "USDfx"], errors="coerce") * 2
    long.loc[long["ISO3"].eq("URY"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("URY"), "USDfx"], errors="coerce") * _pow10_literal(-6)
    long.loc[long["ISO3"].eq("BRA"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("BRA"), "USDfx"], errors="coerce") / 2750
    long.loc[long["ISO3"].eq("BRA"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("BRA"), "USDfx"], errors="coerce") * _pow10_literal(-12)
    long.loc[long["ISO3"].eq("VEN"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("VEN"), "USDfx"], errors="coerce") / 1000
    long.loc[long["ISO3"].eq("MEX"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("MEX"), "USDfx"], errors="coerce") / 1000
    long.loc[long["ISO3"].eq("NIC"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("NIC"), "USDfx"], errors="coerce") / 500000000
    long.loc[long["ISO3"].eq("NIC"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("NIC"), "USDfx"], errors="coerce") / 12.5
    long.loc[long["ISO3"].eq("BGR"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("BGR"), "USDfx"], errors="coerce") * _pow10_literal(-6)
    long.loc[long["ISO3"].eq("PRY"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("PRY"), "USDfx"], errors="coerce") / 100
    long.loc[long["ISO3"].eq("PER"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("PER"), "USDfx"], errors="coerce") / 1000000000
    long.loc[long["ISO3"].eq("CHL"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("CHL"), "USDfx"], errors="coerce") / 1000
    long.loc[long["ISO3"].eq("ISL"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("ISL"), "USDfx"], errors="coerce") / 100
    long.loc[long["ISO3"].eq("FRA"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("FRA"), "USDfx"], errors="coerce") / 100
    long.loc[long["ISO3"].eq("TUR"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("TUR"), "USDfx"], errors="coerce") / 100000
    long.loc[long["ISO3"].eq("ROU"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("ROU"), "USDfx"], errors="coerce") * _pow10_literal(-8)
    long.loc[long["ISO3"].eq("ROU"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("ROU"), "USDfx"], errors="coerce") / 2
    long.loc[long["ISO3"].eq("ARG"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("ARG"), "USDfx"], errors="coerce") * _pow10_literal(-13)
    long.loc[long["ISO3"].eq("VEN"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("VEN"), "USDfx"], errors="coerce") * _pow10_literal(-11, adjust=-1)
    long.loc[long["ISO3"].eq("EST"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("EST"), "USDfx"], errors="coerce") * (10**-1)
    long.loc[long["ISO3"].eq("EST"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("EST"), "USDfx"], errors="coerce") * (10**-1)
    long.loc[long["ISO3"].eq("LVA"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("LVA"), "USDfx"], errors="coerce") / 200
    long.loc[long["ISO3"].eq("LTU"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("LTU"), "USDfx"], errors="coerce") / 200
    long.loc[long["ISO3"].eq("RUS"), "USDfx"] = pd.to_numeric(long.loc[long["ISO3"].eq("RUS"), "USDfx"], errors="coerce") / 10000
    long = long.loc[~long["ISO3"].eq("SRB")].copy()

    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].copy()
    long = long.merge(eur_fx, on="ISO3", how="left")
    fx_mask = long["EUR_irrevocable_FX"].notna() & pd.to_numeric(long["year"], errors="coerce").le(1998)
    long.loc[fx_mask, "USDfx"] = pd.to_numeric(long.loc[fx_mask, "USDfx"], errors="coerce") / pd.to_numeric(long.loc[fx_mask, "EUR_irrevocable_FX"], errors="coerce")
    long = long.drop(columns=["EUR_irrevocable_FX"], errors="ignore")

    long = long.rename(columns={"USDfx": "Tena_USDfx"})
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("int16")
    long["Tena_USDfx"] = pd.to_numeric(long["Tena_USDfx"], errors="coerce").astype("float64")
    long = long[["ISO3", "year", "Tena_USDfx"]].copy()
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("int16")
    long["Tena_USDfx"] = pd.to_numeric(long["Tena_USDfx"], errors="coerce").astype("float64")
    long = _sort_keys(long)
    out_path = clean_dir / "aggregators" / "Tena" / "trade" / "Tena_USDfx.dta"
    _save_dta(long, out_path)
    return long
__all__ = ["clean_tena_usdfx"]
