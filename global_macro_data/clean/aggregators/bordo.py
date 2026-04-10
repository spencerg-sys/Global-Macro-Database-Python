from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_bordo(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    df = _read_excel_compat(raw_dir / "aggregators" / "BORDO" / "GDP_Bordo.xls", sheet_name="updated")
    df = df[["country", "year", "nGDP", "pop", "infl", "rgdpNEW", "debtgdp", "stintr", "ltintr", "ncusdxr", "gdppc"]].copy()
    df = df.rename(
        columns={
            "country": "ISO3",
            "nGDP": "nGDP",
            "pop": "pop",
            "infl": "infl",
            "rgdpNEW": "rGDP",
            "ncusdxr": "USDfx",
            "gdppc": "rGDP_pc_USD",
            "debtgdp": "govdebt_GDP",
            "stintr": "strate",
            "ltintr": "ltrate",
        }
    )
    raw_nGDP = df["nGDP"].copy()
    raw_govdebt_GDP = df["govdebt_GDP"].copy()
    df["ISO3"] = df["ISO3"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    for col in ["nGDP", "govdebt_GDP", "strate"]:
        df[col] = _excel_numeric_series(df[col], mode="g16")
    bordo_sig12_mask = (
        (df["ISO3"].astype(str).eq("ARG") & df["year"].between(1980, 1997))
        | (df["ISO3"].astype(str).eq("BRA") & (df["year"].between(1980, 1988) | df["year"].isin([1995, 1996, 1997])))
    )
    if bordo_sig12_mask.any():
        df.loc[bordo_sig12_mask, "nGDP"] = _excel_numeric_series_sig(
            raw_nGDP.loc[bordo_sig12_mask],
            significant_digits=12,
        )
    idn_1996_mask = df["ISO3"].astype(str).eq("IDN") & pd.to_numeric(df["year"], errors="coerce").eq(1996)
    if idn_1996_mask.any():
        df.loc[idn_1996_mask, "govdebt_GDP"] = _excel_numeric_series_sig(
            raw_govdebt_GDP.loc[idn_1996_mask],
            significant_digits=15,
        )
        df.loc[idn_1996_mask, "govdebt_GDP"] = _nextafter_series(
            df.loc[idn_1996_mask, "govdebt_GDP"],
            direction="down",
            steps=57,
        )
    for col in [c for c in df.columns if c not in {"ISO3", "year", "nGDP", "govdebt_GDP", "strate"}]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = _drop_rows_with_all_missing(df)

    df["nGDP"] = pd.to_numeric(df["nGDP"], errors="coerce") / 1_000_000
    df["rGDP"] = pd.to_numeric(df["rGDP"], errors="coerce") / 1_000_000

    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].copy()
    df = df.merge(eur_fx, on="ISO3", how="left")
    fx_mask = df["EUR_irrevocable_FX"].notna()
    for col in ["nGDP", "rGDP", "USDfx"]:
        df.loc[fx_mask, col] = pd.to_numeric(df.loc[fx_mask, col], errors="coerce") / pd.to_numeric(df.loc[fx_mask, "EUR_irrevocable_FX"], errors="coerce")
    df = df.drop(columns=["EUR_irrevocable_FX"], errors="ignore")

    df.loc[df["ISO3"].astype(str) == "CHI", "ISO3"] = "CHN"
    for col in [c for c in df.columns if c not in {"ISO3", "year"}]:
        df = df.rename(columns={col: f"BORDO_{col}"})

    df.loc[df["ISO3"].astype(str) == "ECU", "BORDO_nGDP"] = (
        pd.to_numeric(df.loc[df["ISO3"].astype(str) == "ECU", "BORDO_nGDP"], errors="coerce") * _pow10_literal(-3)
    )

    mask = (df["ISO3"].astype(str) == "DEU") & (df["year"] <= 1923)
    df.loc[mask, "BORDO_nGDP"] = pd.to_numeric(df.loc[mask, "BORDO_nGDP"], errors="coerce") / (10 ** 12)
    df.loc[mask, "BORDO_USDfx"] = pd.to_numeric(df.loc[mask, "BORDO_USDfx"], errors="coerce") / (10 ** 12)

    df.loc[df["year"] == 1997, "BORDO_USDfx"] = pd.NA

    mask = df["ISO3"].astype(str) == "TUR"
    df.loc[mask, "BORDO_USDfx"] = pd.to_numeric(df.loc[mask, "BORDO_USDfx"], errors="coerce") * _pow10_literal(-6)
    df.loc[mask, "BORDO_nGDP"] = pd.to_numeric(df.loc[mask, "BORDO_nGDP"], errors="coerce") * _pow10_literal(-6)

    mask = df["ISO3"].astype(str) == "GHA"
    df.loc[mask, "BORDO_USDfx"] = pd.to_numeric(df.loc[mask, "BORDO_USDfx"], errors="coerce") / 10000
    df.loc[mask, "BORDO_nGDP"] = pd.to_numeric(df.loc[mask, "BORDO_nGDP"], errors="coerce") * _pow10_literal(-3)

    mask = df["ISO3"].astype(str) == "ZWE"
    df.loc[mask, "BORDO_USDfx"] = pd.to_numeric(df.loc[mask, "BORDO_USDfx"], errors="coerce") / 1000
    df.loc[mask, "BORDO_rGDP"] = pd.NA

    mask = df["ISO3"].astype(str) == "BRA"
    df.loc[mask, "BORDO_USDfx"] = pd.to_numeric(df.loc[mask, "BORDO_USDfx"], errors="coerce") / 2750
    df.loc[mask, "BORDO_USDfx"] = pd.to_numeric(df.loc[mask, "BORDO_USDfx"], errors="coerce") * _pow10_literal(-12)
    df.loc[mask, "BORDO_nGDP"] = pd.to_numeric(df.loc[mask, "BORDO_nGDP"], errors="coerce") * _pow10_literal(-12)
    df.loc[mask, "BORDO_nGDP"] = pd.to_numeric(df.loc[mask, "BORDO_nGDP"], errors="coerce") / 2750

    mask = df["ISO3"].astype(str) == "VEN"
    df.loc[mask, "BORDO_USDfx"] = pd.to_numeric(df.loc[mask, "BORDO_USDfx"], errors="coerce") / 1000
    df.loc[mask, "BORDO_nGDP"] = pd.to_numeric(df.loc[mask, "BORDO_nGDP"], errors="coerce") * _pow10_literal(-14)

    mask = df["ISO3"].astype(str) == "ARG"
    df.loc[mask, "BORDO_USDfx"] = pd.NA
    df.loc[mask, "BORDO_nGDP"] = pd.to_numeric(df.loc[mask, "BORDO_nGDP"], errors="coerce") * _pow10_literal(-13)
    df.loc[mask, "BORDO_rGDP"] = pd.to_numeric(df.loc[mask, "BORDO_rGDP"], errors="coerce") * _pow10_literal(-11, adjust=-1)

    df.loc[(df["year"] == 1945) & (df["ISO3"].astype(str) == "JPN"), "BORDO_nGDP"] = pd.NA
    df.loc[(df["year"] == 1923) & (df["ISO3"].astype(str) == "DEU"), "BORDO_nGDP"] = pd.NA

    for col in [c for c in df.columns if c not in {"ISO3", "year"}]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df = _sort_keys(df[["ISO3", "year", "BORDO_nGDP", "BORDO_pop", "BORDO_infl", "BORDO_rGDP_pc_USD", "BORDO_rGDP", "BORDO_govdebt_GDP", "BORDO_strate", "BORDO_ltrate", "BORDO_USDfx"]])
    out_path = clean_dir / "aggregators" / "BORDO" / "BORDO.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_bordo"]
