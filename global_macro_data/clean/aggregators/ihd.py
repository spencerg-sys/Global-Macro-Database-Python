from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_ihd(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    path = raw_dir / "aggregators" / "IHD" / "IHD.csv"

    raw = pd.read_csv(path, encoding="latin-1", dtype=str)
    raw = raw.loc[raw["title1eng"].isin(["Foreign trade", "Money and Credit", "Prices"])].copy()
    raw = raw.loc[~(raw["title1eng"].eq("Foreign trade") & ~raw["title2eng"].eq("Overall movement"))].copy()
    raw = raw.loc[
        ~(
            raw["title1eng"].eq("Money and Credit")
            & ~raw["title3eng"].isin(
                [
                    "Banknotes in circulation",
                    "Bank Discount",
                    "Discount Bank (Bank of Italy)",
                    "Discount rate of the Banco de la Nacion de comercio pagares for-credit banks",
                    "Discount rate of the Imperial Bank of India",
                    "Market discount (Milan)",
                    "Reichsbank discount",
                ]
            )
        )
    ].copy()
    raw = raw.loc[
        ~(raw["title1eng"].eq("Prices") & ~raw["title3eng"].isin(["Overall index", "Overall index of"]))
    ].copy()

    raw["IHD"] = raw["title3eng"].map(
        {
            "Export": "exports",
            "Importing": "imports",
            "Bank Discount": "strate",
            "Banknotes in circulation": "M0",
            "Discount rate of the Banco de la Nacion de comercio pagares for-credit banks": "strate",
            "Export incl gold bullion and coins": "exports",
            "Re-export": "re-exports",
            "Discount Bank (Bank of Italy)": "cbrate",
            "Discount rate of the Imperial Bank of India": "cbrate",
            "Reichsbank discount": "cbrate",
            "Market discount (Milan)": "strate",
            "Overall index": "CPI",
            "Overall index of": "CPI",
        }
    )
    raw = raw.loc[~raw["title3eng"].eq("Exports excluding gold bullion and coins")].copy()
    raw = raw.loc[raw["IHD"].notna(), ["country", "year", "month", "value", "note1eng", "IHD", "book", "table"]].copy()

    raw.loc[raw["value"].isin([".", "-"]), "value"] = ""
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw["month"] = pd.to_numeric(raw["month"], errors="coerce")

    def _substr_bytes(value: object, *, start: int, length: int) -> object:
        if pd.isna(value):
            return pd.NA
        raw_bytes = str(value).encode("utf-8")
        sub = raw_bytes[start - 1 : start - 1 + length]
        return sub.decode("utf-8", errors="ignore")

    units = raw["note1eng"].astype("string").map(lambda v: _substr_bytes(v, start=1, length=4))
    unit_mask = units.isin(["1000", chr(163) + "1,"])
    raw.loc[unit_mask, "value"] = raw.loc[unit_mask, "value"] / 1000

    raw["ISO3"] = raw["country"].map(
        {
            "Argentinien": "ARG",
            "Australischer Bund": "AUS",
            "Belgien": "BEL",
            "Brasilien": "BRA",
            "Britisch Indien": "IND",
            "Bulgarien": "BGR",
            "Chile": "CHL",
            "Columbien": "COL",
            "Daenemark": "DNK",
            "Deutsches Reich": "DEU",
            "Estland": "EST",
            "Finnland": "FIN",
            "Frankreich": "FRA",
            "Griechenland": "GRC",
            "Grossbritannien": "GBR",
            "Grossbritannien und Nordirland": "GBR",
            "Italien": "ITA",
            "Japan": "JPN",
            "Jugoslawien": "YUG",
            "Kanada": "CAN",
            "Lettland": "LVA",
            "Litauen": "LTU",
            "Mexiko": "MEX",
            "Neuseeland": "NZL",
            "Niederlaendisch Indien": "IDN",
            "Niederlande": "NLD",
            "Norwegen": "NOR",
            "Oesterreich": "AUT",
            "Peru": "PER",
            "Polen": "POL",
            "Portugal": "PRT",
            "Rumaenien": "ROU",
            "Russland (UdSSR)": "RUS",
            "Schweden": "SWE",
            "Schweiz": "CHE",
            "Spanien": "ESP",
            "Suedafrikanische Union": "ZAF",
            "Union von Suedafrika": "ZAF",
            "Tschechoslowakei": "CSK",
            "Ungarn": "HUN",
            "Ver. St. v. Amerika": "USA",
            "Vereinigte Staaten von Amerika": "USA",
            "Irischer Freistaat": "IRL",
        }
    )
    raw = raw.drop(columns=["country", "note1eng"])
    # import delimited infers these as numeric in the reference pipeline and sort is numeric.
    raw[["year", "book", "table"]] = raw[["year", "book", "table"]].apply(pd.to_numeric, errors="coerce")
    raw = raw.sort_values(
        ["ISO3", "year", "month", "IHD", "book", "table"],
        kind="mergesort",
    )
    raw = raw.drop_duplicates(
        ["ISO3", "year", "month", "IHD"],
        keep="first",
    )

    wide = raw.pivot(index=["ISO3", "year", "month"], columns="IHD", values="value").reset_index()
    wide.columns.name = None
    if "re-exports" in wide.columns:
        wide = wide.rename(columns={"re-exports": "re_exports"})
    required_cols = ["exports", "imports", "re_exports", "M0", "CPI"]
    missing_required = [col for col in required_cols if col not in wide.columns]
    if missing_required:
        raise ValueError(f"IHD missing required reshaped columns: {missing_required}")
    for optional_col in ["cbrate", "strate"]:
        if optional_col not in wide.columns:
            wide[optional_col] = np.nan

    wide = wide.sort_values(["ISO3", "year", "month"]).reset_index(drop=True)
    wide["exports_s"] = pd.to_numeric(wide["exports"], errors="coerce").fillna(0.0).groupby([wide["ISO3"], wide["year"]], sort=False).cumsum()
    wide["imports_s"] = pd.to_numeric(wide["imports"], errors="coerce").fillna(0.0).groupby([wide["ISO3"], wide["year"]], sort=False).cumsum()
    wide["exports_s"] = _materialize_storage(wide["exports_s"], storage="float")
    wide["imports_s"] = _materialize_storage(wide["imports_s"], storage="float")
    wide = wide.drop(columns=["imports", "exports"]).rename(columns={"imports_s": "imports", "exports_s": "exports"})
    wide = wide.loc[wide["month"].eq(12)].copy()
    wide.loc[pd.to_numeric(wide["exports"], errors="coerce").eq(0), "exports"] = np.nan
    wide.loc[pd.to_numeric(wide["imports"], errors="coerce").eq(0), "imports"] = np.nan
    has_reexports = pd.to_numeric(wide["re_exports"], errors="coerce").notna()
    wide.loc[has_reexports, "exports"] = _materialize_storage(
        pd.to_numeric(wide.loc[has_reexports, "exports"], errors="coerce")
        + pd.to_numeric(wide.loc[has_reexports, "re_exports"], errors="coerce"),
        storage="float",
    )
    wide = wide.drop(columns=["month", "re_exports"])

    ihd_storage = {"imports": "float", "exports": "float", "M0": "double"}
    for col, storage in ihd_storage.items():
        wide[col] = _materialize_storage(wide[col], storage=storage)

    def _apply_scale(iso3: str, col: str, *, ops: list[tuple[str, float]]) -> None:
        mask = wide["ISO3"].eq(iso3)
        if not mask.any():
            return
        wide.loc[mask, col] = _apply_scale_chain(
            wide.loc[mask, col],
            ops=ops,
            storage=ihd_storage[col],
        )

    for col in ["imports", "exports", "M0"]:
        _apply_scale("AUS", col, ops=[("mul", 2.0)])
        _apply_scale("URY", col, ops=[("div", 1_000_000.0)])
        _apply_scale("PER", col, ops=[("mul", _pow10_literal(-9))])
        _apply_scale("MEX", col, ops=[("div", 1000.0)])
        _apply_scale("BRA", col, ops=[("mul", 2.750e-15)])
        _apply_scale("ARG", col, ops=[("mul", _pow10_literal(-13))])
        _apply_scale("BGR", col, ops=[("mul", _pow10_literal(-6))])
        _apply_scale("CHL", col, ops=[("mul", _pow10_literal(-3))])
        _apply_scale("POL", col, ops=[("mul", _pow10_literal(-3))])
        _apply_scale("GRC", col, ops=[("mul", _pow10_literal(-3))])
        _apply_scale("ROU", col, ops=[("mul", _pow10_literal(-8)), ("div", 2.0)])
        _apply_scale("ZAF", col, ops=[("mul", 2.0)])
    for iso3 in ["FIN", "FRA"]:
        _apply_scale(iso3, "exports", ops=[("div", 100.0)])
        _apply_scale(iso3, "imports", ops=[("div", 100.0), ("div", 100.0)])

    _apply_scale("NIC", "exports", ops=[("div", 500_000_000.0)])
    _apply_scale("NIC", "imports", ops=[("div", 500_000_000.0), ("div", 500_000_000.0)])
    wide.loc[wide["ISO3"].eq("YUG"), ["imports", "exports", "M0"]] = np.nan

    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].drop_duplicates().copy()
    wide = wide.merge(eur_fx, on="ISO3", how="left")
    matched = wide["EUR_irrevocable_FX"].notna()
    for col in ["imports", "exports", "M0"]:
        wide.loc[matched, col] = _materialize_storage(
            pd.to_numeric(wide.loc[matched, col], errors="coerce")
            / pd.to_numeric(wide.loc[matched, "EUR_irrevocable_FX"], errors="coerce"),
            storage=ihd_storage[col],
        )
        wide[col] = _materialize_storage(wide[col], storage=ihd_storage[col])
    wide = wide.drop(columns=["EUR_irrevocable_FX"]).sort_values(["ISO3", "year"]).reset_index(drop=True)

    prev_cpi = _lag_if_consecutive_year(wide, "CPI")
    wide["infl"] = np.where(
        prev_cpi.notna(),
        (
            pd.to_numeric(wide["CPI"], errors="coerce")
            - pd.to_numeric(prev_cpi, errors="coerce")
        )
        / pd.to_numeric(prev_cpi, errors="coerce")
        * 100,
        np.nan,
    )

    out = wide.rename(columns={col: f"IHD_{col}" for col in wide.columns if col not in {"ISO3", "year"}})
    out = out[
        [
            "ISO3",
            "year",
            "IHD_CPI",
            "IHD_M0",
            "IHD_cbrate",
            "IHD_strate",
            "IHD_exports",
            "IHD_imports",
            "IHD_infl",
        ]
    ].copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int16")
    out["IHD_CPI"] = pd.to_numeric(out["IHD_CPI"], errors="coerce").astype("float64")
    out["IHD_M0"] = pd.to_numeric(out["IHD_M0"], errors="coerce").astype("float64")
    out["IHD_cbrate"] = pd.to_numeric(out["IHD_cbrate"], errors="coerce").astype("float64")
    out["IHD_strate"] = pd.to_numeric(out["IHD_strate"], errors="coerce").astype("float64")
    out["IHD_exports"] = pd.to_numeric(out["IHD_exports"], errors="coerce").astype("float32")
    out["IHD_imports"] = pd.to_numeric(out["IHD_imports"], errors="coerce").astype("float32")
    out["IHD_infl"] = pd.to_numeric(out["IHD_infl"], errors="coerce").astype("float32")
    out = _apply_clean_overrides(out, source_name="IHD", data_helper_dir=helper_dir)
    out = _sort_keys(out)
    if out.duplicated(["ISO3", "year"]).any():
        raise ValueError("IHD contains duplicate ISO3-year keys after processing.")
    out_path = clean_dir / "aggregators" / "IHD" / "IHD.dta"
    _save_dta(out, out_path)
    return out
__all__ = ["clean_ihd"]
