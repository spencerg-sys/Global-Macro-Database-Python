from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})


def clean_eus(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    df = _load_dta(raw_dir / "aggregators" / "EUS" / "EUS.dta")
    geo = df["geo"].astype(str)
    drop_mask = geo.str.contains(r"[0-9]", regex=True, na=False)
    drop_mask |= geo.str.contains("FX|EFTA|EA|EU|EL|DE_TOT", regex=True, na=False)
    df = df.loc[~drop_mask].copy()

    dataset = df["dataset_name"].astype("string")

    def _normalize_text(series: pd.Series) -> pd.Series:
        out = series.astype("string").fillna("")
        for token in ["鈥?", "姣?", "每", "–", "—"]:
            out = out.str.replace(token, "-", regex=False)
        out = out.str.replace(r"\s+", " ", regex=True).str.strip()
        return out

    def _eq_any(series: pd.Series, *values: str) -> pd.Series:
        out = pd.Series(False, index=series.index)
        for value in values:
            out = out | series.eq(value)
        return out

    def _dataset_eq_any(*values: str) -> pd.Series:
        return _eq_any(_normalize_text(dataset), *[str(v).strip() for v in values])

    def _series_contains_any(*values: str) -> pd.Series:
        out = pd.Series(False, index=df.index)
        cur = _normalize_text(df["series_name"])
        for value in values:
            out = out | cur.str.contains(str(value).strip(), na=False, regex=False)
        return out

    df.loc[_dataset_eq_any("Real effective exchange rate - index, 42 trading partners"), "series_name"] = "REER"
    df.loc[dataset.eq("Unemployment rate - annual data"), "series_name"] = "unemp"
    df.loc[dataset.eq("Interest rates - monthly data"), "series_name"] = "strate"
    df.loc[_series_contains_any("Population on 1 January"), "series_name"] = "pop"
    df.loc[_series_contains_any("Total receipts from taxes"), "series_name"] = "govtax"
    df.loc[_series_contains_any("government expenditure"), "series_name"] = "govexp"
    df.loc[_series_contains_any("government revenue"), "series_name"] = "govrev"
    df.loc[_series_contains_any("Net lending (+)/net borrowing (-)"), "series_name"] = "govdef_GDP"
    df.loc[_dataset_eq_any("House price index (2015 = 100) - quarterly data"), "series_name"] = "HPI"
    df.loc[_series_contains_any("Exports of goods and services"), "series_name"] = "exports"
    df.loc[_series_contains_any("Imports of goods and services"), "series_name"] = "imports"
    df.loc[_series_contains_any("Gross fixed capital formation"), "series_name"] = "finv"
    df.loc[_series_contains_any("Gross capital formation"), "series_name"] = "inv"
    df.loc[_series_contains_any("Final consumption expenditure"), "series_name"] = "cons"
    df.loc[
        _series_contains_any(
            "Current prices, million units of national currency - Gross domestic product at market prices",
        ),
        "series_name",
    ] = "nGDP"
    df.loc[
        _series_contains_any(
            "Chain linked volumes (2010), million units of national currency - Gross domestic product at market prices",
        ),
        "series_name",
    ] = "rGDP"
    df.loc[_series_contains_any("Growth rate (t/t-12)"), "series_name"] = "infl"
    df.loc[_series_contains_any("Monthly - Harmonized consumer price index, 2015=100"), "series_name"] = "CPI"

    df["year"] = df["period"].astype(str).str.slice(0, 4)
    df = df.drop(columns=["period"])
    # The reference pipeline only sorts on geo/series/year before `by ...: keep if _n == _N`.
    # Using a stable sort preserves the original raw input order within each
    # year-specific group, which is what determines the kept monthly/quarterly
    # observation for CPI/HPI/infl/strate.
    df = df.sort_values(["geo", "series_name", "year"], kind="mergesort")
    df = df.groupby(["geo", "series_name", "year"], as_index=False, sort=False).tail(1)
    df = df[["year", "geo", "value", "series_name"]].copy()

    wide = df.pivot(index=["year", "geo"], columns="series_name", values="value").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={col: f"EUS_{col}" for col in wide.columns if col not in {"year", "geo"}})

    wide["EUS_pop"] = pd.to_numeric(wide["EUS_pop"], errors="coerce") / 1_000_000
    wide["EUS_govdef"] = pd.to_numeric(wide["EUS_govdef_GDP"], errors="coerce") * pd.to_numeric(wide["EUS_nGDP"], errors="coerce") / 100

    wide = wide.rename(columns={"geo": "ISO2"})
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int16")
    wide.loc[wide["ISO2"].astype(str) == "UK", "ISO2"] = "GB"

    countrylist = _load_dta(helper_dir / "countrylist.dta")[["ISO2", "ISO3"]].copy()
    wide = wide.merge(countrylist, on="ISO2", how="left")
    wide = wide.loc[wide["ISO3"].notna()].drop(columns=["ISO2"]).copy()

    for result, num in [
        ("EUS_cons_GDP", "EUS_cons"),
        ("EUS_imports_GDP", "EUS_imports"),
        ("EUS_exports_GDP", "EUS_exports"),
        ("EUS_finv_GDP", "EUS_finv"),
        ("EUS_inv_GDP", "EUS_inv"),
        ("EUS_govrev_GDP", "EUS_govrev"),
        ("EUS_govexp_GDP", "EUS_govexp"),
        ("EUS_govtax_GDP", "EUS_govtax"),
    ]:
        wide[result] = pd.to_numeric(wide[num], errors="coerce") / pd.to_numeric(wide["EUS_nGDP"], errors="coerce") * 100

    for col in [
        "EUS_govdef",
        "EUS_cons_GDP",
        "EUS_imports_GDP",
        "EUS_exports_GDP",
        "EUS_finv_GDP",
        "EUS_inv_GDP",
        "EUS_govrev_GDP",
        "EUS_govexp_GDP",
        "EUS_govtax_GDP",
    ]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float32")

    wide = _apply_clean_overrides(wide, source_name="EUS", data_helper_dir=helper_dir)
    wide = _sort_keys(wide)
    wide = wide[["ISO3", "year"] + [col for col in wide.columns if col not in {"ISO3", "year"}]]
    if wide.duplicated(["ISO3", "year"]).any():
        raise ValueError("EUS contains duplicate ISO3-year keys after processing.")

    out_path = clean_dir / "aggregators" / "EUS" / "EUS.dta"
    _save_dta(wide, out_path)
    return wide


__all__ = ["clean_eus"]
