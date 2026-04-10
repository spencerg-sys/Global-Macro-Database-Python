from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})


def _availability_stats(df: pd.DataFrame, *, current_year: int, exclude_pop: bool = False) -> dict[str, int | None]:
    value_cols = [col for col in df.columns if col not in {"ISO3", "year", "countryname"}]
    if exclude_pop:
        value_cols = [col for col in value_cols if col != "pop"]
    if not value_cols:
        return {"from": None, "from_median": None, "to": None, "forecasts": None, "num_country": None, "num_obs": None}

    work = df[["ISO3", "year", *value_cols]].copy()
    mask = work[value_cols].notna().any(axis=1)
    active = work.loc[mask].copy()
    if active.empty:
        return {"from": None, "from_median": None, "to": None, "forecasts": None, "num_country": None, "num_obs": None}

    active["year"] = pd.to_numeric(active["year"], errors="coerce").astype(int)
    starts = active.groupby("ISO3")["year"].min()
    to_year = int(active["year"].max())
    return {
        "from": int(active["year"].min()),
        "from_median": int(float(starts.median())),
        "to": to_year,
        "forecasts": max(0, to_year - (current_year - 1)),
        "num_country": int(active["ISO3"].astype(str).nunique()),
        "num_obs": int(len(active)),
    }


def _ratio_string(value: float | int | None) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "---"
    text = f"{float(numeric):.1f}"
    return text[:-2] if text.endswith(".0") else text


def _gmd_variable_counts(data_final: pd.DataFrame) -> pd.Series:
    value_cols = [col for col in data_final.columns if col not in {"ISO3", "year", "countryname"}]
    counts = {col: int(pd.to_numeric(data_final[col], errors="coerce").notna().sum()) for col in value_cols}
    return pd.Series(counts, dtype="float64")


def _source_variable_counts(clean_wide: pd.DataFrame, source: str) -> pd.Series:
    counts: dict[str, int] = {}
    frame = _source_frame(clean_wide, source)
    for var in PAPER_VARIABLE_ORDER:
        if var not in frame.columns:
            continue
        counts[var] = _count_nonmissing(frame[var])
    return pd.Series(counts, dtype="float64")


def _gfd_variable_counts(data_helper_dir: Path | str = sh.DATA_HELPER_DIR) -> pd.Series:
    gfd = _gfd_frame(data_helper_dir=data_helper_dir)
    counts: dict[str, int] = {}
    for column in gfd.columns:
        if not column.startswith("GFD_"):
            continue
        variable = column.removeprefix("GFD_")
        counts[variable] = _count_nonmissing(gfd[column])
    return pd.Series(counts, dtype="float64")


def build_paper_tab_obs_count(
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
    tables_dir: Path | str = sh.OUTPUT_TABLES_DIR,
) -> Path:
    clean_wide = _load_clean_wide(
        data_clean_dir=data_clean_dir,
        data_temp_dir=data_temp_dir,
        data_final_dir=data_final_dir,
    )
    data_final = _load_data_final(data_final_dir=data_final_dir)

    gmd_counts = _gmd_variable_counts(data_final)
    source_counts = {
        source: _source_variable_counts(clean_wide, source)
        for source in PAPER_COMPARISON_SOURCES
        if source != "GFD"
    }
    source_counts["GFD"] = _gfd_variable_counts(data_helper_dir=data_helper_dir)

    rows: list[dict[str, object]] = []
    for var in PAPER_VARIABLE_ORDER:
        gmd = pd.to_numeric(pd.Series([gmd_counts.get(var)]), errors="coerce").iloc[0]
        if pd.isna(gmd) or int(gmd) == 0:
            continue
        row: dict[str, object] = {
            "var": var,
            "variable": PAPER_VARIABLE_LABELS.get(var, var),
            "GMD": _comma_int(gmd),
        }
        for source in PAPER_COMPARISON_SOURCES:
            current = pd.to_numeric(pd.Series([source_counts[source].get(var)]), errors="coerce").iloc[0]
            if pd.isna(current) or float(gmd) == 0:
                row[source] = "---"
            else:
                row[source] = _ratio_string(round((float(current) / float(gmd)) * 100, 1))
        rows.append(row)

    out = _paper_variable_frame(rows)
    if out.empty:
        path = _resolve(tables_dir) / "tab_obs_counts.tex"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return path

    gfd_cpi = out.loc[out["var"].eq("CPI"), "GFD"]
    if not gfd_cpi.empty:
        out.loc[out["var"].eq("infl"), "GFD"] = gfd_cpi.iloc[0]
    out.loc[out["var"].isin(["M2", "M3"]), "GFD"] = "---"

    out = out[["variable", "GMD", "IMF_IFS", "IMF_WEO", "OECD_EO", "WDI", "UN", "JST", "Mitchell", "GFD"]].copy()
    path = _resolve(tables_dir) / "tab_obs_counts.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    sh.gmdwriterows(out, out.columns.tolist(), path)
    return path


def build_paper_tab_no_sources(
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
    tables_dir: Path | str = sh.OUTPUT_TABLES_DIR,
) -> Path:
    current_year = date.today().year
    clean_wide = _load_clean_wide(
        data_clean_dir=data_clean_dir,
        data_temp_dir=data_temp_dir,
        data_final_dir=data_final_dir,
    )
    sources = _load_sources_csv(data_helper_dir=data_helper_dir).copy()
    sources["source_abbr"] = sources["source_abbr"].astype(str).str.strip()
    sources["varabbr"] = sources["varabbr"].astype(str).str.strip()
    sources = sources.loc[sources["source_abbr"].ne("IDCM")].copy()

    var_counts = sources.groupby("source_abbr")["varabbr"].nunique().rename("varlist")
    meta = sources.drop_duplicates(subset=["source_abbr"]).copy()
    meta = meta[["source_abbr", "download_date", "digitized", "country_specific", "historical"]].copy()
    meta = meta.merge(var_counts, on="source_abbr", how="left")

    download_dates_path = _resolve(data_temp_dir) / "download_dates.dta"
    if download_dates_path.exists():
        download_dates = pd.read_dta(download_dates_path, convert_categoricals=False)
        meta = meta.drop(columns=["download_date"], errors="ignore").merge(download_dates, on="source_abbr", how="left")

    stats_by_source: dict[str, dict[str, int | None]] = {}
    for source_abbr in meta["source_abbr"].astype(str).tolist():
        current = meta.loc[meta["source_abbr"].eq(source_abbr)].iloc[0]
        if str(current["country_specific"]) == "Yes":
            try:
                iso3, version = source_abbr.split("_", 1)
                source_prefix = f"CS{version}"
            except ValueError:
                stats_by_source[source_abbr] = {"from": None, "to": None, "forecasts": None, "num_country": None, "num_obs": None, "from_median": None}
                continue
            source_frame = _source_frame(clean_wide, source_prefix)
            source_frame = source_frame.loc[source_frame["ISO3"].astype(str) == iso3].copy()
        else:
            source_frame = _source_frame(clean_wide, source_abbr)
        stats_by_source[source_abbr] = _availability_stats(source_frame, current_year=current_year)

    rows: list[dict[str, object]] = []
    ordered_meta = meta.sort_values(["country_specific", "source_abbr"]).reset_index(drop=True)
    for _, row in ordered_meta.iterrows():
        stats = stats_by_source.get(str(row["source_abbr"]), {})
        rows.append(
            {
                "source": f"\\citet{{{row['source_abbr']}}}",
                "source_abbr": f"\\citetalias{{{row['source_abbr']}}}",
                "download_date": "" if pd.isna(row.get("download_date")) else str(row.get("download_date")),
                "digitized": "" if pd.isna(row.get("digitized")) else _safe_int_string(row.get("digitized")),
                "from": _safe_int_string(stats.get("from")),
                "to": _safe_int_string(stats.get("to")),
                "forecasts": "---" if pd.to_numeric(pd.Series([stats.get("forecasts")]), errors="coerce").fillna(0).iloc[0] == 0 else _safe_int_string(stats.get("forecasts")),
                "varlist": _safe_int_string(row.get("varlist")),
                "num_country": _safe_int_string(stats.get("num_country")),
                "historical": "" if pd.isna(row.get("historical")) else str(row.get("historical")),
            }
        )

    out = pd.DataFrame(rows, columns=["source", "source_abbr", "download_date", "digitized", "from", "to", "forecasts", "varlist", "num_country", "historical"])
    path = _resolve(tables_dir) / "tab_no_sources.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    sh.gmdwriterows(out, out.columns.tolist(), path)
    return path


def build_paper_tab_comparison(
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
    tables_dir: Path | str = sh.OUTPUT_TABLES_DIR,
) -> Path:
    current_year = date.today().year
    clean_wide = _load_clean_wide(
        data_clean_dir=data_clean_dir,
        data_temp_dir=data_temp_dir,
        data_final_dir=data_final_dir,
    )
    data_final = _load_data_final(data_final_dir=data_final_dir)
    sources = _load_sources_csv(data_helper_dir=data_helper_dir).copy()
    sources["source_abbr"] = sources["source_abbr"].astype(str).str.strip()
    sources["varabbr"] = sources["varabbr"].astype(str).str.strip()

    alias_lookup = {"EUROSTAT": "EUS"}
    rows: list[dict[str, object]] = []
    for source in PAPER_MAJOR_SOURCES:
        lookup = alias_lookup.get(source, source)
        var_count = int(sources.loc[sources["source_abbr"].eq(lookup), "varabbr"].nunique())
        stats = _availability_stats(_source_frame(clean_wide, source, include_pop=False), current_year=current_year, exclude_pop=True)
        rows.append(
            {
                "source_abbr": source,
                "from": stats["from"],
                "from_median": stats["from_median"],
                "to": stats["to"],
                "forecasts": stats["forecasts"],
                "num_country": stats["num_country"],
                "num_obs": stats["num_obs"],
                "varlist": var_count,
            }
        )

    gfd = _gfd_frame(data_helper_dir=data_helper_dir)
    gfd_value_cols = [col for col in gfd.columns if col.startswith("GFD_")]
    gfd_stats = _availability_stats(gfd[["ISO3", "year", *gfd_value_cols]], current_year=current_year)
    gfd_var_count = len(gfd_value_cols)
    rows.insert(
        0,
        {
            "source_abbr": "GFD",
            "from": gfd_stats["from"],
            "from_median": gfd_stats["from_median"],
            "to": gfd_stats["to"],
            "forecasts": gfd_stats["forecasts"],
            "num_country": gfd_stats["num_country"],
            "num_obs": gfd_stats["num_obs"],
            "varlist": gfd_var_count,
        },
    )

    data_final_work = data_final.copy()
    value_cols = [col for col in data_final_work.columns if col not in {"ISO3", "year", "countryname"}]
    data_final_active = data_final_work.loc[data_final_work[value_cols].notna().any(axis=1), ["ISO3", "year", *value_cols]].copy()
    gmd_stats = _availability_stats(data_final_active, current_year=current_year)
    rows.insert(
        0,
        {
            "source_abbr": "GMD",
            "from": gmd_stats["from"],
            "from_median": gmd_stats["from_median"],
            "to": gmd_stats["to"],
            "forecasts": gmd_stats["forecasts"],
            "num_country": gmd_stats["num_country"],
            "num_obs": gmd_stats["num_obs"],
            "varlist": len(value_cols),
        },
    )

    out = pd.DataFrame(rows)
    out["to_actual"] = pd.to_numeric(out["to"], errors="coerce") - pd.to_numeric(out["forecasts"], errors="coerce").fillna(0)
    out["to_forecast"] = np.where(pd.to_numeric(out["forecasts"], errors="coerce").fillna(0).gt(0), out["to"], np.nan)

    out["source_abbr"] = out["source_abbr"].astype(str)
    out.loc[out["source_abbr"].ne("GMD"), "source_abbr"] = out.loc[out["source_abbr"].ne("GMD"), "source_abbr"].map(lambda value: f"\\citetalias{{{value}}}")

    out["from"] = out["from"].map(_safe_int_string)
    out["from_median"] = out["from_median"].map(_safe_int_string)
    out["to_actual"] = out["to_actual"].map(_safe_int_string)
    out["to_forecast"] = out["to_forecast"].map(lambda value: "---" if _safe_int_string(value) == "" else _safe_int_string(value))
    out["num_country"] = out["num_country"].map(_safe_int_string)
    out["num_obs"] = out["num_obs"].map(_comma_int)
    out["varlist"] = out["varlist"].map(_safe_int_string)

    out = out[["source_abbr", "from", "from_median", "to_actual", "to_forecast", "num_country", "num_obs", "varlist"]].copy()
    path = _resolve(tables_dir) / "tab_comparison.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    sh.gmdwriterows(out, out.columns.tolist(), path)
    return path


def build_paper_tab_variable_descriptions(
    *,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
    tables_dir: Path | str = sh.OUTPUT_TABLES_DIR,
) -> list[Path]:
    docvars = _load_docvars_csv(data_helper_dir=data_helper_dir).copy()
    docvars = docvars.loc[docvars["final_var_list"].fillna("").astype(str).ne("")].copy()
    docvars = docvars[["codes", "units", "label"]].copy()

    docvars["units"] = docvars["units"].astype(str)
    docvars["label"] = docvars["label"].astype(str)
    docvars.loc[docvars["units"].eq("millions"), "units"] = "Millions"
    docvars.loc[docvars["units"].eq("in %"), "units"] = "%"
    docvars.loc[docvars["units"].eq("index, 2010 = 100"), "units"] = "Index, 2010 = 100"
    docvars.loc[docvars["label"].eq("Real GDP Per Capita"), "units"] = "LC"
    docvars.loc[docvars["label"].eq("USD Exchange Rate"), "units"] = "1 USD in LC"
    docvars = docvars.rename(columns={"label": "varname", "codes": "abbr"})

    data_final = _load_data_final(data_final_dir=data_final_dir)
    value_cols = [col for col in data_final.columns if col not in {"ISO3", "year", "countryname"}]
    rows: list[dict[str, object]] = []
    current_year = date.today().year
    for var in value_cols:
        series = pd.to_numeric(data_final[var], errors="coerce")
        available = data_final.loc[series.notna(), ["ISO3", "year"]].copy()
        if available.empty:
            continue
        years = pd.to_numeric(available["year"], errors="coerce")
        rows.append(
            {
                "abbr": var,
                "from": int(years.min()),
                "to": int(years.max()),
                "forecasts": max(0, int(years.max()) - (current_year - 1)),
                "countries": int(available["ISO3"].astype(str).nunique()),
            }
        )

    summary = pd.DataFrame(rows)
    out = docvars.merge(summary, on="abbr", how="inner")
    out["abbr"] = out["abbr"].astype(str)

    order_lookup = {
        "Nominal GDP": 1,
        "Real GDP": 2,
        "Real GDP in USD": 3,
        "Real GDP Per Capita": 4,
        "GDP Deflator": 5,
        "Population": 6,
        "Real final consumption": 7,
        "Final consumption": 8,
        "Final consumption in percent of GDP": 9,
        "Gross capital formation": 10,
        "Gross capital formation in percent of GDP": 11,
        "Gross fixed capital formation": 12,
        "Gross fixed capital formation in percent of GDP": 13,
        "Current account": 14,
        "Current account in percent of GDP": 15,
        "Exports": 16,
        "Exports in percent of GDP": 17,
        "Imports": 18,
        "Imports in percent of GDP": 19,
        "Real effective exchange rate": 20,
        "USD Exchange Rate": 21,
        "Government debt": 22,
        "Government debt in percent of GDP": 23,
        "Government deficit": 24,
        "Government deficit in percent of GDP": 25,
        "Government expenditure": 26,
        "Government expenditure in percent of GDP": 27,
        "Government revenue": 28,
        "Government revenue in percent of GDP": 29,
        "Government tax revenue": 30,
        "Government tax revenue in percent of GDP": 31,
        "M0": 32,
        "M1": 33,
        "M2": 34,
        "M3": 35,
        "M4": 36,
        "Central bank policy rate": 37,
        "Short-term interest rate": 38,
        "Long-term interest rate": 39,
        "Consumer price index": 40,
        "House price index": 41,
        "Inflation": 42,
        "Unemployment rate": 43,
        "Banking crisis dummy": 44,
        "Sovereign debt crisis dummy": 45,
        "Currency crisis dummy": 46,
    }
    out["order"] = out["varname"].map(order_lookup)
    out = out.dropna(subset=["order"]).sort_values("order").reset_index(drop=True)
    out = out.drop(columns=["order"])

    out["varname"] = out["varname"].map(_latex_escape_text)
    out["abbr"] = out["abbr"].map(lambda value: f"\\texttt{{{_latex_escape_text(value)}}}")
    out["units"] = out["units"].map(_latex_escape_text)
    out["from"] = out["from"].map(_safe_int_string)
    out["to"] = out["to"].map(_safe_int_string)
    out["forecasts"] = out["forecasts"].map(lambda value: "---" if _safe_int_string(value) in {"", "0"} else _safe_int_string(value))
    out["countries"] = out["countries"].map(_safe_int_string)

    panels = [("A", 0, 6), ("B", 6, 13), ("C", 13, 21), ("D", 21, 31), ("E", 31, 39), ("F", 39, 43), ("G", 43, 46)]
    paths: list[Path] = []
    target_dir = _ensure_output_dir(tables_dir)
    for panel, start, end in panels:
        subset = out.iloc[start:end].copy()
        path = target_dir / f"tab_variable_descriptions_{panel}.tex"
        sh.gmdwriterows(subset, subset.columns.tolist(), path)
        paths.append(path)
    return paths


__all__ = [
    "build_paper_tab_comparison",
    "build_paper_tab_no_sources",
    "build_paper_tab_obs_count",
    "build_paper_tab_variable_descriptions",
]
