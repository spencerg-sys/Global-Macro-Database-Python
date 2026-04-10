from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import _core as _core
from ..documentation._core import (
    HEATMAP_LABELS,
    HEATMAP_SUFFIXES,
    _match_source_variable,
    build_country_heatmap,
)

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})


PAPER_COMPARISON_SOURCES = ["IMF_IFS", "IMF_WEO", "WDI", "OECD_EO", "UN", "JST", "Mitchell", "GFD"]
PAPER_MAJOR_SOURCES = ["IMF_IFS", "WDI", "PWT", "OECD_EO", "IMF_WEO", "EUROSTAT", "UN", "JST", "MAD"]
PAPER_OUTPUT_GRAPH_DIR = sh.OUTPUT_DIR / "graphs"

PAPER_VARIABLE_SPECS: list[tuple[str, str]] = [
    ("cbrate", "Central bank policy rate"),
    ("strate", "Short-term interest rate"),
    ("ltrate", "Long-term interest rate"),
    ("M0", "Money supply (M0)"),
    ("M1", "Money supply (M1)"),
    ("M2", "Money supply (M2)"),
    ("M3", "Money supply (M3)"),
    ("M4", "Money supply (M4)"),
    ("rGDP", "Real GDP"),
    ("nGDP", "Nominal GDP"),
    ("cons", "Consumption"),
    ("rcons", "Real consumption"),
    ("inv", "Gross capital formation"),
    ("finv", "Gross fixed capital formation"),
    ("CA_GDP", "Current account"),
    ("exports", "Exports"),
    ("imports", "Imports"),
    ("REER", "Real effective exchange rate"),
    ("USDfx", "US dollar exchange rate"),
    ("govrev", "Government revenue"),
    ("govtax", "Government tax revenue"),
    ("govexp", "Government expenditure"),
    ("govdebt_GDP", "Government debt"),
    ("govdef_GDP", "Government deficit"),
    ("unemp", "Unemployment rate"),
    ("infl", "Inflation rate"),
    ("CPI", "Consumer price index"),
    ("HPI", "House price index"),
    ("pop", "Population"),
]

PAPER_VARIABLE_ORDER = [var for var, _ in PAPER_VARIABLE_SPECS]
PAPER_VARIABLE_LABELS = {var: label for var, label in PAPER_VARIABLE_SPECS}
PAPER_VARIABLE_ORDER_MAP = {var: idx for idx, var in enumerate(PAPER_VARIABLE_ORDER, start=1)}
PAPER_COMPARISON_ALIAS_MAP = {"EUROSTAT": "EUS"}

GDP_SHARE_EXCLUSIONS = {"MMR", "SLE", "YUG", "ROM", "ZWE", "SRB", "POL", "RUS"}
GDP_SHARE_DROPS = {"ZWE", "SLE", "ROU", "YUG", "IRQ", "URY"}


def _ensure_output_dir(path: Path | str) -> Path:
    out = _resolve(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _normalize_clean_wide_for_paper(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    rename_map = {
        col: col.replace("WDI_ARC_", "WDIARC_")
        for col in work.columns
        if col.startswith("WDI_ARC_")
    }
    if "OECD_HPI" in work.columns and "OECD_EO_HPI" not in work.columns:
        rename_map["OECD_HPI"] = "OECD_EO_HPI"
    if "WB_CC_infl" in work.columns and "WBCC_infl" not in work.columns:
        rename_map["WB_CC_infl"] = "WBCC_infl"
    if rename_map:
        work = work.rename(columns=rename_map)
    return work


def _country_names(data_helper_dir: Path | str = sh.DATA_HELPER_DIR) -> pd.DataFrame:
    helper_dir = _resolve(data_helper_dir)
    return pd.read_dta(helper_dir / "countrylist.dta", convert_categoricals=False)[["ISO3", "countryname"]]


def _load_sources_csv(data_helper_dir: Path | str = sh.DATA_HELPER_DIR) -> pd.DataFrame:
    helper_dir = _resolve(data_helper_dir)
    return sh._read_csv_compat(helper_dir / "sources.csv")


def _load_docvars_csv(data_helper_dir: Path | str = sh.DATA_HELPER_DIR) -> pd.DataFrame:
    helper_dir = _resolve(data_helper_dir)
    df = sh._read_csv_compat(helper_dir / "docvars.csv")
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    return df


def _load_data_final(data_final_dir: Path | str = sh.DATA_FINAL_DIR) -> pd.DataFrame:
    return _load_dta(_resolve(data_final_dir) / "data_final.dta")


def _load_chainlinked(varname: str, data_final_dir: Path | str = sh.DATA_FINAL_DIR) -> pd.DataFrame:
    return _load_dta(_resolve(data_final_dir) / f"chainlinked_{varname}.dta")


def _load_clean_wide(
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
) -> pd.DataFrame:
    return _normalize_clean_wide_for_paper(
        _load_clean_data_wide(
            data_clean_dir=data_clean_dir,
            data_temp_dir=data_temp_dir,
            data_final_dir=data_final_dir,
        )
    )


def _first_nonmissing(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    for column in columns:
        if column not in df.columns:
            continue
        out = out.where(out.notna(), df[column])
    return out


def _source_columns(df: pd.DataFrame, source_alias: str) -> list[str]:
    actual = PAPER_COMPARISON_ALIAS_MAP.get(source_alias, source_alias)
    prefix = f"{actual}_"
    return [col for col in df.columns if col.startswith(prefix)]


def _available_variables_for_source(df: pd.DataFrame, source_alias: str) -> list[str]:
    variables: set[str] = set()
    for column in _source_columns(df, source_alias):
        match = _match_source_variable(column)
        if match is not None:
            _, variable = match
            variables.add(variable)

    if source_alias == "OECD_EO":
        variables.update({"strate", "ltrate", "cbrate", "M1", "M3"})
    if source_alias == "IMF_IFS":
        variables.update({"govdebt_GDP", "cbrate", "ltrate", "strate", "M0", "M1", "M2", "govexp", "govrev", "govdef_GDP", "govtax"})
    if source_alias == "WDI":
        variables.add("unemp")
    return [var for var in PAPER_VARIABLE_ORDER if var in variables]


def _source_series(df: pd.DataFrame, source_alias: str, variable: str) -> pd.Series:
    actual = PAPER_COMPARISON_ALIAS_MAP.get(source_alias, source_alias)
    column = f"{actual}_{variable}"
    if source_alias == "OECD_EO":
        if variable == "strate":
            return _first_nonmissing(df, [column, "OECD_KEI_strate", "OECD_MEI_ARC_strate"])
        if variable == "ltrate":
            return _first_nonmissing(df, [column, "OECD_MEI_ARC_ltrate", "OECD_KEI_ltrate", "OECD_MEI_ltrate"])
        if variable == "cbrate":
            return _first_nonmissing(df, [column, "OECD_MEI_cbrate", "OECD_MEI_ARC_cbrate"])
        if variable == "M1":
            return _first_nonmissing(df, [column, "OECD_MEI_M1"])
        if variable == "M3":
            return _first_nonmissing(df, [column, "OECD_MEI_M3"])
    if source_alias == "IMF_IFS":
        if variable == "govdebt_GDP":
            return _first_nonmissing(df, [column, "IMF_FPP_govdebt_GDP", "IMF_HDD_govdebt_GDP", "IMF_GDD_govdebt_GDP"])
        if variable == "cbrate":
            return _first_nonmissing(df, [column, "IMF_MFS_cbrate"])
        if variable == "ltrate":
            return _first_nonmissing(df, [column, "IMF_MFS_ltrate"])
        if variable == "strate":
            return _first_nonmissing(df, [column, "IMF_MFS_strate"])
        if variable == "M0":
            return _first_nonmissing(df, [column, "IMF_MFS_M0"])
        if variable == "M1":
            return _first_nonmissing(df, [column, "IMF_MFS_M1"])
        if variable == "M2":
            return _first_nonmissing(df, [column, "IMF_MFS_M2"])
        if variable == "govexp":
            return _first_nonmissing(df, [column, "IMF_GFS_govexp"])
        if variable == "govrev":
            return _first_nonmissing(df, [column, "IMF_GFS_govrev"])
        if variable == "govdef_GDP":
            return _first_nonmissing(df, [column, "IMF_GFS_govdef_GDP"])
        if variable == "govtax":
            return _first_nonmissing(df, [column, "IMF_GFS_govtax"])
    if source_alias == "WDI" and variable == "unemp":
        return _first_nonmissing(df, [column, "ILO_unemp"])
    if column in df.columns:
        return df[column]
    return pd.Series(np.nan, index=df.index)


def _source_frame(df: pd.DataFrame, source_alias: str, *, include_pop: bool = True) -> pd.DataFrame:
    out = df[["ISO3", "year"]].copy()
    for variable in _available_variables_for_source(df, source_alias):
        if not include_pop and variable == "pop":
            continue
        series = _source_series(df, source_alias, variable)
        if series.notna().any():
            out[variable] = series
    return out


def _load_gfd_processed(
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
) -> pd.DataFrame:
    helper_dir = _resolve(data_helper_dir)
    gfd = pd.read_dta(helper_dir / "GFD_processed.dta", convert_categoricals=False)
    if "GFD_nGDP" in gfd.columns:
        gfd["GFD_govdef"] = gfd["GFD_govdef_GDP"] * gfd["GFD_nGDP"]
        gfd["GFD_govdebt"] = gfd["GFD_govdebt_GDP"] * gfd["GFD_nGDP"]
        gfd["GFD_govexp_GDP"] = gfd["GFD_govexp"] / gfd["GFD_nGDP"]
        gfd["GFD_govrev_GDP"] = gfd["GFD_govrev"] / gfd["GFD_nGDP"]
        gfd["GFD_govtax_GDP"] = gfd["GFD_govtax"] / gfd["GFD_nGDP"]
        gfd["GFD_imports_GDP"] = gfd["GFD_imports"] / gfd["GFD_nGDP"]
        gfd["GFD_exports_GDP"] = gfd["GFD_exports"] / gfd["GFD_nGDP"]
        gfd["GFD_CA"] = gfd["GFD_CA_GDP"] * gfd["GFD_nGDP"]
        gfd["GFD_finv_GDP"] = gfd["GFD_finv"] / gfd["GFD_nGDP"]
        gfd["GFD_inv_GDP"] = gfd["GFD_inv"] / gfd["GFD_nGDP"]
        gfd["GFD_rGDP_pc"] = gfd["GFD_rGDP"] / gfd["GFD_pop"]
    if "GFD_USDfx" in gfd.columns:
        gfd.loc[pd.to_numeric(gfd["year"], errors="coerce").le(1791), "GFD_USDfx"] = np.nan
    return gfd


def _gfd_frame(data_helper_dir: Path | str = sh.DATA_HELPER_DIR) -> pd.DataFrame:
    gfd = _load_gfd_processed(data_helper_dir=data_helper_dir)
    keep_cols = ["ISO3", "year"] + [col for col in gfd.columns if col.startswith("GFD_")]
    return gfd[keep_cols].copy()


def _count_nonmissing(series: pd.Series) -> int:
    return int(pd.to_numeric(series, errors="coerce").notna().sum())


def _comma_int(value: int | float | str | None) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return ""
    return f"{int(round(float(numeric))):,}"


def _safe_int_string(value: int | float | str | None) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return ""
    return str(int(round(float(numeric))))


def _latex_escape_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("_", "\\_")
    text = text.replace("%", "\\%")
    return text


def _paper_variable_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["order"] = df["var"].map(PAPER_VARIABLE_ORDER_MAP)
    df = df.dropna(subset=["order"]).sort_values("order").reset_index(drop=True)
    return df


def _list_immediate_python_files(path: Path) -> list[Path]:
    return sorted(
        [
            file
            for file in path.glob("*.py")
            if file.name not in {"__init__.py", "_core.py"}
        ],
        key=lambda item: item.name.lower(),
    )


def _clean_module_counts(repo_root: Path | str = REPO_ROOT) -> dict[str, int]:
    root = _resolve(repo_root)
    clean_root = root / "global_macro_data" / "clean"
    aggregators = _list_immediate_python_files(clean_root / "aggregators")
    country_level = _list_immediate_python_files(clean_root / "country_level")
    return {"aggregators": len(aggregators), "country_level": len(country_level)}


def _download_module_counts(repo_root: Path | str = REPO_ROOT) -> dict[str, int]:
    root = _resolve(repo_root)
    download_root = root / "global_macro_data" / "download"
    aggregators = _list_immediate_python_files(download_root / "aggregators")
    country_level = _list_immediate_python_files(download_root / "country_level")
    return {"aggregators": len(aggregators), "country_level": len(country_level)}


def _figure_path(filename: str, graphs_dir: Path | str = PAPER_OUTPUT_GRAPH_DIR) -> Path:
    graph_dir = _ensure_output_dir(graphs_dir)
    return graph_dir / filename


__all__ = [
    "GDP_SHARE_DROPS",
    "GDP_SHARE_EXCLUSIONS",
    "HEATMAP_LABELS",
    "HEATMAP_SUFFIXES",
    "PAPER_COMPARISON_SOURCES",
    "PAPER_MAJOR_SOURCES",
    "PAPER_OUTPUT_GRAPH_DIR",
    "PAPER_VARIABLE_LABELS",
    "PAPER_VARIABLE_ORDER",
    "PAPER_VARIABLE_ORDER_MAP",
    "_clean_module_counts",
    "_comma_int",
    "_country_names",
    "_count_nonmissing",
    "_download_module_counts",
    "_ensure_output_dir",
    "_figure_path",
    "_gfd_frame",
    "_latex_escape_text",
    "_load_chainlinked",
    "_load_clean_wide",
    "_load_data_final",
    "_load_docvars_csv",
    "_load_sources_csv",
    "_normalize_clean_wide_for_paper",
    "_paper_variable_frame",
    "_safe_int_string",
    "_source_frame",
    "_source_series",
    "build_country_heatmap",
]
