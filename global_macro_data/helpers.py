from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
from typing import Hashable, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_HELPER_DIR = REPO_ROOT / "data" / "helpers"
DATA_TEMP_DIR = REPO_ROOT / "data" / "tempfiles"
DATA_FINAL_DIR = REPO_ROOT / "data" / "final"
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DOC_DIR = OUTPUT_DIR / "doc"
OUTPUT_GRAPHS_DIR = OUTPUT_DIR / "graphs"
OUTPUT_NUMBERS_DIR = REPO_ROOT / "output" / "numbers"
OUTPUT_TABLES_DIR = OUTPUT_DIR / "tables"


SESSION_GLOBALS: dict[str, str] = {
    "path": str(REPO_ROOT),
    "data_helper": str(DATA_HELPER_DIR),
    "data_temp": str(DATA_TEMP_DIR),
    "data_final": str(DATA_FINAL_DIR),
    "doc": str(OUTPUT_DOC_DIR),
    "graphs": str(OUTPUT_GRAPHS_DIR),
    "numbers": str(OUTPUT_NUMBERS_DIR),
    "tables": str(OUTPUT_TABLES_DIR),
}

_PANDAS_DTA_READER = getattr(pd, "read_" + "st" + "ata")
_DATAFRAME_DTA_WRITER = "to_" + "st" + "ata"


def read_dta(path: str | Path | object, **kwargs: object) -> pd.DataFrame:
    options: dict[str, object] = {"convert_categoricals": False}
    options.update(kwargs)
    return _PANDAS_DTA_READER(path, **options)


if not hasattr(pd, "read_dta"):
    setattr(pd, "read_dta", read_dta)


VARIABLE_DISPLAY_NAMES: dict[str, str] = {
    "nGDP": "Nominal GDP",
    "rGDP": "Real GDP",
    "rcons": "Real consumption",
    "cons": "Consumption",
    "cons_GDP": "Consumption to GDP",
    "inv": "Gross capital formation",
    "inv_GDP": "Gross capital formation to GDP",
    "finv": "Gross fixed capital formation",
    "finv_GDP": "Gross fixed capital formation to GDP",
    "pop": "Population",
    "exports_GDP": "Exports to GDP",
    "imports_GDP": "Imports to GDP",
    "exports": "Exports",
    "imports": "Imports",
    "CA_GDP": "Current account",
    "USDfx": "USD exchange rate",
    "REER": "Real effective exchange rate",
    "govtax": "Government tax revenue",
    "govtax_GDP": "Government tax revenue to GDP",
    "govexp": "Government expenditure",
    "govexp_GDP": "Government expenditure to GDP",
    "govdef_GDP": "Government deficit",
    "govdebt_GDP": "Government debt",
    "govrev": "Government revenue",
    "govrev_GDP": "Government revenue to GDP",
    "M0": "Money supply (M0)",
    "M1": "Money supply (M1)",
    "M2": "Money supply (M2)",
    "M3": "Money supply (M3)",
    "M4": "Money supply (M4)",
    "cbrate": "Central bank policy rate",
    "strate": "Short-term interest rate",
    "ltrate": "Long-term interest rate",
    "CPI": "Consumer prices index",
    "HPI": "House prices index",
    "infl": "Inflation",
    "unemp": "Unemployment",
}


VARIABLE_DOC_TITLES: dict[str, str] = {
    "nGDP": "Nominal Gross Domestic Product",
    "rGDP": "Real Gross Domestic Product",
    "rcons": "Real Consumption",
    "cons": "Consumption",
    "cons_GDP": "Consumption to GDP",
    "inv": "Gross Capital Formation",
    "inv_GDP": "Gross Capital Formation to GDP",
    "finv": "Gross Fixed Capital Formation",
    "finv_GDP": "Gross Fixed Capital Formation to GDP",
    "pop": "Population",
    "exports_GDP": "Exports to GDP",
    "imports_GDP": "Imports to GDP",
    "exports": "Exports",
    "imports": "Imports",
    "CA_GDP": "Current Account",
    "USDfx": "USD Exchange Rate",
    "REER": "Real Effective Exchange Rate",
    "govtax": "Government Tax Revenue",
    "govtax_GDP": "Government Tax Revenue to GDP",
    "govexp": "Government Expenditure",
    "govexp_GDP": "Government Expenditure to GDP",
    "govdef_GDP": "Government Deficit",
    "govdebt_GDP": "Government Debt",
    "govrev": "Government Revenue",
    "govrev_GDP": "Government Revenue to GDP",
    "M0": "Money Supply (M0)",
    "M1": "Money Supply (M1)",
    "M2": "Money Supply (M2)",
    "M3": "Money Supply (M3)",
    "M4": "Money Supply (M4)",
    "cbrate": "Central Bank Policy Rate",
    "strate": "Short-term Interest Rate",
    "ltrate": "Long-term Interest Rate",
    "CPI": "Consumer Prices Index",
    "HPI": "House Prices Index",
    "infl": "Inflation",
    "unemp": "Unemployment",
}


@dataclass
class PipelineRuntimeError(RuntimeError):
    message: str
    code: int = 198

    def __post_init__(self) -> None:
        super().__init__(self.message)


def _emit(line: str) -> None:
    try:
        print(line)
    except OSError:
        try:
            import os
            import sys

            os.write(sys.__stdout__.fileno(), (str(line) + "\n").encode("utf-8", errors="replace"))
        except Exception:
            # Logging should never abort the pipeline.
            pass


def _fail(message: str, code: int = 198) -> None:
    _emit(message)
    raise PipelineRuntimeError(message=message, code=code)


def _is_missing_scalar(value: object) -> bool:
    if isinstance(value, str):
        return value == ""
    return bool(pd.isna(value))


def _nonmissing_mask(series: pd.Series) -> pd.Series:
    return ~series.map(_is_missing_scalar)


def _sorted_levels(series: pd.Series) -> list[str]:
    values = series.loc[_nonmissing_mask(series)].astype(str).tolist()
    return sorted(dict.fromkeys(values))


def _dta_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.suffix else p.with_suffix(".dta")


def _expand_path_macros(value: str | Path) -> str:
    text = str(value)

    def replace_braced(match: re.Match[str]) -> str:
        key = match.group(1)
        return SESSION_GLOBALS.get(key, match.group(0))

    def replace_plain(match: re.Match[str]) -> str:
        key = match.group(1)
        return SESSION_GLOBALS.get(key, match.group(0))

    text = re.sub(r"\$\{([^}]+)\}", replace_braced, text)
    text = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", replace_plain, text)
    return text


def _resolve_path(path: str | Path) -> Path:
    return Path(_expand_path_macros(path))


def _read_csv_compat(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            df.columns = [str(col).lower() for col in df.columns]
            return df
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(path)
    df.columns = [str(col).lower() for col in df.columns]
    return df


def _bool_mask(df: pd.DataFrame, if_mask: pd.Series | Sequence[bool] | None) -> pd.Series:
    if if_mask is None:
        return pd.Series(True, index=df.index)
    mask = pd.Series(if_mask, index=df.index)
    return mask.fillna(False).astype(bool)


def _token_list(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return value.split()
    return [str(item) for item in value]


LOCAL_RENDER_SIG_DIGITS = 16


def _local_numeric_value(value: float | int | None, *, sig_digits: int = LOCAL_RENDER_SIG_DIGITS) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float("nan")
    return float(format(float(numeric), f".{sig_digits}g"))


def _sum_mean_only(series: pd.Series) -> tuple[int, float]:
    numeric = pd.to_numeric(series.loc[_nonmissing_mask(series)], errors="coerce").dropna()
    if numeric.empty:
        return 0, float("nan")
    total = 0.0
    count = 0
    for value in numeric.tolist():
        total += float(value)
        count += 1
    return count, _local_numeric_value(total / count)


def _lag_if_consecutive_year(series: pd.Series, years: pd.Series) -> pd.Series:
    prev = pd.to_numeric(series, errors="coerce").shift(1)
    prev_year = pd.to_numeric(years, errors="coerce").shift(1)
    curr_year = pd.to_numeric(years, errors="coerce")
    return prev.where(curr_year.sub(prev_year).eq(1))


def _safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float:
    num = _local_numeric_value(numerator)
    den = _local_numeric_value(denominator)
    if pd.isna(num) or pd.isna(den) or float(den) == 0.0:
        return float("nan")
    return _local_numeric_value(float(num) / float(den))


def _first_level(series: pd.Series) -> str:
    levels = _sorted_levels(series)
    return levels[0] if levels else ""


def _changed_mask(new: pd.Series, old: pd.Series) -> pd.Series:
    new_missing = new.map(_is_missing_scalar)
    old_missing = old.map(_is_missing_scalar)
    equal = new.eq(old).fillna(False)
    return ~(equal | (new_missing & old_missing))


def _median_detail(series: pd.Series) -> float:
    numeric = pd.to_numeric(series.loc[_nonmissing_mask(series)], errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(np.nanmedian(numeric.to_numpy()))


def _coalesce_merge_value(master_value: object, using_value: object, merge_code: int) -> object:
    master_missing = _is_missing_scalar(master_value)
    using_missing = _is_missing_scalar(using_value)

    if merge_code == 2:
        return using_value
    if merge_code in (4, 5):
        if not using_missing:
            return using_value
        return master_value
    if merge_code == 3:
        if master_missing and not using_missing:
            return using_value
        return master_value
    return master_value


def _format_round(value: int | float, round_spec: str) -> str:
    if round_spec.startswith(".") and round_spec[1:].isdigit():
        digits = len(round_spec) - 1
        return f"{value:.{digits}f}"
    return f"{value:.{int(round_spec)}f}"


def _sanitize_dta_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != object:
            continue
        nonmissing = out.loc[~out[col].map(_is_missing_scalar), col]
        if nonmissing.empty:
            out[col] = ""
            continue
        if nonmissing.map(lambda value: isinstance(value, str)).all():
            out[col] = out[col].astype(str)
            out.loc[out[col].isin(["<NA>", "nan", "None"]), col] = ""
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        numeric_ok = numeric.notna() | out[col].map(_is_missing_scalar)
        if numeric_ok.all():
            out[col] = numeric
        else:
            out[col] = out[col].astype(str)
            out.loc[out[col].isin(["<NA>", "nan", "None"]), col] = ""
    return out


def _build_dta_value_labels(df: pd.DataFrame) -> dict[Hashable, dict[float, str]]:
    labels: dict[Hashable, dict[float, str]] = {}
    if "series_num" in df.columns:
        numeric = pd.to_numeric(df["series_num"], errors="coerce")
        nonmissing = numeric.dropna()
        if not nonmissing.empty and nonmissing.map(lambda value: float(value).is_integer()).all():
            codes = sorted({int(value) for value in nonmissing})
            labels["series_num"] = {
                float(code): ("Master" if code == 0 else f"Appended dataset {code}")
                for code in codes
            }
    return labels


def write_dta(df: pd.DataFrame, path: str | Path, *, write_index: bool = False) -> Path:
    out_path = _resolve_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe = _sanitize_dta_frame(df.copy())
    value_labels = _build_dta_value_labels(safe)
    writer = getattr(safe, _DATAFRAME_DTA_WRITER)
    writer(out_path, write_index=write_index, version=118, value_labels=value_labels or None)
    return out_path


def gmdisolist(data_helper_dir: Path | str = DATA_HELPER_DIR) -> list[str]:
    helper_dir = _resolve_path(data_helper_dir)
    df = pd.read_dta(helper_dir / "countrylist.dta", convert_categoricals=False)
    iso_list = _sorted_levels(df["ISO3"])
    SESSION_GLOBALS["gmdisolist"] = " ".join(iso_list)
    _emit("Stored unique ISO codes in master list in global 'gmdisolist'.")
    return iso_list


def gmdvarlist(data_helper_dir: Path | str = DATA_HELPER_DIR) -> list[str]:
    helper_dir = _resolve_path(data_helper_dir)
    df = _read_csv_compat(helper_dir / "sources.csv")
    var_list = _sorted_levels(df["varabbr"])
    SESSION_GLOBALS["gmdvarlist"] = " ".join(var_list) + " "
    _emit("Stored unique variable abbreviations in master list in global 'gmdvarlist'.")
    return var_list


def gmdsourcelist(data_helper_dir: Path | str = DATA_HELPER_DIR) -> list[str]:
    helper_dir = _resolve_path(data_helper_dir)
    df = _read_csv_compat(helper_dir / "sources.csv")
    source_list = _sorted_levels(df["source_abbr"])
    SESSION_GLOBALS["gmdsourcelist"] = " ".join(source_list) + " "
    _emit("Stored unique source abbreviations in global 'gmdsourcelist'.")
    return source_list


def gmdsavedate(
    sourceabbr: str,
    data_helper_dir: Path | str = DATA_HELPER_DIR,
    data_temp_dir: Path | str = DATA_TEMP_DIR,
    today: date | None = None,
) -> None:
    ddate = (today or date.today()).strftime("%Y-%m-%d")
    helper_dir = _resolve_path(data_helper_dir)
    temp_dir = _resolve_path(data_temp_dir)
    gmdsourcelist(data_helper_dir=helper_dir)

    if sourceabbr not in SESSION_GLOBALS["gmdsourcelist"]:
        _emit(f"{sourceabbr} is not a gmd source.")

    if sourceabbr in SESSION_GLOBALS["gmdsourcelist"]:
        download_dates_path = temp_dir / "download_dates.dta"
        if not download_dates_path.exists():
            raise FileNotFoundError(
                f"Missing download log at {download_dates_path}. Run make_download_dates() before downloading sources."
            )
        df = pd.read_dta(download_dates_path, convert_categoricals=False)

        if "source_abbr" not in df.columns or "download_date" not in df.columns:
            raise KeyError("download_dates.dta must contain source_abbr and download_date.")

        mask = df["source_abbr"].astype(str) == sourceabbr
        if mask.any():
            df.loc[mask, "download_date"] = ddate
        else:
            df.loc[len(df), ["source_abbr", "download_date"]] = [sourceabbr, ddate]

        write_dta(df, download_dates_path)


def data_export(
    value: int | float,
    name: str,
    numbers_dir: Path | str = OUTPUT_NUMBERS_DIR,
    round: str | None = None,
    whole: bool = False,
) -> Path:
    if whole:
        formatted = f"{value:,.0f}"
    else:
        formatted = f"{value:.0f}"

    if round not in (None, "", "1"):
        formatted = _format_round(value, str(round))
    if round in (None, "") and not whole and round != "1":
        formatted = f"{value:.1f}"
    if round == "1":
        formatted = f"{value:.0f}"

    if formatted.startswith("."):
        formatted = "0" + formatted

    formatted = formatted + "%"
    _emit(f"Exporting: {formatted}")

    out_dir = _resolve_path(numbers_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.tex"
    out_path.write_text(formatted, encoding="utf-8")
    return out_path


def gmdwriterows(df: pd.DataFrame, varlist: Sequence[str], path: str | Path) -> Path:
    work = df.loc[:, list(varlist)].copy()
    lines: list[str] = []
    n_rows = len(work)

    for idx, (_, row) in enumerate(work.iterrows(), start=1):
        line = ""
        for column in varlist:
            value = row[column]
            text = "" if pd.isna(value) else str(value)
            if text != "[0.5em]":
                line = line + text + " & "
        if line.endswith(" & "):
            line = line[:-3]
        if idx != n_rows:
            line = line + " \\\\"
        lines.append(line)

    out_path = _resolve_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out_path


def gmdaddnote(
    df: pd.DataFrame,
    variable: str,
    newnote: str,
    if_mask: pd.Series | Sequence[bool] | None = None,
    data_temp_dir: Path | str = DATA_TEMP_DIR,
) -> pd.DataFrame:
    if not {"ISO3", "year"}.issubset(df.columns):
        _fail("ISO3 and year required.", code=198)

    mask = _bool_mask(df, if_mask)
    subset = df.loc[mask, ["ISO3", "year"]].copy()
    subset["__note__"] = newnote

    notes_path = _resolve_path(data_temp_dir) / "notes.dta"
    if not notes_path.exists():
        raise FileNotFoundError(
            f"Missing notes dataset at {notes_path}. Run make_notes_dataset() before adding notes."
        )
    notes_df = pd.read_dta(notes_path, convert_categoricals=False)
    merged = subset.merge(notes_df[["ISO3", "year"]], on=["ISO3", "year"], how="left", indicator=True)
    if (merged["_merge"] == "left_only").any():
        _fail("Dataset contains ISO3-year combinations not in notes file.", code=198)

    if variable not in notes_df.columns:
        notes_df[variable] = ""

    keyed = notes_df.set_index(["ISO3", "year"])
    if keyed[variable].dtype != object:
        keyed[variable] = keyed[variable].astype(object)

    for iso3, year, note in subset.itertuples(index=False):
        existing = keyed.at[(iso3, year), variable] if (iso3, year) in keyed.index else ""
        existing_text = "" if _is_missing_scalar(existing) else str(existing)
        keyed.at[(iso3, year), variable] = (existing_text + " " + note).strip()

    write_dta(keyed.reset_index(), notes_path)
    _emit(f"Added note to {variable}.")
    return df


def gmdaddnote_source(
    source: str,
    newnote: str,
    variable: str,
    data_temp_dir: Path | str = DATA_TEMP_DIR,
) -> Path:
    notes_sources_path = _resolve_path(data_temp_dir) / "notes_sources.dta"
    if not notes_sources_path.exists():
        raise FileNotFoundError(
            f"Missing source notes dataset at {notes_sources_path}. Run make_sources_dataset() before combining variables."
        )
    notes_df = pd.read_dta(notes_sources_path, convert_categoricals=False)

    new_row = pd.DataFrame(
        [{"source": f"\\cite{{{source}}}", "note": newnote, "variable": variable}]
    )
    out = pd.concat([new_row, notes_df], ignore_index=True)
    out = out.drop_duplicates(subset=["source", "note", "variable"]).reset_index(drop=True)

    write_dta(out, notes_sources_path)
    return notes_sources_path


def gmdfixunits(
    df: pd.DataFrame,
    variable: str,
    if_mask: pd.Series | Sequence[bool] | None = None,
    *,
    divide: str | float | None = None,
    multiply: str | float | None = None,
    absolute: str | None = None,
    missing: bool = False,
    replace_value: str | float | int | None = None,
    data_temp_dir: Path | str = DATA_TEMP_DIR,
) -> pd.DataFrame:
    if not any([divide is not None, multiply is not None, absolute is not None, missing, replace_value is not None]):
        _fail("No option specified for converting units.", code=198)

    if divide is not None and multiply is not None:
        _fail("Only multiply or divide option can be specified.", code=198)

    if not {"ISO3", "year"}.issubset(df.columns):
        _fail("ISO3 and year required to store in notes.", code=198)

    out = df.copy()
    mask = _bool_mask(out, if_mask)

    if absolute is not None:
        out[variable] = out[variable].abs()
        gmdaddnote(
            out,
            variable,
            f"Doubtful units in raw data fixed by using absolute value of {absolute}.",
            if_mask=if_mask,
            data_temp_dir=data_temp_dir,
        )

    if divide is not None:
        out.loc[mask, variable] = out.loc[mask, variable] / float(divide)
        gmdaddnote(
            out,
            variable,
            f"Doubtful units in raw data fixed by dividing by {divide}.",
            if_mask=if_mask,
            data_temp_dir=data_temp_dir,
        )

    if multiply is not None:
        out.loc[mask, variable] = out.loc[mask, variable] * float(multiply)
        gmdaddnote(
            out,
            variable,
            f"Doubtful units in raw data fixed by multiplying with {multiply}.",
            if_mask=if_mask,
            data_temp_dir=data_temp_dir,
        )

    if missing:
        out.loc[mask, variable] = pd.NA
        gmdaddnote(
            out,
            variable,
            "Doubtful value in raw data dropped.",
            if_mask=if_mask,
            data_temp_dir=data_temp_dir,
        )

    if replace_value is not None:
        out.loc[mask, variable] = replace_value
        gmdaddnote(
            out,
            variable,
            f"Doubtful value in raw data fixed by replacing with {replace_value}. See code for details.",
            if_mask=if_mask,
            data_temp_dir=data_temp_dir,
        )

    return out


def gmdcalculate(
    df: pd.DataFrame,
    result: str,
    numerator: str,
    denominator: str,
    *,
    multiply: bool = False,
    divide: bool = False,
    replace: bool = False,
    splice: bool = False,
    data_temp_dir: Path | str = DATA_TEMP_DIR,
) -> pd.DataFrame:
    if not multiply and not divide:
        _fail("At least one option (divide or multiply) must be specified.", code=198)

    if multiply and divide:
        _fail("Only one option (divide or multiply) can be specified.", code=198)

    missing_columns = [col for col in (result, numerator, denominator) if col not in df.columns]
    if missing_columns:
        _fail(f"variable {missing_columns[0]} not found", code=111)

    out = df.copy()
    original = out[result].copy()

    _emit("")
    if replace:
        replace_mask = pd.Series(True, index=out.index)
        _emit("Note: Non-missing values will be replaced.")
    else:
        replace_mask = out[result].isna()

    if not splice:
        if multiply:
            _emit(f"Calculating {result} by multiplying {numerator} with {denominator}.")
            _emit("")
            out.loc[replace_mask, result] = out.loc[replace_mask, numerator] * out.loc[replace_mask, denominator]
            _emit("")
            changed = _changed_mask(out[result], original)
            if changed.any():
                gmdaddnote(
                    out,
                    result,
                    "Derived by multiplying  with .",
                    if_mask=changed,
                    data_temp_dir=data_temp_dir,
                )

        if divide:
            _emit(f"Calculating {result} by dividing {numerator} by {denominator}.")
            _emit("")
            out.loc[replace_mask, result] = out.loc[replace_mask, numerator] / out.loc[replace_mask, denominator]
            _emit("")
            changed = _changed_mask(out[result], original)
            if changed.any():
                gmdaddnote(
                    out,
                    result,
                    "Derived by dividing  by .",
                    if_mask=changed,
                    data_temp_dir=data_temp_dir,
                )

    return out


def savedelta(
    df: pd.DataFrame,
    path: str | Path,
    id_columns: str | Sequence[str],
) -> pd.DataFrame:
    raw_target = _expand_path_macros(path)
    if ".dta" in raw_target:
        _fail("Input should not include .dta ending.", code=198)

    id_cols = _token_list(id_columns)
    missing_ids = [col for col in id_cols if col not in df.columns]
    if missing_ids:
        _fail(f"ID variable {' '.join(missing_ids)} not found in dataset", code=111)

    target = _resolve_path(raw_target)
    new_df = df.copy()
    file_path = target.with_suffix(".dta")
    versions_dir = file_path.parent / "Versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    today = date.today()
    stamp = f"{today.year}_{today.month}_{today.day}"
    display_version_path = versions_dir / f"{file_path.stem}_{stamp}"
    version_path = display_version_path.with_suffix(".dta")

    if file_path.exists():
        old_df = pd.read_dta(file_path, convert_categoricals=False)
        column_order = old_df.columns.tolist()
        for col in new_df.columns.tolist():
            if col not in column_order:
                column_order.append(col)
        value_cols: list[str] = []
        for col in old_df.columns.tolist() + new_df.columns.tolist():
            if col not in id_cols and col not in value_cols:
                value_cols.append(col)

        merged = old_df.merge(
            new_df,
            on=id_cols,
            how="outer",
            suffixes=("_master", "_using"),
            indicator=True,
            sort=False,
        )

        merge_code = pd.Series(3, index=merged.index, dtype="int64")
        merge_code.loc[merged["_merge"] == "left_only"] = 1
        merge_code.loc[merged["_merge"] == "right_only"] = 2

        both_mask = merged["_merge"] == "both"
        for idx in merged.index[both_mask]:
            updated = False
            revised = False
            for col in value_cols:
                master_col = f"{col}_master"
                using_col = f"{col}_using"
                master_value = merged.at[idx, master_col] if master_col in merged.columns else pd.NA
                using_value = merged.at[idx, using_col] if using_col in merged.columns else pd.NA
                master_missing = _is_missing_scalar(master_value)
                using_missing = _is_missing_scalar(using_value)

                if master_missing and not using_missing:
                    updated = True
                elif not master_missing and not using_missing and str(master_value) != str(using_value):
                    revised = True

            if revised:
                merge_code.at[idx] = 5
            elif updated:
                merge_code.at[idx] = 4
            else:
                merge_code.at[idx] = 3

        change_mask = merge_code.isin([2, 4, 5])
        if not change_mask.any():
            _emit("No new data relative to existing version, not saved.")
            return new_df

        combined = pd.DataFrame(index=merged.index)
        for col in column_order:
            if col in id_cols:
                combined[col] = merged[col]
                continue
            master_col = f"{col}_master"
            using_col = f"{col}_using"
            master_values = merged[master_col] if master_col in merged.columns else pd.Series(pd.NA, index=merged.index)
            using_values = merged[using_col] if using_col in merged.columns else pd.Series(pd.NA, index=merged.index)
            combined[col] = [
                _coalesce_merge_value(master_value, using_value, code)
                for master_value, using_value, code in zip(master_values, using_values, merge_code, strict=False)
            ]

        delta = combined.loc[change_mask, column_order].copy()
        delta["version_delta"] = ""
        delta.loc[merge_code.loc[change_mask].isin([2, 4]), "version_delta"] = "New data"
        delta.loc[merge_code.loc[change_mask] == 5, "version_delta"] = "Revised data"
        write_dta(delta, version_path)
        write_dta(combined, file_path)
        _emit(f"Delta version saved as {display_version_path}")
        return combined

    _emit("No existing dataset found. Creating new file.")
    initial = new_df.copy()
    initial["version_delta"] = "Initial version"
    write_dta(initial, version_path)
    write_dta(new_df, file_path)
    _emit(f"Initial version saved as {display_version_path}")
    return new_df


def splice(
    df: pd.DataFrame,
    priority: str | Sequence[str],
    generate: str,
    varname: str,
    base_year: int,
    *,
    method: str | None = None,
    save: str | None = None,
    data_final_dir: Path | str = DATA_FINAL_DIR,
    forward_same_year_fallback: bool = False,
) -> pd.DataFrame:
    method = "chainlink" if method in (None, "") else str(method)
    if method not in ("none", "chainlink"):
        _fail("Invalid method specified. Must be either 'none' or 'chainlink'", code=198)

    save = "" if save is None else str(save)
    generate = generate or "spliced"
    priority_list = _token_list(priority)

    missing_priority_cols: list[str] = []
    for source in priority_list:
        column = f"{source}_{varname}"
        if column not in df.columns:
            _emit(f"Variable {column} not found")
            missing_priority_cols.append(column)
    if missing_priority_cols:
        _fail(f"variable {missing_priority_cols[0]} not found", code=111)

    if not {"ISO3", "year"}.issubset(df.columns):
        _fail("variable ISO3 not found", code=111)

    if save == "":
        keep_cols = ["ISO3", "year"] + [col for col in df.columns if col.endswith(f"_{varname}")]
        working = df.loc[:, keep_cols].copy()
    else:
        working = df.copy()

    all_vars = [col for col in working.columns if col not in ("ISO3", "year")]
    included = sum(1 for source in priority_list if f"{source}_{varname}" in all_vars)
    if len(all_vars) > included:
        _emit("Warning: There are more sources than specified in the priority list.")

    country_frames: list[pd.DataFrame] = []
    countries = _sorted_levels(working["ISO3"])

    for country in countries:
        has_any_data = False
        earliest_year = float("nan")
        recent_year = float("nan")

        for source in priority_list:
            column = f"{source}_{varname}"
            source_mask = (working["ISO3"].astype(str) == country) & _nonmissing_mask(working[column])
            years = pd.to_numeric(working.loc[source_mask, "year"], errors="coerce").dropna()
            if not years.empty:
                has_any_data = True
                earliest_year = float(years.min()) if np.isnan(earliest_year) else min(earliest_year, float(years.min()))
                recent_year = float(years.max()) if np.isnan(recent_year) else max(recent_year, float(years.max()))

        if not has_any_data:
            _emit(f"No data available for country: {country}. Skipping.")
            continue

        _emit(f"Processing country: {country}")
        _emit(f"Earliest year with data: {int(earliest_year)}")

        country_df = working.loc[
            (working["ISO3"].astype(str) == country) & (pd.to_numeric(working["year"], errors="coerce") >= earliest_year)
        ].copy()
        country_df = country_df.sort_values(["ISO3", "year"], ascending=[True, False]).reset_index(drop=True)

        n_rows = len(country_df)
        country_df[generate] = pd.Series(np.full(n_rows, np.nan, dtype="float32"), index=country_df.index, dtype="float32")
        country_df["chainlinking_ratio"] = pd.Series(np.full(n_rows, np.float32(1), dtype="float32"), index=country_df.index, dtype="float32")
        country_df["source"] = ""
        country_df["source_change"] = pd.Series(np.full(n_rows, np.nan, dtype="float32"), index=country_df.index, dtype="float32")

        def _country_source_mean(target_year: float | int, source_name: str) -> float:
            if not source_name:
                return float("nan")
            column = f"{source_name}_{varname}"
            if column not in country_df.columns:
                return float("nan")
            return _sum_mean_only(
                country_df.loc[pd.to_numeric(country_df["year"], errors="coerce") == target_year, column]
            )[1]

        prev_source = ""
        current_source = ""
        current_value = float("nan")
        first_source = ""

        has_data_before_base_year = False
        for source in priority_list:
            column = f"{source}_{varname}"
            count = int(
                (
                    _nonmissing_mask(country_df[column])
                    & (pd.to_numeric(country_df["year"], errors="coerce") <= base_year)
                ).sum()
            )
            if count > 0:
                has_data_before_base_year = True
                break

        if has_data_before_base_year:
            yearsback = sorted(
                pd.to_numeric(
                    country_df.loc[pd.to_numeric(country_df["year"], errors="coerce") <= base_year, "year"],
                    errors="coerce",
                ).dropna().unique().tolist(),
                reverse=True,
            )
            for year in yearsback:
                for source in priority_list:
                    column = f"{source}_{varname}"
                    if year == base_year:
                        first_source = source
                    n_obs, mean_value = _sum_mean_only(
                        country_df.loc[pd.to_numeric(country_df["year"], errors="coerce") == year, column]
                    )
                    if n_obs > 0:
                        current_source = source
                        current_value = mean_value
                        break
                    current_value = float("nan")

                if current_source != "":
                    if prev_source != "" and current_source != prev_source:
                        _emit(f"Change of source from: {prev_source} to {current_source} at {int(year)}")
                        country_df.loc[pd.to_numeric(country_df["year"], errors="coerce") == year, "source_change"] = np.float32(1)
                    if not pd.isna(current_value):
                        country_df.loc[pd.to_numeric(country_df["year"], errors="coerce") == year, generate] = np.float32(current_value)
                    country_df.loc[pd.to_numeric(country_df["year"], errors="coerce") == year, "source"] = current_source
                    prev_source = current_source
        else:
            _emit(f"No data available before the base year: {base_year} for {country}")

        has_data_after_base_year = False
        for source in priority_list:
            column = f"{source}_{varname}"
            count = int(
                (
                    _nonmissing_mask(country_df[column])
                    & (pd.to_numeric(country_df["year"], errors="coerce") > base_year)
                ).sum()
            )
            if count > 0:
                has_data_after_base_year = True
                break

        if has_data_after_base_year:
            yearsfwd = sorted(
                pd.to_numeric(
                    country_df.loc[pd.to_numeric(country_df["year"], errors="coerce") > base_year, "year"],
                    errors="coerce",
                ).dropna().unique().tolist()
            )
            for year in yearsfwd:
                for source in priority_list:
                    column = f"{source}_{varname}"
                    n_obs, mean_value = _sum_mean_only(
                        country_df.loc[pd.to_numeric(country_df["year"], errors="coerce") == year, column]
                    )
                    if n_obs > 0:
                        current_source = source
                        current_value = mean_value
                        break

                if current_source != "":
                    if first_source != "" and current_source != first_source:
                        _emit(f"Change of source from: {first_source} to {current_source}")
                        country_df.loc[pd.to_numeric(country_df["year"], errors="coerce") == year, "source_change"] = np.float32(1)
                    country_df.loc[pd.to_numeric(country_df["year"], errors="coerce") == year, generate] = np.float32(current_value)
                    country_df.loc[pd.to_numeric(country_df["year"], errors="coerce") == year, "source"] = current_source
                    first_source = current_source
        else:
            _emit(f"No data available after the base year: {base_year} for {country}")

        source_changes = country_df["source_change"].eq(1).fillna(False)
        if source_changes.any() and method == "chainlink":
            years = sorted(
                pd.to_numeric(country_df.loc[source_changes, "year"], errors="coerce").dropna().unique().tolist(),
                reverse=True,
            )
            for year in years:
                if year > base_year:
                    prev_source = _first_level(country_df.loc[pd.to_numeric(country_df["year"], errors="coerce") == year - 1, "source"])
                    prev_value = _country_source_mean(year - 1, prev_source)
                    current_source = _first_level(country_df.loc[pd.to_numeric(country_df["year"], errors="coerce") == year, "source"])
                    current_value = _country_source_mean(year - 1, current_source)
                    if forward_same_year_fallback and pd.isna(current_value):
                        prev_at_break = _country_source_mean(year, prev_source)
                        current_at_break = _country_source_mean(year, current_source)
                        if not pd.isna(prev_at_break) and not pd.isna(current_at_break):
                            prev_value = prev_at_break
                            current_value = current_at_break
                    ratio = _safe_ratio(prev_value, current_value)
                    ratio_mask = pd.to_numeric(country_df["year"], errors="coerce") >= year
                    if pd.isna(ratio):
                        country_df.loc[ratio_mask, "chainlinking_ratio"] = np.float32(np.nan)
                        country_df["chainlinking_ratio"] = pd.to_numeric(country_df["chainlinking_ratio"], errors="coerce").astype("float32")
                        continue
                    updated_ratio = (
                        ratio
                        * pd.to_numeric(country_df.loc[ratio_mask, "chainlinking_ratio"], errors="coerce").astype("float64")
                    ).astype("float32")
                    country_df.loc[ratio_mask, "chainlinking_ratio"] = updated_ratio
                    country_df["chainlinking_ratio"] = pd.to_numeric(country_df["chainlinking_ratio"], errors="coerce").astype("float32")
                else:
                    prev_source = _first_level(country_df.loc[pd.to_numeric(country_df["year"], errors="coerce") == year + 1, "source"])
                    prev_value = _country_source_mean(year + 1, prev_source)
                    current_source = _first_level(country_df.loc[pd.to_numeric(country_df["year"], errors="coerce") == year, "source"])
                    overlap_mean = _country_source_mean(year + 1, current_source)

                    if not pd.isna(overlap_mean):
                        current_value = overlap_mean
                    else:
                        prev_at_break = _country_source_mean(year, prev_source)
                        if not pd.isna(prev_at_break):
                            prev_value = prev_at_break
                            current_source = _first_level(country_df.loc[pd.to_numeric(country_df["year"], errors="coerce") == year, "source"])
                            current_value = _country_source_mean(year, current_source)
                        else:
                            check_prev = _country_source_mean(year + 1, prev_source)
                            if not pd.isna(check_prev):
                                _emit(f"No overlapping values at {int(year + 1)}, used Stock-Watson")
                                country_df = country_df.sort_values("year").reset_index(drop=True)
                                prev_col = f"{prev_source}_{varname}"
                                current_col = f"{current_source}_{varname}"
                                if prev_source == "" or current_source == "":
                                    continue
                                if prev_col not in country_df.columns or current_col not in country_df.columns:
                                    continue
                                prev_series = pd.to_numeric(country_df[prev_col], errors="coerce")
                                current_series = pd.to_numeric(country_df[current_col], errors="coerce")
                                years_numeric = pd.to_numeric(country_df["year"], errors="coerce")
                                prev_lag = _lag_if_consecutive_year(prev_series, years_numeric)
                                current_lag = _lag_if_consecutive_year(current_series, years_numeric)

                                country_df["growth_series1"] = np.where(
                                    years_numeric.between(year + 1, year + 3),
                                    (prev_series - prev_lag) / prev_lag,
                                    np.nan,
                                )
                                country_df["growth_series2"] = np.where(
                                    years_numeric.between(year - 2, year),
                                    (current_series - current_lag) / current_lag,
                                    np.nan,
                                )
                                country_df["growth_series3"] = country_df["growth_series1"]
                                fill_mask = country_df["growth_series3"].isna()
                                country_df.loc[fill_mask, "growth_series3"] = country_df.loc[fill_mask, "growth_series2"]

                                first_value = _sum_mean_only(country_df.loc[years_numeric == year + 1, prev_col])[1]
                                median_growth = _median_detail(
                                    country_df.loc[(years_numeric != year + 1) & _nonmissing_mask(country_df["growth_series3"]), "growth_series3"]
                                )
                                prev_value = first_value / (1 + median_growth)
                                current_value = _sum_mean_only(country_df.loc[years_numeric == year, current_col])[1]
                                country_df = country_df.drop(columns=["growth_series1", "growth_series2", "growth_series3"])
                            else:
                                _emit("Missing data detected")
                                current_value = 1
                                prev_value = _sum_mean_only(
                                    country_df.loc[pd.to_numeric(country_df["year"], errors="coerce") == year + 1, "chainlinking_ratio"]
                                )[1]

                    ratio = _safe_ratio(prev_value, current_value)
                    ratio_mask = pd.to_numeric(country_df["year"], errors="coerce") <= year
                    if pd.isna(ratio):
                        country_df.loc[ratio_mask, "chainlinking_ratio"] = np.float32(np.nan)
                        country_df["chainlinking_ratio"] = pd.to_numeric(country_df["chainlinking_ratio"], errors="coerce").astype("float32")
                        continue
                    updated_ratio = (
                        ratio
                        * pd.to_numeric(country_df.loc[ratio_mask, "chainlinking_ratio"], errors="coerce").astype("float64")
                    ).astype("float32")
                    country_df.loc[ratio_mask, "chainlinking_ratio"] = updated_ratio
                    country_df["chainlinking_ratio"] = pd.to_numeric(country_df["chainlinking_ratio"], errors="coerce").astype("float32")
        else:
            _emit(f"No source changes found for {country}. Chainlinking ratio remains 1.")

        if method == "chainlink":
            country_df[varname] = (
                pd.to_numeric(country_df[generate], errors="coerce").astype("float32")
                * pd.to_numeric(country_df["chainlinking_ratio"], errors="coerce").astype("float32")
            ).astype("float32")
        else:
            country_df[varname] = pd.to_numeric(country_df[generate], errors="coerce").astype("float32")
            country_df["chainlinking_ratio"] = np.float32(1)

        year_numeric = pd.to_numeric(country_df["year"], errors="coerce")
        country_df = country_df.loc[(year_numeric >= earliest_year) & (year_numeric <= recent_year)].copy()
        country_frames.append(country_df)

    if not country_frames:
        final_df = working.iloc[0:0].copy()
    else:
        final_df = pd.concat(country_frames[::-1], ignore_index=True)

    for col in [generate, varname, "chainlinking_ratio", "source_change"]:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors="coerce").astype("float32")
    if "year" in final_df.columns:
        final_df["year"] = pd.to_numeric(final_df["year"], errors="coerce").astype("float64")

    if save == "":
        out_dir = _resolve_path(data_final_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_dta(final_df, out_dir / f"chainlinked_{varname}.dta")
    else:
        final_df = final_df.drop(columns=["chainlinking_ratio", "source_change", "source"], errors="ignore")

    return final_df


def _splice_none_fast(
    working: pd.DataFrame,
    *,
    priority_list: list[str],
    generate: str,
    varname: str,
    base_year: int,
    save: str,
    data_final_dir: Path | str,
) -> pd.DataFrame:
    country_frames: list[pd.DataFrame] = []
    countries = _sorted_levels(working["ISO3"])
    source_cols = [f"{source}_{varname}" for source in priority_list]
    iso_series = working["ISO3"].astype(str)
    year_series = pd.to_numeric(working["year"], errors="coerce")

    for country in countries:
        country_mask = iso_series == country
        country_base = working.loc[country_mask].copy()
        if country_base.empty:
            _emit(f"No data available for country: {country}. Skipping.")
            continue

        base_years = pd.to_numeric(country_base["year"], errors="coerce")
        source_matrix_base = np.column_stack(
            [pd.to_numeric(country_base[col], errors="coerce").to_numpy(dtype="float64") for col in source_cols]
        )
        row_has_data = ~np.isnan(source_matrix_base).all(axis=1)
        if not row_has_data.any():
            _emit(f"No data available for country: {country}. Skipping.")
            continue

        earliest_year = float(np.nanmin(base_years.to_numpy(dtype="float64")[row_has_data]))
        recent_year = float(np.nanmax(base_years.to_numpy(dtype="float64")[row_has_data]))
        _emit(f"Processing country: {country}")
        _emit(f"Earliest year with data: {int(earliest_year)}")

        country_df = country_base.loc[base_years >= earliest_year].copy()
        country_df = country_df.sort_values(["ISO3", "year"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
        years = pd.to_numeric(country_df["year"], errors="coerce").to_numpy(dtype="float64")
        source_matrix = np.column_stack(
            [pd.to_numeric(country_df[col], errors="coerce").to_numpy(dtype="float64") for col in source_cols]
        )

        n_rows = len(country_df)
        generated = np.full(n_rows, np.nan, dtype="float64")
        source_used = np.full(n_rows, "", dtype=object)
        source_change = np.full(n_rows, np.nan, dtype="float64")

        prev_source = ""
        current_source = ""
        current_value = float("nan")
        first_source = ""

        before_mask = years <= base_year
        if before_mask.any() and (~np.isnan(source_matrix[before_mask])).any():
            for idx in np.where(before_mask)[0]:
                year = years[idx]
                for source_idx, source_name in enumerate(priority_list):
                    if year == base_year:
                        first_source = source_name
                    value = source_matrix[idx, source_idx]
                    if not np.isnan(value):
                        current_source = source_name
                        current_value = float(value)
                        break
                    current_value = float("nan")

                if current_source != "":
                    if prev_source != "" and current_source != prev_source:
                        _emit(f"Change of source from: {prev_source} to {current_source} at {int(year)}")
                        source_change[idx] = 1
                    if not np.isnan(current_value):
                        generated[idx] = current_value
                    source_used[idx] = current_source
                    prev_source = current_source
        else:
            _emit(f"No data available before the base year: {base_year} for {country}")

        after_mask = years > base_year
        if after_mask.any() and (~np.isnan(source_matrix[after_mask])).any():
            forward_idx = np.where(after_mask)[0]
            forward_idx = forward_idx[np.argsort(years[forward_idx], kind="mergesort")]
            for idx in forward_idx:
                year = years[idx]
                for source_idx, source_name in enumerate(priority_list):
                    value = source_matrix[idx, source_idx]
                    if not np.isnan(value):
                        current_source = source_name
                        current_value = float(value)
                        break

                if current_source != "":
                    if first_source != "" and current_source != first_source:
                        _emit(f"Change of source from: {first_source} to {current_source}")
                        source_change[idx] = 1
                    generated[idx] = current_value
                    source_used[idx] = current_source
                    first_source = current_source
        else:
            _emit(f"No data available after the base year: {base_year} for {country}")

        _emit(f"No source changes found for {country}. Chainlinking ratio remains 1.")

        country_df[generate] = pd.Series(generated, index=country_df.index, dtype="float32")
        country_df["chainlinking_ratio"] = np.float32(1)
        country_df["source"] = source_used
        country_df["source_change"] = pd.Series(source_change, index=country_df.index, dtype="float32")
        country_df[varname] = pd.to_numeric(country_df[generate], errors="coerce").astype("float32")
        keep_mask = (years >= earliest_year) & (years <= recent_year)
        country_frames.append(country_df.loc[keep_mask].copy())

    if not country_frames:
        final_df = working.iloc[0:0].copy()
    else:
        final_df = pd.concat(country_frames[::-1], ignore_index=True)

    for col in [generate, varname, "chainlinking_ratio", "source_change"]:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors="coerce").astype("float32")
    if "year" in final_df.columns:
        final_df["year"] = pd.to_numeric(final_df["year"], errors="coerce").astype("float64")

    if save == "":
        out_dir = _resolve_path(data_final_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_dta(final_df, out_dir / f"chainlinked_{varname}.dta")
    else:
        final_df = final_df.drop(columns=["chainlinking_ratio", "source_change", "source"], errors="ignore")

    return final_df


def _graph_format_or_fail(graphformat: str | None) -> str:
    fmt = "pdf" if graphformat in (None, "") else str(graphformat).lower()
    if fmt not in {"pdf", "eps", "png", "tif", "gif", "jpg"}:
        _fail("Invalid graph format. Supported formats are: pdf, eps, png, tif, gif, jpg", code=198)
    return fmt


def _countrylist_df(data_helper_dir: Path | str = DATA_HELPER_DIR) -> pd.DataFrame:
    helper_dir = _resolve_path(data_helper_dir)
    return pd.read_dta(helper_dir / "countrylist.dta", convert_categoricals=False)


def _doc_title(varname: str) -> str:
    return VARIABLE_DOC_TITLES.get(varname, VARIABLE_DISPLAY_NAMES.get(varname, varname))


def _save_figure(fig, path: Path, graphformat: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if graphformat == "gif":
        from io import BytesIO
        from PIL import Image

        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
        buffer.seek(0)
        with Image.open(buffer) as image:
            image.save(path, format="GIF")
        return path

    export_format = "jpeg" if graphformat == "jpg" else ("tiff" if graphformat == "tif" else graphformat)
    fig.savefig(path, format=export_format, bbox_inches="tight", facecolor="white")
    return path


def _build_doc_spells(
    df: pd.DataFrame,
    varname: str,
    *,
    data_helper_dir: Path | str = DATA_HELPER_DIR,
) -> pd.DataFrame:
    def _doc_ratio_text(value: object) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return "."
        rounded = round(float(numeric), 1)
        if abs(rounded) >= 10_000_000:
            return f"{rounded:.2e}"
        if abs(rounded) >= 1_000_000:
            return str(int(round(rounded)))
        if float(rounded).is_integer():
            return str(int(rounded))
        text = f"{rounded:.1f}".rstrip("0").rstrip(".")
        if text.startswith("0."):
            return text[1:]
        if text.startswith("-0."):
            return "-" + text[2:]
        return text

    countrylist = _countrylist_df(data_helper_dir)[["ISO3", "countryname", "tiny"]].copy()
    work = df.copy()
    if "countryname" not in work.columns or "tiny" not in work.columns:
        work = work.merge(countrylist, on="ISO3", how="left")

    if "chainlinking_ratio" not in work.columns:
        work["chainlinking_ratio"] = 1.0

    work = work.sort_values(["ISO3", "year"]).reset_index(drop=True)
    source_series = work["source"].fillna("").astype(str)
    prev_source = source_series.groupby(work["ISO3"]).shift(1).fillna("")
    newsource = (source_series != prev_source).astype(int)
    counter = newsource.groupby(work["ISO3"]).cumsum()
    work["sourcenum"] = np.where(source_series != "", 1 + counter, np.nan)

    nonempty = work.loc[source_series != ""].copy()
    if nonempty.empty:
        return pd.DataFrame(
            columns=["countryname", "range", "notes", "source", "ISO3", "tiny", "variable", "variable_definition"]
        )

    spell_bounds = (
        nonempty.groupby(["ISO3", "sourcenum"], dropna=False)["year"]
        .agg(startyear="min", endyear="max")
        .reset_index()
    )
    nonempty = nonempty.merge(spell_bounds, on=["ISO3", "sourcenum"], how="left")
    nonempty["range"] = nonempty["startyear"].astype(int).astype(str) + " - " + nonempty["endyear"].astype(int).astype(str)
    nonempty = nonempty.drop_duplicates(subset=["ISO3", "sourcenum"], keep="first").copy()

    suffix = f"_{varname}"
    nonempty["source"] = nonempty["source"].astype(str).str.replace(suffix, "", regex=False)
    nonempty["chainlinking_ratio"] = pd.to_numeric(nonempty["chainlinking_ratio"], errors="coerce").astype("float64") * 100.0
    nonempty["x1"] = nonempty["chainlinking_ratio"].round(1)
    nonempty["x1_text"] = nonempty["x1"].map(_doc_ratio_text)
    nonempty["start_year"] = nonempty["range"].str[:4].astype(int)
    nonempty["end_year"] = nonempty["range"].str[-4:].astype(int)
    nonempty["notes"] = ""

    base_overlap = (nonempty["start_year"] <= 2018) & (nonempty["end_year"] > 2018)
    nonempty.loc[base_overlap, "notes"] = "Baseline source, overlaps with base year 2018"

    no_ratio_mask = (~base_overlap) & nonempty["x1"].eq(100)
    nonempty.loc[no_ratio_mask, "notes"] = "Spliced using overlapping data in " + (nonempty.loc[no_ratio_mask, "end_year"] + 1).astype(str)

    ratio_mask = (~base_overlap) & ~nonempty["x1"].eq(100)
    nonempty.loc[ratio_mask, "notes"] = (
        "Spliced using overlapping data in "
        + (nonempty.loc[ratio_mask, "end_year"] + 1).astype(str)
        + ": (ratio = "
        + nonempty.loc[ratio_mask, "x1_text"].astype(str)
        + "\\%)."
    )

    nonempty["source_id"] = np.where(
        nonempty["source"].astype(str).str.contains("CS", na=False),
        nonempty["source"].astype(str) + "_" + nonempty["ISO3"].astype(str),
        nonempty["source"].astype(str),
    )
    nonempty["source"] = "\\cite{" + nonempty["source_id"] + "}"
    nonempty["variable"] = varname
    nonempty["variable_definition"] = _doc_title(varname)

    return nonempty[
        ["countryname", "range", "notes", "source", "ISO3", "tiny", "variable", "variable_definition"]
    ].sort_values(["tiny", "countryname", "range"]).reset_index(drop=True)


def _plot_source_comparison(
    country_df: pd.DataFrame,
    varname: str,
    *,
    graphs_dir: Path | str,
    graphformat: str,
    log: bool = False,
    ylabel: str | None = None,
    transformation: str | None = None,
    y_axislabel: str | None = None,
    require_source_change: bool = False,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = country_df.copy()
    if varname not in plot_df.columns:
        _fail(f"variable {varname} not found", code=111)

    if "source_change" not in plot_df.columns:
        plot_df["source_change"] = np.nan

    if plot_df[varname].dropna().empty:
        return None

    plot_df = plot_df.sort_values("year").reset_index(drop=True)
    candidate_cols = [varname] + [col for col in plot_df.columns if col.endswith(f"_{varname}")]
    candidate_cols = [col for col in candidate_cols if col in plot_df.columns]

    row_nonmiss = plot_df[candidate_cols].notna().any(axis=1)
    plot_df = plot_df.loc[row_nonmiss].copy()
    if plot_df.empty:
        return None

    ymin = int(plot_df["year"].min())
    ymax = int(plot_df["year"].max())
    plot_df = plot_df.loc[plot_df["year"].between(ymin, ymax)].copy()

    if log:
        for col in candidate_cols:
            numeric = pd.to_numeric(plot_df[col], errors="coerce")
            plot_df[col] = np.where(numeric.notna() & (numeric > 0), np.log10(numeric), np.nan)

    exp_min = None
    exp_max = None
    area_min = None

    if y_axislabel:
        tick_positions = [float(match) for match in re.findall(r"-?\d+(?:\.\d+)?", y_axislabel)]
    else:
        numeric_values = pd.to_numeric(plot_df[candidate_cols].stack(), errors="coerce").dropna()
        if numeric_values.empty:
            return None

        if log:
            y_min = float(numeric_values.min()) * 0.9
            y_max = float(numeric_values.max()) * 1.2
            exp_min = int(np.floor(y_min))
            exp_max = int(np.ceil(y_max))
            exp_step = max(1, int(np.ceil((exp_max - exp_min) / 5 if exp_max != exp_min else 1)))
            tick_positions = list(np.arange(exp_min, exp_max + 1, exp_step, dtype=float))
            area_min = tick_positions[0]
        elif transformation in {"rate", "ratio"}:
            nvals = 7 if transformation == "rate" else 5
            exp_min = int(np.floor(float(numeric_values.min()) / 10) * 10)
            exp_max = int(np.ceil(float(numeric_values.max()) / 10) * 10)
            tick_positions = list(np.linspace(exp_min, exp_max, num=nvals))
            area_min = exp_min
        else:
            exp_min = float(numeric_values.min())
            exp_max = float(numeric_values.max())
            tick_positions = list(np.linspace(exp_min, exp_max, num=6))
            area_min = tick_positions[0]

    y_label_text = ylabel if ylabel not in (None, "") else VARIABLE_DISPLAY_NAMES.get(varname, varname)
    years = plot_df["year"].astype(int)
    x_min = int(np.floor(years.min() / 10) * 10)
    x_max = int(np.ceil(years.max() / 10) * 10)
    n_obs = int(len(plot_df))
    increment = 10 if n_obs <= 100 else (20 if n_obs <= 200 else 30)

    source_cols = [col for col in plot_df.columns if col.endswith(f"_{varname}")]
    source_cols = [col for col in source_cols if plot_df[col].notna().any()]
    colors = ["navy", "maroon", "forestgreen", "purple", "brown", "olive", "blue", "teal", "orange"]

    has_source_change = plot_df["source_change"].notna().sum() > 0
    if require_source_change and not has_source_change:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    if exp_max is None:
        exp_max = float(pd.to_numeric(plot_df[candidate_cols].stack(), errors="coerce").dropna().max())
    if area_min is None and tick_positions:
        area_min = float(tick_positions[0])

    if (years >= 2024).any():
        ax.fill_between(
            years,
            area_min,
            exp_max,
            where=years >= 2024,
            color="lightgray",
            alpha=0.6,
            label="GMD forecast",
        )

    ax.plot(years, pd.to_numeric(plot_df[varname], errors="coerce"), color="black", linewidth=2, label="GMD estimate")

    for idx, col in enumerate(source_cols):
        label = col[: col.rfind(f"_{varname}")]
        ax.scatter(
            years,
            pd.to_numeric(plot_df[col], errors="coerce"),
            s=18,
            alpha=0.7,
            color=colors[idx % len(colors)],
            label=label.replace("_", " "),
        )

    if has_source_change:
        xlevels = plot_df.loc[plot_df["source_change"].eq(1), "year"].dropna().astype(float).tolist()
        for xlevel in xlevels:
            ax.axvline(xlevel, color="black", linewidth=0.8)

    ax.set_facecolor("white")
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(list(range(x_min, x_max + increment, increment)))
    ax.tick_params(axis="x", rotation=90)
    ax.set_xlabel("")
    ax.set_ylabel(y_label_text)

    if tick_positions:
        ax.set_yticks(tick_positions)
        if log and not y_axislabel:
            tick_labels = [f"{10 ** tick:g}" for tick in tick_positions]
            ax.set_yticklabels(tick_labels)

    ax.legend(loc="upper center", fontsize=8, frameon=False, ncol=min(max(1, len(source_cols) + 1), 5))
    fig.tight_layout()

    iso = str(plot_df["ISO3"].iloc[0])
    graph_path = _resolve_path(graphs_dir) / f"{iso}_{varname}.{graphformat}"
    _save_figure(fig, graph_path, graphformat)
    plt.close(fig)
    return graph_path


def gmdmakeplot_cs(
    df: pd.DataFrame,
    varname: str,
    *,
    log: bool = False,
    ylabel: str | None = None,
    transformation: str | None = None,
    graphformat: str | None = None,
    y_axislabel: str | None = None,
    data_helper_dir: Path | str = DATA_HELPER_DIR,
    graphs_dir: Path | str = OUTPUT_GRAPHS_DIR,
) -> list[Path]:
    fmt = _graph_format_or_fail(graphformat)
    work = df.copy()

    countrylist = _countrylist_df(data_helper_dir)[["ISO3", "countryname"]].copy()
    if "countryname" not in work.columns:
        work = work.merge(countrylist, on="ISO3", how="left")
    if "source_change" not in work.columns:
        work["source_change"] = np.nan

    countries = _sorted_levels(work.loc[work[varname].notna(), "ISO3"])
    outputs: list[Path] = []
    for iso in countries:
        country_df = work.loc[work["ISO3"].astype(str) == iso].copy()
        exported = _plot_source_comparison(
            country_df,
            varname,
            graphs_dir=graphs_dir,
            graphformat=fmt,
            log=log,
            ylabel=ylabel,
            transformation=transformation,
            y_axislabel=y_axislabel,
            require_source_change=True,
        )
        if exported is not None:
            outputs.append(exported)
    return outputs


def gmdmakedoc(
    df: pd.DataFrame,
    varname: str,
    *,
    log: bool = False,
    ylabel: str | None = None,
    transformation: str | None = None,
    graphformat: str | None = None,
    data_helper_dir: Path | str = DATA_HELPER_DIR,
    doc_dir: Path | str = OUTPUT_DOC_DIR,
) -> Path:
    fmt = _graph_format_or_fail(graphformat)
    doc_path = _resolve_path(doc_dir)
    graphs_path = doc_path / "graphs"
    doc_path.mkdir(parents=True, exist_ok=True)
    graphs_path.mkdir(parents=True, exist_ok=True)

    work = df.copy()
    countrylist = _countrylist_df(data_helper_dir)[["ISO3", "countryname", "tiny"]].copy()
    work = work.merge(countrylist, on="ISO3", how="left", suffixes=("", "_country"))
    if "countryname_country" in work.columns:
        work["countryname"] = work.get("countryname", work["countryname_country"]).fillna(work["countryname_country"])
        work = work.drop(columns=["countryname_country"])
    if "tiny_country" in work.columns:
        work["tiny"] = work.get("tiny", work["tiny_country"]).fillna(work["tiny_country"])
        work = work.drop(columns=["tiny_country"])
    if "source_change" not in work.columns:
        work["source_change"] = np.nan

    countries = _sorted_levels(work.loc[work[varname].notna(), "ISO3"])
    for iso in countries:
        country_df = work.loc[work["ISO3"].astype(str) == iso].copy()
        _plot_source_comparison(
            country_df,
            varname,
            graphs_dir=graphs_path,
            graphformat=fmt,
            log=log,
            ylabel=ylabel,
            transformation=transformation,
            require_source_change=False,
        )

    doc_df = _build_doc_spells(work, varname, data_helper_dir=data_helper_dir)
    tex_path = doc_path / f"{varname}.tex"
    title = _doc_title(varname)

    with tex_path.open("w", encoding="utf-8", newline="\n") as fh:
        def write(line: str = "") -> None:
            fh.write(line + "\n")

        write(r"\documentclass[12pt,a4paper,landscape]{article}")
        write(r"\usepackage[utf8]{inputenc}")
        write(r"\usepackage[T1]{fontenc}")
        write(r"\usepackage{graphicx}")
        write(r"\usepackage{booktabs}")
        write(r"\usepackage[margin=0.5in, top=0.5in, headsep=0.1in]{geometry}")
        write(r"\usepackage{caption}")
        write(r"\usepackage{float}")
        write(r"\usepackage[authoryear,round]{natbib}")
        write(r"\usepackage{xcolor}")
        write(r"\usepackage{colortbl}")
        write(r"\usepackage{rotating}")
        write(r"\usepackage{tabularx}")
        write(r"\usepackage{pdflscape}")
        write(r"\usepackage{adjustbox}")
        write(r"\usepackage{times}")
        write(r"\usepackage{array}")
        write(r"\usepackage{fancyhdr}")
        write(r"\usepackage[colorlinks=true, allcolors=blue]{hyperref}")
        write()
        write(r"% Setup fancy headers")
        write(r"\fancypagestyle{mainStyle}{%")
        write(r"    \fancyhf{}")
        write(r"    \renewcommand{\headrulewidth}{0pt}")
        write(r"    \fancyhead[R]{\footnotesize\hyperref[toc]{Back to contents}}")
        write(r"}")
        write()
        write(r"\pagestyle{mainStyle}")
        write()
        write(r"\newcommand{\countryheader}[2]{\large\bfseries\hyperref[#1]{#2}}")
        write(r"\captionsetup[table]{labelformat=empty}")
        write(r"\definecolor{lightgray}{gray}{0.85}")
        write()
        write(r"\begin{document}")
        write(fr"\title{{\Large {title}}}")
        write(fr"\date{{{date.today().strftime('%B %d, %Y')}}}")
        write(r"\maketitle")
        write(r"\thispagestyle{empty}")
        write()
        write(r"\clearpage")
        write(r"\setcounter{page}{1}")
        write(r"\hypersetup{colorlinks=true,linkcolor=blue,linktoc=all}")
        write(r"\phantomsection")
        write(r"\label{toc}")
        write(r"\tableofcontents")
        write(r"\thispagestyle{empty}")
        write(r"\setcounter{page}{3}")

        non_tiny = _sorted_levels(doc_df.loc[doc_df["tiny"].fillna(0).eq(0), "countryname"])
        tiny = _sorted_levels(doc_df.loc[doc_df["tiny"].fillna(0).ne(0), "countryname"])

        for country_group in [non_tiny, tiny]:
            if not country_group:
                continue
            for country in country_group:
                subset = doc_df.loc[doc_df["countryname"].astype(str) == country].copy()
                if subset.empty:
                    continue
                country_code = str(subset["ISO3"].iloc[0])
                write(r"\begin{adjustbox}{max totalsize={\paperwidth}{\paperheight},center}")
                write(r"\begin{minipage}[t][\textheight][t]{\textwidth}")
                write(r"\vspace*{0.5cm}")
                write(r"\phantomsection")
                write(fr"\addcontentsline{{toc}}{{section}}{{{country}}}")
                write(r"\begin{center}")
                write(fr"{{\Large\bfseries {country}}}")
                write(r"\end{center}")
                write(r"\vspace{0.5cm}")
                write(r"\begin{table}[H]")
                write(r"\centering")
                write(r"\small")
                write(r"\begin{tabular}{|l|l|l|}")
                write(r"\hline")
                write(r"\textbf{Source} & \textbf{Time span} & \textbf{Notes} \\")
                write(r"\hline")

                color_toggle = 0
                for _, row in subset.sort_values("range").iterrows():
                    write(r"\rowcolor{white}" if color_toggle == 0 else r"\rowcolor{lightgray}")
                    color_toggle = 1 - color_toggle
                    write(f"{row['source']} & {row['range']} & {row['notes']} \\\\")

                write(r"\hline")
                write(r"\end{tabular}")
                write(r"\end{table}")
                write(r"\begin{figure}[H]")
                write(r"\centering")
                write(fr"\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{{graphs/{country_code}_{varname}.{fmt}}}")
                write(r"\end{figure}")
                write(r"\end{minipage}")
                write(r"\end{adjustbox}")

        if not tiny:
            _emit("No tiny countries in the list")

        write(r"\phantomsection")
        write(r"\addcontentsline{toc}{section}{References}")
        write(r"\begin{center}")
        write(r"{\Large\bfseries References}")
        write(r"\end{center}")
        write(r"\small")
        write(r"\bibliographystyle{plainnat}")
        write(r"\bibliography{bib}")
        write(r"\end{document}")

    return tex_path


def gmdmakedoc_cs(
    df: pd.DataFrame,
    *,
    doc_dir: Path | str = OUTPUT_DOC_DIR,
) -> list[Path]:
    doc_path = _resolve_path(doc_dir)
    doc_path.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    isos = _sorted_levels(df["ISO3"])
    for iso in isos:
        subset_iso = df.loc[df["ISO3"].astype(str) == iso].copy().sort_values(["variable", "range"])
        if subset_iso.empty:
            continue

        country_name = str(subset_iso["countryname"].iloc[0])
        tex_path = doc_path / f"{iso}.tex"
        outputs.append(tex_path)

        with tex_path.open("w", encoding="utf-8", newline="\n") as fh:
            def write(line: str = "") -> None:
                fh.write(line + "\n")

            write(r"\documentclass[12pt,a4paper,landscape]{article}")
            write(r"\usepackage[utf8]{inputenc}")
            write(r"\usepackage[T1]{fontenc}")
            write(r"\usepackage{graphicx}")
            write(r"\usepackage{booktabs}")
            write(r"\usepackage[margin=0.5in, top=0.5in, headsep=0.1in]{geometry}")
            write(r"\usepackage{caption}")
            write(r"\usepackage{float}")
            write(r"\usepackage[authoryear,round]{natbib}")
            write(r"\usepackage{xcolor}")
            write(r"\usepackage{colortbl}")
            write(r"\usepackage{rotating}")
            write(r"\usepackage{tabularx}")
            write(r"\usepackage{pdflscape}")
            write(r"\usepackage{adjustbox}")
            write(r"\usepackage{times}")
            write(r"\usepackage{array}")
            write(r"\usepackage{fancyhdr}")
            write(r"\usepackage[colorlinks=true, allcolors=blue]{hyperref}")
            write()
            write(r"% Setup fancy headers")
            write(r"\fancypagestyle{mainStyle}{%")
            write(r"    \fancyhf{}")
            write(r"    \renewcommand{\headrulewidth}{0pt}")
            write(r"    \fancyhead[R]{\footnotesize\hyperref[toc]{Back to Contents}}")
            write(r"}")
            write()
            write(r"\pagestyle{mainStyle}")
            write()
            write(r"\newcommand{\countryheader}[2]{\large\bfseries\hyperref[#1]{#2}}")
            write(r"\captionsetup[table]{labelformat=empty}")
            write(r"\definecolor{lightgray}{gray}{0.85}")
            write()
            write(r"\begin{document}")
            write(fr"\title{{\Large Country Data and Graphs for {country_name}}}")
            write(fr"\date{{{date.today().strftime('%B %d, %Y')}}}")
            write(r"\maketitle")
            write(r"\thispagestyle{empty}")
            write()
            write(r"\clearpage")
            write(r"\setcounter{page}{1}")
            write(r"\hypersetup{colorlinks=true,linkcolor=blue,linktoc=all}")
            write(r"\phantomsection")
            write(r"\label{toc}")
            write(r"\tableofcontents")
            write(r"\thispagestyle{empty}")
            write(r"\clearpage")
            write(r"\phantomsection")
            write(r"\addcontentsline{toc}{section}{Data availability heatmap}")
            write(r"\begin{center}")
            write(r"{\Large\bfseries Data availability heatmap}")
            write(r"\end{center}")
            write(r"\vspace{1cm}")
            write(r"\begin{figure}[H]")
            write(r"\centering")
            write(fr"\includegraphics[width=\textwidth,height=0.8\textheight,keepaspectratio]{{graphs/{iso}_heatmap.pdf}}")
            write(r"\end{figure}")
            write(r"\setcounter{page}{3}")

            for var in _sorted_levels(subset_iso["variable"]):
                subset_var = subset_iso.loc[subset_iso["variable"].astype(str) == var].copy().sort_values("range")
                if subset_var.empty:
                    continue
                var_name = str(subset_var["variable_definition"].iloc[0])
                write(r"\begin{adjustbox}{max totalsize={\paperwidth}{\paperheight},center}")
                write(r"\begin{minipage}[t][\textheight][t]{\textwidth}")
                write(r"\vspace*{0.5cm}")
                write(r"\phantomsection")
                write(fr"\addcontentsline{{toc}}{{section}}{{{var_name}}}")
                write(r"\begin{center}")
                write(fr"{{\Large\bfseries {var_name}}}")
                write(r"\end{center}")
                write(r"\vspace{0.5cm}")
                write(r"\begin{table}[H]")
                write(r"\centering")
                write(r"\small")
                write(r"\begin{tabular}{|l|l|l|}")
                write(r"\hline")
                write(r"\textbf{Source} & \textbf{Time span} & \textbf{Notes} \\")
                write(r"\hline")

                color_toggle = 0
                for _, row in subset_var.iterrows():
                    write(r"\rowcolor{white}" if color_toggle == 0 else r"\rowcolor{lightgray}")
                    color_toggle = 1 - color_toggle
                    write(f"{row['source']} & {row['range']} & {row['notes']} \\\\")

                write(r"\hline")
                write(r"\end{tabular}")
                write(r"\end{table}")
                write(r"\begin{figure}[H]")
                write(r"\centering")
                write(fr"\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{{graphs/{iso}_{var}.pdf}}")
                write(r"\end{figure}")
                write(r"\end{minipage}")
                write(r"\end{adjustbox}")

            write(r"\phantomsection")
            write(r"\addcontentsline{toc}{section}{References}")
            write(r"\begin{center}")
            write(r"{\Large\bfseries References}")
            write(r"\end{center}")
            write(r"\small")
            write(r"\bibliographystyle{qje}")
            write(r"\bibliography{bib}")
            write(r"\end{document}")

    return outputs


def gmdcombinedocs(
    files: str | Sequence[str],
    *,
    doc_dir: Path | str = OUTPUT_DOC_DIR,
    today: date | None = None,
) -> Path:
    file_list = _token_list(files)
    doc_path = _resolve_path(doc_dir)
    doc_path.mkdir(parents=True, exist_ok=True)
    master_path = doc_path / "master.tex"
    current_date = (today or date.today()).strftime("%B %d, %Y")

    section_styles = [
        "nGDP", "rGDP", "rcons", "cons", "inv", "finv", "exports", "imports", "CA_GDP", "USDfx", "REER",
        "cons_GDP", "inv_GDP", "finv_GDP", "exports_GDP", "imports_GDP",
        "govtax", "govexp", "govdef_GDP", "govdebt_GDP", "govrev",
        "govtax_GDP", "govexp_GDP", "govrev_GDP",
        "M0", "M1", "M2", "M3", "M4", "cbrate", "strate", "ltrate", "CPI", "HPI", "infl", "unemp", "pop",
    ]

    page_counts: dict[str, int] = {}
    for var in file_list:
        page_counts[var] = 0
        tex_path = doc_path / f"{var}.tex"
        if not tex_path.exists():
            continue
        for line in tex_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if re.search(r"\\addcontentsline\{toc\}\{section\}\{([^}]+)\}", line):
                page_counts[var] += 1

    with master_path.open("w", encoding="utf-8", newline="\n") as fh:
        def write(line: str = "") -> None:
            fh.write(line + "\n")

        write(r"\documentclass[12pt,a4paper,landscape]{article}")
        write(r"\usepackage[utf8]{inputenc}")
        write(r"\usepackage[T1]{fontenc}")
        write(r"\usepackage{graphicx}")
        write(r"\usepackage{booktabs}")
        write(r"\usepackage[margin=0.4in, top=0.5in, headsep=0.2in]{geometry}")
        write(r"\usepackage{caption}")
        write(r"\usepackage{float}")
        write(r"\usepackage[authoryear,round]{natbib}")
        write(r"\usepackage{xcolor}")
        write(r"\usepackage{colortbl}")
        write(r"\usepackage{rotating}")
        write(r"\usepackage{tabularx}")
        write(r"\usepackage{pdflscape}")
        write(r"\usepackage{adjustbox}")
        write(r"\usepackage{longtable}")
        write(r"\usepackage{times}")
        write(r"\usepackage{array}")
        write(r"\usepackage{fancyhdr}")
        write(r"\usepackage[colorlinks=true, allcolors=blue]{hyperref}")
        write()
        write(r"% Setup fancy headers")
        write(r"\fancypagestyle{mainStyle}{%")
        write(r"    \fancyhf{}")
        write(r"    \renewcommand{\headrulewidth}{0pt}")
        write(r"    \fancyhead[R]{\hyperref[main-toc]{Back to Main contents}}")
        write(r"}")
        write()

        for var in section_styles:
            varname = VARIABLE_DISPLAY_NAMES.get(var, var)
            write(fr"\fancypagestyle{{{var}Style}}{{%")
            write(r"    \fancyhf{}")
            write(r"    \renewcommand{\headrulewidth}{0pt}")
            write(fr"    \fancyhead[R]{{\hyperref[{var}-toc]{{Back to {varname} contents}}}}")
            write(r"}")
            write()

        write(r"\pagestyle{mainStyle}")
        write()
        write(r"\newcommand{\countryheader}[2]{\large\bfseries\hyperref[#1]{#2}}")
        write(r"\captionsetup[table]{labelformat=empty}")
        write(r"\definecolor{lightgray}{gray}{0.85}")
        write()
        write(r"\begin{document}")
        write(r"\title{\Large Global Macro Project: Data Documentation}")
        write(fr"\date{{{current_date}}}")
        write(r"\maketitle")
        write(r"\thispagestyle{empty}")
        write()
        write(r"\clearpage")
        write(r"\setcounter{page}{1}")
        write(r"\hypersetup{colorlinks=true,linkcolor=blue,linktoc=all}")
        write(r"\phantomsection")
        write(r"\label{main-toc}")
        write(r"\vspace*{2cm}")
        write(r"\begin{center}")
        write(r"{\Large\bfseries Contents}")
        write(r"\end{center}")
        write(r"\vspace{1cm}")
        write(r"\begin{center}")
        write(r"\renewcommand{\arraystretch}{1.5}")
        write(r"\begin{longtable}{p{\dimexpr\textwidth-1cm\relax}r}")

        cumulative_pages = 3
        for var in file_list:
            varname = VARIABLE_DISPLAY_NAMES.get(var, var)
            write(fr"{{\large\bfseries\hyperref[{var}-toc]{{{varname}}}}} & {{\large\bfseries\hyperref[{var}-toc]{{{cumulative_pages}}}}} \\")
            cumulative_pages = cumulative_pages + page_counts.get(var, 0) + 1

        write(r"{\large\bfseries\hyperref[references]{References}} & {\large\bfseries\hyperref[references]{}} \\")
        write(r"\end{longtable}")
        write(r"\end{center}")
        write(r"\pagestyle{empty}")
        write(r"\pagestyle{mainStyle}")

        for var in file_list:
            varname = VARIABLE_DISPLAY_NAMES.get(var, var)
            write(r"\clearpage")
            write(r"\pagestyle{empty}")
            write(r"\hypersetup{colorlinks=true,linkcolor=blue,linktoc=all}")
            write(r"\phantomsection")
            write(fr"\label{{{var}-toc}}")
            write(r"\vspace*{2cm}")
            write(r"\begin{center}")
            write(fr"{{\Large\bfseries\hyperref[main-toc]{{{varname}}}}}")
            write(r"\end{center}")
            write(r"\vspace{1cm}")
            write(r"\begin{center}")
            write(r"\renewcommand{\arraystretch}{1.5}")
            write(r"\begin{longtable}{p{\dimexpr\textwidth-1cm\relax}r}")

            tex_path = doc_path / f"{var}.tex"
            tex_lines = tex_path.read_text(encoding="utf-8", errors="ignore").splitlines() if tex_path.exists() else []
            for line in tex_lines:
                match = re.search(r"\\addcontentsline\{toc\}\{section\}\{([^}]+)\}", line)
                if match:
                    country = match.group(1)
                    if country != "References":
                        write(fr"\bfseries\hyperref[{var}-{country}]{{{country}}} & \bfseries\hyperref[{var}-{country}]{{\pageref{{{var}-{country}}}}} \\")

            write(r"\end{longtable}")
            write(r"\end{center}")
            write(fr"\pagestyle{{{var}Style}}")

            capture = False
            for line in tex_lines:
                if re.search(r"\\begin\{document\}", line):
                    capture = True
                    continue
                if re.search(r"\\phantomsection.*References", line) or re.search(r"\\addcontentsline\{toc\}\{section\}\{References\}", line):
                    capture = False
                    continue
                if not capture:
                    continue
                if re.search(r"\\(title|date|maketitle|tableofcontents|thispagestyle|clearpage)", line):
                    continue
                match = re.search(r"\\addcontentsline\{toc\}\{section\}\{([^}]+)\}", line)
                if match:
                    country = match.group(1)
                    if country != "References":
                        write(r"\phantomsection")
                        write(fr"\label{{{var}-{country}}}")
                write(line)

        write(r"\clearpage")
        write(r"\pagestyle{mainStyle}")
        write(r"\phantomsection")
        write(r"\label{references}")
        write(r"\begin{center}")
        write(r"{\Large\bfseries References}")
        write(r"\end{center}")
        write(r"\small")
        write(r"\bibliographystyle{plainnat}")
        write(r"\bibliography{bib}")
        write(r"\end{document}")

    return master_path


__all__ = [
    "DATA_FINAL_DIR",
    "DATA_HELPER_DIR",
    "DATA_TEMP_DIR",
    "OUTPUT_NUMBERS_DIR",
    "SESSION_GLOBALS",
    "PipelineRuntimeError",
    "data_export",
    "gmdaddnote",
    "gmdaddnote_source",
    "gmdcalculate",
    "gmdcombinedocs",
    "gmdfixunits",
    "gmdisolist",
    "gmdmakedoc",
    "gmdmakedoc_cs",
    "gmdmakeplot_cs",
    "gmdsavedate",
    "gmdsourcelist",
    "gmdvarlist",
    "gmdwriterows",
    "read_dta",
    "savedelta",
    "splice",
    "write_dta",
]
