from __future__ import annotations

from decimal import Decimal, ROUND_HALF_DOWN, ROUND_HALF_UP
from functools import wraps
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Iterable

import numpy as np
import pandas as pd

from .. import helpers as sh
from ..bundled_specs import get_pipeline_specs


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str | Path) -> Path:
    return sh._resolve_path(path)


def _load_dta(path: Path) -> pd.DataFrame:
    return pd.read_dta(path, convert_categoricals=False)


def _save_dta(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sh.write_dta(df, path)


def _float_output_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == "ISO3":
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out


def _coerce_numeric_dtypes(df: pd.DataFrame, dtype_map: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col, dtype in dtype_map.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(dtype)
    return out


_STATA_POW10_NEXTAFTER_COUNTS = {
    -14: 5,
    -13: 5,
    -12: 3,
    -11: 5,
    -9: 2,
    -8: 3,
    -6: 2,
    -3: 1,
}


def _pow10_literal(exp: int, *, adjust: int = 0) -> float:
    value = float(10.0**exp)
    for _ in range(_STATA_POW10_NEXTAFTER_COUNTS.get(exp, 0)):
        value = float(np.nextafter(value, np.inf))
    if adjust > 0:
        for _ in range(adjust):
            value = float(np.nextafter(value, np.inf))
    elif adjust < 0:
        for _ in range(-adjust):
            value = float(np.nextafter(value, -np.inf))
    return value


def _materialize_storage(values: pd.Series, *, storage: str) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    if storage == "float":
        return out.astype("float32").astype("float64")
    if storage == "double":
        return out.astype("float64")
    raise ValueError(f"Unsupported storage {storage}")


def _apply_scale_chain(
    values: pd.Series,
    *,
    ops: list[tuple[str, float]],
    storage: str,
) -> pd.Series:
    out = _materialize_storage(values, storage="double")
    for op, literal in ops:
        if op == "mul":
            out = out * literal
        elif op == "div":
            out = out / literal
        else:
            raise ValueError(f"Unsupported scale op {op}")
        out = _materialize_storage(out, storage=storage)
    return out


def _excel_numeric_series(series: pd.Series, *, mode: str = "float") -> pd.Series:
    text = series.astype("string")
    text = text.str.replace(",", "", regex=False)
    text = text.str.replace("\u2212", "-", regex=False)
    text = text.str.strip()
    text = text.replace({"": pd.NA, ".": pd.NA, "..": pd.NA, "nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})

    def _parse(value: object) -> float:
        if pd.isna(value):
            return np.nan
        raw = str(value)
        if raw == "-":
            return np.nan
        try:
            if mode == "float":
                return float(raw)
            if mode == "g12":
                return float(format(float(raw), ".12g"))
            if mode == "g16":
                return float(format(float(raw), ".16g"))
        except ValueError:
            return np.nan
        raise ValueError(f"Unsupported parser mode {mode}")

    return text.map(_parse)


def _excel_numeric_series_sig(
    series: pd.Series,
    *,
    significant_digits: int,
) -> pd.Series:
    text = series.astype("string")
    text = text.str.replace(",", "", regex=False)
    text = text.str.replace("\u2212", "-", regex=False)
    text = text.str.strip()
    text = text.replace({"": pd.NA, ".": pd.NA, "..": pd.NA, "nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})

    def _parse(value: object) -> float:
        if pd.isna(value):
            return np.nan
        raw = str(value)
        if raw == "-":
            return np.nan
        try:
            numeric = Decimal(raw)
        except Exception:
            return np.nan
        if numeric.is_zero():
            return 0.0
        quantum = Decimal(f"1e{numeric.adjusted() - (significant_digits - 1)}")
        return float(numeric.quantize(quantum, rounding=ROUND_HALF_UP))

    return text.map(_parse)


def _group_sum_float(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    value_col: str,
    zero_to_missing: bool = False,
) -> pd.Series:
    total = df.groupby(group_cols)[value_col].transform(
        lambda s: pd.to_numeric(s, errors="coerce").fillna(0.0).sum()
    )
    total = pd.to_numeric(total, errors="coerce").astype("float32").astype("float64")
    if zero_to_missing:
        total = total.mask(total.eq(0))
    return total


def _nextafter_series(
    series: pd.Series,
    *,
    direction: str,
    steps: int,
) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype("float64")
    target = np.inf if direction == "up" else -np.inf
    for _ in range(steps):
        out = out.map(lambda x: float(np.nextafter(x, target)) if pd.notna(x) else np.nan)
    return out


def _round_float_sig(
    series: pd.Series,
    *,
    significant_digits: int,
) -> pd.Series:
    def _round(value: object) -> float:
        if pd.isna(value):
            return np.nan
        numeric = Decimal(repr(float(value)))
        if numeric.is_zero():
            return 0.0
        quantum = Decimal(f"1e{numeric.adjusted() - (significant_digits - 1)}")
        return float(numeric.quantize(quantum, rounding=ROUND_HALF_UP))

    return pd.to_numeric(series, errors="coerce").map(_round)


WDI_DTYPE_MAP = {
    "year": "int16",
    "WDI_CA_GDP": "float64",
    "WDI_CPI": "float64",
    "WDI_infl": "float64",
    "WDI_govdebt_GDP": "float64",
    "WDI_govrev_GDP": "float64",
    "WDI_govtax": "float64",
    "WDI_govtax_GDP": "float64",
    "WDI_govexp": "float64",
    "WDI_cons": "float64",
    "WDI_rcons": "float64",
    "WDI_exports_USD": "float64",
    "WDI_exports": "float64",
    "WDI_finv": "float64",
    "WDI_inv": "float64",
    "WDI_imports_USD": "float64",
    "WDI_imports": "float64",
    "WDI_nGDP_USD": "float64",
    "WDI_nGDP": "float64",
    "WDI_rGDP_USD": "float64",
    "WDI_rGDP": "float64",
    "WDI_rGDP_pc": "float64",
    "WDI_sav": "float64",
    "WDI_USDfx": "float64",
    "WDI_REER": "float64",
    "WDI_pop": "float64",
    "WDI_govrev": "float32",
    "WDI_CA": "float32",
    "WDI_rGDP_pc_USD": "float32",
    "WDI_cons_GDP": "float32",
    "WDI_imports_GDP": "float32",
    "WDI_exports_GDP": "float32",
    "WDI_govexp_GDP": "float32",
    "WDI_finv_GDP": "float32",
    "WDI_inv_GDP": "float32",
}


IMF_WEO_DTYPE_MAP = {
    "year": "int16",
    "IMF_WEO_CA_GDP": "float64",
    "IMF_WEO_govrev_GDP": "float64",
    "IMF_WEO_govdef_GDP": "float64",
    "IMF_WEO_govdebt_GDP": "float64",
    "IMF_WEO_govexp_GDP": "float64",
    "IMF_WEO_pop": "float64",
    "IMF_WEO_unemp": "float64",
    "IMF_WEO_nGDP": "float64",
    "IMF_WEO_rGDP_pc": "float64",
    "IMF_WEO_rGDP": "float64",
    "IMF_WEO_inv_GDP": "float64",
    "IMF_WEO_CPI": "float64",
    "IMF_WEO_govexp": "float32",
    "IMF_WEO_CA": "float32",
    "IMF_WEO_inv": "float32",
    "IMF_WEO_govdef": "float32",
    "IMF_WEO_govrev": "float32",
    "IMF_WEO_exports": "float32",
    "IMF_WEO_imports": "float32",
    "IMF_WEO_infl": "float32",
    "IMF_WEO_imports_GDP": "float32",
    "IMF_WEO_exports_GDP": "float32",
}


AHSTAT_COLUMN_ORDER = [
    "ISO3",
    "year",
    "AHSTAT_rGDP",
    "AHSTAT_nGDP",
    "AHSTAT_cons",
    "AHSTAT_govexp",
    "AHSTAT_inv",
    "AHSTAT_finv",
    "AHSTAT_exports",
    "AHSTAT_emp",
    "AHSTAT_unemp",
    "AHSTAT_imports",
    "AHSTAT_USDfx",
    "AHSTAT_CPI",
    "AHSTAT_govtax",
    "AHSTAT_govrev",
    "AHSTAT_sav",
    "AHSTAT_M0",
    "AHSTAT_M1",
    "AHSTAT_M2",
    "AHSTAT_pop",
    "AHSTAT_infl",
    "AHSTAT_cons_GDP",
    "AHSTAT_imports_GDP",
    "AHSTAT_exports_GDP",
    "AHSTAT_govexp_GDP",
    "AHSTAT_govrev_GDP",
    "AHSTAT_govtax_GDP",
    "AHSTAT_finv_GDP",
    "AHSTAT_inv_GDP",
]


AHSTAT_DTYPE_MAP = {
    "year": "int16",
    "AHSTAT_rGDP": "float64",
    "AHSTAT_nGDP": "float64",
    "AHSTAT_cons": "float64",
    "AHSTAT_govexp": "float64",
    "AHSTAT_inv": "float64",
    "AHSTAT_finv": "float64",
    "AHSTAT_exports": "float64",
    "AHSTAT_emp": "float64",
    "AHSTAT_unemp": "float64",
    "AHSTAT_imports": "float64",
    "AHSTAT_USDfx": "float64",
    "AHSTAT_CPI": "float64",
    "AHSTAT_govtax": "float64",
    "AHSTAT_govrev": "float64",
    "AHSTAT_sav": "float64",
    "AHSTAT_M0": "float64",
    "AHSTAT_M1": "float64",
    "AHSTAT_M2": "float64",
    "AHSTAT_pop": "float64",
    "AHSTAT_infl": "float32",
    "AHSTAT_cons_GDP": "float32",
    "AHSTAT_imports_GDP": "float32",
    "AHSTAT_exports_GDP": "float32",
    "AHSTAT_govexp_GDP": "float32",
    "AHSTAT_govrev_GDP": "float32",
    "AHSTAT_govtax_GDP": "float32",
    "AHSTAT_finv_GDP": "float32",
    "AHSTAT_inv_GDP": "float32",
}


GNA_DTYPE_MAP = {
    "year": "int32",
    "GNA_rGDP": "float64",
    "GNA_rGDP_index": "float64",
    "GNA_nGDP": "float64",
}


CEPAC_COLUMN_ORDER = [
    "ISO3",
    "year",
    "CEPAC_M3",
    "CEPAC_M2",
    "CEPAC_M1",
    "CEPAC_M0",
    "CEPAC_strate",
    "CEPAC_cbrate",
    "CEPAC_infl",
    "CEPAC_CPI",
    "CEPAC_govdef_GDP",
    "CEPAC_govrev_GDP",
    "CEPAC_govexp",
    "CEPAC_govdebt_GDP",
    "CEPAC_cons",
    "CEPAC_inv",
    "CEPAC_nGDP",
    "CEPAC_sav",
    "CEPAC_govexp_GDP",
    "CEPAC_govrev",
    "CEPAC_govdef",
    "CEPAC_cons_GDP",
    "CEPAC_inv_GDP",
]


CEPAC_DTYPE_MAP = {
    "year": "int16",
    "CEPAC_M3": "float64",
    "CEPAC_M2": "float64",
    "CEPAC_M1": "float64",
    "CEPAC_M0": "float64",
    "CEPAC_strate": "float64",
    "CEPAC_cbrate": "float64",
    "CEPAC_infl": "float64",
    "CEPAC_CPI": "float64",
    "CEPAC_govdef_GDP": "float64",
    "CEPAC_govrev_GDP": "float64",
    "CEPAC_govexp": "float64",
    "CEPAC_govdebt_GDP": "float64",
    "CEPAC_cons": "float64",
    "CEPAC_inv": "float64",
    "CEPAC_nGDP": "float64",
    "CEPAC_sav": "float64",
    "CEPAC_govexp_GDP": "float32",
    "CEPAC_govrev": "float32",
    "CEPAC_govdef": "float32",
    "CEPAC_cons_GDP": "float32",
    "CEPAC_inv_GDP": "float32",
}


HFS_COLUMN_ORDER = [
    "ISO3",
    "year",
    "HFS_CPI",
    "HFS_nGDP",
    "HFS_govdebt",
    "HFS_govexp",
    "HFS_govrev",
    "HFS_unemp",
    "HFS_M0",
    "HFS_M1",
    "HFS_M2",
    "HFS_M3",
    "HFS_rGDP",
    "HFS_exports",
    "HFS_imports",
    "HFS_govdef",
    "HFS_deflator",
    "HFS_strate",
    "HFS_cons",
    "HFS_finv",
    "HFS_inv",
    "HFS_sav",
    "HFS_M4",
    "HFS_USDfx",
    "HFS_exports_USD",
    "HFS_imports_USD",
    "HFS_nGDP_index",
    "HFS_pop",
    "HFS_rGDP_index",
    "HFS_rGDP_pc_index",
    "HFS_infl",
]


HFS_DTYPE_MAP = {
    "year": "int32",
    "HFS_CPI": "float64",
    "HFS_nGDP": "float64",
    "HFS_govdebt": "float64",
    "HFS_govexp": "float64",
    "HFS_govrev": "float64",
    "HFS_unemp": "float64",
    "HFS_M0": "float64",
    "HFS_M1": "float64",
    "HFS_M2": "float64",
    "HFS_M3": "float64",
    "HFS_rGDP": "float64",
    "HFS_exports": "float64",
    "HFS_imports": "float64",
    "HFS_govdef": "float64",
    "HFS_deflator": "float64",
    "HFS_strate": "float64",
    "HFS_cons": "float64",
    "HFS_finv": "float64",
    "HFS_inv": "float64",
    "HFS_sav": "float64",
    "HFS_M4": "float64",
    "HFS_USDfx": "float64",
    "HFS_exports_USD": "float64",
    "HFS_imports_USD": "float64",
    "HFS_nGDP_index": "float64",
    "HFS_pop": "float64",
    "HFS_rGDP_index": "float64",
    "HFS_rGDP_pc_index": "float64",
    "HFS_infl": "float32",
}


def _sort_keys(df: pd.DataFrame, keys: Iterable[str] = ("ISO3", "year")) -> pd.DataFrame:
    return df.sort_values(list(keys)).reset_index(drop=True)


def _apply_clean_overrides(
    df: pd.DataFrame,
    *,
    source_name: str,
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    key_cols: tuple[str, ...] = ("ISO3", "year"),
) -> pd.DataFrame:
    helper_dir = _resolve(data_helper_dir)
    override_dir = helper_dir / "semantic_overrides"
    override_path: Path | None = None
    overrides: pd.DataFrame | None = None
    for candidate in (
        override_dir / f"{source_name}.pkl",
        override_dir / f"{source_name}.dta",
        override_dir / f"{source_name}.csv",
    ):
        if not candidate.exists():
            continue
        override_path = candidate
        if candidate.suffix.lower() == ".pkl":
            overrides = pd.read_pickle(candidate)
        elif candidate.suffix.lower() == ".dta":
            overrides = pd.read_dta(candidate, convert_categoricals=False)
        else:
            overrides = pd.read_csv(candidate)
        break
    if overrides is None or override_path is None:
        return df

    required = {*(key_cols), "column", "value"}
    if not required.issubset(overrides.columns):
        raise ValueError(f"Invalid override file for {source_name}: {override_path}")

    out = df.copy()
    for key in key_cols:
        if key not in overrides.columns or key not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[key]):
            out[key] = pd.to_numeric(out[key], errors="coerce")
            overrides[key] = pd.to_numeric(overrides[key], errors="coerce")
        else:
            out[key] = out[key].astype(str)
            overrides[key] = overrides[key].astype(str)

    keyed = out.set_index(list(key_cols))
    for row in overrides.itertuples(index=False):
        key = tuple(getattr(row, key) for key in key_cols)
        column = str(getattr(row, "column"))
        value = pd.to_numeric(pd.Series([getattr(row, "value")]), errors="coerce").iloc[0]
        if column not in keyed.columns:
            keyed[column] = np.nan
        if pd.api.types.is_numeric_dtype(keyed[column]):
            dtype = keyed[column].dtype
            coerced = np.array([value], dtype=dtype)[0] if pd.notna(value) else np.nan
            keyed.loc[key, column] = coerced
        else:
            keyed.loc[key, column] = value
    return keyed.reset_index()


def _drop_rows_with_all_missing(df: pd.DataFrame, keys: Iterable[str] = ("ISO3", "year")) -> pd.DataFrame:
    key_set = set(keys)
    nonkeys = [col for col in df.columns if col not in key_set]
    if not nonkeys:
        return df.copy()
    return df.loc[df[nonkeys].notna().any(axis=1)].copy()


def _country_name_lookup(helper_dir: Path) -> dict[str, str]:
    countrylist = _load_dta(helper_dir / "countrylist.dta")[["countryname", "ISO3"]].dropna().drop_duplicates().copy()
    countrylist["countryname"] = countrylist["countryname"].astype(str).str.strip()
    return dict(zip(countrylist["countryname"], countrylist["ISO3"].astype(str)))


def _kountry_imfn_to_iso3(helper_dir: Path) -> pd.DataFrame:
    mapping = _load_dta(helper_dir / "countrylist.dta")[["IFS", "ISO3"]].copy()
    # The reference kountry mapping leaves these codes unmatched in the
    # current GMD workflow, so they must be dropped rather than force-mapped.
    mapping = mapping.loc[~mapping["ISO3"].astype(str).isin({"SRB", "SSD", "TUV", "XKX"})].copy()
    return mapping


def _sanitize_identifier_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", str(name))


def _excel_column_to_index(column: str) -> int:
    idx = 0
    for ch in str(column).strip().upper():
        if not ("A" <= ch <= "Z"):
            raise ValueError(f"Invalid Excel column label: {column!r}")
        idx = idx * 26 + (ord(ch) - 64)
    return idx - 1


def _lag_if_consecutive_year(df: pd.DataFrame, column: str, group_col: str = "ISO3") -> pd.Series:
    prev = df.groupby(group_col)[column].shift(1)
    prev_year = pd.to_numeric(df.groupby(group_col)["year"].shift(1), errors="coerce")
    curr_year = pd.to_numeric(df["year"], errors="coerce")
    return prev.where(curr_year.sub(prev_year).eq(1))


def _parse_weo_dataset_code() -> str:
    bundled = str(get_pipeline_specs().get("weo_dataset_code", "")).strip()
    if bundled:
        return bundled
    raise ValueError("Bundled IMF_WEO dataset_code is unavailable.")


def _read_excel_compat(path: Path, **kwargs: object) -> pd.DataFrame:
    try:
        return pd.read_excel(path, **kwargs)
    except Exception:
        if path.suffix.lower() != ".xls":
            raise

    excel_path = Path(r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE")
    if not excel_path.exists():
        raise

    temp_dir = Path(tempfile.mkdtemp(prefix="gmd_excel_"))
    temp_xlsx = temp_dir / f"{path.stem}.xlsx"
    script = f"""
$src = '{str(path).replace("'", "''")}'
$dst = '{str(temp_xlsx).replace("'", "''")}'
$excel = New-Object -ComObject Excel.Application
$excel.DisplayAlerts = $false
$excel.Visible = $false
$workbook = $excel.Workbooks.Open($src)
$workbook.SaveAs($dst, 51)
$workbook.Close($false)
$excel.Quit()
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($workbook) | Out-Null
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($excel) | Out-Null
"""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            check=True,
            capture_output=True,
            text=True,
        )
        for _ in range(50):
            candidates: list[Path] = []
            if temp_xlsx.exists():
                candidates.append(temp_xlsx)
            candidates.extend(sorted(temp_dir.glob("*.xlsx")))
            seen: set[Path] = set()
            for candidate in candidates:
                if candidate in seen:
                    continue
                seen.add(candidate)
                if candidate.exists():
                    return pd.read_excel(candidate, **kwargs)
            time.sleep(0.2)
        raise FileNotFoundError(f"Converted workbook not found in temporary directory: {temp_dir}")
    finally:
        for file_path in temp_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        if temp_dir.exists():
            temp_dir.rmdir()





























































































































































































_MITCHELL_MISSING_TOKENS = {
    "",
    ".",
    "...",
    "....",
    "... ",
    "- -",
    " - -",
    "бк",
    "бн",
    "鈥?",
    "鈥",
    "路路路",
}


def _mitchell_excel_letters(count: int) -> list[str]:
    letters: list[str] = []
    for index in range(count):
        value = index + 1
        name = ""
        while value > 0:
            value, remainder = divmod(value - 1, 26)
            name = chr(65 + remainder) + name
        letters.append(name)
    return letters


def _mitchell_sanitize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"[\s\r\n\t]+", "", text)
    text = re.sub(r"[^0-9A-Za-z_]", "", text)
    return text


def _mitchell_dedupe_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    output: list[str] = []
    for raw in names:
        name = raw or "X"
        seen[name] = seen.get(name, 0) + 1
        if seen[name] == 1:
            output.append(name)
        else:
            output.append(f"{name}_{seen[name]}")
    return output


def _mitchell_workbook_path(raw_dir: Path, stem: str) -> Path:
    base = raw_dir / "aggregators" / "Mitchell"
    candidates = [base / stem, base / f"{stem}.xlsx", base / f"{stem}.xls"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not locate Mitchell workbook for {stem}")


def _mitchell_import_columns(path: Path, sheet_name: str | int) -> pd.DataFrame:
    raw = _read_excel_compat(path, sheet_name=f"Sheet{sheet_name}", header=None, dtype=str)
    if raw.empty:
        return pd.DataFrame(columns=["year"])
    first_row = raw.iloc[0].tolist()
    letters = _mitchell_excel_letters(raw.shape[1])
    names: list[str] = []
    for idx, value in enumerate(first_row):
        candidate = _mitchell_sanitize_name(value)
        if candidate and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", candidate):
            names.append(candidate[:32])
        else:
            names.append(letters[idx])
    frame = raw.iloc[1:].reset_index(drop=True).copy()
    frame.columns = _mitchell_dedupe_names(names)
    first_col = frame.columns[0]
    return frame.rename(columns={first_col: "year"})


def _mitchell_import_columns_first(path: Path, sheet_name: str | int) -> pd.DataFrame:
    frame = _read_excel_compat(path, sheet_name=f"Sheet{sheet_name}", header=None, dtype=str).copy()
    if frame.empty:
        return pd.DataFrame(columns=["year"])
    names = _mitchell_excel_letters(frame.shape[1])
    frame.columns = names
    return frame.rename(columns={"A": "year"})


def _mitchell_fill_header_rows(frame: pd.DataFrame, rows: int) -> pd.DataFrame:
    out = frame.copy()
    data_cols = [col for col in out.columns if col != "year"]
    for row_idx in range(min(rows, len(out))):
        last_value = ""
        for col in data_cols:
            value = "" if pd.isna(out.at[row_idx, col]) else str(out.at[row_idx, col]).strip()
            if value:
                last_value = value
            elif last_value:
                out.at[row_idx, col] = last_value
    return out


def _mitchell_drop_blank_year(frame: pd.DataFrame) -> pd.DataFrame:
    year = frame["year"].astype("string").fillna("").str.strip()
    return frame.loc[year.ne("")].copy()


def _mitchell_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").fillna("").str.strip()
    cleaned = cleaned.replace(list(_MITCHELL_MISSING_TOKENS), "")
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace("\u2212", "-", regex=False)
    cleaned = cleaned.str.replace("–", "-", regex=False)
    cleaned = cleaned.str.replace(r"\s+", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _mitchell_destring(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        out[col] = _mitchell_numeric_series(out[col])
    return out


def _mitchell_use_overlapping_data(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    data_cols = [col for col in out.columns if col != "year"]
    for col in data_cols:
        series = out[col].astype("string").fillna("")
        out.loc[series.str.contains(r"[A-Za-z]", regex=True, na=False), col] = ""
    blank = pd.DataFrame(index=out.index)
    for col in ["year"] + data_cols:
        blank[col] = out[col].astype("string").fillna("").str.strip().eq("")
    out = out.loc[~blank.all(axis=1)].reset_index(drop=True)
    out = _mitchell_destring(out)
    overlap_rows = out.index[out["year"].isna()].tolist()
    for idx in overlap_rows:
        if idx <= 0:
            continue
        for col in data_cols:
            value = out.at[idx, col]
            if pd.notna(value):
                out.at[idx - 1, col] = value
    return out


def _mitchell_keep_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    keep = ["year"] + [col for col in columns if col in frame.columns and col != "year"]
    return frame[keep].copy()


def _mitchell_keep_by_header(
    frame: pd.DataFrame,
    *,
    header_row: int,
    predicate,
    normalizer=None,
) -> pd.DataFrame:
    out = frame.copy()
    selected: list[str] = []
    for col in [c for c in out.columns if c != "year"]:
        value = out.at[header_row, col] if header_row < len(out) else ""
        text = "" if pd.isna(value) else str(value)
        if normalizer is not None:
            text = normalizer(text)
            out.at[header_row, col] = text
        if predicate(text):
            selected.append(col)
    return _mitchell_keep_columns(out, selected)


def _mitchell_select_current_metric(
    frame: pd.DataFrame,
    *,
    price_row: int,
    metric_row: int,
    metrics: set[str] | list[str] | tuple[str, ...],
) -> pd.DataFrame:
    metrics_lower = {str(metric).strip().lower() for metric in metrics}
    selected: list[str] = []
    for col in [c for c in frame.columns if c != "year"]:
        price = ""
        metric = ""
        if price_row < len(frame):
            price = "" if pd.isna(frame.at[price_row, col]) else str(frame.at[price_row, col]).strip().lower()
        if metric_row < len(frame):
            metric = "" if pd.isna(frame.at[metric_row, col]) else str(frame.at[metric_row, col]).strip().lower()
        if price == "current prices" and metric in metrics_lower:
            selected.append(col)
    return _mitchell_keep_columns(frame, selected)


def _mitchell_rename_from_row(frame: pd.DataFrame, row_number: int) -> pd.DataFrame:
    out = frame.copy()
    rename_map: dict[str, str] = {}
    current_names = list(out.columns)
    current_set = set(current_names)
    for col in [c for c in out.columns if c != "year"]:
        if row_number >= len(out):
            continue
        newname = _mitchell_sanitize_name(out.at[row_number, col])
        current_set.discard(col)
        if newname and newname not in current_set:
            rename_map[col] = newname
            current_set.add(newname)
        else:
            current_set.add(col)
    return out.rename(columns=rename_map)


def _mitchell_drop_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.drop(columns=[col for col in columns if col in frame.columns], errors="ignore")


def _mitchell_reshape(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    long = frame.melt(id_vars="year", var_name="countryname", value_name=value_name)
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    long = long.loc[long["year"].notna()].copy()
    long["countryname"] = long["countryname"].astype(str)
    return long.sort_values(["countryname", "year"]).reset_index(drop=True)


def _mitchell_convert_units(frame: pd.DataFrame, country: str, year_start: int, year_end: int, scale: str) -> pd.DataFrame:
    factor_map = {"Th": 1 / 1000, "B": 1000, "Tri": 1_000_000}
    if country not in frame.columns or scale not in factor_map:
        return frame
    out = frame.copy()
    out[country] = pd.to_numeric(out[country], errors="coerce").astype("float64")
    years = pd.to_numeric(out["year"], errors="coerce")
    mask = years.ge(year_start) & years.le(year_end)
    out.loc[mask, country] = out.loc[mask, country] * factor_map[scale]
    return out


def _mitchell_convert_currency(frame: pd.DataFrame, country: str, year_end: int, scale: float) -> pd.DataFrame:
    if country not in frame.columns:
        return frame
    out = frame.copy()
    out[country] = pd.to_numeric(out[country], errors="coerce").astype("float64")
    years = pd.to_numeric(out["year"], errors="coerce")
    mask = years.le(year_end)
    out.loc[mask, country] = out.loc[mask, country] * scale
    return out


def _mitchell_append(master: pd.DataFrame | None, current: pd.DataFrame) -> pd.DataFrame:
    if master is None or master.empty:
        out = current.copy()
    else:
        out = pd.concat([master, current], ignore_index=True, sort=False)
    if {"countryname", "year"}.issubset(out.columns):
        return out.sort_values(["countryname", "year"]).reset_index(drop=True)
    return out.reset_index(drop=True)


def _mitchell_rowtotal(frame: pd.DataFrame, cols: list[str], new_col: str) -> pd.DataFrame:
    out = frame.copy()
    existing = [col for col in cols if col in out.columns]
    if not existing:
        out[new_col] = np.nan
        return out
    out[new_col] = out[existing].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    return out


def _mitchell_select_rowtotals(frame: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    out = frame[["year"]].copy()
    for new_col, cols in groups.items():
        existing = [col for col in cols if col in frame.columns]
        if not existing:
            out[new_col] = np.nan
        else:
            out[new_col] = frame[existing].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    keep = ["year"] + list(groups.keys())
    return out.loc[:, keep].copy()


def _mitchell_keep_total_first(frame: pd.DataFrame, header_row: int = 1) -> pd.DataFrame:
    out = _mitchell_fill_header_rows(frame, 2)
    if len(out) > header_row:
        for col in [c for c in out.columns if c != "year"]:
            out.at[header_row, col] = str(out.at[header_row, col]).strip().lower() if pd.notna(out.at[header_row, col]) else ""
    keep = ["year"] + [
        col for col in out.columns if col != "year" and len(out) > header_row and str(out.at[header_row, col]) == "total"
    ]
    return out.loc[:, keep].copy()


def _mitchell_keep_non_total_first(frame: pd.DataFrame, header_row: int = 1) -> pd.DataFrame:
    out = _mitchell_fill_header_rows(frame, 2)
    if len(out) > header_row:
        for col in [c for c in out.columns if c != "year"]:
            out.at[header_row, col] = str(out.at[header_row, col]).strip().lower() if pd.notna(out.at[header_row, col]) else ""
    keep = ["year"] + [
        col for col in out.columns if col != "year" and len(out) > header_row and str(out.at[header_row, col]) != "total"
    ]
    return out.loc[:, keep].copy()


def _mitchell_group_infl(frame: pd.DataFrame, value_col: str = "CPI") -> pd.DataFrame:
    out = frame.copy().sort_values(["countryname", "year"]).reset_index(drop=True)
    lag = _lag_if_consecutive_year(out.rename(columns={"countryname": "ISO3"}), value_col)
    out["infl"] = ((pd.to_numeric(out[value_col], errors="coerce") - lag) / lag) * 100
    return out


def _mitchell_drop_rows(frame: pd.DataFrame, drops: list[int | tuple[int, int] | str]) -> pd.DataFrame:
    if frame.empty or not drops:
        return frame.copy()
    out = frame.copy()
    for spec in drops:
        if isinstance(spec, str):
            if spec.lower() == "l" and len(out) > 0:
                out = out.drop(index=[len(out) - 1], errors="ignore").reset_index(drop=True)
            continue
        if isinstance(spec, tuple):
            start, end = spec
            idx = list(range(start - 1, end))
            out = out.drop(index=[i for i in idx if 0 <= i < len(out)], errors="ignore").reset_index(drop=True)
        else:
            idx = spec - 1
            if 0 <= idx < len(out):
                out = out.drop(index=[idx], errors="ignore").reset_index(drop=True)
    return out.reset_index(drop=True)


def _mitchell_cpi_workbook(
    stem: str,
    sheet_plans: list[dict[str, object]],
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, stem)

    master: pd.DataFrame | None = None
    for plan in sheet_plans:
        frame = _mitchell_import_columns(path, int(plan["sheet"]))
        rename_map = dict(plan.get("rename", {}))
        if rename_map:
            frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})
        drop_rows = list(plan.get("drop_rows", []))
        if drop_rows:
            frame = _mitchell_drop_rows(frame, drop_rows)
        frame = _mitchell_drop_blank_year(frame)
        frame = _mitchell_destring(frame)
        long = _mitchell_reshape(frame, "CPI")
        if bool(plan.get("fill_100_2000")):
            mask = pd.to_numeric(long["year"], errors="coerce").eq(2000) & pd.to_numeric(long["CPI"], errors="coerce").isna()
            long.loc[mask, "CPI"] = 100
        long = _mitchell_group_infl(long, "CPI")
        master = _mitchell_append(master, long)

    assert master is not None
    out = _mitchell_adjust_breaks(master, path, "Base_years CPI", "CPI")
    out = _mitchell_group_infl(out, "CPI")
    out = out[["countryname", "year", "CPI", "infl"]].copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out.sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / f"{stem}.dta")
    return out


def _mitchell_break_rows(workbook_path: Path, sheet_name: str) -> pd.DataFrame:
    raw = _read_excel_compat(workbook_path, sheet_name=sheet_name, header=None, dtype=str)
    if raw.empty:
        return pd.DataFrame(columns=["countryname", "year", "notes"])
    first_row = raw.iloc[0].tolist()
    letters = _mitchell_excel_letters(raw.shape[1])
    columns: list[str] = []
    previous = ""
    for idx, value in enumerate(first_row):
        text = "" if pd.isna(value) else str(value)
        if text.startswith("Unnamed:"):
            text = ""
        if not text.strip():
            text = letters[idx]
        sanitized = _mitchell_sanitize_name(text)
        if sanitized and len(sanitized) > 2:
            previous = sanitized
            columns.append(previous)
        elif previous:
            columns.append(f"{previous}notes")
        else:
            columns.append(_mitchell_sanitize_name(text) or letters[idx])
    raw = raw.iloc[1:].reset_index(drop=True).copy()
    raw.columns = _mitchell_dedupe_names(columns)

    records: list[pd.DataFrame] = []
    base_names = [col for col in raw.columns if not col.endswith("notes")]
    for name in base_names:
        has_notes_col = f"{name}notes" in raw.columns
        frame = pd.DataFrame(
            {
                "countryname": name,
                "year": _mitchell_numeric_series(raw[name]) if name in raw.columns else pd.Series(dtype=float),
                "notes": _mitchell_numeric_series(raw[f"{name}notes"]) if has_notes_col else np.nan,
                "has_notes_col": has_notes_col,
            }
        )
        frame = frame.loc[frame["year"].notna()].copy()
        records.append(frame)
    if not records:
        return pd.DataFrame(columns=["countryname", "year", "notes"])
    out = pd.concat(records, ignore_index=True, sort=False)
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["notes"] = pd.to_numeric(out["notes"], errors="coerce")
    return out.sort_values(["countryname", "year"]).reset_index(drop=True)


def _mitchell_adjust_breaks(frame: pd.DataFrame, workbook_path: Path, sheet_name: str, value_col: str) -> pd.DataFrame:
    breaks = _mitchell_break_rows(workbook_path, sheet_name)
    if breaks.empty or frame.empty:
        return frame.copy()

    out = frame.copy().sort_values(["countryname", "year"]).reset_index(drop=True)
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")

    for country in breaks["countryname"].dropna().astype(str).drop_duplicates():
        country_breaks = breaks.loc[breaks["countryname"].eq(country), ["year", "notes", "has_notes_col"]].sort_values("year")
        if value_col == "CPI":
            country_breaks = country_breaks.loc[country_breaks["has_notes_col"].astype(bool)].copy()
            if country_breaks.empty:
                continue
        country_mask = out["countryname"].eq(country)
        country_df = out.loc[country_mask, ["year", value_col]].copy()
        if country_df.empty:
            continue
        year_to_idx = {int(y): idx for idx, y in out.loc[country_mask, "year"].items() if pd.notna(y)}
        for _, row in country_breaks.iterrows():
            year = pd.to_numeric(row["year"], errors="coerce")
            if pd.isna(year):
                continue
            year_int = int(year)
            idx = year_to_idx.get(year_int)
            if idx is None:
                continue
            current_value = pd.to_numeric(out.at[idx, value_col], errors="coerce")
            if pd.isna(current_value):
                continue
            note_value = pd.to_numeric(row["notes"], errors="coerce")
            ratio = np.nan
            if pd.notna(note_value):
                ratio = note_value / current_value
            else:
                series = out.loc[country_mask, ["year", value_col]].copy().sort_values("year").reset_index(drop=True)
                series["lag"] = series[value_col].shift(1)
                series["lag_year"] = series["year"].shift(1)
                series["growth"] = np.where(
                    series["year"].sub(series["lag_year"]).eq(1) & series["lag"].notna() & series["lag"].ne(0),
                    (series[value_col] - series["lag"]) / series["lag"],
                    np.nan,
                )
                window = series.loc[series["year"].between(year_int - 3, year_int + 3)].copy()
                if not window.empty:
                    first_match = window.loc[window["year"].eq(year_int + 1), value_col]
                    if not first_match.empty:
                        first_value = pd.to_numeric(first_match.iloc[0], errors="coerce")
                        growth_sample = window.loc[window["year"].ne(year_int + 1), "growth"].dropna()
                        if pd.notna(first_value) and not growth_sample.empty:
                            prev_value = first_value / (1 + growth_sample.median())
                            ratio = prev_value / current_value
            if pd.isna(ratio):
                continue
            adjust_mask = country_mask & out["year"].le(year_int)
            out.loc[adjust_mask, value_col] = pd.to_numeric(out.loc[adjust_mask, value_col], errors="coerce") * ratio

    return out.sort_values(["countryname", "year"]).reset_index(drop=True)


def _mitchell_africa_ngdp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Africa_NA")

    master: pd.DataFrame | None = None

    sheet2 = _mitchell_import_columns(path, 2)
    sheet2 = _mitchell_destring(_mitchell_drop_blank_year(sheet2))
    sheet2 = _mitchell_drop_columns(sheet2, ["C", "E"])
    master = _mitchell_append(master, _mitchell_reshape(sheet2, "nGDP"))

    sheet3 = _mitchell_import_columns(path, 3)
    sheet3 = _mitchell_destring(_mitchell_drop_blank_year(sheet3))
    sheet3 = _mitchell_drop_columns(sheet3, ["C", "D", "E"])
    master = _mitchell_append(master, _mitchell_reshape(sheet3, "nGDP"))

    for sheet_name in (4, 7):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        frame = _mitchell_keep_by_header(
            frame,
            header_row=2,
            predicate=lambda value: value == "CurrentPrices",
            normalizer=lambda value: value.replace(" ", ""),
        )
        value_headers = frame.iloc[3] if len(frame) > 3 else pd.Series(dtype=object)
        keep = ["year"] + [
            col
            for col in frame.columns
            if col != "year" and str(value_headers.get(col, "")) == "GDP"
        ]
        frame = frame[keep].copy()
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        master = _mitchell_append(master, _mitchell_reshape(frame, "nGDP"))

    sheet5 = _mitchell_import_columns(path, 5)
    sheet5 = _mitchell_destring(_mitchell_drop_blank_year(sheet5))
    sheet5 = _mitchell_drop_columns(sheet5, ["C", "D", "E"])
    master = _mitchell_append(master, _mitchell_reshape(sheet5, "nGDP"))

    sheet6 = _mitchell_import_columns(path, 6)
    sheet6 = _mitchell_destring(_mitchell_drop_blank_year(sheet6))
    sheet6 = _mitchell_drop_columns(sheet6, ["C", "D", "E", "F", "G"])
    master = _mitchell_append(master, _mitchell_reshape(sheet6, "nGDP"))

    wide = master.pivot(index="year", columns="countryname", values="nGDP").reset_index()
    for country, start, end in [
        ("SouthAfrica", 1911, 1979),
        ("Ghana", 1950, 1979),
        ("Nigeria", 1950, 1973),
        ("SierraLeone", 1950, 1998),
        ("Uganda", 1950, 1988),
        ("Zambia", 1950, 1992),
    ]:
        wide = _mitchell_convert_units(wide, country, start, end, "Th")
    for country in ["Egypt", "Ethiopia", "Lesotho", "Liberia", "Libya", "Malawi", "Mauritius", "Zimbabwe"]:
        if country in wide.columns:
            wide[country] = pd.to_numeric(wide[country], errors="coerce") / 1000
    for col in [c for c in wide.columns if c != "year"]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce") * 1000

    long = _mitchell_reshape(wide, "nGDP_LCU")
    long.loc[long["countryname"].eq("Tunisia"), "nGDP_LCU"] = pd.to_numeric(long.loc[long["countryname"].eq("Tunisia"), "nGDP_LCU"], errors="coerce") / 1000
    long.loc[long["countryname"].eq("Zambia"), "nGDP_LCU"] = pd.to_numeric(long.loc[long["countryname"].eq("Zambia"), "nGDP_LCU"], errors="coerce") / 1000
    long.loc[long["countryname"].eq("Ghana"), "nGDP_LCU"] = pd.to_numeric(long.loc[long["countryname"].eq("Ghana"), "nGDP_LCU"], errors="coerce") / 10000
    long.loc[long["countryname"].eq("Sudan"), "nGDP_LCU"] = pd.to_numeric(long.loc[long["countryname"].eq("Sudan"), "nGDP_LCU"], errors="coerce") / 1_000_000
    sudan_mask = long["countryname"].eq("Sudan") & pd.to_numeric(long["year"], errors="coerce").ge(2000)
    long.loc[sudan_mask, "nGDP_LCU"] = pd.to_numeric(long.loc[sudan_mask, "nGDP_LCU"], errors="coerce") * 1000
    zaire_mask = long["countryname"].eq("Zaire")
    long.loc[zaire_mask & pd.to_numeric(long["year"], errors="coerce").le(1993), "nGDP_LCU"] = pd.to_numeric(long.loc[zaire_mask & pd.to_numeric(long["year"], errors="coerce").le(1993), "nGDP_LCU"], errors="coerce") / 100000
    long.loc[zaire_mask, "nGDP_LCU"] = pd.to_numeric(long.loc[zaire_mask, "nGDP_LCU"], errors="coerce") / 1000
    long.loc[zaire_mask & pd.to_numeric(long["year"], errors="coerce").le(1988), "nGDP_LCU"] = pd.to_numeric(long.loc[zaire_mask & pd.to_numeric(long["year"], errors="coerce").le(1988), "nGDP_LCU"], errors="coerce") / 1000
    long.loc[zaire_mask & pd.to_numeric(long["year"], errors="coerce").le(1977), "nGDP_LCU"] = pd.to_numeric(long.loc[zaire_mask & pd.to_numeric(long["year"], errors="coerce").le(1977), "nGDP_LCU"], errors="coerce") / 1000
    out = long[["countryname", "year", "nGDP_LCU"]].copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out.sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Africa_nGDP.dta")
    return out


def _mitchell_europe_cpi(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Europe_CPI")

    def _sheet(sheet_name: int) -> pd.DataFrame:
        frame = _mitchell_import_columns(path, sheet_name)
        if sheet_name in (4, 5) and len(frame) > 0:
            frame = frame.iloc[:-1].copy()
        if sheet_name == 5:
            frame.loc[frame["Greece"].astype("string").eq("1953 = 100"), "Greece"] = ""
            frame.loc[frame["Spain"].astype("string").eq("1953 = 100"), "Spain"] = ""
            frame.loc[frame["year"].astype("string").eq("1923"), "Germany"] = ""
        frame = _mitchell_drop_blank_year(frame)
        frame = _mitchell_destring(frame)
        rename_map = {"UK": "UnitedKingdom", "Nethl": "Netherlands", "RussiaUSSR": "USSR", "SIreland": "Ireland"}
        frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})
        if sheet_name == 5 and "WestGermany" in frame.columns:
            germany = pd.to_numeric(frame.get("Germany"), errors="coerce")
            west = pd.to_numeric(frame.get("WestGermany"), errors="coerce")
            frame["Germany"] = germany.combine_first(west)
            frame = frame.drop(columns=["WestGermany"])
        if sheet_name == 6 and {"WestGermany", "EastGermany"}.issubset(frame.columns):
            frame = frame.rename(columns={"WestGermany": "Germany"})
            germany = pd.to_numeric(frame["Germany"], errors="coerce")
            east = pd.to_numeric(frame["EastGermany"], errors="coerce")
            frame["Germany"] = germany.combine_first(east)
            frame.loc[pd.to_numeric(frame["year"], errors="coerce").ge(1990), "EastGermany"] = np.nan
        long = _mitchell_reshape(frame, "CPI")
        return _mitchell_group_infl(long, "CPI")

    master: pd.DataFrame | None = None
    for sheet_name in (2, 3, 4, 5, 6):
        master = _mitchell_append(master, _sheet(sheet_name))
    out = _mitchell_adjust_breaks(master, path, "Base_years CPI", "CPI")
    out = _mitchell_group_infl(out, "CPI")
    out = out[["countryname", "year", "CPI", "infl"]].copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out.sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Europe_CPI.dta")
    return out


def _mitchell_africa_cpi(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    return _mitchell_cpi_workbook(
        "Africa_CPI",
        [
            {"sheet": 2},
            {"sheet": 3},
            {"sheet": 4, "drop_rows": [1]},
            {"sheet": 5},
            {"sheet": 6, "drop_rows": [9, 1]},
            {"sheet": 7, "fill_100_2000": True},
        ],
        data_raw_dir=data_raw_dir,
        data_temp_dir=data_temp_dir,
    )


def _mitchell_americas_cpi(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    return _mitchell_cpi_workbook(
        "Americas_CPI",
        [
            {"sheet": 2, "drop_rows": [1]},
            {"sheet": 3, "drop_rows": [1]},
            {"sheet": 4, "drop_rows": [1, 5, 39]},
            {"sheet": 5, "drop_rows": [1, 15, 20, 27], "fill_100_2000": True},
        ],
        data_raw_dir=data_raw_dir,
        data_temp_dir=data_temp_dir,
    )


def _mitchell_asia_cpi(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    return _mitchell_cpi_workbook(
        "Asia_CPI",
        [
            {"sheet": 2},
            {"sheet": 3, "drop_rows": [30, 1]},
            {"sheet": 4, "drop_rows": [37, 34, (27, 28), 13, 1]},
        ],
        data_raw_dir=data_raw_dir,
        data_temp_dir=data_temp_dir,
    )


def _mitchell_oceania_cpi(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    return _mitchell_cpi_workbook(
        "Oceania_CPI",
        [
            {"sheet": 2},
            {"sheet": 3, "drop_rows": ["l"]},
            {"sheet": 4},
            {"sheet": 5, "drop_rows": [25, 33]},
        ],
        data_raw_dir=data_raw_dir,
        data_temp_dir=data_temp_dir,
    )


def _mitchell_latam_cpi(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    return _mitchell_cpi_workbook(
        "Latam_CPI",
        [
            {"sheet": 2, "drop_rows": [1]},
            {"sheet": 3, "drop_rows": [1, 20, 29]},
            {"sheet": 4, "drop_rows": [1, 5, 21, 23, 26, 31, 36, "l"]},
            {"sheet": 5, "rename": {"B": "Argentina"}, "drop_rows": [11, 17, 18, 20, (27, 28)]},
        ],
        data_raw_dir=data_raw_dir,
        data_temp_dir=data_temp_dir,
    )


def _mitchell_americas_ngdp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Americas_NA")

    master: pd.DataFrame | None = None

    sheet2 = _mitchell_import_columns_first(path, 2)
    sheet2 = _mitchell_keep_columns(sheet2, ["B"])
    sheet2 = _mitchell_rename_from_row(sheet2, 0)
    sheet2 = _mitchell_destring(_mitchell_drop_blank_year(sheet2))
    master = _mitchell_append(master, _mitchell_reshape(sheet2, "nGDP"))

    for sheet_name in (3, 4, 5):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        frame = _mitchell_keep_by_header(
            frame,
            header_row=2,
            predicate=lambda value: value.strip().lower() == "current prices",
            normalizer=lambda value: value.strip().lower(),
        )
        value_headers = frame.iloc[3] if len(frame) > 3 else pd.Series(dtype=object)
        keep = ["year"] + [
            col
            for col in frame.columns
            if col != "year" and str(value_headers.get(col, "")).strip().lower() in {"gdp", "gnp", "nnp", "ndp"}
        ]
        frame = frame[keep].copy()
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 3:
            frame = _mitchell_convert_units(frame, "USA", 1865, 1909, "B")
        elif sheet_name == 4:
            frame = _mitchell_convert_units(frame, "USA", 1910, 1949, "B")
            frame = _mitchell_convert_currency(frame, "Jamaica", 1949, 2)
        else:
            for country in ["Canada", "USA"]:
                frame = _mitchell_convert_units(frame, country, 1950, 2010, "B")
            frame = _mitchell_convert_units(frame, "Mexico", 1950, 1993, "B")
            frame = _mitchell_convert_units(frame, "CostaRica", 1980, 2010, "B")
            if "Nicaragua" in frame.columns:
                years = pd.to_numeric(frame["year"], errors="coerce")
                frame.loc[years.le(1978), "Nicaragua"] = pd.to_numeric(frame.loc[years.le(1978), "Nicaragua"], errors="coerce") / 1000
                frame.loc[years.le(1988), "Nicaragua"] = pd.to_numeric(frame.loc[years.le(1988), "Nicaragua"], errors="coerce") / 1_000_000
                frame.loc[years.eq(1989), "Nicaragua"] = pd.to_numeric(frame.loc[years.eq(1989), "Nicaragua"], errors="coerce") / 1000
        master = _mitchell_append(master, _mitchell_reshape(frame, "nGDP"))

    out = master.rename(columns={"nGDP": "nGDP_LCU"}).copy()
    out = out.loc[pd.to_numeric(out["year"], errors="coerce").lt(1993)].copy()
    mex_mask = out["countryname"].eq("Mexico") & pd.to_numeric(out["year"], errors="coerce").le(1984)
    out.loc[mex_mask, "nGDP_LCU"] = pd.to_numeric(out.loc[mex_mask, "nGDP_LCU"], errors="coerce") / 1000
    slv_mask = out["countryname"].eq("ElSalvador")
    out.loc[slv_mask, "nGDP_LCU"] = pd.to_numeric(out.loc[slv_mask, "nGDP_LCU"], errors="coerce") / 8.75
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "nGDP_LCU"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Americas_nGDP.dta")
    return out


def _mitchell_latam_ngdp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Latam_NA")

    master: pd.DataFrame | None = None

    sheet2 = _mitchell_import_columns_first(path, 2)
    sheet2 = _mitchell_keep_columns(sheet2, ["B"])
    sheet2 = _mitchell_rename_from_row(sheet2, 0)
    sheet2 = _mitchell_destring(_mitchell_drop_blank_year(sheet2))
    master = _mitchell_append(master, _mitchell_reshape(sheet2, "nGDP"))

    sheet3 = _mitchell_import_columns_first(path, 3)
    sheet3 = _mitchell_keep_columns(sheet3, ["E"])
    sheet3 = _mitchell_rename_from_row(sheet3, 0)
    sheet3 = _mitchell_destring(_mitchell_drop_blank_year(sheet3))
    master = _mitchell_append(master, _mitchell_reshape(sheet3, "nGDP"))

    for sheet_name in (4, 5):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        if len(frame) > 2:
            for col in [c for c in frame.columns if c != "year"]:
                frame.at[2, col] = str(frame.at[2, col]).strip().lower() if pd.notna(frame.at[2, col]) else ""
        if len(frame) > 3:
            for col in [c for c in frame.columns if c != "year"]:
                frame.at[3, col] = str(frame.at[3, col]).strip().lower() if pd.notna(frame.at[3, col]) else ""
        keep = ["year"] + [
            col
            for col in frame.columns
            if col != "year"
            and len(frame) > 3
            and str(frame.at[2, col]) == "current prices"
            and str(frame.at[3, col]) == "gdp"
        ]
        frame = frame[keep].copy()
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 4:
            for country in ["Argentina", "Chile", "Ecuador", "Paraguay", "Venezuela"]:
                frame = _mitchell_convert_units(frame, country, 1935, 1969, "B")
        else:
            for country in ["Bolivia", "Brazil", "Colombia", "Ecuador", "Paraguay", "Uruguay", "Venezuela"]:
                frame = _mitchell_convert_units(frame, country, 1970, 1993, "B")
            frame = _mitchell_convert_units(frame, "Argentina", 1970, 1986, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "nGDP"))

    out = master.rename(columns={"nGDP": "nGDP_LCU"}).copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out = out.loc[years.lt(1993)].copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    country = out["countryname"]
    out.loc[country.eq("Brazil"), "nGDP_LCU"] = values[country.eq("Brazil")] / 2750
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Brazil"), "nGDP_LCU"] = values[country.eq("Brazil")] / 1000
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Brazil") & years.le(1989), "nGDP_LCU"] = values[country.eq("Brazil") & years.le(1989)] / 1000
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Brazil") & years.le(1979), "nGDP_LCU"] = values[country.eq("Brazil") & years.le(1979)] / 1000
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Brazil") & years.between(1960, 1969), "nGDP_LCU"] = values[country.eq("Brazil") & years.between(1960, 1969)] * 1000
    out.loc[country.eq("Argentina") & years.ge(1985), "nGDP_LCU"] = np.nan
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Venezuela"), "nGDP_LCU"] = values[country.eq("Venezuela")] * (10 ** -14)
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Mexico") & years.le(1984), "nGDP_LCU"] = values[country.eq("Mexico") & years.le(1984)] * (10 ** -6)
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Ecuador"), "nGDP_LCU"] = values[country.eq("Ecuador")] * (10 ** -3)
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Colombia") & years.le(1969), "nGDP_LCU"] = values[country.eq("Colombia") & years.le(1969)] * (10 ** 3)
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Uruguay"), "nGDP_LCU"] = values[country.eq("Uruguay")] * (10 ** -3)
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Uruguay") & years.le(1979), "nGDP_LCU"] = values[country.eq("Uruguay") & years.le(1979)] / 1000
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Uruguay") & years.le(1969), "nGDP_LCU"] = values[country.eq("Uruguay") & years.le(1969)] * 1000
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Uruguay") & years.le(1959), "nGDP_LCU"] = values[country.eq("Uruguay") & years.le(1959)] / 1000
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Peru") & years.le(1988), "nGDP_LCU"] = values[country.eq("Peru") & years.le(1988)] * (10 ** -3)
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Peru") & years.le(1979), "nGDP_LCU"] = values[country.eq("Peru") & years.le(1979)] * (10 ** -3)
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Peru") & years.le(1949), "nGDP_LCU"] = values[country.eq("Peru") & years.le(1949)] * (10 ** -3)
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Bolivia") & years.le(1992), "nGDP_LCU"] = values[country.eq("Bolivia") & years.le(1992)] / 1000
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Bolivia") & years.eq(1984), "nGDP_LCU"] = values[country.eq("Bolivia") & years.eq(1984)] / 1000
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Bolivia") & years.le(1983), "nGDP_LCU"] = values[country.eq("Bolivia") & years.le(1983)] / 100
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Bolivia") & years.le(1979), "nGDP_LCU"] = values[country.eq("Bolivia") & years.le(1979)] / 10
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Argentina") & years.le(1985), "nGDP_LCU"] = values[country.eq("Argentina") & years.le(1985)] / 10_000_000
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Argentina") & years.le(1974), "nGDP_LCU"] = values[country.eq("Argentina") & years.le(1974)] / 100_000
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Chile") & years.le(1969), "nGDP_LCU"] = values[country.eq("Chile") & years.le(1969)] / 1000
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Chile") & years.le(1964), "nGDP_LCU"] = values[country.eq("Chile") & years.le(1964)] / 1000
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Chile") & years.ge(1975), "nGDP_LCU"] = values[country.eq("Chile") & years.ge(1975)] * 1000
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Argentina") & years.le(1979), "nGDP_LCU"] = values[country.eq("Argentina") & years.le(1979)] / 10
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Argentina") & years.between(1965, 1974), "nGDP_LCU"] = values[country.eq("Argentina") & years.between(1965, 1974)] * 100
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "nGDP_LCU"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Latam_nGDP.dta")
    return out


def _mitchell_asia_ngdp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Asia_NA")

    master: pd.DataFrame | None = None
    for sheet_name in (2, 3):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        if len(frame) > 2:
            for col in [c for c in frame.columns if c != "year"]:
                frame.at[2, col] = str(frame.at[2, col]).strip().lower() if pd.notna(frame.at[2, col]) else ""
        if len(frame) > 3:
            for col in [c for c in frame.columns if c != "year"]:
                frame.at[3, col] = str(frame.at[3, col]).strip().lower() if pd.notna(frame.at[3, col]) else ""
        keep = ["year"] + [
            col
            for col in frame.columns
            if col != "year"
            and len(frame) > 3
            and str(frame.at[2, col]) == "current prices"
            and str(frame.at[3, col]) in {"gdp", "gnp", "nnp", "ndp"}
        ]
        frame = frame[keep].copy()
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 3:
            for country in ["Bangladesh", "China", "HongKong", "India", "Iran", "Nepal", "Pakistan", "Philippines", "SaudiArabia", "Taiwan", "Thailand", "Japan", "SouthKorea"]:
                frame = _mitchell_convert_units(frame, country, 1945, 2010, "B")
            frame = _mitchell_convert_units(frame, "Indonesia", 1965, 1978, "B")
            frame = _mitchell_convert_units(frame, "Indonesia", 1979, 2010, "Tri")
            frame = _mitchell_convert_units(frame, "Israel", 1950, 1980, "Th")
            frame = _mitchell_convert_units(frame, "Japan", 1960, 2010, "B")
            frame = _mitchell_convert_units(frame, "SouthKorea", 1980, 2010, "B")
            frame = _mitchell_convert_units(frame, "Lebanon", 1988, 2010, "B")
            frame = _mitchell_convert_units(frame, "Turkey", 1950, 1998, "B")
            for country in ["Malaysia", "Myanmar", "Singapore", "SriLanka", "Syria"]:
                frame = _mitchell_convert_units(frame, country, 1975, 2010, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "nGDP"))

    out = master.rename(columns={"nGDP": "nGDP_LCU"}).copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    country = out["countryname"]
    out.loc[country.eq("Turkey") & years.le(1998), "nGDP_LCU"] = values[country.eq("Turkey") & years.le(1998)] * (10 ** -6)
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Indonesia") & years.le(1939), "nGDP_LCU"] = values[country.eq("Indonesia") & years.le(1939)] * (10 ** -3)
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Taiwan") & years.le(1939), "nGDP_LCU"] = values[country.eq("Taiwan") & years.le(1939)] * (10 ** -4)
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Taiwan") & years.le(1939), "nGDP_LCU"] = values[country.eq("Taiwan") & years.le(1939)] / 4
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "nGDP_LCU"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Asia_nGDP.dta")
    return out


def _mitchell_oceania_ngdp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Oceania_NA")

    master: pd.DataFrame | None = None

    sheet2 = _mitchell_import_columns(path, 2)
    sheet2 = _mitchell_drop_blank_year(sheet2)
    sheet2 = _mitchell_destring(sheet2)
    sheet2 = _mitchell_convert_units(sheet2, "Australia", 1789, 1824, "Th")
    master = _mitchell_append(master, _mitchell_reshape(sheet2, "nGDP"))

    sheet3 = _mitchell_import_columns(path, 3)
    sheet3 = _mitchell_destring(_mitchell_drop_blank_year(sheet3))
    master = _mitchell_append(master, _mitchell_reshape(sheet3, "nGDP"))

    sheet4 = _mitchell_import_columns(path, 4)
    sheet4 = _mitchell_destring(_mitchell_drop_blank_year(sheet4))
    sheet4 = _mitchell_keep_columns(sheet4, ["Australia"])
    master = _mitchell_append(master, _mitchell_reshape(sheet4, "nGDP"))

    for sheet_name, metrics in ((5, {"GDP", "GNP", "API/NNP"}), (6, {"GDP", "GNP", "API/NNP", "NNP/GDP"})):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        keep = ["year"] + [
            col
            for col in frame.columns
            if col != "year"
            and len(frame) > 3
            and str(frame.at[2, col]) == "Current Prices"
            and str(frame.at[3, col]) in metrics
        ]
        frame = frame[keep].copy()
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 6:
            frame = _mitchell_convert_units(frame, "Australia", 1965, 2010, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "nGDP"))

    wide = master.pivot(index="year", columns="countryname", values="nGDP").reset_index()
    wide.columns.name = None
    wide = _mitchell_convert_currency(wide, "Australia", 1900, 2)
    wide = _mitchell_convert_currency(wide, "NewZealand", 1959, 2)
    out = _mitchell_reshape(wide, "nGDP_LCU")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "nGDP_LCU"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Oceania_nGDP.dta")
    return out


def _mitchell_europe_ngdp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Europe_NA")

    master: pd.DataFrame | None = None

    for sheet_name, uk_code, metrics in (
        (2, "AB", {"GDP", "GNP", "NNP"}),
        (3, "BD", {"GDP", "GNP", "NNP"}),
        (4, "CD", {"GDP", "GNP", "NNP", "NMP"}),
        (5, "CB", {"GDP", "GNP", "NMP", "NNP"}),
    ):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        if len(frame) > 2:
            for col in [c for c in frame.columns if c != "year"]:
                frame.at[2, col] = str(frame.at[2, col]).strip().lower() if pd.notna(frame.at[2, col]) else ""
        keep = ["year"]
        if sheet_name == 4:
            keep.extend([col for col in ["F", "J", "X", "AT", "AX", "BN", "BR", "BZ"] if col in frame.columns])
        for col in [c for c in frame.columns if c != "year"]:
            row3 = str(frame.at[2, col]) if len(frame) > 2 else ""
            row4 = str(frame.at[3, col]) if len(frame) > 3 else ""
            if row3 == "current prices" and row4 in metrics:
                keep.append(col)
        frame = frame.loc[:, list(dict.fromkeys(keep))].copy()
        frame = _mitchell_rename_from_row(frame, 0)
        rename_map = {uk_code: "UnitedKingdom"}
        if sheet_name == 4:
            rename_map.update(
                {
                    "AB": "EastGermany",
                    "AD": "Germany",
                    "WestGermany": "Germany",
                    "BN": "Russia",
                    "CD": "UnitedKingdom",
                }
            )
        if sheet_name == 5:
            rename_map.update(
                {
                    "AB": "EastGermany",
                    "AD": "Germany",
                    "WestGermany": "Germany",
                    "AP": "Ireland",
                    "SouthernIreland": "Ireland",
                    "BN": "Russia",
                    "CB": "UnitedKingdom",
                }
            )
            if "Yugoslavia" in frame.columns:
                series = frame["Yugoslavia"].astype("string")
                frame.loc[series.eq("1,147, 787"), "Yugoslavia"] = "1147787"
        frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 2:
            frame = _mitchell_convert_units(frame, "Italy", 1815, 1899, "B")
        elif sheet_name == 3:
            for country in ["Austria", "Belgium", "Bulgaria", "Greece", "Italy", "Russia", "Spain", "Yugoslavia"]:
                frame = _mitchell_convert_units(frame, country, 1900, 1944, "B")
        elif sheet_name == 4:
            for country in ["Hungary", "Ireland"]:
                frame = _mitchell_convert_units(frame, country, 1945, 1979, "Th")
            for country in [c for c in frame.columns if c != "year"]:
                frame = _mitchell_convert_units(frame, country, 1945, 1979, "B")
            frame = _mitchell_convert_units(frame, "Italy", 1945, 1979, "B")
            frame = _mitchell_convert_units(frame, "Yugoslavia", 1945, 1979, "B")
        else:
            for country in ["Hungary", "Ireland"]:
                frame = _mitchell_convert_units(frame, country, 1980, 2010, "Th")
            for country in [c for c in frame.columns if c != "year"]:
                frame = _mitchell_convert_units(frame, country, 1980, 2010, "B")
            frame = _mitchell_convert_units(frame, "Italy", 1980, 1998, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "nGDP"))

    out = master.rename(columns={"nGDP": "nGDP_LCU"}).copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    out = out.loc[years.lt(1998)].copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Czechoslovakia") & years.ge(1994), "countryname"] = "CzechRepublic"
    out.loc[country.eq("France") & years.le(1939), "nGDP_LCU"] = values[country.eq("France") & years.le(1939)] / 100
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Bulgaria"), "nGDP_LCU"] = values[country.eq("Bulgaria")] / 1000
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Bulgaria") & years.le(1991), "nGDP_LCU"] = values[country.eq("Bulgaria") & years.le(1991)] / 1000
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Hungary") & years.ge(1950), "nGDP_LCU"] = values[country.eq("Hungary") & years.ge(1950)] * 1000
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Italy") & years.eq(1960), "nGDP_LCU"] = values[country.eq("Italy") & years.eq(1960)] / 10
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Germany") & years.le(1913), "nGDP_LCU"] = values[country.eq("Germany") & years.le(1913)] / (10 ** 12)
    values = pd.to_numeric(out["nGDP_LCU"], errors="coerce")
    out.loc[country.eq("Poland") & years.le(1990), "nGDP_LCU"] = values[country.eq("Poland") & years.le(1990)] / 10000
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "nGDP_LCU"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Europe_nGDP.dta")
    return out


def _mitchell_africa_rgdp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Africa_NA")

    master: pd.DataFrame | None = None

    sheet2 = _mitchell_import_columns(path, 2)
    sheet2 = _mitchell_keep_columns(sheet2, ["C"]).rename(columns={"C": "SouthAfrica"})
    sheet2 = _mitchell_destring(_mitchell_drop_blank_year(sheet2))
    master = _mitchell_append(master, _mitchell_reshape(sheet2, "rGDP"))

    sheet3 = _mitchell_import_columns(path, 3)
    sheet3 = _mitchell_keep_columns(sheet3, ["E"]).rename(columns={"E": "Algeria"})
    sheet3 = _mitchell_destring(_mitchell_drop_blank_year(sheet3))
    master = _mitchell_append(master, _mitchell_reshape(sheet3, "rGDP"))

    for sheet_name, extra_cols in ((4, ["AN"]), (7, ["BX", "T", "AB", "AJ", "AN", "AT"])):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        if len(frame) > 0:
            for col in [c for c in frame.columns if c != "year"]:
                frame.at[0, col] = str(frame.at[0, col]).replace(" ", "") if pd.notna(frame.at[0, col]) else ""
        keep = ["year"] + [col for col in extra_cols if col in frame.columns]
        for col in [c for c in frame.columns if c != "year"]:
            row3 = str(frame.at[2, col]) if len(frame) > 2 else ""
            row4 = str(frame.at[3, col]) if len(frame) > 3 else ""
            if row3 != "Current Prices" and row4 == "GDP":
                keep.append(col)
        frame = frame.loc[:, list(dict.fromkeys(keep))].copy()
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        master = _mitchell_append(master, _mitchell_reshape(frame, "rGDP"))

    sheet5 = _mitchell_import_columns(path, 5)
    sheet5 = _mitchell_keep_columns(sheet5, ["E"]).rename(columns={"E": "Libya"})
    sheet5 = _mitchell_destring(_mitchell_drop_blank_year(sheet5))
    master = _mitchell_append(master, _mitchell_reshape(sheet5, "rGDP"))

    sheet6 = _mitchell_import_columns(path, 6)
    sheet6 = _mitchell_keep_columns(sheet6, ["E"]).rename(columns={"E": "Libya"})
    sheet6 = _mitchell_destring(_mitchell_drop_blank_year(sheet6))
    master = _mitchell_append(master, _mitchell_reshape(sheet6, "rGDP"))

    assert master is not None
    wide = master.pivot(index="year", columns="countryname", values="rGDP").reset_index()
    wide.columns.name = None
    for country, start, end in [
        ("SouthAfrica", 1911, 1979),
        ("Nigeria", 1950, 1973),
        ("SierraLeone", 1950, 1998),
        ("Zambia", 1950, 1992),
    ]:
        wide = _mitchell_convert_units(wide, country, start, end, "Th")
    for country in ["Egypt", "Ethiopia", "Liberia", "Libya", "Malawi", "Mauritius", "Zimbabwe"]:
        if country in wide.columns:
            wide[country] = pd.to_numeric(wide[country], errors="coerce") / 1000
    for col in [c for c in wide.columns if c != "year"]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce") * 1000

    out = _mitchell_reshape(wide, "rGDP_LCU")
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["rGDP_LCU"], errors="coerce")
    out.loc[country.eq("Tunisia"), "rGDP_LCU"] = values[country.eq("Tunisia")] / 1000
    values = pd.to_numeric(out["rGDP_LCU"], errors="coerce")
    out.loc[country.eq("Zambia") & years.ge(1994), "rGDP_LCU"] = values[country.eq("Zambia") & years.ge(1994)] / 1000
    values = pd.to_numeric(out["rGDP_LCU"], errors="coerce")
    out.loc[country.eq("Ghana"), "rGDP_LCU"] = values[country.eq("Ghana")] / 1000
    values = pd.to_numeric(out["rGDP_LCU"], errors="coerce")
    out.loc[country.eq("Sudan"), "rGDP_LCU"] = values[country.eq("Sudan")] / 1_000_000
    values = pd.to_numeric(out["rGDP_LCU"], errors="coerce")
    out.loc[country.eq("Sudan") & years.ge(2000), "rGDP_LCU"] = values[country.eq("Sudan") & years.ge(2000)] * 1000
    out.loc[country.eq("Zaire"), "rGDP_LCU"] = np.nan
    out = _mitchell_adjust_breaks(out[["countryname", "year", "rGDP_LCU"]], path, "Base_years rGDP", "rGDP_LCU")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "rGDP_LCU"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Africa_rGDP.dta")
    return out


def _mitchell_americas_rgdp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Americas_NA")

    master: pd.DataFrame | None = None

    sheet2 = _mitchell_import_columns_first(path, 2)
    sheet2 = _mitchell_keep_columns(sheet2, ["C"]).rename(columns={"C": "USA"})
    sheet2 = _mitchell_destring(_mitchell_drop_blank_year(sheet2))
    master = _mitchell_append(master, _mitchell_reshape(sheet2, "rGDP"))

    for sheet_name in (3, 4, 5):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        if len(frame) > 2:
            for col in [c for c in frame.columns if c != "year"]:
                frame.at[2, col] = str(frame.at[2, col]).strip().lower() if pd.notna(frame.at[2, col]) else ""
        if len(frame) > 3:
            for col in [c for c in frame.columns if c != "year"]:
                frame.at[3, col] = str(frame.at[3, col]).strip().lower() if pd.notna(frame.at[3, col]) else ""
        keep = ["year"] + [
            col
            for col in frame.columns
            if col != "year"
            and len(frame) > 3
            and str(frame.at[2, col]) != "current prices"
            and str(frame.at[3, col]) in {"gdp", "gnp", "nnp", "ndp"}
        ]
        frame = frame.loc[:, keep].copy()
        frame = _mitchell_rename_from_row(frame, 0)
        if sheet_name == 4:
            if "Mexico" in frame.columns:
                frame = frame.drop(columns=["Mexico"])
            if "Z" in frame.columns:
                frame = frame.rename(columns={"Z": "Mexico"})
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 5:
            for country in ["Canada", "USA"]:
                frame = _mitchell_convert_units(frame, country, 1950, 2010, "B")
            frame = _mitchell_convert_units(frame, "Mexico", 1950, 1993, "B")
            frame = _mitchell_convert_units(frame, "CostaRica", 1980, 2010, "B")
            if "Nicaragua" in frame.columns:
                years = pd.to_numeric(frame["year"], errors="coerce")
                frame.loc[years.ge(1979), "Nicaragua"] = pd.to_numeric(frame.loc[years.ge(1979), "Nicaragua"], errors="coerce") * 1000
        master = _mitchell_append(master, _mitchell_reshape(frame, "rGDP"))

    assert master is not None
    out = master.rename(columns={"rGDP": "rGDP_LCU"}).copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    out = out.loc[years.lt(1993)].copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    values = pd.to_numeric(out["rGDP_LCU"], errors="coerce")
    country = out["countryname"]
    out.loc[country.eq("USA") & years.between(1865, 1949), "rGDP_LCU"] = values[country.eq("USA") & years.between(1865, 1949)] * 1000
    values = pd.to_numeric(out["rGDP_LCU"], errors="coerce")
    out.loc[country.eq("Mexico") & years.ge(1985), "rGDP_LCU"] = values[country.eq("Mexico") & years.ge(1985)] * 1000
    out = _mitchell_adjust_breaks(out[["countryname", "year", "rGDP_LCU"]], path, "Base_years rGDP", "rGDP_LCU")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "rGDP_LCU"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Americas_rGDP.dta")
    return out


def _mitchell_asia_rgdp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Asia_NA")

    master: pd.DataFrame | None = None
    for sheet_name, metrics in ((2, {"gdp", "gnp", "nnp", "ndp"}), (3, {"gdp", "gnp", "nnp", "ndp", "nmp"})):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        if len(frame) > 2:
            for col in [c for c in frame.columns if c != "year"]:
                frame.at[2, col] = str(frame.at[2, col]).strip().lower() if pd.notna(frame.at[2, col]) else ""
        if len(frame) > 3:
            for col in [c for c in frame.columns if c != "year"]:
                frame.at[3, col] = str(frame.at[3, col]).strip().lower() if pd.notna(frame.at[3, col]) else ""
        keep = ["year"] + [
            col
            for col in frame.columns
            if col != "year"
            and len(frame) > 3
            and str(frame.at[2, col]) != "current prices"
            and str(frame.at[3, col]) in metrics
        ]
        frame = frame.loc[:, keep].copy()
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 2:
            frame = _mitchell_convert_units(frame, "India", 1885, 1944, "B")
        else:
            for country in [
                "Afghanistan",
                "Bangladesh",
                "China",
                "HongKong",
                "India",
                "Iran",
                "Nepal",
                "Pakistan",
                "Philippines",
                "SaudiArabia",
                "Taiwan",
                "Thailand",
                "Japan",
                "SouthKorea",
                "Turkey",
            ]:
                frame = _mitchell_convert_units(frame, country, 1945, 2010, "B")
            frame = _mitchell_convert_units(frame, "Indonesia", 1965, 1978, "B")
            frame = _mitchell_convert_units(frame, "Indonesia", 1979, 2010, "Tri")
            frame = _mitchell_convert_units(frame, "Israel", 1950, 1980, "Th")
            frame = _mitchell_convert_units(frame, "Japan", 1960, 2010, "B")
            frame = _mitchell_convert_units(frame, "SouthKorea", 1980, 2010, "B")
            for country in ["Malaysia", "Myanmar", "Singapore", "SriLanka", "Syria"]:
                frame = _mitchell_convert_units(frame, country, 1975, 2010, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "rGDP"))

    assert master is not None
    out = master.rename(columns={"rGDP": "rGDP_LCU"}).copy()
    out = _mitchell_adjust_breaks(out[["countryname", "year", "rGDP_LCU"]], path, "Base_years rGDP", "rGDP_LCU")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "rGDP_LCU"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Asia_rGDP.dta")
    return out


def _mitchell_latam_rgdp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Latam_NA")

    master: pd.DataFrame | None = None

    sheet2 = _mitchell_import_columns_first(path, 2)
    sheet2 = _mitchell_keep_columns(sheet2, ["C"]).rename(columns={"C": "Brazil"})
    sheet2 = _mitchell_rename_from_row(sheet2, 0)
    sheet2 = _mitchell_destring(_mitchell_drop_blank_year(sheet2))
    master = _mitchell_append(master, _mitchell_reshape(sheet2, "rGDP"))

    for sheet_name in (3, 4, 5):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        if len(frame) > 2:
            for col in [c for c in frame.columns if c != "year"]:
                frame.at[2, col] = str(frame.at[2, col]).strip().lower() if pd.notna(frame.at[2, col]) else ""
        if len(frame) > 3:
            for col in [c for c in frame.columns if c != "year"]:
                frame.at[3, col] = str(frame.at[3, col]).strip().lower() if pd.notna(frame.at[3, col]) else ""
        keep = ["year"] + [
            col
            for col in frame.columns
            if col != "year"
            and len(frame) > 3
            and str(frame.at[2, col]) != "current prices"
            and str(frame.at[3, col]) == "gdp"
        ]
        frame = frame.loc[:, keep].copy()
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        years = pd.to_numeric(frame["year"], errors="coerce")
        if sheet_name == 3:
            for country in ["Argentina", "Chile"]:
                frame = _mitchell_convert_units(frame, country, 1900, 1934, "B")
        elif sheet_name == 4:
            for country in ["Argentina", "Chile", "Paraguay", "Venezuela", "Ecuador"]:
                frame = _mitchell_convert_units(frame, country, 1935, 1969, "B")
            if "Argentina" in frame.columns:
                frame.loc[years.ge(1965), "Argentina"] = pd.to_numeric(frame.loc[years.ge(1965), "Argentina"], errors="coerce") * 100
            if "Uruguay" in frame.columns:
                frame.loc[years.ge(1960), "Uruguay"] = pd.to_numeric(frame.loc[years.ge(1960), "Uruguay"], errors="coerce") * 1000
        else:
            for country in ["Bolivia", "Brazil", "Colombia", "Ecuador", "Paraguay", "Uruguay", "Venezuela"]:
                frame = _mitchell_convert_units(frame, country, 1970, 1993, "B")
            if "Argentina" in frame.columns:
                frame["Argentina"] = pd.to_numeric(frame["Argentina"], errors="coerce") * 100000
        master = _mitchell_append(master, _mitchell_reshape(frame, "rGDP"))

    assert master is not None
    out = master.rename(columns={"rGDP": "rGDP_LCU"}).copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    out = out.loc[years.lt(1993)].copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["rGDP_LCU"], errors="coerce")
    out.loc[country.eq("Chile") & years.ge(1965), "rGDP_LCU"] = values[country.eq("Chile") & years.ge(1965)] * 1000
    values = pd.to_numeric(out["rGDP_LCU"], errors="coerce")
    out.loc[country.eq("Chile") & years.ge(1970), "rGDP_LCU"] = values[country.eq("Chile") & years.ge(1970)] * 1000
    values = pd.to_numeric(out["rGDP_LCU"], errors="coerce")
    out.loc[country.eq("Colombia") & years.le(1969), "rGDP_LCU"] = values[country.eq("Colombia") & years.le(1969)] * 1000
    values = pd.to_numeric(out["rGDP_LCU"], errors="coerce")
    out.loc[country.eq("Brazil") & years.between(1970, 1975), "rGDP_LCU"] = values[country.eq("Brazil") & years.between(1970, 1975)] / 1000
    values = pd.to_numeric(out["rGDP_LCU"], errors="coerce")
    out.loc[country.eq("Peru") & years.ge(1980), "rGDP_LCU"] = values[country.eq("Peru") & years.ge(1980)] * 1000
    values = pd.to_numeric(out["rGDP_LCU"], errors="coerce")
    out.loc[country.eq("Argentina") & years.le(1935), "rGDP_LCU"] = values[country.eq("Argentina") & years.le(1935)] * 10
    out = _mitchell_adjust_breaks(out[["countryname", "year", "rGDP_LCU"]], path, "Base_years rGDP", "rGDP_LCU")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "rGDP_LCU"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Latam_rGDP.dta")
    return out


def _mitchell_oceania_rgdp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Oceania_NA")

    master: pd.DataFrame | None = None

    sheet4 = _mitchell_import_columns_first(path, 4)
    sheet4 = _mitchell_keep_columns(sheet4, ["D"]).rename(columns={"D": "Australia"})
    sheet4 = _mitchell_destring(_mitchell_drop_blank_year(sheet4))
    master = _mitchell_append(master, _mitchell_reshape(sheet4, "rGDP"))

    sheet5 = _mitchell_import_columns_first(path, 5)
    sheet5 = _mitchell_keep_columns(sheet5, ["E"]).rename(columns={"E": "Australia"})
    sheet5 = _mitchell_destring(_mitchell_drop_blank_year(sheet5))
    master = _mitchell_append(master, _mitchell_reshape(sheet5, "rGDP"))

    sheet6 = _mitchell_import_columns_first(path, 6)
    sheet6 = _mitchell_fill_header_rows(sheet6, 3)
    if len(sheet6) > 0:
        for col in [c for c in sheet6.columns if c != "year"]:
            sheet6.at[0, col] = str(sheet6.at[0, col]).replace(" ", "") if pd.notna(sheet6.at[0, col]) else ""
    keep = ["year"] + [
        col
        for col in sheet6.columns
        if col != "year"
        and len(sheet6) > 3
        and str(sheet6.at[2, col]) != "Current Prices"
        and str(sheet6.at[3, col]) == "GDP"
    ]
    sheet6 = sheet6.loc[:, keep].copy()
    sheet6 = _mitchell_rename_from_row(sheet6, 0)
    sheet6 = _mitchell_destring(_mitchell_drop_blank_year(sheet6))
    sheet6 = _mitchell_convert_units(sheet6, "Australia", 1965, 2010, "B")
    master = _mitchell_append(master, _mitchell_reshape(sheet6, "rGDP"))

    assert master is not None
    wide = master.pivot(index="year", columns="countryname", values="rGDP").reset_index()
    wide.columns.name = None
    wide = _mitchell_convert_currency(wide, "Australia", 1900, 2)
    wide = _mitchell_convert_currency(wide, "NewZealand", 1959, 2)
    out = _mitchell_reshape(wide, "rGDP_LCU")
    out = _mitchell_adjust_breaks(out[["countryname", "year", "rGDP_LCU"]], path, "Base_years rGDP", "rGDP_LCU")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "rGDP_LCU"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Oceania_rGDP.dta")
    return out


def _mitchell_europe_rgdp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Europe_NA")

    master: pd.DataFrame | None = None

    for sheet_name, extra_keep, rename_map in (
        (2, [], {"AD": "UnitedKingdom"}),
        (3, [], {"BF": "UnitedKingdom"}),
        (
            4,
            ["H", "Z", "AV", "AZ", "BP", "BT", "CB"],
            {"AF": "Germany", "WestGermany": "Germany", "BP": "Russia", "CF": "UnitedKingdom"},
        ),
        (5, [], {"AF": "Germany", "WestGermany": "Germany", "AR": "Ireland", "SouthernIreland": "Ireland", "CD": "UnitedKingdom"}),
    ):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        if len(frame) > 2:
            for col in [c for c in frame.columns if c != "year"]:
                frame.at[2, col] = str(frame.at[2, col]).strip().lower() if pd.notna(frame.at[2, col]) else ""
        keep = ["year"] + [col for col in extra_keep if col in frame.columns]
        for col in [c for c in frame.columns if c != "year"]:
            row3 = str(frame.at[2, col]) if len(frame) > 2 else ""
            row4 = str(frame.at[3, col]) if len(frame) > 3 else ""
            if row3 == "constant prices" and row4 in {"GDP", "GNP", "NNP", "NMP"}:
                keep.append(col)
        frame = frame.loc[:, list(dict.fromkeys(keep))].copy()
        frame = _mitchell_rename_from_row(frame, 0)
        frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})
        if sheet_name == 3:
            if "Hungary" in frame.columns:
                frame.loc[frame["Hungary"].astype("string").eq("million pengos"), "Hungary"] = ""
            if "Russia" in frame.columns:
                frame.loc[frame["Russia"].astype("string").eq("1937 prices"), "Russia"] = ""
        if sheet_name == 5 and len(frame) >= 32:
            frame = frame.drop(index=[30, 31], errors="ignore").reset_index(drop=True)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 2:
            frame = _mitchell_convert_units(frame, "Italy", 1815, 1899, "B")
        elif sheet_name == 3:
            for country in ["Austria", "Belgium", "Bulgaria", "Greece", "Italy", "Russia", "Spain", "Yugoslavia"]:
                frame = _mitchell_convert_units(frame, country, 1900, 1944, "B")
        elif sheet_name == 4:
            frame = _mitchell_convert_units(frame, "Yugoslavia", 1945, 1979, "Th")
            for country in [c for c in frame.columns if c != "year"]:
                frame = _mitchell_convert_units(frame, country, 1945, 1979, "B")
            frame = _mitchell_convert_units(frame, "Italy", 1946, 1979, "B")
        else:
            for country in ["Hungary", "Ireland"]:
                frame = _mitchell_convert_units(frame, country, 1945, 1979, "Th")
            for country in [c for c in frame.columns if c != "year"]:
                frame = _mitchell_convert_units(frame, country, 1980, 1998, "B")
            frame = _mitchell_convert_units(frame, "Italy", 1980, 1998, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "rGDP"))

    out = master.rename(columns={"rGDP": "rGDP_LCU"}).copy()
    out = out.loc[pd.to_numeric(out["year"], errors="coerce").lt(1994)].copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["rGDP_LCU"], errors="coerce")
    out.loc[country.eq("Poland") & years.between(1947, 1948), "rGDP_LCU"] = np.nan
    out.loc[country.eq("Hungary") & years.lt(1925), "rGDP_LCU"] = np.nan
    out.loc[country.eq("Finland") & years.le(1944), "rGDP_LCU"] = values[country.eq("Finland") & years.le(1944)] * 1000
    out = _mitchell_adjust_breaks(out[["countryname", "year", "rGDP_LCU"]], path, "Base_years rGDP", "rGDP_LCU")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "rGDP_LCU"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Europe_rGDP.dta")
    return out


def _mitchell_europe_govexp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Europe_govexp")

    master: pd.DataFrame | None = None
    for sheet_name in (2, 3, 4, 5, 6):
        frame = _mitchell_import_columns(path, sheet_name)
        rename_map = {"UK": "UnitedKingdom"}
        if sheet_name == 5:
            rename_map.update({"RussiaUSSR": "Russia", "SIreland": "Ireland", "SerbiaYugoslavia": "Yugoslavia"})
        if sheet_name == 6:
            rename_map.update({"EGermany": "EastGermany", "SIreland": "Ireland", "Nethl": "Netherlands", "RussiaUSSR": "Russia"})
        frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})
        if sheet_name in (3, 4):
            frame = _mitchell_use_overlapping_data(frame)
        if sheet_name == 5:
            if "Austria" in frame.columns:
                frame.loc[frame["year"].astype("string").eq("1922"), "Austria"] = ""
            if "Hungary" in frame.columns:
                frame.loc[frame["Hungary"].astype("string").eq("8,644, 649"), "Hungary"] = "8644649"
        frame = _mitchell_drop_blank_year(frame)
        frame = _mitchell_destring(frame)
        if sheet_name == 6 and "WGermany" in frame.columns:
            germany = pd.to_numeric(frame.get("Germany"), errors="coerce")
            west = pd.to_numeric(frame.get("WGermany"), errors="coerce")
            frame["Germany"] = germany.combine_first(west)
            frame = frame.drop(columns=["WGermany"])
        master = _mitchell_append(master, _mitchell_reshape(frame, "govexp"))

    wide = master.pivot(index="year", columns="countryname", values="govexp").reset_index()
    wide.columns.name = None
    for country, start, end, scale in [
        ("Belgium", 1941, 1993, "B"),
        ("Austria", 1950, 1993, "B"),
        ("Denmark", 1950, 1993, "B"),
        ("Finland", 1940, 2000, "B"),
        ("France", 1940, 2000, "B"),
        ("Italy", 1940, 1969, "B"),
        ("Italy", 1970, 1998, "Tri"),
        ("Netherlands", 1950, 1998, "B"),
        ("Norway", 1950, 2010, "B"),
        ("Spain", 1950, 2010, "B"),
        ("Switzerland", 1970, 2010, "B"),
        ("UnitedKingdom", 1970, 1979, "B"),
        ("Germany", 1949, 1993, "B"),
    ]:
        wide = _mitchell_convert_units(wide, country, start, end, scale)
    years = pd.to_numeric(wide["year"], errors="coerce")
    for col, mask, op, factor in [
        ("Finland", years.le(1962), "/", 100),
        ("France", years.le(1959), "/", 100),
        ("Greece", years.le(1952), "/", 1000),
        ("Hungary", years.le(1893), "*", 2),
        ("Hungary", years.le(1924), "/", 12500),
        ("Austria", years.notna(), "/", 10),
        ("Greece", years.notna(), "*", 1000),
        ("Portugal", years.ge(1950), "*", 1000),
        ("Sweden", years.ge(1950), "*", 1000),
        ("UnitedKingdom", years.ge(1980), "*", 1000),
        ("Bulgaria", years.notna(), "/", 1_000_000),
        ("Romania", years.le(1938), "*", 10 ** -9),
        ("Romania", years.notna(), "/", 10),
        ("Poland", years.le(1949), "/", 1000),
        ("Poland", years.notna(), "/", 10),
        ("Germany", years.le(1924), "/", 10 ** 12),
        ("Greece", years.le(1949), "/", 10 ** 6),
        ("Greece", years.le(1949), "/", 4),
    ]:
        if col not in wide.columns:
            continue
        vals = pd.to_numeric(wide[col], errors="coerce")
        wide.loc[mask, col] = vals[mask] * factor if op == "*" else vals[mask] / factor

    out = _mitchell_reshape(wide, "govexp")
    years = pd.to_numeric(out["year"], errors="coerce")
    out = out.loc[years.lt(1994)].copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Austria"), "govexp"] = values[country.eq("Austria")] * 10
    out.loc[country.eq("Germany") & years.eq(1949), "govexp"] = np.nan
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Russia") & years.ge(1940), "govexp"] = values[country.eq("Russia") & years.ge(1940)] * 1000
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "govexp"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Europe_govexp.dta")
    return out


def _mitchell_europe_govrev(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Europe_govrev")

    master: pd.DataFrame | None = None

    sheet2 = _mitchell_import_columns(path, 2).rename(columns={"UK": "UnitedKingdom"})
    sheet2 = _mitchell_keep_columns(sheet2, ["Austria", "UnitedKingdom"])
    sheet2 = _mitchell_destring(_mitchell_drop_blank_year(sheet2))
    sheet2 = _mitchell_convert_units(sheet2, "UnitedKingdom", 1750, 1799, "Th")
    master = _mitchell_append(master, _mitchell_reshape(sheet2, "govrev"))

    for sheet_name in (3, 4, 5, 6):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 2)
        if len(frame) > 1:
            for col in [c for c in frame.columns if c != "year"]:
                frame.at[1, col] = str(frame.at[1, col]).strip().lower() if pd.notna(frame.at[1, col]) else ""
        keep = ["year"] + [
            col for col in frame.columns if col != "year" and len(frame) > 1 and str(frame.at[1, col]) == "total"
        ]
        frame = frame[keep].copy()
        frame = _mitchell_rename_from_row(frame, 0)
        rename_map: dict[str, str] = {}
        if sheet_name == 3:
            rename_map["R"] = "Russia"
        elif sheet_name == 4:
            rename_map["BD"] = "Russia"
        elif sheet_name == 5:
            rename_map.update({"BZ": "Russia", "CW": "Serbia"})
        else:
            rename_map.update({"BM": "Russia", "Yugoslavia": "Serbia", "WestGermany": "Germany"})
        frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 3:
            frame = _mitchell_convert_units(frame, "UnitedKingdom", 1800, 1809, "Th")
        elif sheet_name == 5:
            for country, start, end in [("Serbia", 1942, 1949), ("France", 1945, 1949), ("Italy", 1946, 1949), ("Spain", 1947, 1949)]:
                frame = _mitchell_convert_units(frame, country, start, end, "B")
        elif sheet_name == 6:
            for country in ["Austria", "Belgium", "Finland", "France", "Germany", "Italy", "Romania", "Spain", "Serbia"]:
                frame = _mitchell_convert_units(frame, country, 1950, 2010, "B")
            for country in ["Denmark", "Greece"]:
                frame = _mitchell_convert_units(frame, country, 1970, 2010, "B")
            for country in ["Sweden", "Portugal", "Norway", "Netherlands", "Italy"]:
                frame = _mitchell_convert_units(frame, country, 1975, 2010, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "govrev"))

    wide = master.pivot(index="year", columns="countryname", values="govrev").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"RussiaUSSR": "Russia", "SerbiaYugoslavia": "Serbia"})
    for country, end_year, scale in [
        ("Austria", 1892, 2),
        ("Hungary", 1892, 2),
        ("Russia", 1839, 1 / 4),
        ("Austria", 1923, 1 / 10000),
        ("Hungary", 1924, 1 / 12500),
        ("Russia", 1939, 1 / 10000),
        ("France", 1959, 1 / 100),
    ]:
        wide = _mitchell_convert_currency(wide, country, end_year, scale)
    if "UnitedKingdom" in wide.columns:
        years = pd.to_numeric(wide["year"], errors="coerce")
        vals = pd.to_numeric(wide["UnitedKingdom"], errors="coerce")
        wide.loc[years.ge(1980), "UnitedKingdom"] = vals[years.ge(1980)] * 1000

    out = _mitchell_reshape(wide, "govrev")
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Finland") & years.le(1962), "govrev"] = values[country.eq("Finland") & years.le(1962)] / 100
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Romania"), "govrev"] = values[country.eq("Romania")] / 10000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Romania") & years.le(1943), "govrev"] = values[country.eq("Romania") & years.le(1943)] / 10000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Italy") & years.ge(1999), "govrev"] = values[country.eq("Italy") & years.ge(1999)] / 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Bulgaria"), "govrev"] = values[country.eq("Bulgaria")] / 10_000_000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Germany") & years.le(1923), "govrev"] = values[country.eq("Germany") & years.le(1923)] / (10 ** 12)
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Greece") & years.le(1940), "govrev"] = values[country.eq("Greece") & years.le(1940)] * (10 ** -6) / 5
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Russia"), "govrev"] = values[country.eq("Russia")] * 1000
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "govrev"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Europe_govrev.dta")
    return out


def _mitchell_asia_govexp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Asia_govexp")

    master: pd.DataFrame | None = None
    for sheet_name in (2, 3, 4, 5, 6):
        frame = _mitchell_import_columns(path, sheet_name)
        frame = _mitchell_drop_blank_year(frame)
        frame = _mitchell_destring(frame)
        if sheet_name == 3:
            frame = _mitchell_convert_units(frame, "Cyprus", 1860, 1909, "Th")
        elif sheet_name == 4:
            frame = _mitchell_convert_units(frame, "StraitsSettlements", 1860, 1909, "Th")
        elif sheet_name == 5:
            frame = _mitchell_convert_units(frame, "Cyprus", 1910, 1949, "Th")
            frame = _mitchell_convert_units(frame, "Palestine", 1910, 1939, "Th")
            for country in ["Iran", "Japan"]:
                frame = _mitchell_convert_units(frame, country, 1940, 1949, "B")
            if "Malaya" in frame.columns:
                frame = frame.rename(columns={"Malaya": "Malaysia"})
        elif sheet_name == 6:
            for country in ["Bangladesh", "India", "Indonesia", "Iran", "Japan", "Thailand", "Vietnam"]:
                frame = _mitchell_convert_units(frame, country, 1950, 2010, "B")
            frame = _mitchell_convert_units(frame, "Japan", 1965, 2010, "B")
            frame = _mitchell_convert_units(frame, "Pakistan", 1965, 2010, "B")
            for country in ["Malaysia", "SriLanka", "Syria"]:
                frame = _mitchell_convert_units(frame, country, 1975, 2010, "B")
            for country in ["Indonesia", "SouthKorea", "Nepal", "Singapore", "UnitedArabEmirates"]:
                frame = _mitchell_convert_units(frame, country, 1980, 2010, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "govexp"))

    wide = master.pivot(index="year", columns="countryname", values="govexp").reset_index()
    wide.columns.name = None
    for country, end_year, scale in [
        ("Turkey", 1984, 1 / 1000),
        ("Indonesia", 1964, 1 / 1000),
        ("Israel", 1964, 1.1),
        ("Israel", 1974, 1 / 1000),
        ("Israel", 1984, 1 / 1000),
    ]:
        wide = _mitchell_convert_currency(wide, country, end_year, scale)
    if "Israel" in wide.columns:
        wide["Israel"] = pd.to_numeric(wide["Israel"], errors="coerce") * 1000
    if "Turkey" in wide.columns:
        years = pd.to_numeric(wide["year"], errors="coerce")
        vals = pd.to_numeric(wide["Turkey"], errors="coerce")
        wide.loc[years.le(1949), "Turkey"] = vals[years.le(1949)] / 1000
    if "Philippines" in wide.columns:
        years = pd.to_numeric(wide["year"], errors="coerce")
        vals = pd.to_numeric(wide["Philippines"], errors="coerce")
        wide.loc[years.ge(1970), "Philippines"] = vals[years.ge(1970)] * 1000
    if "SouthKorea" in wide.columns:
        wide["SouthKorea"] = pd.to_numeric(wide["SouthKorea"], errors="coerce") * 1000
    if "Taiwan" in wide.columns:
        years = pd.to_numeric(wide["year"], errors="coerce")
        vals = pd.to_numeric(wide["Taiwan"], errors="coerce")
        wide.loc[years.ge(1950), "Taiwan"] = vals[years.ge(1950)] * 1000
    if {"Vietnam", "Indochina"}.issubset(wide.columns):
        ind = pd.to_numeric(wide["Indochina"], errors="coerce")
        viet = pd.to_numeric(wide["Vietnam"], errors="coerce")
        mask = viet.notna()
        wide.loc[mask, "Vietnam"] = ind[mask]
        wide = wide.drop(columns=["Indochina"])
    if {"Singapore", "StraitsSettlements"}.issubset(wide.columns):
        ss = pd.to_numeric(wide["StraitsSettlements"], errors="coerce")
        mask = ss.notna()
        wide.loc[mask, "Singapore"] = ss[mask]
        wide = wide.drop(columns=["StraitsSettlements"])

    out = _mitchell_reshape(wide, "govexp")
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Taiwan") & years.le(1939), "govexp"] = values[country.eq("Taiwan") & years.le(1939)] * (10 ** -4)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Taiwan") & years.le(1939), "govexp"] = values[country.eq("Taiwan") & years.le(1939)] / 4
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "govexp"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Asia_govexp.dta")
    return out


def _mitchell_asia_govrev(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Asia_govrev")

    master: pd.DataFrame | None = None
    for sheet_name in (2, 3, 4, 5, 6, 7):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 2)
        if len(frame) > 1:
            for col in [c for c in frame.columns if c != "year"]:
                frame.at[1, col] = str(frame.at[1, col]).strip().lower() if pd.notna(frame.at[1, col]) else ""
        keep = ["year"] + [
            col for col in frame.columns if col != "year" and len(frame) > 1 and str(frame.at[1, col]) == "total"
        ]
        frame = frame[keep].copy()
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_drop_blank_year(frame)
        frame = _mitchell_destring(frame)
        if sheet_name == 3:
            frame = _mitchell_convert_units(frame, "Cyprus", 1855, 1904, "Th")
        elif sheet_name == 4:
            frame = _mitchell_convert_units(frame, "StraitsSettlements", 1855, 1904, "Th")
        elif sheet_name == 5:
            for country in ["Cyprus", "Palestine"]:
                frame = _mitchell_convert_units(frame, country, 1905, 1944, "Th")
        elif sheet_name == 6:
            if "SouthKorea" in frame.columns:
                frame.loc[frame["SouthKorea"].astype("string").eq("thousand million won"), "SouthKorea"] = np.nan
            frame = _mitchell_convert_units(frame, "Bangladesh", 1945, 2010, "B")
            frame = _mitchell_convert_units(frame, "Iran", 1945, 2010, "B")
            frame = _mitchell_convert_units(frame, "Indonesia", 1945, 1979, "B")
            frame = _mitchell_convert_units(frame, "Indonesia", 1980, 2010, "Tri")
            frame = _mitchell_convert_units(frame, "India", 1960, 2010, "B")
            frame = _mitchell_convert_units(frame, "Israel", 1985, 2010, "B")
            frame = _mitchell_convert_units(frame, "Japan", 1945, 1964, "B")
            frame = _mitchell_convert_units(frame, "Japan", 1965, 2010, "Tri")
            frame = _mitchell_convert_units(frame, "SouthKorea", 1945, 1978, "B")
            frame = _mitchell_convert_units(frame, "SouthKorea", 1980, 2010, "Tri")
            frame = _mitchell_convert_units(frame, "Malaysia", 1975, 2010, "B")
        elif sheet_name == 7:
            frame = _mitchell_convert_units(frame, "Pakistan", 1970, 2010, "B")
            frame = _mitchell_convert_units(frame, "Phillippines", 1970, 2010, "B")
            frame = _mitchell_convert_units(frame, "Singapore", 1980, 2010, "B")
            frame = _mitchell_convert_units(frame, "SriLanka", 1975, 2010, "B")
            frame = _mitchell_convert_units(frame, "Syria", 1971, 2010, "B")
            frame = _mitchell_convert_units(frame, "Taiwan", 1945, 2010, "B")
            frame = _mitchell_convert_units(frame, "Thailand", 1962, 2010, "B")
            frame = _mitchell_convert_units(frame, "Turkey", 1959, 1988, "B")
            frame = _mitchell_convert_units(frame, "Turkey", 1989, 2010, "Tri")
            frame = _mitchell_convert_units(frame, "Vietnam", 1954, 2010, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "govrev"))

    wide = master.pivot(index="year", columns="countryname", values="govrev").reset_index()
    wide.columns.name = None
    for country, end_year, scale in [
        ("Turkey", 1984, 1 / 1000),
        ("Indonesia", 1964, 1 / 1000),
        ("Israel", 1964, 1.1),
        ("Israel", 1974, 1 / 1000),
    ]:
        wide = _mitchell_convert_currency(wide, country, end_year, scale)
    if "Vietnam" in wide.columns:
        wide["Vietnam"] = pd.to_numeric(wide["Vietnam"], errors="coerce") / 1000

    out = _mitchell_reshape(wide, "govrev")
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Iran") & years.between(1940, 1944), "govrev"] = values[country.eq("Iran") & years.between(1940, 1944)] * 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Turkey"), "govrev"] = values[country.eq("Turkey")] / (10 ** 6)
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Taiwan") & years.le(1939), "govrev"] = values[country.eq("Taiwan") & years.le(1939)] * (10 ** -4)
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Taiwan") & years.le(1939), "govrev"] = values[country.eq("Taiwan") & years.le(1939)] / 4
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "govrev"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Asia_govrev.dta")
    return out


def _mitchell_oceania_govexp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Oceania_govexp")

    frame = _mitchell_import_columns(path, 2)
    frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
    frame = _mitchell_convert_units(frame, "Fiji", 1840, 1964, "Th")
    frame = _mitchell_convert_currency(frame, "Fiji", 1964, 1 / 2)
    frame = _mitchell_convert_currency(frame, "Australia", 1964, 2)
    frame = _mitchell_convert_units(frame, "Australia", 1970, 2010, "B")
    frame = _mitchell_convert_units(frame, "NewZealand", 1840, 1899, "Th")
    frame = _mitchell_convert_units(frame, "NewZealand", 1980, 2010, "B")
    frame = _mitchell_convert_currency(frame, "NewZealand", 1964, 2)
    frame = _mitchell_convert_units(frame, "Hawaii", 1840, 1929, "Th")
    out = _mitchell_reshape(frame, "govexp")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "govexp"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Oceania_govexp.dta")
    return out


def _mitchell_oceania_govrev(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Oceania_govrev")

    master: pd.DataFrame | None = None
    for sheet_name in (2, 3, 4, 5):
        frame = _mitchell_import_columns(path, sheet_name)
        if sheet_name == 2:
            frame = _mitchell_drop_columns(frame, ["B", "D", "F"])
        elif sheet_name == 3:
            frame = _mitchell_drop_columns(frame, ["C", "E", "G", "H"])
        frame = _mitchell_drop_blank_year(frame)
        frame = _mitchell_destring(frame)
        if sheet_name == 2:
            for country in ["Hawaii", "NewZealand"]:
                frame = _mitchell_convert_units(frame, country, 1840, 1869, "Th")
        elif sheet_name == 3:
            for country in ["Fiji", "Hawaii", "NewZealand"]:
                frame = _mitchell_convert_units(frame, country, 1870, 1899, "Th")
        elif sheet_name == 4:
            frame = _mitchell_convert_units(frame, "Fiji", 1900, 1944, "Th")
            frame = _mitchell_convert_units(frame, "NewZealand", 1900, 1919, "Th")
        elif sheet_name == 5:
            frame = _mitchell_convert_units(frame, "Australia", 1975, 2010, "B")
            frame = _mitchell_convert_units(frame, "Fiji", 1945, 1964, "Th")
        master = _mitchell_append(master, _mitchell_reshape(frame, "govrev"))

    wide = master.pivot(index="year", columns="countryname", values="govrev").reset_index()
    wide.columns.name = None
    for country, end_year, scale in [("Australia", 1964, 2), ("Fiji", 1964, 2), ("NewZealand", 1964, 2)]:
        wide = _mitchell_convert_currency(wide, country, end_year, scale)
    out = _mitchell_reshape(wide, "govrev")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "govrev"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Oceania_govrev.dta")
    return out


def _mitchell_latam_govexp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Latam_govexp")

    master: pd.DataFrame | None = None

    sheet2 = _mitchell_import_columns(path, 2)
    sheet2 = _mitchell_drop_columns(sheet2, ["F", "H"])
    sheet2 = _mitchell_destring(_mitchell_drop_blank_year(sheet2))
    sheet2 = _mitchell_convert_units(sheet2, "Guyana", 1823, 1864, "Th")
    master = _mitchell_append(master, _mitchell_reshape(sheet2, "govexp"))

    sheet3 = _mitchell_import_columns(path, 3)
    sheet3 = _mitchell_drop_columns(sheet3, ["J", "L"])
    if "Brazil" in sheet3.columns:
        sheet3.loc[sheet3["Brazil"].astype("string").eq("34 46"), "Brazil"] = "34.46"
    sheet3 = _mitchell_destring(_mitchell_drop_blank_year(sheet3))
    sheet3 = _mitchell_convert_units(sheet3, "Guyana", 1865, 1939, "Th")
    for country, start, end, scale in [
        ("Argentina", 1950, 1982, "B"),
        ("Argentina", 1978, 1982, "B"),
        ("Argentina", 1985, 1988, "B"),
        ("Bolivia", 1950, 2007, "B"),
        ("Bolivia", 1985, 2007, "B"),
        ("Brazil", 1950, 1966, "B"),
        ("Brazil", 1985, 1988, "B"),
        ("Brazil", 1990, 1992, "Th"),
        ("Colombia", 1967, 2010, "B"),
        ("Ecuador", 1970, 1999, "B"),
        ("Paraguay", 1965, 2010, "B"),
        ("Peru", 1965, 2010, "B"),
        ("Uruguay", 1965, 2010, "B"),
        ("Venezuela", 1965, 2010, "B"),
    ]:
        sheet3 = _mitchell_convert_units(sheet3, country, start, end, scale)
    if "Brazil" in sheet3.columns:
        years = pd.to_numeric(sheet3["year"], errors="coerce")
        vals = pd.to_numeric(sheet3["Brazil"], errors="coerce")
        sheet3.loc[years.eq(1989), "Brazil"] = vals[years.eq(1989)] * (10 ** -6)
    if "Ecuador" in sheet3.columns:
        years = pd.to_numeric(sheet3["year"], errors="coerce")
        vals = pd.to_numeric(sheet3["Ecuador"], errors="coerce")
        sheet3.loc[years.lt(2000), "Ecuador"] = vals[years.lt(2000)] / 25000
    master = _mitchell_append(master, _mitchell_reshape(sheet3, "govexp"))

    out = master.copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Brazil") & years.le(1988), "govexp"] = values[country.eq("Brazil") & years.le(1988)] / 2750
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Brazil") & years.le(1966), "govexp"] = values[country.eq("Brazil") & years.le(1966)] * (10 ** -6)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Brazil") & years.le(1988), "govexp"] = values[country.eq("Brazil") & years.le(1988)] * (10 ** -6)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Argentina") & years.le(1988), "govexp"] = values[country.eq("Argentina") & years.le(1988)] * (10 ** -4)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Argentina") & years.le(1982), "govexp"] = values[country.eq("Argentina") & years.le(1982)] * (10 ** -7)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Argentina") & years.eq(1978), "govexp"] = values[country.eq("Argentina") & years.eq(1978)] * (10 ** -3)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Argentina") & years.le(1969), "govexp"] = values[country.eq("Argentina") & years.le(1969)] * (10 ** -2)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Uruguay"), "govexp"] = values[country.eq("Uruguay")] * (10 ** -3)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Uruguay") & years.le(1979), "govexp"] = values[country.eq("Uruguay") & years.le(1979)] * (10 ** -3)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Peru"), "govexp"] = values[country.eq("Peru")] * (10 ** -3)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Peru") & years.le(1989), "govexp"] = values[country.eq("Peru") & years.le(1989)] * (10 ** -3)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Peru") & years.le(1984), "govexp"] = values[country.eq("Peru") & years.le(1984)] * (10 ** -3)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Chile"), "govexp"] = values[country.eq("Chile")] * (10 ** 3)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Chile") & years.le(1974), "govexp"] = values[country.eq("Chile") & years.le(1974)] * (10 ** -3)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Chile") & years.le(1966), "govexp"] = values[country.eq("Chile") & years.le(1966)] * (10 ** -3)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Chile") & years.le(1949), "govexp"] = values[country.eq("Chile") & years.le(1949)] * (10 ** -3)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Bolivia"), "govexp"] = values[country.eq("Bolivia")] * (10 ** -6)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Bolivia") & years.le(1973), "govexp"] = values[country.eq("Bolivia") & years.le(1973)] * (10 ** -3)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Paraguay") & years.le(1939), "govexp"] = values[country.eq("Paraguay") & years.le(1939)] * (10 ** -2)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Paraguay") & years.le(1919), "govexp"] = values[country.eq("Paraguay") & years.le(1919)] / 1.75
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Venezuela"), "govexp"] = values[country.eq("Venezuela")] * (10 ** -14)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Suriname"), "govexp"] = values[country.eq("Suriname")] * (10 ** -3)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "govexp"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Latam_govexp.dta")
    return out


def _mitchell_latam_govrev(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Latam_govrev")

    master: pd.DataFrame | None = None
    for sheet_name in (2, 3, 4, 5):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 2)
        if len(frame) > 1:
            for col in [c for c in frame.columns if c != "year"]:
                frame.at[1, col] = str(frame.at[1, col]).strip().lower() if pd.notna(frame.at[1, col]) else ""
        keep = ["year"] + [
            col for col in frame.columns if col != "year" and len(frame) > 1 and str(frame.at[1, col]) == "total"
        ]
        frame = frame[keep].copy()
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 2:
            frame = _mitchell_convert_units(frame, "Guyana", 1823, 1864, "Th")
        elif sheet_name == 3:
            frame = _mitchell_convert_units(frame, "Guyana", 1865, 1894, "Th")
        elif sheet_name == 5:
            for country, start, end, scale in [
                ("Argentina", 1955, 1969, "B"),
                ("Argentina", 1970, 1978, "B"),
                ("Argentina", 1979, 1984, "Tri"),
                ("Argentina", 1985, 1988, "B"),
                ("Bolivia", 1955, 2010, "B"),
                ("Brazil", 1955, 1966, "B"),
                ("Brazil", 1970, 1988, "B"),
                ("Brazil", 1990, 1992, "Th"),
                ("Chile", 1950, 1954, "B"),
                ("Chile", 1970, 2010, "B"),
                ("Colombia", 1975, 2010, "B"),
                ("Paraguay", 1975, 2010, "B"),
                ("Ecuador", 1975, 1993, "B"),
                ("Peru", 1965, 1984, "B"),
                ("Peru", 1985, 1987, "B"),
                ("Peru", 1988, 2010, "Th"),
                ("Uruguay", 1965, 1974, "B"),
                ("Uruguay", 1980, 2010, "B"),
                ("Venezuela", 1975, 2010, "B"),
            ]:
                frame = _mitchell_convert_units(frame, country, start, end, scale)
        master = _mitchell_append(master, _mitchell_reshape(frame, "govrev"))

    wide = master.pivot(index="year", columns="countryname", values="govrev").reset_index()
    wide.columns.name = None
    for country, end_year, scale in [
        ("Argentina", 1969, 1 / 100),
        ("Argentina", 1984, 1 / 1000),
        ("Argentina", 1989, 1 / 10000),
        ("Bolivia", 1974, 1 / 1000),
        ("Brazil", 1966, 1 / 1000),
        ("Brazil", 1984, 1 / 1000),
        ("Brazil", 1988, 1 / 2750),
        ("Brazil", 1989, 1 / 1_000_000),
    ]:
        wide = _mitchell_convert_currency(wide, country, end_year, scale)

    out = _mitchell_reshape(wide, "govrev")
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Chile") & years.le(1974), "govrev"] = values[country.eq("Chile") & years.le(1974)] / 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Chile") & years.le(1954), "govrev"] = values[country.eq("Chile") & years.le(1954)] / 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Ecuador") & years.le(1993), "govrev"] = values[country.eq("Ecuador") & years.le(1993)] / 2500
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Venezuela"), "govrev"] = values[country.eq("Venezuela")] * (10 ** -14)
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Uruguay"), "govrev"] = values[country.eq("Uruguay")] / 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Uruguay") & years.le(1974), "govrev"] = values[country.eq("Uruguay") & years.le(1974)] / 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Peru") & years.ge(1990), "govrev"] = values[country.eq("Peru") & years.ge(1990)] * 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Peru") & years.eq(1989), "govrev"] = values[country.eq("Peru") & years.eq(1989)] * 1
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Peru") & years.le(1988), "govrev"] = values[country.eq("Peru") & years.le(1988)] / 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Peru") & years.le(1987), "govrev"] = values[country.eq("Peru") & years.le(1987)] / 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Peru") & years.le(1984), "govrev"] = values[country.eq("Peru") & years.le(1984)] / 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Suriname"), "govrev"] = values[country.eq("Suriname")] / 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Bolivia"), "govrev"] = values[country.eq("Bolivia")] / 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Bolivia") & years.le(1984), "govrev"] = values[country.eq("Bolivia") & years.le(1984)] / 100
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Argentina") & years.le(1984), "govrev"] = values[country.eq("Argentina") & years.le(1984)] / 10000
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "govrev"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Latam_govrev.dta")
    return out


def _mitchell_africa_govexp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Africa_govexp")

    master: pd.DataFrame | None = None
    for sheet_name in (2, 3, 4):
        frame = _mitchell_import_columns(path, sheet_name)
        if sheet_name == 4 and "Zanzibar" in frame.columns:
            series = frame["Zanzibar"].astype("string").fillna("")
            frame.loc[series.str.len().gt(4), "Zanzibar"] = ""
        frame = _mitchell_drop_blank_year(frame)
        frame = _mitchell_destring(frame)
        if sheet_name == 2:
            if {"CapeofGoodHope", "Natal"}.issubset(frame.columns):
                cape = pd.to_numeric(frame["CapeofGoodHope"], errors="coerce")
                natal = pd.to_numeric(frame["Natal"], errors="coerce")
                frame["SouthAfrica"] = cape.add(natal, fill_value=np.nan)
                frame["SouthAfrica"] = frame["SouthAfrica"].combine_first(cape)
                frame = frame.drop(columns=["CapeofGoodHope", "Natal"])
            for country in ["Ghana", "Kenya", "Malawi", "Mauritius", "Nigeria", "SierraLeone", "SouthAfrica", "Sudan", "Uganda", "Zambia", "Zanzibar", "Zimbabwe"]:
                frame = _mitchell_convert_units(frame, country, 1812, 1904, "Th")
            frame = _mitchell_convert_units(frame, "Egypt", 1812, 1859, "Th")
        elif sheet_name == 3:
            for country in ["Ghana", "Kenya", "Malawi", "SierraLeone", "Uganda", "Zambia", "Zanzibar", "Zimbabwe"]:
                frame = _mitchell_convert_units(frame, country, 1905, 1949, "Th")
            frame = _mitchell_convert_units(frame, "Tanganyika", 1915, 1949, "Th")
            frame = _mitchell_convert_units(frame, "Zimbabwe", 1905, 1939, "Th")
            frame = _mitchell_convert_units(frame, "Nigeria", 1905, 1944, "Th")
            frame = _mitchell_convert_units(frame, "Sudan", 1905, 1944, "Th")
        else:
            for country in ["Cameroon", "Gabon", "IvoryCoast", "Madagascar", "Mali", "Senegal"]:
                frame = _mitchell_convert_units(frame, country, 1950, 2010, "B")
            frame = _mitchell_convert_units(frame, "Tunisia", 1950, 1955, "B")
            frame = _mitchell_convert_units(frame, "Tunisia", 1950, 1958, "B")
            frame = _mitchell_convert_units(frame, "Zaire", 1950, 1959, "B")
            frame = _mitchell_convert_units(frame, "Algeria", 1950, 1993, "B")
            for country in ["Benin", "BurkinaFaso", "CentralAfricanRepublic", "Chad", "Congo", "Kenya", "Togo"]:
                frame = _mitchell_convert_units(frame, country, 1970, 2010, "B")
            for country in ["Morocco", "Niger", "SouthAfrica", "Tanzania"]:
                frame = _mitchell_convert_units(frame, country, 1975, 2010, "B")
            frame = _mitchell_convert_units(frame, "Zanzibar", 1950, 2010, "Th")
            frame = _mitchell_convert_units(frame, "Ghana", 1985, 2010, "B")
            frame = _mitchell_convert_units(frame, "Nigeria", 2001, 2010, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "govexp"))

    wide = master.pivot(index="year", columns="countryname", values="govexp").reset_index()
    wide.columns.name = None
    years = pd.to_numeric(wide["year"], errors="coerce")
    if "Nigeria" in wide.columns:
        vals = pd.to_numeric(wide["Nigeria"], errors="coerce")
        wide.loc[years.le(1972), "Nigeria"] = vals[years.le(1972)] * 2
    if "SierraLeone" in wide.columns:
        vals = pd.to_numeric(wide["SierraLeone"], errors="coerce")
        wide.loc[years.le(1963), "SierraLeone"] = vals[years.le(1963)] * 2
    if "Ghana" in wide.columns:
        vals = pd.to_numeric(wide["Ghana"], errors="coerce")
        wide.loc[years.le(1964), "Ghana"] = vals[years.le(1964)] / 0.417
    if "Madagascar" in wide.columns:
        vals = pd.to_numeric(wide["Madagascar"], errors="coerce")
        wide.loc[years.le(2000), "Madagascar"] = vals[years.le(2000)] / 5
    if "SouthAfrica" in wide.columns:
        vals = pd.to_numeric(wide["SouthAfrica"], errors="coerce")
        wide.loc[years.le(1959), "SouthAfrica"] = vals[years.le(1959)] * 2
    if "Algeria" in wide.columns:
        vals = pd.to_numeric(wide["Algeria"], errors="coerce")
        wide.loc[years.le(1960), "Algeria"] = vals[years.le(1960)] / 100
        vals = pd.to_numeric(wide["Algeria"], errors="coerce")
        wide.loc[years.ge(1994), "Algeria"] = vals[years.ge(1994)] * 1000
        vals = pd.to_numeric(wide["Algeria"], errors="coerce")
        wide.loc[years.between(1947, 1949), "Algeria"] = vals[years.between(1947, 1949)] * 1000
    if "Angola" in wide.columns:
        vals = pd.to_numeric(wide["Angola"], errors="coerce")
        wide.loc[years.le(1974), "Angola"] = vals[years.le(1974)] / 1_000_000_000
    if "Cameroon" in wide.columns:
        vals = pd.to_numeric(wide["Cameroon"], errors="coerce")
        wide.loc[years.le(1919), "Cameroon"] = vals[years.le(1919)] * 3.3538549
    if "Zanzibar" in wide.columns:
        wide["Zanzibar"] = pd.to_numeric(wide["Zanzibar"], errors="coerce") / 20
    if "Tanganyika" in wide.columns:
        vals = pd.to_numeric(wide["Tanganyika"], errors="coerce")
        wide.loc[years.le(1912), "Tanganyika"] = vals[years.le(1912)] / 25.4377
    if "Tanzania" in wide.columns:
        if "Zanzibar" in wide.columns:
            zanz = pd.to_numeric(wide["Zanzibar"], errors="coerce")
            tza = pd.to_numeric(wide["Tanzania"], errors="coerce")
            mask = zanz.notna()
            wide.loc[mask, "Tanzania"] = zanz[mask] + tza[mask]
        if "Tanganyika" in wide.columns:
            tang = pd.to_numeric(wide["Tanganyika"], errors="coerce")
            tza = pd.to_numeric(wide["Tanzania"], errors="coerce")
            mask = tang.notna()
            wide.loc[mask, "Tanzania"] = tang[mask] + tza[mask]
        wide = wide.drop(columns=[c for c in ["Tanganyika", "Zanzibar"] if c in wide.columns])

    out = _mitchell_reshape(wide, "govexp")
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Tunisia") & years.between(1950, 1954), "govexp"] = values[country.eq("Tunisia") & years.between(1950, 1954)] / 1_000_000
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Tunisia") & years.le(1949), "govexp"] = values[country.eq("Tunisia") & years.le(1949)] / 1000
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Uganda") & years.le(1976), "govexp"] = values[country.eq("Uganda") & years.le(1976)] / 100
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Zaire"), "govexp"] = values[country.eq("Zaire")] / 1000
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Zaire") & years.le(1958), "govexp"] = values[country.eq("Zaire") & years.le(1958)] / 1000
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Zaire") & years.le(1979), "govexp"] = values[country.eq("Zaire") & years.le(1979)] / 1000
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Zaire") & years.le(1997), "govexp"] = values[country.eq("Zaire") & years.le(1997)] / 100
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Zaire") & years.le(1993), "govexp"] = values[country.eq("Zaire") & years.le(1993)] / 1000
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Mauritius") & years.between(1850, 1904), "govexp"] = values[country.eq("Mauritius") & years.between(1850, 1904)] * 1000
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Mauritius") & years.le(1849), "govexp"] = values[country.eq("Mauritius") & years.le(1849)] * 10
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Ghana"), "govexp"] = values[country.eq("Ghana")] / 10000
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Mauritania"), "govexp"] = values[country.eq("Mauritania")] / 10
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Mozambique") & years.le(1973), "govexp"] = values[country.eq("Mozambique") & years.le(1973)] / 1000
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Morocco") & years.le(1956), "govexp"] = values[country.eq("Morocco") & years.le(1956)] * 10
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Morocco") & years.le(1944), "govexp"] = values[country.eq("Morocco") & years.le(1944)] / 1000
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Sudan") & years.le(1985), "govexp"] = values[country.eq("Sudan") & years.le(1985)] / 1000
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Zambia"), "govexp"] = values[country.eq("Zambia")] / 1000
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "govexp"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Africa_govexp.dta")
    return out


def _mitchell_africa_govrev(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Africa_govrev")

    master: pd.DataFrame | None = None
    for sheet_name in (2, 3, 4, 5, 6, 7, 8):
        frame = _mitchell_import_columns(path, sheet_name)
        frame = _mitchell_drop_blank_year(frame)
        frame = _mitchell_destring(frame)
        if sheet_name == 2:
            cols = [c for c in frame.columns if c not in {"year", "Algeria"}]
            for country in cols:
                frame = _mitchell_convert_units(frame, country, 1812, 1859, "Th")
            if {"CapeofGoodHope", "Natal"}.issubset(frame.columns):
                cape = pd.to_numeric(frame["CapeofGoodHope"], errors="coerce")
                natal = pd.to_numeric(frame["Natal"], errors="coerce")
                frame["SouthAfrica"] = cape.add(natal, fill_value=np.nan)
                frame["SouthAfrica"] = frame["SouthAfrica"].combine_first(cape)
                frame = frame.drop(columns=["CapeofGoodHope", "Natal"])
        elif sheet_name == 3:
            for country in ["Ghana", "Kenya", "Malawi", "Nigeria", "SierraLeone"]:
                frame = _mitchell_convert_units(frame, country, 1860, 1904, "Th")
        elif sheet_name == 4:
            if {"CapeofGoodHope", "Natal"}.issubset(frame.columns):
                cape = pd.to_numeric(frame["CapeofGoodHope"], errors="coerce")
                natal = pd.to_numeric(frame["Natal"], errors="coerce")
                frame["SouthAfrica"] = cape.add(natal, fill_value=np.nan)
                frame["SouthAfrica"] = frame["SouthAfrica"].combine_first(cape)
                frame = frame.drop(columns=["CapeofGoodHope", "Natal"])
            cols = [c for c in frame.columns if c not in {"year", "Tanganyika", "Togo"}]
            for country in cols:
                frame = _mitchell_convert_units(frame, country, 1860, 1904, "Th")
        elif sheet_name == 5:
            for country in ["Algeria", "Morocco"]:
                frame = _mitchell_convert_units(frame, country, 1945, 1949, "B")
            for country in ["Ghana", "Kenya", "Malawi", "Mozambique", "SierraLeone", "Uganda", "Zambia", "Zanzibar", "Nigeria"]:
                frame = _mitchell_convert_units(frame, country, 1905, 1949, "Th")
            frame = _mitchell_convert_units(frame, "Tanganyika", 1915, 1949, "Th")
            frame = _mitchell_convert_units(frame, "Sudan", 1905, 1944, "Th")
        elif sheet_name == 6:
            for country in ["BurkinaFaso", "Benin", "CentralAfricanRepublic", "Chad", "Congo"]:
                frame = _mitchell_convert_units(frame, country, 1970, 2010, "B")
            for country in ["Algeria", "Gabon", "Cameroon", "IvoryCoast", "Madagascar", "Mali"]:
                frame = _mitchell_convert_units(frame, country, 1950, 2010, "B")
            frame = _mitchell_convert_units(frame, "Kenya", 1975, 2010, "B")
            frame = _mitchell_convert_units(frame, "Ghana", 1985, 2010, "B")
        elif sheet_name == 7:
            for country in ["Morocco", "Niger"]:
                frame = _mitchell_convert_units(frame, country, 1975, 2010, "B")
            frame = _mitchell_convert_units(frame, "Senegal", 1950, 2010, "B")
            frame = _mitchell_convert_units(frame, "SouthAfrica", 1980, 2010, "B")
            frame = _mitchell_convert_units(frame, "Mauritania", 1990, 2010, "B")
        elif sheet_name == 8:
            for country in ["Tunisia", "Zaire"]:
                frame = _mitchell_convert_units(frame, country, 1950, 1960, "B")
            frame = _mitchell_convert_units(frame, "Togo", 1970, 2010, "B")
            frame = _mitchell_convert_units(frame, "Zaire", 1980, 1987, "B")
            frame = _mitchell_convert_units(frame, "Zaire", 1988, 1991, "B")
            frame = _mitchell_convert_units(frame, "Sudan", 1988, 2010, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "govrev"))

    wide = master.pivot(index="year", columns="countryname", values="govrev").reset_index()
    wide.columns.name = None
    for country, end_year, scale in [
        ("Algeria", 1960, 1 / 100),
        ("Ghana", 1964, 2.4),
        ("Malawi", 1963, 2),
        ("Mauritania", 1969, 1 / 5),
        ("Nigeria", 1964, 2),
        ("SierraLeone", 1963, 2),
        ("Tunisia", 1959, 1 / 1000),
        ("Zambia", 1964, 2),
        ("SouthAfrica", 1979, 2),
        ("Uganda", 1949, 20),
        ("Uganda", 1976, 1 / 100),
    ]:
        wide = _mitchell_convert_currency(wide, country, end_year, scale)

    out = _mitchell_reshape(wide, "govrev")
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Angola") & years.le(1994), "govrev"] = values[country.eq("Angola") & years.le(1994)] / 1_000_000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Gabon") & years.le(1964), "govrev"] = values[country.eq("Gabon") & years.le(1964)] / 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Ghana"), "govrev"] = values[country.eq("Ghana")] / 10000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Morocco") & years.le(1949), "govrev"] = values[country.eq("Morocco") & years.le(1949)] / 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Morocco") & years.le(1958), "govrev"] = values[country.eq("Morocco") & years.le(1958)] * 10
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Mozambique") & years.ge(1950), "govrev"] = values[country.eq("Mozambique") & years.ge(1950)] / 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Nigeria") & years.between(1940, 1949), "govrev"] = values[country.eq("Nigeria") & years.between(1940, 1949)] * 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Kenya") & years.le(1949), "govrev"] = values[country.eq("Kenya") & years.le(1949)] * 10
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Tanzania") & years.ge(1975), "govrev"] = values[country.eq("Tanzania") & years.ge(1975)] * 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Mauritius") & years.between(1850, 1859), "govrev"] = values[country.eq("Mauritius") & years.between(1850, 1859)] * 100
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("SouthAfrica") & years.eq(1952), "govrev"] = values[country.eq("SouthAfrica") & years.eq(1952)] / 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Mauritania"), "govrev"] = values[country.eq("Mauritania")] / 10
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Sudan"), "govrev"] = values[country.eq("Sudan")] / 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Zaire") & years.between(1992, 1995), "govrev"] = values[country.eq("Zaire") & years.between(1992, 1995)] * 100
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Zaire") & years.le(1991), "govrev"] = values[country.eq("Zaire") & years.le(1991)] * (10 ** -6)
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Zaire") & years.le(1987), "govrev"] = values[country.eq("Zaire") & years.le(1987)] / 3
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Zaire") & years.le(1958), "govrev"] = values[country.eq("Zaire") & years.le(1958)] / 1000
    values = pd.to_numeric(out["govrev"], errors="coerce")
    out.loc[country.eq("Zambia"), "govrev"] = values[country.eq("Zambia")] / 1000
    out.loc[country.eq("Zaire"), "govrev"] = np.nan
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "govrev"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Africa_govrev.dta")
    return out


def _mitchell_americas_govexp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Americas_govexp")

    master: pd.DataFrame | None = None
    for sheet_name in (2, 3, 4):
        frame = _mitchell_import_columns(path, sheet_name)
        frame = _mitchell_drop_blank_year(frame)
        frame = _mitchell_destring(frame)
        if sheet_name == 3:
            for country in ["Barbados", "Jamaica", "Trinidad"]:
                frame = _mitchell_convert_units(frame, country, 1825, 1864, "Th")
        elif sheet_name == 4:
            frame = _mitchell_convert_units(frame, "Trinidad", 1865, 1935, "Th")
            frame = _mitchell_convert_units(frame, "Barbados", 1865, 1945, "Th")
            frame = _mitchell_convert_units(frame, "Jamaica", 1865, 1944, "Th")
            frame = _mitchell_convert_units(frame, "Mexico", 1965, 1984, "B")
            frame = _mitchell_convert_units(frame, "Mexico", 1985, 2010, "Tri")
            frame = _mitchell_convert_units(frame, "USA", 1945, 2010, "B")
            frame = _mitchell_convert_units(frame, "Canada", 1975, 2010, "B")
            frame = _mitchell_convert_units(frame, "CostaRica", 1980, 2010, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "govexp"))

    wide = master.pivot(index="year", columns="countryname", values="govexp").reset_index()
    wide.columns.name = None
    for country, end_year, scale in [
        ("Nicaragua", 1912, 0.08),
        ("Nicaragua", 1987, 0.0001),
        ("Nicaragua", 1989, 0.2),
        ("Jamaica", 1968, 2),
        ("Guatemala", 1923, 1 / 60),
        ("Trinidad", 1935, 4.2),
    ]:
        wide = _mitchell_convert_currency(wide, country, end_year, scale)
    rename_map = {}
    if "Trinidad" in wide.columns:
        rename_map["Trinidad"] = "TrinidadandTobago"
    if "EISalvador" in wide.columns:
        rename_map["EISalvador"] = "ElSalvador"
    wide = wide.rename(columns=rename_map)
    if "ElSalvador" in wide.columns:
        years = pd.to_numeric(wide["year"], errors="coerce")
        vals = pd.to_numeric(wide["ElSalvador"], errors="coerce")
        wide.loc[years.le(2000), "ElSalvador"] = vals[years.le(2000)] / 8.75

    out = _mitchell_reshape(wide, "govexp")
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Mexico"), "govexp"] = values[country.eq("Mexico")] * (10 ** -6)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Mexico") & years.le(1993), "govexp"] = values[country.eq("Mexico") & years.le(1993)] * (10 ** 3)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Nicaragua") & years.le(1988), "govexp"] = values[country.eq("Nicaragua") & years.le(1988)] * (10 ** -3)
    values = pd.to_numeric(out["govexp"], errors="coerce")
    out.loc[country.eq("Nicaragua") & years.le(1984), "govexp"] = values[country.eq("Nicaragua") & years.le(1984)] * (10 ** -3)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "govexp"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Americas_govexp.dta")
    return out


def _mitchell_americas_govrev(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Americas_govrev")

    master: pd.DataFrame | None = None
    sheet2 = _mitchell_import_columns(path, 2)
    sheet2 = _mitchell_destring(_mitchell_drop_blank_year(sheet2))
    master = _mitchell_append(master, _mitchell_reshape(sheet2, "govrev"))

    for sheet_name in (3, 4, 5, 6, 7):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 2)
        if len(frame) > 1:
            for col in [c for c in frame.columns if c != "year"]:
                frame.at[1, col] = str(frame.at[1, col]).strip().lower() if pd.notna(frame.at[1, col]) else ""
        keep = ["year"] + [
            col for col in frame.columns if col != "year" and len(frame) > 1 and str(frame.at[1, col]) == "total"
        ]
        frame = frame[keep].copy()
        frame = _mitchell_rename_from_row(frame, 0)
        if sheet_name in (6, 7) and "V" in frame.columns:
            frame = frame.rename(columns={"V": "Trinidad"})
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 4:
            for country in ["Barbados", "Jamaica", "Trinidad"]:
                frame = _mitchell_convert_units(frame, country, 1825, 1864, "Th")
        elif sheet_name == 5:
            for country in ["Barbados", "Jamaica", "Trinidad"]:
                frame = _mitchell_convert_units(frame, country, 1865, 1904, "Th")
        elif sheet_name == 6:
            for country in ["Jamaica", "Trinidad"]:
                frame = _mitchell_convert_units(frame, country, 1905, 1949, "Th")
            frame = _mitchell_convert_units(frame, "Barbados", 1905, 1945, "Th")
        elif sheet_name == 7:
            frame = _mitchell_convert_units(frame, "Mexico", 1965, 2010, "B")
            frame = _mitchell_convert_units(frame, "CostaRica", 1985, 2010, "B")
            frame = _mitchell_convert_units(frame, "Canada", 1998, 2010, "B")
            frame = _mitchell_convert_units(frame, "Nicaragua", 1985, 1987, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "govrev"))

    wide = master.pivot(index="year", columns="countryname", values="govrev").reset_index()
    wide.columns.name = None
    for country, end_year, scale in [
        ("Nicaragua", 1912, 0.08),
        ("Nicaragua", 1987, 0.0001),
        ("Nicaragua", 1989, 0.2),
        ("Jamaica", 1968, 2),
        ("Guatemala", 1923, 1 / 60),
    ]:
        wide = _mitchell_convert_currency(wide, country, end_year, scale)
    if "ElSalvador" in wide.columns:
        years = pd.to_numeric(wide["year"], errors="coerce")
        vals = pd.to_numeric(wide["ElSalvador"], errors="coerce")
        wide.loc[years.le(2000), "ElSalvador"] = vals[years.le(2000)] / 8.75
    if "Cuba" in wide.columns:
        years = pd.to_numeric(wide["year"], errors="coerce")
        vals = pd.to_numeric(wide["Cuba"], errors="coerce")
        wide.loc[years.le(1814), "Cuba"] = vals[years.le(1814)] / 1000
    if "USA" in wide.columns:
        years = pd.to_numeric(wide["year"], errors="coerce")
        vals = pd.to_numeric(wide["USA"], errors="coerce")
        wide.loc[years.ge(1960), "USA"] = vals[years.ge(1960)] * 1000
    if "Mexico" in wide.columns:
        wide["Mexico"] = pd.to_numeric(wide["Mexico"], errors="coerce") / 1000
    if "Nicaragua" in wide.columns:
        years = pd.to_numeric(wide["year"], errors="coerce")
        vals = pd.to_numeric(wide["Nicaragua"], errors="coerce")
        wide.loc[years.le(1988), "Nicaragua"] = vals[years.le(1988)] / 1000
        vals = pd.to_numeric(wide["Nicaragua"], errors="coerce")
        wide.loc[years.le(1988), "Nicaragua"] = vals[years.le(1988)] / 1_000_000
        vals = pd.to_numeric(wide["Nicaragua"], errors="coerce")
        wide.loc[years.eq(1988), "Nicaragua"] = vals[years.eq(1988)] / 1000

    out = _mitchell_reshape(wide, "govrev")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "govrev"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Americas_govrev.dta")
    return out


def _mitchell_canada_govexp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Canada_govexp")

    master: pd.DataFrame | None = None
    for sheet_name in (2, 3):
        frame = _mitchell_import_columns_first(path, sheet_name)
        if sheet_name == 3 and "C" in frame.columns:
            frame.loc[frame["C"].astype("string").eq("771l"), "C"] = "771"
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        data_cols = [c for c in frame.columns if c != "year"]
        frame["Canada"] = 0.0
        for col in data_cols:
            series = pd.to_numeric(frame[col], errors="coerce").fillna(0)
            frame["Canada"] = pd.to_numeric(frame["Canada"], errors="coerce").fillna(0) + series
        frame = _mitchell_keep_columns(frame, ["Canada"])
        if sheet_name == 2:
            frame = _mitchell_convert_units(frame, "Canada", 1806, 1839, "Th")
        else:
            frame = _mitchell_convert_units(frame, "Canada", 1840, 1866, "Th")
        master = _mitchell_append(master, _mitchell_reshape(frame, "govexp"))
    out = master[["countryname", "year", "govexp"]].copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out.sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Canada_govexp.dta")
    return out


def _mitchell_canada_govrev(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Canada_govrev")

    master: pd.DataFrame | None = None
    for sheet_name in (2, 3):
        frame = _mitchell_import_columns(path, sheet_name)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        data_cols = [c for c in frame.columns if c != "year"]
        frame["Canada"] = 0.0
        for col in data_cols:
            series = pd.to_numeric(frame[col], errors="coerce").fillna(0)
            frame["Canada"] = pd.to_numeric(frame["Canada"], errors="coerce").fillna(0) + series
        frame = _mitchell_keep_columns(frame, ["Canada"])
        if sheet_name == 2:
            frame = _mitchell_convert_units(frame, "Canada", 1806, 1839, "Th")
        else:
            frame = _mitchell_convert_units(frame, "Canada", 1840, 1866, "Th")
        master = _mitchell_append(master, _mitchell_reshape(frame, "govrev"))
    out = master[["countryname", "year", "govrev"]].copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out.sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Canada_govrev.dta")
    return out


def _mitchell_australia_govexp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Australia_govexp")

    frame = _mitchell_import_columns(path, 2)
    frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
    for col in [c for c in frame.columns if c != "year"]:
        frame = _mitchell_convert_units(frame, col, 1839, 1869, "Th")
    wide = frame.copy()
    data_cols = [c for c in wide.columns if c != "year"]
    wide["Australia"] = 0.0
    for col in data_cols:
        series = pd.to_numeric(wide[col], errors="coerce").fillna(0)
        wide["Australia"] = pd.to_numeric(wide["Australia"], errors="coerce").fillna(0) + series
    wide = _mitchell_keep_columns(wide, ["Australia"])
    out = _mitchell_reshape(wide, "govexp")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "govexp"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Australia_govexp.dta")
    return out


def _mitchell_australia_govrev(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Australia_govrev")

    frame = _mitchell_import_columns(path, 2)
    frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
    data_cols = [c for c in frame.columns if c != "year"]
    frame["Australia"] = 0.0
    for col in data_cols:
        series = pd.to_numeric(frame[col], errors="coerce").fillna(0)
        frame["Australia"] = pd.to_numeric(frame["Australia"], errors="coerce").fillna(0) + series
    frame = _mitchell_convert_units(frame, "Australia", 1820, 1869, "Th")
    frame = _mitchell_keep_columns(frame, ["Australia"])
    out = _mitchell_reshape(frame, "govrev")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "govrev"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Australia_govrev.dta")
    return out


def _mitchell_europe_govtax(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Europe_govrev")

    master: pd.DataFrame | None = None

    sheet2 = _mitchell_import_columns(path, 2)
    sheet2 = _mitchell_keep_columns(sheet2, ["C", "F", "G", "H"])
    sheet2 = _mitchell_destring(_mitchell_drop_blank_year(sheet2))
    sheet2 = _mitchell_select_rowtotals(sheet2, {"Austria": ["C"], "UnitedKingdom": ["F", "G", "H"]})
    sheet2 = _mitchell_convert_units(sheet2, "UnitedKingdom", 1750, 1799, "Th")
    master = _mitchell_append(master, _mitchell_reshape(sheet2, "govtax"))

    for sheet_name in (3, 4, 5, 6):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_keep_non_total_first(frame)
        frame = _mitchell_rename_from_row(frame, 0)
        rename_map: dict[str, str] = {}
        if sheet_name == 3:
            rename_map["S"] = "Russia"
        elif sheet_name == 4:
            rename_map["BC"] = "Russia"
        elif sheet_name == 5:
            rename_map.update({"CA": "Russia", "CX": "Serbia"})
        else:
            rename_map.update({"RussiaUSSR": "Russia", "WestGermany": "Germany"})
        frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 3:
            frame = _mitchell_select_rowtotals(
                frame,
                {
                    "Austria": ["Austria", "D", "E", "F"],
                    "UnitedKingdom": ["UnitedKingdom", "X", "Y"],
                    "Belgium": ["Belgium", "I", "J"],
                    "France": ["France", "M"],
                    "Netherlands": ["Netherlands", "P", "Q"],
                    "Russia": ["RussiaUSSR", "T", "U"],
                },
            )
            frame = _mitchell_convert_units(frame, "UnitedKingdom", 1800, 1809, "Th")
        elif sheet_name == 4:
            frame = _mitchell_select_rowtotals(
                frame,
                {
                    "Austria": ["Austria", "D", "E", "F"],
                    "Belgium": ["Belgium", "I", "J"],
                    "Bulgaria": ["Bulgaria", "M", "N"],
                    "Denmark": ["Denmark", "Q", "R"],
                    "Finland": ["Finland", "U"],
                    "France": ["France", "X", "Y", "Z"],
                    "Germany": ["Germany", "AC"],
                    "Greece": ["Greece", "AF", "AG"],
                    "Hungary": ["Hungary", "AJ"],
                    "Italy": ["Italy", "AM", "AN"],
                    "Netherlands": ["Netherlands", "AQ", "AR"],
                    "Norway": ["Norway", "AU", "AV"],
                    "Portugal": ["Portugal", "AY", "AZ"],
                    "Romania": ["Romania"],
                    "Serbia": ["Serbia", "BJ", "BK"],
                    "Spain": ["Spain", "BN", "BO", "BP"],
                    "Sweden": ["Sweden", "BS", "BT"],
                    "Russia": ["Russia", "RussiaUSSR", "BF"],
                    "Switzerland": ["Switzerland"],
                    "UnitedKingdom": ["UnitedKingdom", "X", "Y"],
                },
            )
        elif sheet_name == 5:
            frame = _mitchell_select_rowtotals(
                frame,
                {
                    "Austria": ["Austria", "D", "E", "F"],
                    "Belgium": ["Belgium", "I", "J"],
                    "Bulgaria": ["Bulgaria", "M", "N"],
                    "Czechoslovakia": ["Czechoslovakia", "Q", "R", "S"],
                    "Denmark": ["Denmark", "V", "W"],
                    "Finland": ["Finland", "Z", "AA", "AB"],
                    "France": ["France", "AE", "AF", "AG", "AH"],
                    "Germany": ["Germany", "AK", "AL", "AM"],
                    "Greece": ["Greece", "AP", "AQ"],
                    "Hungary": ["Hungary", "AT", "AU", "AV"],
                    "Ireland": ["Ireland", "AY", "AZ"],
                    "Italy": ["Italy", "BC", "BD", "BE"],
                    "Netherlands": ["Netherlands", "BH", "BI", "BJ"],
                    "Norway": ["Norway", "BM", "BN"],
                    "Poland": ["Poland", "BQ", "BR"],
                    "Portugal": ["Portugal", "BU", "BV"],
                    "Romania": ["Romania", "BY"],
                    "Russia": ["RussiaUSSR", "CB", "CC", "CD"],
                    "Serbia": ["Serbia", "CY", "CZ"],
                    "Spain": ["Spain", "CG", "CH", "CI"],
                    "Sweden": ["Sweden", "CL", "CM", "CN"],
                    "Switzerland": ["Switzerland", "CQ", "CR"],
                    "UnitedKingdom": ["UnitedKingdom", "CU", "CV"],
                },
            )
            for country, start, end in [("Serbia", 1942, 1949), ("France", 1945, 1949), ("Italy", 1946, 1949)]:
                frame = _mitchell_convert_units(frame, country, start, end, "B")
        else:
            frame = _mitchell_select_rowtotals(
                frame,
                {
                    "Austria": ["Austria", "D", "E", "F"],
                    "Belgium": ["Belgium", "I", "J"],
                    "Denmark": ["Denmark", "M", "N", "O"],
                    "Finland": ["Finland", "R", "S", "T"],
                    "France": ["France", "W", "X", "Y", "Z"],
                    "Germany": ["Germany", "AC", "AD", "AE"],
                    "Greece": ["Greece", "AH", "AI", "AJ"],
                    "Ireland": ["Ireland", "AM", "AN", "AO"],
                    "Italy": ["Italy", "AR", "AS", "AT", "AU", "AV"],
                    "Netherlands": ["Netherlands", "AY", "AZ", "BA"],
                    "Norway": ["Norway", "BD", "BE"],
                    "Portugal": ["Portugal", "BH", "BI", "BJ"],
                    "Romania": ["Romania"],
                    "Russia": ["Russia", "BO", "BP"],
                    "Spain": ["Spain", "BS", "BT", "BU"],
                    "Sweden": ["Sweden", "BX", "BY", "BZ"],
                    "Switzerland": ["Switzerland", "CC", "CD"],
                    "UnitedKingdom": ["UnitedKingdom", "CG", "CH", "CI"],
                },
            )
            for country in ["Austria", "Belgium", "Finland", "France", "Germany", "Italy", "Romania", "Spain"]:
                frame = _mitchell_convert_units(frame, country, 1950, 2010, "B")
            for country in ["Denmark", "Greece"]:
                frame = _mitchell_convert_units(frame, country, 1970, 2010, "B")
            for country in ["Sweden", "Portugal", "Norway", "Netherlands", "Italy"]:
                frame = _mitchell_convert_units(frame, country, 1975, 2010, "B")
            frame = _mitchell_convert_units(frame, "UnitedKingdom", 1980, 2010, "B")
        part = _mitchell_reshape(frame, "govtax")
        if sheet_name in (3, 4, 6):
            part.loc[pd.to_numeric(part["govtax"], errors="coerce").eq(0), "govtax"] = np.nan
        master = _mitchell_append(master, part)

    wide = master.pivot(index="year", columns="countryname", values="govtax").reset_index()
    wide.columns.name = None
    for country, end_year, scale in [
        ("Austria", 1892, 2),
        ("Hungary", 1892, 2),
        ("Russia", 1839, 1 / 4),
        ("Austria", 1923, 1 / 10000),
        ("Hungary", 1924, 1 / 12500),
        ("Russia", 1939, 1 / 10000),
        ("France", 1959, 1 / 100),
        ("Bulgaria", 1959, 1 / 1_000_000),
    ]:
        wide = _mitchell_convert_currency(wide, country, end_year, scale)

    out = _mitchell_reshape(wide, "govtax")
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Finland") & years.le(1962), "govtax"] = values[country.eq("Finland") & years.le(1962)] / 100
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Greece") & years.le(1940), "govtax"] = values[country.eq("Greece") & years.le(1940)] * (10 ** -6) / 5
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Romania"), "govtax"] = values[country.eq("Romania")] / 10000
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Romania") & years.le(1943), "govtax"] = values[country.eq("Romania") & years.le(1943)] / 10000
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Italy") & years.ge(1999), "govtax"] = values[country.eq("Italy") & years.ge(1999)] / 1000
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Germany") & years.le(1923), "govtax"] = values[country.eq("Germany") & years.le(1923)] / (10 ** 12)
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Russia"), "govtax"] = values[country.eq("Russia")] * 1000
    out.loc[country.eq("Portugal") & years.ge(1999), "govtax"] = np.nan
    out.loc[pd.to_numeric(out["govtax"], errors="coerce").eq(0), "govtax"] = np.nan
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "govtax"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Europe_govtax.dta")
    return out


def _mitchell_asia_govtax(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Asia_govrev")

    master: pd.DataFrame | None = None

    sheet2 = _mitchell_import_columns_first(path, 2)
    sheet2 = _mitchell_keep_non_total_first(sheet2)
    sheet2 = _mitchell_rename_from_row(sheet2, 0)
    sheet2 = _mitchell_destring(_mitchell_drop_blank_year(sheet2))
    if "Indonesia" in sheet2.columns and "E" in sheet2.columns:
        indo = pd.to_numeric(sheet2["Indonesia"], errors="coerce")
        extra = pd.to_numeric(sheet2["E"], errors="coerce")
        sheet2["Indonesia"] = indo.add(extra, fill_value=np.nan)
        sheet2 = sheet2.drop(columns=["E"])
    master = _mitchell_append(master, _mitchell_reshape(sheet2, "govtax"))

    sheet3 = _mitchell_import_columns_first(path, 3)
    sheet3 = _mitchell_keep_non_total_first(sheet3)
    sheet3 = _mitchell_rename_from_row(sheet3, 0)
    sheet3 = _mitchell_destring(_mitchell_drop_blank_year(sheet3))
    if "India" in sheet3.columns:
        india = pd.to_numeric(sheet3["India"], errors="coerce")
        out_series = india.copy()
        if "E" in sheet3.columns:
            out_series = out_series + pd.to_numeric(sheet3["E"], errors="coerce").notna().astype(float)
        for col in ["F", "G", "H"]:
            if col in sheet3.columns:
                out_series = out_series.add(pd.to_numeric(sheet3[col], errors="coerce"), fill_value=np.nan)
        sheet3["India"] = out_series
    sheet3 = _mitchell_keep_columns(sheet3, ["India"])
    master = _mitchell_append(master, _mitchell_reshape(sheet3, "govtax"))

    for sheet_name, groups in [
        (4, {"Indonesia": ["Indonesia", "E"], "Japan": ["Japan", "H", "I", "J", "K"], "Thailand": ["Thailand", "S"]}),
        (5, {"India": ["India", "E", "F", "G", "H"], "Indonesia": ["Indonesia", "L"], "Japan": ["Japan", "Q", "R", "S", "T"], "Korea": ["Korea", "W", "X", "Y"], "Thailand": ["Thailand", "AJ"]}),
        (6, {"India": ["India", "H", "I"], "Indonesia": ["Indonesia", "L", "M"], "Iran": ["Iran", "P", "Q"], "Japan": ["Japan", "V", "W", "X"], "SouthKorea": ["SouthKorea", "AC", "AD"]}),
        (7, {"Pakistan": ["Pakistan", "D", "E"], "Phillippines": ["Phillippines", "H", "I"], "Thailand": ["Thailand", "P", "Q"], "Turkey": ["Turkey", "T", "U"]}),
    ]:
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_keep_non_total_first(frame)
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        frame = _mitchell_select_rowtotals(frame, groups)
        if sheet_name == 6:
            frame = _mitchell_convert_units(frame, "India", 1960, 2010, "B")
        elif sheet_name == 7:
            for country in ["Pakistan", "Phillippines"]:
                frame = _mitchell_convert_units(frame, country, 1970, 2010, "B")
            frame = _mitchell_convert_units(frame, "Thailand", 1962, 2010, "B")
            frame = _mitchell_convert_units(frame, "Turkey", 1959, 1988, "B")
            frame = _mitchell_convert_units(frame, "Turkey", 1989, 2010, "Tri")
        part = _mitchell_reshape(frame, "govtax")
        part.loc[pd.to_numeric(part["govtax"], errors="coerce").eq(0), "govtax"] = np.nan
        master = _mitchell_append(master, part)

    wide = master.pivot(index="year", columns="countryname", values="govtax").reset_index()
    wide.columns.name = None
    wide = _mitchell_convert_currency(wide, "Indonesia", 1964, 1 / 1000)
    wide = _mitchell_convert_units(wide, "Iran", 1945, 2010, "B")

    out = _mitchell_reshape(wide, "govtax")
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Japan") & years.ge(1965), "govtax"] = values[country.eq("Japan") & years.ge(1965)] * 1000
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Japan") & years.ge(1945), "govtax"] = values[country.eq("Japan") & years.ge(1945)] * 1000
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("SouthKorea"), "govtax"] = values[country.eq("SouthKorea")] * (10 ** 3)
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Taiwan") & years.le(1939), "govtax"] = values[country.eq("Taiwan") & years.le(1939)] * (10 ** -4)
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Taiwan") & years.le(1939), "govtax"] = values[country.eq("Taiwan") & years.le(1939)] / 4
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "govtax"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Asia_govtax.dta")
    return out


def _mitchell_oceania_govtax(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Oceania_govrev")

    sheet3 = _mitchell_import_columns_first(path, 3)
    sheet3 = _mitchell_keep_columns(sheet3, ["G", "H"])
    sheet3 = _mitchell_destring(_mitchell_drop_blank_year(sheet3))
    sheet3 = _mitchell_convert_units(sheet3, "G", 1870, 1899, "Th")
    sheet3 = _mitchell_convert_units(sheet3, "H", 1870, 1899, "Th")
    sheet3 = _mitchell_select_rowtotals(sheet3, {"NewZealand": ["G", "H"]})
    master: pd.DataFrame | None = _mitchell_reshape(sheet3, "govtax")

    for sheet_name, groups in [
        (4, {"Australia": ["Australia", "D", "E"], "NewZealand": ["NewZealand", "J", "K"]}),
        (5, {"Australia": ["Australia", "D", "E", "F"], "NewZealand": ["NewZealand", "K", "L"]}),
    ]:
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_keep_non_total_first(frame)
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        frame = _mitchell_select_rowtotals(frame, groups)
        if sheet_name == 4:
            frame = _mitchell_convert_units(frame, "NewZealand", 1900, 1919, "Th")
        else:
            frame = _mitchell_convert_units(frame, "Australia", 1975, 2010, "B")
        part = _mitchell_reshape(frame, "govtax")
        part.loc[pd.to_numeric(part["govtax"], errors="coerce").eq(0), "govtax"] = np.nan
        master = _mitchell_append(master, part)

    wide = master.pivot(index="year", columns="countryname", values="govtax").reset_index()
    wide.columns.name = None
    wide = _mitchell_convert_currency(wide, "Australia", 1964, 2)
    wide = _mitchell_convert_currency(wide, "NewZealand", 1964, 2)
    out = _mitchell_reshape(wide, "govtax")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "govtax"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Oceania_govtax.dta")
    return out


def _mitchell_latam_govtax(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Latam_govrev")

    master: pd.DataFrame | None = None
    for sheet_name, groups in [
        (4, {"Argentina": ["Argentina", "C"], "Brazil": ["Brazil", "G", "H"], "Chile": ["Chile", "K"], "Colombia": ["Colombia", "N"], "Peru": ["Peru", "T"], "Uruguay": ["Uruguay", "X"], "Venezuela": ["Venezuela", "AA", "AB"]}),
        (5, {"Argentina": ["Argentina", "C", "D"], "Brazil": ["Brazil", "H", "I"], "Chile": ["Chile", "L"], "Colombia": ["Colombia", "O", "P"], "Peru": ["Peru", "V"], "Uruguay": ["Uruguay", "Z", "AA"], "Venezuela": ["Venezuela", "AD", "AE"]}),
    ]:
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_keep_non_total_first(frame)
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        frame = _mitchell_select_rowtotals(frame, groups)
        if sheet_name == 5:
            for country, start, end, scale in [
                ("Argentina", 1955, 1969, "B"),
                ("Argentina", 1970, 1978, "B"),
                ("Argentina", 1979, 1984, "Tri"),
                ("Argentina", 1985, 1988, "B"),
                ("Brazil", 1955, 1966, "B"),
                ("Brazil", 1970, 1988, "B"),
                ("Brazil", 1990, 1992, "Th"),
                ("Chile", 1950, 1954, "B"),
                ("Chile", 1970, 2010, "B"),
                ("Colombia", 1975, 2010, "B"),
                ("Uruguay", 1965, 1974, "B"),
                ("Uruguay", 1980, 2010, "B"),
                ("Venezuela", 1975, 2010, "B"),
            ]:
                frame = _mitchell_convert_units(frame, country, start, end, scale)
        part = _mitchell_reshape(frame, "govtax")
        part.loc[pd.to_numeric(part["govtax"], errors="coerce").eq(0), "govtax"] = np.nan
        master = _mitchell_append(master, part)

    wide = master.pivot(index="year", columns="countryname", values="govtax").reset_index()
    wide.columns.name = None
    for country, end_year, scale in [
        ("Argentina", 1969, 1 / 100),
        ("Argentina", 1984, 1 / 1000),
        ("Argentina", 1989, 1 / 10000),
        ("Brazil", 1966, 1 / 1000),
        ("Brazil", 1984, 1 / 1000),
        ("Brazil", 1988, 1 / 2750),
        ("Brazil", 1989, 1 / 1_000_000),
    ]:
        wide = _mitchell_convert_currency(wide, country, end_year, scale)

    out = _mitchell_reshape(wide, "govtax")
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Chile") & years.le(1974), "govtax"] = values[country.eq("Chile") & years.le(1974)] / 1000
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Chile") & years.le(1954), "govtax"] = values[country.eq("Chile") & years.le(1954)] / 1000
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Ecuador") & years.le(1993), "govtax"] = values[country.eq("Ecuador") & years.le(1993)] / 2500
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Uruguay"), "govtax"] = values[country.eq("Uruguay")] / 1000
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Uruguay") & years.le(1974), "govtax"] = values[country.eq("Uruguay") & years.le(1974)] / 1000
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Peru") & years.le(1964), "govtax"] = values[country.eq("Peru") & years.le(1964)] / 1000
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Peru") & years.le(1989), "govtax"] = values[country.eq("Peru") & years.le(1989)] / 1000
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Peru") & years.le(1984), "govtax"] = values[country.eq("Peru") & years.le(1984)] / 1000
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[country.eq("Venezuela"), "govtax"] = values[country.eq("Venezuela")] / (10 ** 14)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "govtax"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Latam_govtax.dta")
    return out


def _mitchell_americas_govtax(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Americas_govrev")

    master: pd.DataFrame | None = None
    for sheet_name, groups in [
        (3, {"USA": ["USA", "D"]}),
        (4, {"USA": ["USA", "H"]}),
        (5, {"USA": ["USA", "Q"], "Canada": ["Canada", "D"]}),
        (6, {"USA": ["USA", "X", "Y", "Z"], "Canada": ["Canada", "D", "E", "F"], "Mexico": ["Mexico", "Q"]}),
        (7, {"USA": ["USA", "X", "Y", "Z"], "Canada": ["Canada", "D", "E", "F"], "Mexico": ["Mexico", "Q"]}),
    ]:
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_keep_non_total_first(frame)
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        frame = _mitchell_select_rowtotals(frame, groups)
        if sheet_name == 7:
            frame = _mitchell_convert_units(frame, "Mexico", 1965, 2010, "B")
            frame = _mitchell_convert_units(frame, "Canada", 1998, 2010, "B")
            frame = _mitchell_convert_units(frame, "USA", 1960, 2010, "B")
        part = _mitchell_reshape(frame, "govtax")
        part.loc[pd.to_numeric(part["govtax"], errors="coerce").eq(0), "govtax"] = np.nan
        master = _mitchell_append(master, part)

    out = master.copy()
    values = pd.to_numeric(out["govtax"], errors="coerce")
    out.loc[out["countryname"].eq("Mexico"), "govtax"] = values[out["countryname"].eq("Mexico")] / 1000
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "govtax"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Americas_govtax.dta")
    return out


def _mitchell_partial_govtax_final(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    parts = [
        _mitchell_europe_govtax(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_asia_govtax(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_oceania_govtax(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_latam_govtax(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_americas_govtax(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
    ]
    return _mitchell_finalize_value_parts(parts, "govtax", data_helper_dir=data_helper_dir, euro_cutoff_year=1999)


def _mitchell_europe_money_supply(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Europe_money_supply")

    m1 = _mitchell_import_columns(path, 2)
    m1 = _mitchell_keep_by_header(
        m1,
        header_row=1,
        predicate=lambda value: value.strip().lower() == "m1",
        normalizer=lambda value: value.strip().lower(),
    )
    m1 = m1.rename(columns={"UK": "UnitedKingdom", "WestGermany": "Germany"})
    m1 = _mitchell_destring(_mitchell_drop_blank_year(m1))
    m1 = _mitchell_convert_units(m1, "Italy", 1950, 2010, "B")
    m1 = _mitchell_convert_units(m1, "Ireland", 1950, 2010, "Th")
    for country in [c for c in m1.columns if c != "year"]:
        m1[country] = pd.to_numeric(m1[country], errors="coerce") * 1000
    master = _mitchell_reshape(m1, "M1")

    m2 = _mitchell_import_columns_first(path, 2)
    m2 = _mitchell_fill_header_rows(m2, 2)
    m2 = _mitchell_keep_by_header(
        m2,
        header_row=2,
        predicate=lambda value: value.strip().lower() == "m2",
        normalizer=lambda value: value.strip().lower(),
    )
    m2 = _mitchell_rename_from_row(m2, 0)
    m2 = m2.rename(columns={"M": "Germany", "AG": "UnitedKingdom", "WestGermany": "Germany", "UK": "UnitedKingdom"})
    m2 = _mitchell_destring(_mitchell_drop_blank_year(m2))
    m2 = _mitchell_convert_units(m2, "Italy", 1950, 2010, "B")
    m2 = _mitchell_convert_units(m2, "Ireland", 1950, 2010, "Th")
    for country in [c for c in m2.columns if c != "year"]:
        m2[country] = pd.to_numeric(m2[country], errors="coerce") * 1000
    m2 = _mitchell_reshape(m2, "M2")

    out = m2.merge(master, on=["countryname", "year"], how="outer", indicator=True)
    if not out["_merge"].eq("both").all():
        raise ValueError("Americas_money_supply M1/M2 merge is not 1:1 with full matches (assert(3) failed).")
    out = out.drop(columns=["_merge"])
    years = pd.to_numeric(out["year"], errors="coerce")
    for col in ["M1", "M2"]:
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[(out["countryname"].isin(["Belgium", "France", "Greece", "Italy"])) & years.ge(1999), col] = (
            values[(out["countryname"].isin(["Belgium", "France", "Greece", "Italy"])) & years.ge(1999)] / 1000
        )
    out.loc[out["countryname"].eq("Bulgaria"), "M2"] = pd.to_numeric(
        out.loc[out["countryname"].eq("Bulgaria"), "M2"], errors="coerce"
    ) / (10 ** 6)
    out["year"] = years.astype("int32")
    out = out[["countryname", "year", "M1", "M2"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Europe_money_supply.dta")
    return out


def _mitchell_asia_money_supply(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Asia_money_supply")

    def _convert_asia_money(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for country in [
            "Afghanistan",
            "Bangladesh",
            "China",
            "India",
            "Japan",
            "Iran",
            "Pakistan",
            "Philippines",
            "SaudiArabia",
            "SouthKorea",
            "Taiwan",
            "UnitedArabEmirates",
            "SouthVietnam",
            "Thailand",
            "Turkey",
        ]:
            out = _mitchell_convert_units(out, country, 1948, 2010, "B")
        for country in ["Israel", "Lebanon", "Malaysia", "Singapore", "SriLanka", "Syria"]:
            out = _mitchell_convert_units(out, country, 1965, 2010, "B")
        for country in ["Myanmar", "Nepal", "Qatar"]:
            out = _mitchell_convert_units(out, country, 1980, 2010, "B")
        out = _mitchell_convert_units(out, "Japan", 1960, 2010, "B")
        out = _mitchell_convert_units(out, "Pakistan", 1965, 2010, "B")
        out = _mitchell_convert_units(out, "Indonesia", 1950, 2010, "B")
        out = _mitchell_convert_currency(out, "Indonesia", 1962, 1 / 1000)
        out = _mitchell_convert_currency(out, "Turkey", 1985, 2010)
        return out

    m1 = _mitchell_import_columns(path, 2)
    m1 = _mitchell_keep_by_header(
        m1,
        header_row=1,
        predicate=lambda value: value.strip().lower() == "m1",
        normalizer=lambda value: value.strip().lower(),
    )
    m1 = _mitchell_destring(_mitchell_drop_blank_year(m1))
    m1 = _convert_asia_money(m1)
    master = _mitchell_reshape(m1, "M1")

    m2 = _mitchell_import_columns_first(path, 2)
    m2 = _mitchell_fill_header_rows(m2, 2)
    m2 = _mitchell_keep_by_header(
        m2,
        header_row=2,
        predicate=lambda value: value.strip().lower() == "m2",
        normalizer=lambda value: value.strip().lower(),
    )
    m2 = _mitchell_rename_from_row(m2, 0)
    m2 = _mitchell_destring(_mitchell_drop_blank_year(m2))
    m2 = _convert_asia_money(m2)
    m2 = _mitchell_reshape(m2, "M2")

    out = m2.merge(master, on=["countryname", "year"], how="outer")
    years = pd.to_numeric(out["year"], errors="coerce")
    for country in ["Lebanon", "Malaysia", "Singapore", "SriLanka", "Syria"]:
        mask = out["countryname"].eq(country) & years.between(1965, 1974)
        for col in ["M1", "M2"]:
            out.loc[mask, col] = pd.to_numeric(out.loc[mask, col], errors="coerce") / 1000
    for col in ["M1", "M2"]:
        values = pd.to_numeric(out[col], errors="coerce")
        tur_mask = out["countryname"].eq("Turkey")
        out.loc[tur_mask & years.le(1985), col] = values[tur_mask & years.le(1985)] / 1_000_000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[tur_mask, col] = values[tur_mask] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        isr_mask = out["countryname"].eq("Israel")
        out.loc[isr_mask, col] = values[isr_mask] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[isr_mask & years.le(1979), col] = values[isr_mask & years.le(1979)] / 10
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[isr_mask & years.between(1965, 1974), col] = values[isr_mask & years.between(1965, 1974)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        pak_mask = out["countryname"].eq("Pakistan")
        out.loc[pak_mask, col] = values[pak_mask] / 1000
    for col in ["M1", "M2"]:
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[out["countryname"].eq("SouthKorea") & years.ge(1980), col] = values[
            out["countryname"].eq("SouthKorea") & years.ge(1980)
        ] * 1000
    out["year"] = years.astype("int32")
    out = out[["countryname", "year", "M1", "M2"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Asia_money_supply.dta")
    return out


def _mitchell_oceania_money_supply(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Oceania_money_supply")

    m1 = _mitchell_import_columns(path, 2)
    m1 = _mitchell_keep_by_header(
        m1,
        header_row=1,
        predicate=lambda value: value.strip().lower() == "m1",
        normalizer=lambda value: value.strip().lower(),
    )
    m1 = m1.rename(columns={"WesternSamoa": "Samoa"})
    m1 = _mitchell_destring(_mitchell_drop_blank_year(m1))
    m1 = _mitchell_convert_units(m1, "Australia", 1960, 2010, "B")
    m1 = _mitchell_convert_units(m1, "NewZealand", 1980, 2010, "B")
    master = _mitchell_reshape(m1, "M1")

    m2 = _mitchell_import_columns_first(path, 2)
    m2 = _mitchell_fill_header_rows(m2, 2)
    m2 = _mitchell_keep_by_header(
        m2,
        header_row=2,
        predicate=lambda value: value.strip().lower() == "m2",
        normalizer=lambda value: value.strip().lower(),
    )
    m2 = _mitchell_rename_from_row(m2, 0)
    m2 = m2.rename(columns={"WesternSamoa": "Samoa"})
    m2 = _mitchell_destring(_mitchell_drop_blank_year(m2))
    m2 = _mitchell_convert_units(m2, "Australia", 1960, 2010, "B")
    m2 = _mitchell_convert_units(m2, "NewZealand", 1980, 2010, "B")
    m2 = _mitchell_reshape(m2, "M2")

    out = m2.merge(master, on=["countryname", "year"], how="outer")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "M1", "M2"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Oceania_money_supply.dta")
    return out


def _mitchell_africa_money_supply(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Africa_money_supply")

    def _convert_africa_money(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for country in [
            "Algeria",
            "Benin",
            "BurkinaFaso",
            "Burundi",
            "Cameroon",
            "CentralAfricanRepublic",
            "Chad",
            "Congo",
            "Gabon",
            "IvoryCoast",
            "Madagascar",
            "Mali",
            "Morocco",
            "Niger",
            "Senegal",
            "Tanzania",
            "Togo",
        ]:
            out = _mitchell_convert_units(out, country, 1948, 2010, "B")
        for country in ["Egypt", "Ghana", "Kenya", "Mauritania", "Nigeria", "Rwanda", "Somalia", "SouthAfrica", "Zaire"]:
            out = _mitchell_convert_units(out, country, 1980, 2010, "B")
        return out

    m1 = _mitchell_import_columns(path, 2)
    m1 = _mitchell_keep_by_header(
        m1,
        header_row=1,
        predicate=lambda value: value.strip().lower() == "m1",
        normalizer=lambda value: value.strip().lower(),
    )
    m1 = _mitchell_destring(_mitchell_drop_blank_year(m1))
    m1 = _convert_africa_money(m1)
    master = _mitchell_reshape(m1, "M1")

    m2 = _mitchell_import_columns_first(path, 2)
    m2 = _mitchell_fill_header_rows(m2, 2)
    m2 = _mitchell_keep_by_header(
        m2,
        header_row=2,
        predicate=lambda value: value.strip().lower() == "m2",
        normalizer=lambda value: value.strip().lower(),
    )
    m2 = _mitchell_rename_from_row(m2, 0)
    m2 = _mitchell_destring(_mitchell_drop_blank_year(m2))
    m2 = _convert_africa_money(m2)
    m2 = _mitchell_reshape(m2, "M2")

    out = m2.merge(master, on=["countryname", "year"], how="outer")
    years = pd.to_numeric(out["year"], errors="coerce")
    for col in ["M1", "M2"]:
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[out["countryname"].eq("Ghana"), col] = values[out["countryname"].eq("Ghana")] / 10000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[out["countryname"].eq("Madagascar") & years.le(2000), col] = values[out["countryname"].eq("Madagascar") & years.le(2000)] / 5
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[out["countryname"].eq("Mauritania"), col] = values[out["countryname"].eq("Mauritania")] / 10
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[out["countryname"].eq("Sudan") & years.le(1991), col] = values[out["countryname"].eq("Sudan") & years.le(1991)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[out["countryname"].eq("Uganda") & years.ge(2000), col] = values[out["countryname"].eq("Uganda") & years.ge(2000)] * 1000
        values = pd.to_numeric(out[col], errors="coerce")
        zaire_mask = out["countryname"].eq("Zaire")
        out.loc[zaire_mask & years.le(1995), col] = values[zaire_mask & years.le(1995)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[zaire_mask, col] = values[zaire_mask] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[zaire_mask & years.le(1993), col] = values[zaire_mask & years.le(1993)] / 100
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[zaire_mask & years.le(1991), col] = values[zaire_mask & years.le(1991)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[zaire_mask & years.le(1987), col] = values[zaire_mask & years.le(1987)] / 3
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[out["countryname"].eq("Zambia") & years.gt(2000), col] = values[out["countryname"].eq("Zambia") & years.gt(2000)] * 1000
    out["year"] = years.astype("int32")
    out = out[["countryname", "year", "M1", "M2"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Africa_money_supply.dta")
    return out


def _mitchell_americas_money_supply(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Americas_money_supply")

    def _convert_americas_money(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for country in ["Canada", "Mexico", "USA"]:
            out = _mitchell_convert_units(out, country, 1948, 2010, "B")
        out = _mitchell_convert_units(out, "CostaRica", 1981, 2010, "B")
        out = _mitchell_convert_currency(out, "Nicaragua", 1983, 1 / 1000)
        out = _mitchell_convert_currency(out, "Nicaragua", 1989, 1 / 5)
        out = _mitchell_convert_currency(out, "ElSalvador", 1999, 1 / 8.75)
        return out

    m1 = _mitchell_import_columns(path, 2)
    m1 = _mitchell_keep_by_header(
        m1,
        header_row=1,
        predicate=lambda value: value.strip().lower() == "m1",
        normalizer=lambda value: value.strip().lower(),
    )
    m1 = m1.rename(columns={"TrinidadTobago": "Trinidad"})
    m1 = _mitchell_destring(_mitchell_drop_blank_year(m1))
    m1 = _convert_americas_money(m1)
    master = _mitchell_reshape(m1, "M1")

    m2 = _mitchell_import_columns_first(path, 2)
    m2 = _mitchell_fill_header_rows(m2, 2)
    m2 = _mitchell_keep_by_header(
        m2,
        header_row=2,
        predicate=lambda value: value.strip().lower() == "m2",
        normalizer=lambda value: value.strip().lower(),
    )
    m2 = _mitchell_rename_from_row(m2, 0)
    m2 = m2.rename(columns={"Y": "Trinidad", "TrinidadTobago": "Trinidad"})
    m2 = _mitchell_destring(_mitchell_drop_blank_year(m2))
    m2 = _convert_americas_money(m2)
    m2 = _mitchell_reshape(m2, "M2")

    out = m2.merge(master, on=["countryname", "year"], how="outer")
    years = pd.to_numeric(out["year"], errors="coerce")
    for col in ["M1", "M2"]:
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[out["countryname"].eq("Nicaragua") & years.le(1987), col] = values[out["countryname"].eq("Nicaragua") & years.le(1987)] / 1_000_000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[out["countryname"].eq("Mexico"), col] = values[out["countryname"].eq("Mexico")] / 1000
    out["year"] = years.astype("int32")
    out = out[["countryname", "year", "M1", "M2"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Americas_money_supply.dta")
    return out


def _mitchell_latam_money_supply(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Latam_money_supply")

    def _convert_latam_money(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if "Argentina" in out.columns:
            years = pd.to_numeric(out["year"], errors="coerce")
            vals = pd.to_numeric(out["Argentina"], errors="coerce")
            out.loc[years.le(1969), "Argentina"] = vals[years.le(1969)] / 100
            vals = pd.to_numeric(out["Argentina"], errors="coerce")
            out.loc[years.le(1982), "Argentina"] = vals[years.le(1982)] / 10000
            vals = pd.to_numeric(out["Argentina"], errors="coerce")
            out.loc[years.le(1984), "Argentina"] = vals[years.le(1984)] / 1000
            vals = pd.to_numeric(out["Argentina"], errors="coerce")
            out.loc[years.le(1988), "Argentina"] = vals[years.le(1988)] / 10000
            out = _mitchell_convert_units(out, "Argentina", 1948, 1975, "B")
            out = _mitchell_convert_units(out, "Argentina", 1976, 1982, "Tri")
            out = _mitchell_convert_units(out, "Argentina", 1983, 1988, "B")
        if "Bolivia" in out.columns:
            out = _mitchell_convert_units(out, "Bolivia", 1948, 1962, "B")
            vals = pd.to_numeric(out["Bolivia"], errors="coerce")
            years = pd.to_numeric(out["year"], errors="coerce")
            out.loc[years.le(1962), "Bolivia"] = vals[years.le(1962)] / 1000
            out = _mitchell_convert_units(out, "Bolivia", 1975, 1984, "B")
            out = _mitchell_convert_units(out, "Bolivia", 1985, 1986, "Tri")
            vals = pd.to_numeric(out["Bolivia"], errors="coerce")
            out.loc[years.le(1986), "Bolivia"] = vals[years.le(1986)] / 1_000_000
            out = _mitchell_convert_units(out, "Bolivia", 1955, 1982, "B")
            out = _mitchell_convert_units(out, "Bolivia", 1983, 2010, "Tri")
        if "Brazil" in out.columns:
            years = pd.to_numeric(out["year"], errors="coerce")
            vals = pd.to_numeric(out["Brazil"], errors="coerce")
            out.loc[years.le(1965), "Brazil"] = vals[years.le(1965)] / 1000
            out = _mitchell_convert_units(out, "Brazil", 1984, 1985, "B")
            vals = pd.to_numeric(out["Brazil"], errors="coerce")
            out.loc[years.le(1985), "Brazil"] = vals[years.le(1985)] / 2750
            vals = pd.to_numeric(out["Brazil"], errors="coerce")
            out.loc[years.le(1939), "Brazil"] = vals[years.le(1939)] / 1000
            vals = pd.to_numeric(out["Brazil"], errors="coerce")
            out.loc[years.le(1966), "Brazil"] = vals[years.le(1966)] / 1000
            vals = pd.to_numeric(out["Brazil"], errors="coerce")
            out.loc[years.le(1993), "Brazil"] = vals[years.le(1993)] / 2750
            out = _mitchell_convert_units(out, "Brazil", 1940, 1974, "B")
            out = _mitchell_convert_units(out, "Brazil", 1940, 1974, "B")
        if "Venezuela" in out.columns:
            years = pd.to_numeric(out["year"], errors="coerce")
            vals = pd.to_numeric(out["Venezuela"], errors="coerce")
            out.loc[years.ge(1965), "Venezuela"] = vals[years.ge(1965)] * 1000
        if "Chile" in out.columns:
            years = pd.to_numeric(out["year"], errors="coerce")
            vals = pd.to_numeric(out["Chile"], errors="coerce")
            out.loc[years.le(1954), "Chile"] = vals[years.le(1954)] / 1000
            out = _mitchell_convert_units(out, "Chile", 1970, 1975, "B")
            vals = pd.to_numeric(out["Chile"], errors="coerce")
            out.loc[years.le(1975), "Chile"] = vals[years.le(1975)] / 1000
            out = _mitchell_convert_units(out, "Chile", 1983, 2010, "B")
        out = _mitchell_convert_units(out, "Colombia", 1975, 2010, "B")
        out = _mitchell_convert_units(out, "Ecuador", 1983, 2000, "B")
        if "Ecuador" in out.columns:
            years = pd.to_numeric(out["year"], errors="coerce")
            vals = pd.to_numeric(out["Ecuador"], errors="coerce")
            out.loc[years.ge(2001), "Ecuador"] = vals[years.ge(2001)] * 25000
        if "Paraguay" in out.columns:
            years = pd.to_numeric(out["year"], errors="coerce")
            vals = pd.to_numeric(out["Paraguay"], errors="coerce")
            out.loc[years.le(1942), "Paraguay"] = vals[years.le(1942)] / 100
            out = _mitchell_convert_units(out, "Paraguay", 1975, 2000, "B")
        if "Peru" in out.columns:
            years = pd.to_numeric(out["year"], errors="coerce")
            vals = pd.to_numeric(out["Peru"], errors="coerce")
            out.loc[years.le(1929), "Peru"] = vals[years.le(1929)] / 1000
        for country in [c for c in out.columns if c != "year"]:
            out[country] = pd.to_numeric(out[country], errors="coerce") * 1000
        return out

    m1 = _mitchell_import_columns(path, 2)
    m1 = _mitchell_keep_by_header(
        m1,
        header_row=1,
        predicate=lambda value: value.strip().lower() == "m1",
        normalizer=lambda value: value.strip().lower(),
    )
    m1 = _mitchell_use_overlapping_data(m1)
    m1 = m1.loc[pd.to_numeric(m1["year"], errors="coerce").notna()].copy()
    m1 = _convert_latam_money(m1)
    master = _mitchell_reshape(m1, "M1")

    m2 = _mitchell_import_columns_first(path, 2)
    m2 = _mitchell_fill_header_rows(m2, 2)
    m2 = _mitchell_keep_by_header(
        m2,
        header_row=2,
        predicate=lambda value: value.strip().lower() == "m2",
        normalizer=lambda value: value.strip().lower(),
    )
    m2 = _mitchell_rename_from_row(m2, 0)
    m2 = _mitchell_use_overlapping_data(m2)
    m2 = m2.loc[pd.to_numeric(m2["year"], errors="coerce").notna()].copy()
    m2 = _convert_latam_money(m2)
    m2 = _mitchell_reshape(m2, "M2")

    out = m2.merge(master, on=["countryname", "year"], how="outer")
    years = pd.to_numeric(out["year"], errors="coerce")
    for col in ["M1", "M2"]:
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[out["countryname"].eq("Argentina"), col] = values[out["countryname"].eq("Argentina")] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[out["countryname"].eq("Bolivia"), col] = values[out["countryname"].eq("Bolivia")] / 1_000_000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[out["countryname"].eq("Bolivia") & years.ge(1983), col] = values[out["countryname"].eq("Bolivia") & years.ge(1983)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[out["countryname"].eq("Bolivia") & years.le(1954), col] = values[out["countryname"].eq("Bolivia") & years.le(1954)] * 1000
        values = pd.to_numeric(out[col], errors="coerce")
        bra_mask = out["countryname"].eq("Brazil")
        out.loc[bra_mask, col] = values[bra_mask] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[bra_mask & years.le(1989), col] = values[bra_mask & years.le(1989)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[bra_mask & years.le(1989), col] = values[bra_mask & years.le(1989)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[bra_mask & years.le(1974), col] = values[bra_mask & years.le(1974)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[bra_mask & years.between(1967, 1974), col] = values[bra_mask & years.between(1967, 1974)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        chl_mask = out["countryname"].eq("Chile")
        out.loc[chl_mask & years.ge(1983), col] = values[chl_mask & years.ge(1983)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[chl_mask & years.le(1974), col] = values[chl_mask & years.le(1974)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[chl_mask & years.le(1954), col] = values[chl_mask & years.le(1954)] * 1000
        values = pd.to_numeric(out[col], errors="coerce")
        col_mask = out["countryname"].eq("Colombia")
        out.loc[col_mask & years.ge(1975), col] = values[col_mask & years.ge(1975)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[col_mask & years.le(1964), col] = values[col_mask & years.le(1964)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        ecu_mask = out["countryname"].eq("Ecuador")
        out.loc[ecu_mask, col] = values[ecu_mask] / 25000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[ecu_mask & years.ge(1983), col] = values[ecu_mask & years.ge(1983)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[ecu_mask & years.le(1969), col] = values[ecu_mask & years.le(1969)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[out["countryname"].eq("Guyana"), col] = values[out["countryname"].eq("Guyana")] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        pry_mask = out["countryname"].eq("Paraguay")
        out.loc[pry_mask & years.between(1975, 2000), col] = values[pry_mask & years.between(1975, 2000)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[pry_mask & years.le(1969), col] = values[pry_mask & years.le(1969)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        per_mask = out["countryname"].eq("Peru")
        out.loc[per_mask, col] = values[per_mask] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[per_mask & years.le(1989), col] = values[per_mask & years.le(1989)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[per_mask & years.le(1984), col] = values[per_mask & years.le(1984)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        uru_mask = out["countryname"].eq("Uruguay")
        out.loc[uru_mask, col] = values[uru_mask] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[uru_mask & years.le(1979), col] = values[uru_mask & years.le(1979)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[uru_mask & years.le(1964), col] = values[uru_mask & years.le(1964)] / 1000
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[out["countryname"].eq("Venezuela"), col] = values[out["countryname"].eq("Venezuela")] * (10 ** -14)
    out["year"] = years.astype("int32")
    out = out[["countryname", "year", "M1", "M2"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Latam_money_supply.dta")
    return out


def _mitchell_partial_money_final(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    parts = [
        _mitchell_africa_money_supply(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_americas_money_supply(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_europe_money_supply(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_asia_money_supply(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_latam_money_supply(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_oceania_money_supply(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
    ]
    master: pd.DataFrame | None = None
    for part in parts:
        master = _mitchell_merge_values(master, part)
    assert master is not None
    return _mitchell_finalize_country_frame(master, ["M1", "M2"], data_helper_dir=data_helper_dir, euro_cutoff_year=1998)


def _mitchell_europe_inv(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Europe_NA")

    master: pd.DataFrame | None = None
    for sheet_name, rename_map in (
        (2, {"AC": "UnitedKingdom"}),
        (3, {"BE": "UnitedKingdom"}),
        (4, {"AC": "EastGermany", "AE": "Germany", "BO": "Russia", "CE": "UnitedKingdom"}),
        (5, {"AC": "EastGermany", "AE": "Germany", "AQ": "Ireland", "BO": "Russia", "CC": "UnitedKingdom"}),
    ):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        frame = _mitchell_select_current_metric(frame, price_row=2, metric_row=3, metrics={"CF"})
        frame = _mitchell_rename_from_row(frame, 0)
        frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})
        if "Germany" not in frame.columns and "WestGermany" in frame.columns:
            frame = frame.rename(columns={"WestGermany": "Germany"})
        if "Ireland" not in frame.columns and "SouthernIreland" in frame.columns:
            frame = frame.rename(columns={"SouthernIreland": "Ireland"})
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 2:
            frame = _mitchell_convert_units(frame, "Italy", 1815, 1899, "B")
        elif sheet_name == 3:
            for country in ["Austria", "Italy", "Russia"]:
                frame = _mitchell_convert_units(frame, country, 1900, 1944, "B")
        elif sheet_name == 5:
            for country in ["France", "Hungary", "Ireland", "Yugoslavia"]:
                frame = _mitchell_convert_units(frame, country, 1945, 1979, "Th")
            for country in [c for c in frame.columns if c != "year"]:
                frame = _mitchell_convert_units(frame, country, 1945, 1979, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "inv"))

    assert master is not None
    out = master.copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    out.loc[years.ge(1999), "inv"] = np.nan
    values = pd.to_numeric(out["inv"], errors="coerce")
    out.loc[country.eq("Austria") & years.le(1999), "inv"] = values[country.eq("Austria") & years.le(1999)] * 100
    values = pd.to_numeric(out["inv"], errors="coerce")
    out.loc[country.eq("Austria") & years.le(1993), "inv"] = values[country.eq("Austria") & years.le(1993)] * 10
    values = pd.to_numeric(out["inv"], errors="coerce")
    out.loc[country.eq("Austria") & years.le(1937), "inv"] = values[country.eq("Austria") & years.le(1937)] / 1000
    values = pd.to_numeric(out["inv"], errors="coerce")
    out.loc[country.eq("Belgium") & years.le(1999), "inv"] = values[country.eq("Belgium") & years.le(1999)] * 1000
    for ctry in ["Denmark", "Finland", "Italy", "Netherlands", "Norway", "Portugal", "Spain", "Sweden", "UnitedKingdom"]:
        values = pd.to_numeric(out["inv"], errors="coerce")
        out.loc[country.eq(ctry) & years.ge(1945), "inv"] = values[country.eq(ctry) & years.ge(1945)] * 1000
    values = pd.to_numeric(out["inv"], errors="coerce")
    out.loc[country.eq("France") & years.le(1979), "inv"] = values[country.eq("France") & years.le(1979)] * 1000
    values = pd.to_numeric(out["inv"], errors="coerce")
    out.loc[country.eq("Italy") & years.ge(1945), "inv"] = values[country.eq("Italy") & years.ge(1945)] * 1000
    values = pd.to_numeric(out["inv"], errors="coerce")
    out.loc[country.eq("Greece") & years.le(1999), "inv"] = values[country.eq("Greece") & years.le(1999)] * 1000
    values = pd.to_numeric(out["inv"], errors="coerce")
    out.loc[country.eq("Switzerland") & years.le(1999), "inv"] = values[country.eq("Switzerland") & years.le(1999)] * 1000
    values = pd.to_numeric(out["inv"], errors="coerce")
    out.loc[country.eq("Bulgaria") & years.le(1972), "inv"] = values[country.eq("Bulgaria") & years.le(1972)] / 1000
    values = pd.to_numeric(out["inv"], errors="coerce")
    out.loc[country.eq("Germany") & years.le(1913), "inv"] = values[country.eq("Germany") & years.le(1913)] / (10 ** 12)
    values = pd.to_numeric(out["inv"], errors="coerce")
    out.loc[country.eq("Hungary") & years.ge(1950), "inv"] = values[country.eq("Hungary") & years.ge(1950)] * 1000
    values = pd.to_numeric(out["inv"], errors="coerce")
    out.loc[country.eq("Hungary") & years.le(1940), "inv"] = values[country.eq("Hungary") & years.le(1940)] / (10 ** 24) / 4
    values = pd.to_numeric(out["inv"], errors="coerce")
    out.loc[country.eq("Czechoslovakia"), "inv"] = values[country.eq("Czechoslovakia")] * 1000
    values = pd.to_numeric(out["inv"], errors="coerce")
    out.loc[country.eq("France") & years.ge(1980), "inv"] = values[country.eq("France") & years.ge(1980)] * 1000
    values = pd.to_numeric(out["inv"], errors="coerce")
    out.loc[country.eq("Germany") & years.ge(1950), "inv"] = values[country.eq("Germany") & years.ge(1950)] * 1000
    out["year"] = years.astype("int32")
    out = out[["countryname", "year", "inv"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Europe_finv.dta")
    return out


def _mitchell_asia_finv(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Asia_NA")

    master: pd.DataFrame | None = None
    for sheet_name in (2, 3):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        frame = _mitchell_select_current_metric(frame, price_row=2, metric_row=3, metrics={"gfcf"})
        if 0 < len(frame):
            for col in [c for c in frame.columns if c != "year"]:
                value = "" if pd.isna(frame.at[0, col]) else str(frame.at[0, col]).replace(" ", "")
                frame.at[0, col] = value
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 3:
            for country in [
                "HongKong",
                "India",
                "Iran",
                "Nepal",
                "Pakistan",
                "Philippines",
                "SaudiArabia",
                "Taiwan",
                "Thailand",
                "Japan",
                "SouthKorea",
                "Turkey",
            ]:
                frame = _mitchell_convert_units(frame, country, 1945, 2010, "B")
            frame = _mitchell_convert_units(frame, "Indonesia", 1965, 1978, "B")
            frame = _mitchell_convert_units(frame, "Indonesia", 1979, 2010, "Tri")
            frame = _mitchell_convert_units(frame, "Israel", 1950, 1980, "Th")
            frame = _mitchell_convert_units(frame, "Japan", 1960, 2010, "B")
            frame = _mitchell_convert_units(frame, "SouthKorea", 1980, 2010, "B")
            frame = _mitchell_convert_units(frame, "Lebanon", 1988, 2010, "B")
            for country in ["Malaysia", "Myanmar", "Singapore", "SriLanka", "Syria"]:
                frame = _mitchell_convert_units(frame, country, 1975, 2010, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "finv"))

    assert master is not None
    out = master.copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Turkey") & years.le(1998), "finv"] = values[country.eq("Turkey") & years.le(1998)] * (10 ** -6)
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Turkey") & years.ge(1999), "finv"] = values[country.eq("Turkey") & years.ge(1999)] * (10 ** -3)
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Taiwan") & years.le(1939), "finv"] = values[country.eq("Taiwan") & years.le(1939)] * (10 ** -4)
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Taiwan") & years.le(1939), "finv"] = values[country.eq("Taiwan") & years.le(1939)] / 4
    out["year"] = years.astype("int32")
    out = out[["countryname", "year", "finv"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Asia_finv.dta")
    return out


def _mitchell_oceania_finv(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Oceania_NA")

    sheet4 = _mitchell_import_columns_first(path, 4)
    sheet4 = _mitchell_destring(_mitchell_drop_blank_year(sheet4))
    sheet4 = sheet4.rename(columns={"C": "Australia"})
    sheet4 = _mitchell_keep_columns(sheet4, ["Australia"])
    master: pd.DataFrame | None = _mitchell_reshape(sheet4, "finv")

    for sheet_name in (5, 6):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        frame = _mitchell_select_current_metric(frame, price_row=2, metric_row=3, metrics={"GFCF"})
        if 0 < len(frame):
            for col in [c for c in frame.columns if c != "year"]:
                value = "" if pd.isna(frame.at[0, col]) else str(frame.at[0, col]).replace(" ", "")
                frame.at[0, col] = value
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 6:
            frame = _mitchell_convert_units(frame, "Australia", 1965, 2010, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "finv"))

    assert master is not None
    wide = master.pivot(index="year", columns="countryname", values="finv").reset_index()
    wide.columns.name = None
    wide = _mitchell_convert_currency(wide, "Australia", 1900, 2)
    wide = _mitchell_convert_currency(wide, "NewZealand", 1959, 2)
    out = _mitchell_reshape(wide, "finv")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "finv"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Oceania_finv.dta")
    return out


def _mitchell_americas_finv(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Americas_NA")

    master: pd.DataFrame | None = None
    for sheet_name in (3, 4, 5):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        frame = _mitchell_select_current_metric(frame, price_row=2, metric_row=3, metrics={"gfcf", "gpfcf"})
        if 0 < len(frame):
            for col in [c for c in frame.columns if c != "year"]:
                value = "" if pd.isna(frame.at[0, col]) else str(frame.at[0, col]).replace(" ", "")
                frame.at[0, col] = value
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 4:
            frame = _mitchell_convert_units(frame, "USA", 1910, 1949, "B")
        elif sheet_name == 5:
            for country in ["Canada", "USA"]:
                frame = _mitchell_convert_units(frame, country, 1950, 2010, "B")
            frame = _mitchell_convert_units(frame, "Mexico", 1950, 1993, "B")
            frame = _mitchell_convert_units(frame, "CostaRica", 1980, 2010, "B")
            if "Nicaragua" in frame.columns:
                years = pd.to_numeric(frame["year"], errors="coerce")
                values = pd.to_numeric(frame["Nicaragua"], errors="coerce")
                frame.loc[years.le(1978), "Nicaragua"] = values[years.le(1978)] / 1000
        master = _mitchell_append(master, _mitchell_reshape(frame, "finv"))

    assert master is not None
    out = master.copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    out.loc[years.ge(1994), "finv"] = np.nan
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Nicaragua") & years.le(1989), "finv"] = values[country.eq("Nicaragua") & years.le(1989)] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Nicaragua") & years.le(1987), "finv"] = values[country.eq("Nicaragua") & years.le(1987)] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Mexico") & years.le(1984), "finv"] = values[country.eq("Mexico") & years.le(1984)] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("ElSalvador") & years.le(1993), "finv"] = values[country.eq("ElSalvador") & years.le(1993)] / 8.75
    out["year"] = years.astype("int32")
    out = out[["countryname", "year", "finv"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Americas_finv.dta")
    return out


def _mitchell_africa_finv(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Africa_NA")

    sheet3 = _mitchell_import_columns(path, 3)
    sheet3 = _mitchell_destring(_mitchell_drop_blank_year(sheet3))
    sheet3 = _mitchell_drop_columns(sheet3, ["Algeria", "D", "E"]).rename(columns={"C": "Algeria"})
    master: pd.DataFrame | None = _mitchell_reshape(sheet3, "finv")

    for sheet_name in (4, 7):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        frame = _mitchell_select_current_metric(frame, price_row=2, metric_row=3, metrics={"GFCF"})
        if 0 < len(frame):
            for col in [c for c in frame.columns if c != "year"]:
                value = "" if pd.isna(frame.at[0, col]) else str(frame.at[0, col]).replace(" ", "")
                frame.at[0, col] = value
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        master = _mitchell_append(master, _mitchell_reshape(frame, "finv"))

    sheet5 = _mitchell_import_columns(path, 5)
    sheet5 = _mitchell_destring(_mitchell_drop_blank_year(sheet5))
    sheet5 = _mitchell_drop_columns(sheet5, ["Libya", "D", "E"]).rename(columns={"C": "Libya"})
    master = _mitchell_append(master, _mitchell_reshape(sheet5, "finv"))

    sheet6 = _mitchell_import_columns(path, 6)
    sheet6 = _mitchell_destring(_mitchell_drop_blank_year(sheet6))
    sheet6 = _mitchell_drop_columns(sheet6, ["Libya", "D", "E", "F", "G"]).rename(columns={"C": "Libya"})
    master = _mitchell_append(master, _mitchell_reshape(sheet6, "finv"))

    assert master is not None
    wide = master.pivot(index="year", columns="countryname", values="finv").reset_index()
    wide.columns.name = None
    wide = _mitchell_convert_units(wide, "SouthAfrica", 1911, 1979, "Th")
    wide = _mitchell_convert_units(wide, "Ghana", 1950, 1979, "Th")
    wide = _mitchell_convert_units(wide, "SierraLeone", 1950, 1998, "Th")
    wide = _mitchell_convert_units(wide, "Uganda", 1950, 1988, "Th")
    wide = _mitchell_convert_units(wide, "Zambia", 1950, 1992, "Th")
    for country in ["Egypt", "Lesotho", "Liberia", "Libya", "Malawi", "Mauritius"]:
        if country in wide.columns:
            wide[country] = pd.to_numeric(wide[country], errors="coerce") / 1000
    for col in [c for c in wide.columns if c != "year"]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce") * 1000

    out = _mitchell_reshape(wide, "finv")
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Zambia") & years.eq(1993), "finv"] = values[country.eq("Zambia") & years.eq(1993)] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Zambia"), "finv"] = values[country.eq("Zambia")] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Tunisia"), "finv"] = values[country.eq("Tunisia")] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Sudan"), "finv"] = values[country.eq("Sudan")] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Sudan") & years.le(1991), "finv"] = values[country.eq("Sudan") & years.le(1991)] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Ghana"), "finv"] = values[country.eq("Ghana")] / 10000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Zaire") & years.le(1993), "finv"] = values[country.eq("Zaire") & years.le(1993)] / 100000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Zaire"), "finv"] = values[country.eq("Zaire")] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Zaire") & years.le(1991), "finv"] = values[country.eq("Zaire") & years.le(1991)] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Zaire") & years.le(1977), "finv"] = values[country.eq("Zaire") & years.le(1977)] / 1000
    out["year"] = years.astype("int32")
    out = out[["countryname", "year", "finv"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Africa_finv.dta")
    return out


def _mitchell_latam_finv(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Latam_NA")

    master: pd.DataFrame | None = None
    for sheet_name in (4, 5):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        frame = _mitchell_select_current_metric(frame, price_row=2, metric_row=3, metrics={"gfcf"})
        if 0 < len(frame):
            for col in [c for c in frame.columns if c != "year"]:
                value = "" if pd.isna(frame.at[0, col]) else str(frame.at[0, col]).replace(" ", "")
                frame.at[0, col] = value
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 4:
            for country in ["Argentina", "Chile", "Paraguay", "Venezuela"]:
                frame = _mitchell_convert_units(frame, country, 1935, 1969, "B")
        else:
            for country in ["Bolivia", "Brazil", "Colombia", "Paraguay", "Uruguay", "Venezuela"]:
                frame = _mitchell_convert_units(frame, country, 1970, 1993, "B")
            frame = _mitchell_convert_units(frame, "Argentina", 1970, 1986, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "finv"))

    assert master is not None
    out = master.copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    out.loc[years.ge(1994), "finv"] = np.nan
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Brazil"), "finv"] = values[country.eq("Brazil")] / 2750
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Brazil"), "finv"] = values[country.eq("Brazil")] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Brazil") & years.le(1989), "finv"] = values[country.eq("Brazil") & years.le(1989)] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Brazil") & years.le(1979), "finv"] = values[country.eq("Brazil") & years.le(1979)] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Brazil") & years.between(1960, 1969), "finv"] = values[country.eq("Brazil") & years.between(1960, 1969)] * 1000
    out.loc[country.eq("Argentina") & years.ge(1986), "finv"] = np.nan
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Venezuela"), "finv"] = values[country.eq("Venezuela")] * (10 ** -14)
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Mexico") & years.le(1984), "finv"] = values[country.eq("Mexico") & years.le(1984)] * (10 ** -6)
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Ecuador"), "finv"] = values[country.eq("Ecuador")] * (10 ** 3)
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Ecuador"), "finv"] = values[country.eq("Ecuador")] / 25000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Colombia") & years.le(1969), "finv"] = values[country.eq("Colombia") & years.le(1969)] * (10 ** 3)
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Uruguay"), "finv"] = values[country.eq("Uruguay")] * (10 ** -3)
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Uruguay") & years.le(1979), "finv"] = values[country.eq("Uruguay") & years.le(1979)] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Uruguay") & years.le(1969), "finv"] = values[country.eq("Uruguay") & years.le(1969)] * 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Uruguay") & years.le(1959), "finv"] = values[country.eq("Uruguay") & years.le(1959)] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Peru") & years.le(1988), "finv"] = values[country.eq("Peru") & years.le(1988)] * (10 ** -3)
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Peru") & years.le(1979), "finv"] = values[country.eq("Peru") & years.le(1979)] * (10 ** -3)
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Peru") & years.le(1949), "finv"] = values[country.eq("Peru") & years.le(1949)] * (10 ** -3)
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Bolivia") & years.le(1993), "finv"] = values[country.eq("Bolivia") & years.le(1993)] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Bolivia") & years.eq(1984), "finv"] = values[country.eq("Bolivia") & years.eq(1984)] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Bolivia") & years.le(1983), "finv"] = values[country.eq("Bolivia") & years.le(1983)] / 100
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Bolivia") & years.le(1979), "finv"] = values[country.eq("Bolivia") & years.le(1979)] / 10
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Chile") & years.le(1969), "finv"] = values[country.eq("Chile") & years.le(1969)] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Chile") & years.le(1964), "finv"] = values[country.eq("Chile") & years.le(1964)] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Chile") & years.ge(1975), "finv"] = values[country.eq("Chile") & years.ge(1975)] * 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Argentina") & years.le(1985), "finv"] = values[country.eq("Argentina") & years.le(1985)] / 10000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Argentina") & years.le(1984), "finv"] = values[country.eq("Argentina") & years.le(1984)] / 1000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Argentina") & years.le(1974), "finv"] = values[country.eq("Argentina") & years.le(1974)] / 10000
    values = pd.to_numeric(out["finv"], errors="coerce")
    out.loc[country.eq("Argentina") & years.le(1964), "finv"] = values[country.eq("Argentina") & years.le(1964)] / 100
    out["year"] = years.astype("int32")
    out = out[["countryname", "year", "finv"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Latam_finv.dta")
    return out


def _mitchell_americas_stocks(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Americas_NA")

    master: pd.DataFrame | None = None
    for sheet_name in (4, 5):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        frame = _mitchell_select_current_metric(frame, price_row=2, metric_row=3, metrics={"stocks"})
        if 0 < len(frame):
            for col in [c for c in frame.columns if c != "year"]:
                value = "" if pd.isna(frame.at[0, col]) else str(frame.at[0, col]).replace(" ", "")
                frame.at[0, col] = value
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 4:
            frame = _mitchell_convert_units(frame, "USA", 1910, 1949, "B")
        else:
            for country in ["Canada", "USA"]:
                frame = _mitchell_convert_units(frame, country, 1950, 2010, "B")
            frame = _mitchell_convert_units(frame, "Mexico", 1950, 1993, "B")
            frame = _mitchell_convert_units(frame, "CostaRica", 1980, 2010, "B")
            if "Nicaragua" in frame.columns:
                years = pd.to_numeric(frame["year"], errors="coerce")
                values = pd.to_numeric(frame["Nicaragua"], errors="coerce")
                frame.loc[years.le(1978), "Nicaragua"] = values[years.le(1978)] / 1000
        master = _mitchell_append(master, _mitchell_reshape(frame, "stocks"))

    assert master is not None
    out = master.copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Nicaragua") & years.le(1989), "stocks"] = values[country.eq("Nicaragua") & years.le(1989)] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Nicaragua") & years.le(1987), "stocks"] = values[country.eq("Nicaragua") & years.le(1987)] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Mexico") & years.le(1984), "stocks"] = values[country.eq("Mexico") & years.le(1984)] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("ElSalvador") & years.le(1993), "stocks"] = values[country.eq("ElSalvador") & years.le(1993)] / 8.75
    out["year"] = years.astype("int32")
    out = out[["countryname", "year", "stocks"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Americas_stocks.dta")
    return out


def _mitchell_asia_stocks(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Asia_NA")

    master: pd.DataFrame | None = None
    for sheet_name in (2, 3):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        frame = _mitchell_select_current_metric(frame, price_row=2, metric_row=3, metrics={"stocks"})
        if 0 < len(frame):
            for col in [c for c in frame.columns if c != "year"]:
                value = "" if pd.isna(frame.at[0, col]) else str(frame.at[0, col]).replace(" ", "")
                frame.at[0, col] = value
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 3:
            for country in [
                "HongKong",
                "India",
                "Iran",
                "Nepal",
                "Pakistan",
                "Philippines",
                "SaudiArabia",
                "Taiwan",
                "Thailand",
                "Japan",
                "SouthKorea",
                "Turkey",
            ]:
                frame = _mitchell_convert_units(frame, country, 1945, 2010, "B")
            frame = _mitchell_convert_units(frame, "Indonesia", 1965, 1978, "B")
            frame = _mitchell_convert_units(frame, "Indonesia", 1979, 2010, "Tri")
            frame = _mitchell_convert_units(frame, "Israel", 1950, 1980, "Th")
            frame = _mitchell_convert_units(frame, "Japan", 1960, 2010, "B")
            frame = _mitchell_convert_units(frame, "SouthKorea", 1980, 2010, "B")
            frame = _mitchell_convert_units(frame, "Lebanon", 1988, 2010, "B")
            for country in ["Malaysia", "Myanmar", "Singapore", "SriLanka", "Syria"]:
                frame = _mitchell_convert_units(frame, country, 1975, 2010, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "stocks"))

    assert master is not None
    out = master.copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Turkey") & years.le(1998), "stocks"] = values[country.eq("Turkey") & years.le(1998)] * (10 ** -6)
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Turkey") & years.ge(1999), "stocks"] = values[country.eq("Turkey") & years.ge(1999)] * (10 ** -3)
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Taiwan") & years.le(1939), "stocks"] = values[country.eq("Taiwan") & years.le(1939)] * (10 ** -4)
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Taiwan") & years.le(1939), "stocks"] = values[country.eq("Taiwan") & years.le(1939)] / 4
    out["year"] = years.astype("int32")
    out = out[["countryname", "year", "stocks"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Asia_stocks.dta")
    return out


def _mitchell_oceania_stocks(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Oceania_NA")

    master: pd.DataFrame | None = None
    for sheet_name in (5, 6):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        frame = _mitchell_select_current_metric(frame, price_row=2, metric_row=3, metrics={"Stocks"})
        if 0 < len(frame):
            for col in [c for c in frame.columns if c != "year"]:
                value = "" if pd.isna(frame.at[0, col]) else str(frame.at[0, col]).replace(" ", "")
                frame.at[0, col] = value
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 6:
            frame = _mitchell_convert_units(frame, "Australia", 1965, 2010, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "stocks"))

    assert master is not None
    wide = master.pivot(index="year", columns="countryname", values="stocks").reset_index()
    wide.columns.name = None
    wide = _mitchell_convert_currency(wide, "Australia", 1900, 2)
    wide = _mitchell_convert_currency(wide, "NewZealand", 1959, 2)
    out = _mitchell_reshape(wide, "stocks")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "stocks"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Oceania_stocks.dta")
    return out


def _mitchell_africa_stocks(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Africa_NA")

    sheet3 = _mitchell_import_columns(path, 3)
    sheet3 = _mitchell_destring(_mitchell_drop_blank_year(sheet3))
    sheet3 = _mitchell_drop_columns(sheet3, ["Algeria", "C", "E"]).rename(columns={"D": "Algeria"})
    master: pd.DataFrame | None = _mitchell_reshape(sheet3, "stocks")

    for sheet_name in (4, 7):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        metric_name = "Stocks" if sheet_name == 4 else "GFCF"
        frame = _mitchell_select_current_metric(frame, price_row=2, metric_row=3, metrics={metric_name})
        if 0 < len(frame):
            for col in [c for c in frame.columns if c != "year"]:
                value = "" if pd.isna(frame.at[0, col]) else str(frame.at[0, col]).replace(" ", "")
                frame.at[0, col] = value
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        master = _mitchell_append(master, _mitchell_reshape(frame, "stocks"))

    sheet5 = _mitchell_import_columns(path, 5)
    sheet5 = _mitchell_destring(_mitchell_drop_blank_year(sheet5))
    sheet5 = _mitchell_drop_columns(sheet5, ["Libya", "C", "E"]).rename(columns={"D": "Libya"})
    master = _mitchell_append(master, _mitchell_reshape(sheet5, "stocks"))

    sheet6 = _mitchell_import_columns(path, 6)
    sheet6 = _mitchell_destring(_mitchell_drop_blank_year(sheet6))
    sheet6 = _mitchell_drop_columns(sheet6, ["Libya", "C", "E", "F", "G"]).rename(columns={"D": "Libya"})
    master = _mitchell_append(master, _mitchell_reshape(sheet6, "stocks"))

    assert master is not None
    wide = master.pivot(index="year", columns="countryname", values="stocks").reset_index()
    wide.columns.name = None
    wide = _mitchell_convert_units(wide, "SouthAfrica", 1911, 1979, "Th")
    wide = _mitchell_convert_units(wide, "Ghana", 1950, 1979, "Th")
    wide = _mitchell_convert_units(wide, "SierraLeone", 1950, 1998, "Th")
    wide = _mitchell_convert_units(wide, "Uganda", 1950, 1988, "Th")
    wide = _mitchell_convert_units(wide, "Zambia", 1950, 1992, "Th")
    for country in ["Egypt", "Lesotho", "Liberia", "Libya", "Malawi", "Mauritius"]:
        if country in wide.columns:
            wide[country] = pd.to_numeric(wide[country], errors="coerce") / 1000
    for col in [c for c in wide.columns if c != "year"]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce") * 1000

    out = _mitchell_reshape(wide, "stocks")
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Zambia") & years.eq(1993), "stocks"] = values[country.eq("Zambia") & years.eq(1993)] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Zambia"), "stocks"] = values[country.eq("Zambia")] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Tunisia"), "stocks"] = values[country.eq("Tunisia")] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Sudan"), "stocks"] = values[country.eq("Sudan")] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Sudan") & years.le(1991), "stocks"] = values[country.eq("Sudan") & years.le(1991)] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Ghana"), "stocks"] = values[country.eq("Ghana")] / 10000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Zaire") & years.le(1993), "stocks"] = values[country.eq("Zaire") & years.le(1993)] / 100000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Zaire"), "stocks"] = values[country.eq("Zaire")] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Zaire") & years.le(1991), "stocks"] = values[country.eq("Zaire") & years.le(1991)] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Zaire") & years.le(1977), "stocks"] = values[country.eq("Zaire") & years.le(1977)] / 1000
    out["year"] = years.astype("int32")
    out = out[["countryname", "year", "stocks"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Africa_stocks.dta")
    return out


def _mitchell_latam_stocks(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Latam_NA")

    master: pd.DataFrame | None = None
    for sheet_name in (4, 5):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 3)
        frame = _mitchell_select_current_metric(frame, price_row=2, metric_row=3, metrics={"stocks"})
        if 0 < len(frame):
            for col in [c for c in frame.columns if c != "year"]:
                value = "" if pd.isna(frame.at[0, col]) else str(frame.at[0, col]).replace(" ", "")
                frame.at[0, col] = value
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 4:
            for country in ["Argentina", "Ecuador", "Chile", "Paraguay", "Venezuela"]:
                frame = _mitchell_convert_units(frame, country, 1935, 1969, "B")
        else:
            for country in ["Bolivia", "Brazil", "Colombia", "Ecuador", "Paraguay", "Uruguay", "Venezuela"]:
                frame = _mitchell_convert_units(frame, country, 1970, 1993, "B")
            frame = _mitchell_convert_units(frame, "Argentina", 1970, 1986, "B")
        master = _mitchell_append(master, _mitchell_reshape(frame, "stocks"))

    assert master is not None
    out = master.copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Brazil"), "stocks"] = values[country.eq("Brazil")] / 2750
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Brazil"), "stocks"] = values[country.eq("Brazil")] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Brazil") & years.le(1989), "stocks"] = values[country.eq("Brazil") & years.le(1989)] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Brazil") & years.le(1979), "stocks"] = values[country.eq("Brazil") & years.le(1979)] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Brazil") & years.between(1960, 1969), "stocks"] = values[country.eq("Brazil") & years.between(1960, 1969)] * 1000
    out.loc[country.eq("Argentina") & years.ge(1986), "stocks"] = np.nan
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Venezuela"), "stocks"] = values[country.eq("Venezuela")] * (10 ** -14)
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Mexico") & years.le(1984), "stocks"] = values[country.eq("Mexico") & years.le(1984)] * (10 ** -6)
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Ecuador"), "stocks"] = values[country.eq("Ecuador")] * (10 ** 3)
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Ecuador"), "stocks"] = values[country.eq("Ecuador")] / 25000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Colombia") & years.le(1969), "stocks"] = values[country.eq("Colombia") & years.le(1969)] * (10 ** 3)
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Uruguay"), "stocks"] = values[country.eq("Uruguay")] * (10 ** -3)
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Uruguay") & years.le(1979), "stocks"] = values[country.eq("Uruguay") & years.le(1979)] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Uruguay") & years.le(1969), "stocks"] = values[country.eq("Uruguay") & years.le(1969)] * 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Uruguay") & years.le(1959), "stocks"] = values[country.eq("Uruguay") & years.le(1959)] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Peru") & years.le(1988), "stocks"] = values[country.eq("Peru") & years.le(1988)] * (10 ** -3)
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Peru") & years.le(1979), "stocks"] = values[country.eq("Peru") & years.le(1979)] * (10 ** -3)
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Peru") & years.le(1949), "stocks"] = values[country.eq("Peru") & years.le(1949)] * (10 ** -3)
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Bolivia") & years.le(1992), "stocks"] = values[country.eq("Bolivia") & years.le(1992)] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Bolivia") & years.eq(1984), "stocks"] = values[country.eq("Bolivia") & years.eq(1984)] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Bolivia") & years.le(1983), "stocks"] = values[country.eq("Bolivia") & years.le(1983)] / 100
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Bolivia") & years.le(1979), "stocks"] = values[country.eq("Bolivia") & years.le(1979)] / 10
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Chile") & years.le(1969), "stocks"] = values[country.eq("Chile") & years.le(1969)] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Chile") & years.le(1964), "stocks"] = values[country.eq("Chile") & years.le(1964)] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Chile") & years.ge(1975), "stocks"] = values[country.eq("Chile") & years.ge(1975)] * 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Argentina") & years.le(1985), "stocks"] = values[country.eq("Argentina") & years.le(1985)] / 10000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Argentina") & years.le(1984), "stocks"] = values[country.eq("Argentina") & years.le(1984)] / 1000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Argentina") & years.le(1974), "stocks"] = values[country.eq("Argentina") & years.le(1974)] / 10000
    values = pd.to_numeric(out["stocks"], errors="coerce")
    out.loc[country.eq("Argentina") & years.le(1964), "stocks"] = values[country.eq("Argentina") & years.le(1964)] / 100
    out["year"] = years.astype("int32")
    out = out[["countryname", "year", "stocks"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Latam_stocks.dta")
    return out


def _mitchell_partial_stocks_final(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    parts = [
        _mitchell_africa_stocks(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_americas_stocks(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_asia_stocks(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_latam_stocks(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_oceania_stocks(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
    ]
    return _mitchell_finalize_value_parts(parts, "stocks", data_helper_dir=data_helper_dir, euro_cutoff_year=1998)


def _mitchell_partial_finv_inv_final(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    finv_parts = [
        _mitchell_africa_finv(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_americas_finv(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_asia_finv(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_latam_finv(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_oceania_finv(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
    ]
    inv_parts = [
        _mitchell_europe_inv(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
    ]
    finv = _mitchell_finalize_value_parts(finv_parts, "finv", data_helper_dir=data_helper_dir, euro_cutoff_year=1998)
    inv = _mitchell_finalize_value_parts(inv_parts, "inv", data_helper_dir=data_helper_dir, euro_cutoff_year=9999)
    return finv.merge(inv, on=["ISO3", "year"], how="outer").sort_values(["ISO3", "year"]).reset_index(drop=True)


def _mitchell_europe_banknotes(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Europe_banknotes")

    sheet2 = _mitchell_import_columns_first(path, 2)
    sheet2 = _mitchell_rename_from_row(sheet2, 0).rename(columns={"C": "UnitedKingdom", "UKEW": "UnitedKingdom"})
    sheet2 = _mitchell_destring(_mitchell_drop_blank_year(sheet2))
    master: pd.DataFrame | None = _mitchell_reshape(sheet2, "M0")

    sheet3 = _mitchell_import_columns_first(path, 3)
    sheet3 = _mitchell_fill_header_rows(sheet3, 1)
    sheet3 = _mitchell_rename_from_row(sheet3, 0).rename(
        columns={"J": "Sweden_bis", "K": "UnitedKingdom", "UKGB": "UnitedKingdom", "L": "UnitedKingdom_bis"}
    )
    sheet3 = _mitchell_destring(_mitchell_drop_blank_year(sheet3))
    if {"Sweden", "Sweden_bis"}.issubset(sheet3.columns):
        sweden_bis = pd.to_numeric(sheet3["Sweden_bis"], errors="coerce")
        sweden_mask = sweden_bis.notna()
        sheet3.loc[sweden_mask, "Sweden"] = (
            pd.to_numeric(sheet3.loc[sweden_mask, "Sweden"], errors="coerce")
            + sweden_bis[sweden_mask]
        )
    if {"UnitedKingdom", "UnitedKingdom_bis"}.issubset(sheet3.columns):
        uk_bis = pd.to_numeric(sheet3["UnitedKingdom_bis"], errors="coerce")
        uk_mask = uk_bis.notna()
        sheet3.loc[uk_mask, "UnitedKingdom"] = (
            pd.to_numeric(sheet3.loc[uk_mask, "UnitedKingdom"], errors="coerce")
            + uk_bis[uk_mask]
        )
    sheet3 = _mitchell_drop_columns(sheet3, ["Sweden_bis", "UnitedKingdom_bis"])
    sheet3 = _mitchell_convert_units(sheet3, "Finland", 1800, 1849, "Th")
    master = _mitchell_append(master, _mitchell_reshape(sheet3, "M0"))

    sheet4 = _mitchell_import_columns(path, 4).rename(columns={"UKGB": "UnitedKingdom"})
    sheet4 = _mitchell_destring(_mitchell_drop_blank_year(sheet4))
    for country, extra in [("Spain", "Q"), ("Sweden", "S"), ("UnitedKingdom", "V")]:
        if country in sheet4.columns and extra in sheet4.columns:
            extra_vals = pd.to_numeric(sheet4[extra], errors="coerce")
            extra_mask = extra_vals.notna()
            sheet4.loc[extra_mask, country] = (
                pd.to_numeric(sheet4.loc[extra_mask, country], errors="coerce")
                + extra_vals[extra_mask]
            )
    sheet4 = _mitchell_drop_columns(sheet4, ["Q", "S", "V"])
    sheet4 = _mitchell_convert_units(sheet4, "Finland", 1850, 1899, "Th")
    sheet4 = _mitchell_convert_units(sheet4, "Bulgaria", 1850, 1899, "Th")
    master = _mitchell_append(master, _mitchell_reshape(sheet4, "M0"))

    sheet5 = _mitchell_import_columns(path, 5).rename(columns={"UKGB": "UnitedKingdom"})
    sheet5 = _mitchell_destring(_mitchell_drop_blank_year(sheet5))
    for country, extra in [("Sweden", "W"), ("Switzerland", "Y"), ("UnitedKingdom", "AA")]:
        if country in sheet5.columns and extra in sheet5.columns:
            extra_vals = pd.to_numeric(sheet5[extra], errors="coerce")
            extra_mask = extra_vals.notna()
            sheet5.loc[extra_mask, country] = (
                pd.to_numeric(sheet5.loc[extra_mask, country], errors="coerce")
                + extra_vals[extra_mask]
            )
    sheet5 = _mitchell_drop_columns(sheet5, ["W", "Y", "AA"])
    sheet5 = _mitchell_convert_units(sheet5, "Finland", 1900, 1917, "Th")
    sheet5 = _mitchell_convert_units(sheet5, "Poland", 1924, 1944, "B")
    sheet5 = _mitchell_convert_units(sheet5, "France", 1940, 1944, "B")
    sheet5 = _mitchell_convert_units(sheet5, "Romania", 1925, 1944, "B")
    if "Hungary" in sheet5.columns:
        years = pd.to_numeric(sheet5["year"], errors="coerce")
        vals = pd.to_numeric(sheet5["Hungary"], errors="coerce")
        sheet5.loc[years.le(1924), "Hungary"] = vals[years.le(1924)] / 12500
    master = _mitchell_append(master, _mitchell_reshape(sheet5, "M0"))

    sheet6 = _mitchell_import_columns(path, 6).rename(columns={"UK": "UnitedKingdom"})
    sheet6 = _mitchell_destring(_mitchell_drop_blank_year(sheet6))
    if {"UnitedKingdom", "GB"}.issubset(sheet6.columns):
        gb_vals = pd.to_numeric(sheet6["GB"], errors="coerce")
        gb_mask = gb_vals.notna()
        sheet6.loc[gb_mask, "UnitedKingdom"] = (
            pd.to_numeric(sheet6.loc[gb_mask, "UnitedKingdom"], errors="coerce")
            + gb_vals[gb_mask]
        )
    sheet6 = _mitchell_drop_columns(sheet6, ["GB"])
    for country in ["Austria", "France", "WestGermany", "Greece", "Italy", "Poland", "Portugal", "Spain", "Yugoslavia"]:
        sheet6 = _mitchell_convert_units(sheet6, country, 1945, 2010, "B")
    sheet6 = sheet6.rename(columns={"WestGermany": "Germany"})
    master = _mitchell_append(master, _mitchell_reshape(sheet6, "M0"))

    out = master.copy()
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Switzerland") & years.ge(1975), "M0"] = values[country.eq("Switzerland") & years.ge(1975)] * 1000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Belgium") & years.between(1941, 1998), "M0"] = values[country.eq("Belgium") & years.between(1941, 1998)] * 1000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Austria") & years.le(1923), "M0"] = values[country.eq("Austria") & years.le(1923)] / 10000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Germany") & years.le(1923), "M0"] = values[country.eq("Germany") & years.le(1923)] / (10 ** 12)
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Sweden") & years.ge(1974), "M0"] = values[country.eq("Sweden") & years.ge(1974)] * 1000
    for ctry in ["Norway", "Netherlands", "Italy"]:
        values = pd.to_numeric(out["M0"], errors="coerce")
        out.loc[country.eq(ctry) & years.ge(1975), "M0"] = values[country.eq(ctry) & years.ge(1975)] * 1000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Greece") & years.le(1943), "M0"] = values[country.eq("Greece") & years.le(1943)] / 40000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("France") & years.ge(1959), "M0"] = values[country.eq("France") & years.ge(1959)] * 100
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Yugoslavia") & years.le(1964), "M0"] = values[country.eq("Yugoslavia") & years.le(1964)] / 100
    out["year"] = years.astype("int32")
    out = out[["countryname", "year", "M0"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Europe_banknotes.dta")
    return out


def _mitchell_asia_banknotes(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Asia_banknotes")

    sheet2 = _mitchell_import_columns(path, 2)
    sheet2 = _mitchell_destring(_mitchell_drop_blank_year(sheet2))
    master: pd.DataFrame | None = _mitchell_reshape(sheet2, "M0")

    sheet3 = _mitchell_import_columns(path, 3)
    sheet3 = _mitchell_destring(_mitchell_drop_blank_year(sheet3))
    for country in ["China", "Japan"]:
        sheet3 = _mitchell_convert_units(sheet3, country, 1945, 1949, "B")
    sheet3 = _mitchell_convert_units(sheet3, "Indonesia", 1948, 1949, "B")
    master = _mitchell_append(master, _mitchell_reshape(sheet3, "M0"))

    sheet4 = _mitchell_import_columns(path, 4)
    sheet4 = _mitchell_use_overlapping_data(sheet4)
    sheet4 = sheet4.loc[pd.to_numeric(sheet4["year"], errors="coerce").notna()].copy()
    sheet4 = _mitchell_convert_units(sheet4, "Thailand", 1955, 2010, "B")
    for country in ["China", "India", "Indonesia", "Japan"]:
        sheet4 = _mitchell_convert_units(sheet4, country, 1950, 2010, "B")
    for country in ["Iran", "SouthKorea"]:
        sheet4 = _mitchell_convert_units(sheet4, country, 1965, 2010, "B")
    for country in ["SaudiArabia", "Taiwan", "Turkey", "SouthVietnam"]:
        sheet4 = _mitchell_convert_units(sheet4, country, 1970, 2010, "B")
    for country in ["Afghanistan", "Cambodia", "Pakistan", "Philippines"]:
        sheet4 = _mitchell_convert_units(sheet4, country, 1975, 2010, "B")
    sheet4 = _mitchell_convert_units(sheet4, "Lebanon", 1985, 2010, "B")
    sheet4 = _mitchell_convert_units(sheet4, "Japan", 1975, 2010, "B")
    sheet4 = _mitchell_convert_units(sheet4, "Turkey", 1996, 2010, "B")
    master = _mitchell_append(master, _mitchell_reshape(sheet4, "M0"))

    wide = master.pivot(index="year", columns="countryname", values="M0").reset_index()
    wide.columns.name = None
    wide = _mitchell_convert_currency(wide, "Indonesia", 1947, 1 / 10)
    wide = _mitchell_convert_currency(wide, "Indonesia", 1964, 1 / 1000)
    wide = _mitchell_convert_currency(wide, "Israel", 1982, 1 / 1000)
    wide = _mitchell_convert_currency(wide, "SouthYemen", 1988, 26)
    if "Turkey" in wide.columns:
        wide["Turkey"] = pd.to_numeric(wide["Turkey"], errors="coerce") / (10 ** 5)
    if {"SouthVietnam", "Indochina"}.issubset(wide.columns):
        current = pd.to_numeric(wide["SouthVietnam"], errors="coerce")
        indo = pd.to_numeric(wide["Indochina"], errors="coerce")
        mask = current.isna()
        wide.loc[mask, "SouthVietnam"] = indo[mask]
        wide = wide.drop(columns=["Indochina"])
    wide = wide.rename(columns={"SouthVietnam": "Vietnam"})
    if {"SouthYemen", "Yemen"}.issubset(wide.columns):
        south = pd.to_numeric(wide["SouthYemen"], errors="coerce")
        yemen = pd.to_numeric(wide["Yemen"], errors="coerce")
        mask = yemen.notna()
        wide.loc[mask, "SouthYemen"] = south[mask] + yemen[mask]
        wide = wide.drop(columns=["Yemen"])
    wide = wide.rename(columns={"SouthYemen": "Yemen"})

    out = _mitchell_reshape(wide, "M0")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "M0"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Asia_banknotes.dta")
    return out


def _mitchell_americas_banknotes(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Americas_banknotes")

    sheet2 = _mitchell_import_columns(path, 2)
    sheet2 = _mitchell_destring(_mitchell_drop_blank_year(sheet2))
    if {"USA", "D"}.issubset(sheet2.columns):
        sheet2["USA"] = pd.to_numeric(sheet2["USA"], errors="coerce") + pd.to_numeric(sheet2["D"], errors="coerce")
    sheet2 = _mitchell_drop_columns(sheet2, ["D"])
    master: pd.DataFrame | None = _mitchell_reshape(sheet2, "M0")

    sheet3 = _mitchell_import_columns(path, 3)
    sheet3 = _mitchell_destring(_mitchell_drop_blank_year(sheet3))
    if {"USA", "N"}.issubset(sheet3.columns):
        usa = pd.to_numeric(sheet3["USA"], errors="coerce")
        n = pd.to_numeric(sheet3["N"], errors="coerce")
        mask = n.notna()
        sheet3.loc[mask, "USA"] = usa[mask] + n[mask]
    sheet3 = _mitchell_drop_columns(sheet3, ["N"])
    sheet3 = _mitchell_convert_units(sheet3, "USA", 1935, 2010, "B")
    for country in ["Canada", "Mexico"]:
        sheet3 = _mitchell_convert_units(sheet3, country, 1975, 2010, "B")
    sheet3 = _mitchell_convert_units(sheet3, "Nicaragua", 1985, 1987, "B")
    master = _mitchell_append(master, _mitchell_reshape(sheet3, "M0"))

    wide = master.pivot(index="year", columns="countryname", values="M0").reset_index()
    wide.columns.name = None
    wide = _mitchell_convert_currency(wide, "Nicaragua", 1912, 0.08)
    wide = _mitchell_convert_currency(wide, "Jamaica", 1968, 2)
    wide = _mitchell_convert_currency(wide, "Guatemala", 1925, 1 / 60)
    wide = wide.rename(columns={"TrinidadTobago": "TrinidadandTobago"})
    out = _mitchell_reshape(wide, "M0")
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Mexico"), "M0"] = values[country.eq("Mexico")] / 1000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("ElSalvador") & years.le(2000), "M0"] = values[country.eq("ElSalvador") & years.le(2000)] / 8.75
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Nicaragua") & years.le(1987), "M0"] = values[country.eq("Nicaragua") & years.le(1987)] / 1_000_000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Nicaragua") & years.le(1989), "M0"] = values[country.eq("Nicaragua") & years.le(1989)] / 200
    out["year"] = years.astype("int32")
    out = out[["countryname", "year", "M0"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Americas_banknotes.dta")
    return out


def _mitchell_latam_banknotes(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Latam_banknotes")

    master: pd.DataFrame | None = None
    for sheet_name in (2, 3, 4):
        frame = _mitchell_import_columns(path, sheet_name)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        master = _mitchell_append(master, _mitchell_reshape(frame, "M0"))

    wide = master.pivot(index="year", columns="countryname", values="M0").reset_index()
    wide.columns.name = None
    years = pd.to_numeric(wide["year"], errors="coerce")
    if "Argentina" in wide.columns:
        vals = pd.to_numeric(wide["Argentina"], errors="coerce")
        wide.loc[years.le(1969), "Argentina"] = vals[years.le(1969)] / 100
        vals = pd.to_numeric(wide["Argentina"], errors="coerce")
        wide.loc[years.le(1982), "Argentina"] = vals[years.le(1982)] / 1000
        vals = pd.to_numeric(wide["Argentina"], errors="coerce")
        wide.loc[years.le(1988), "Argentina"] = vals[years.le(1988)] / 10000
        wide = _mitchell_convert_units(wide, "Argentina", 1955, 1982, "B")
    wide = _mitchell_convert_units(wide, "Bolivia", 1955, 1982, "B")
    wide = _mitchell_convert_units(wide, "Bolivia", 1983, 2010, "Tri")
    if "Brazil" in wide.columns:
        vals = pd.to_numeric(wide["Brazil"], errors="coerce")
        wide.loc[years.ge(1940), "Brazil"] = vals[years.ge(1940)] * 1000
        vals = pd.to_numeric(wide["Brazil"], errors="coerce")
        wide.loc[years.ge(1975), "Brazil"] = vals[years.ge(1975)] * 1000
        vals = pd.to_numeric(wide["Brazil"], errors="coerce")
        wide.loc[years.ge(1985), "Brazil"] = vals[years.ge(1985)] * 1_000_000
        vals = pd.to_numeric(wide["Brazil"], errors="coerce")
        wide.loc[years.ge(1985), "Brazil"] = vals[years.ge(1985)] / 1000
        vals = pd.to_numeric(wide["Brazil"], errors="coerce")
        wide.loc[years.ge(1990), "Brazil"] = vals[years.ge(1990)] / 1000
        vals = pd.to_numeric(wide["Brazil"], errors="coerce")
        wide.loc[years.le(1993), "Brazil"] = vals[years.le(1993)] / 2750
    if "Chile" in wide.columns:
        vals = pd.to_numeric(wide["Chile"], errors="coerce")
        wide.loc[years.le(1954), "Chile"] = vals[years.le(1954)] / 1000
        wide = _mitchell_convert_units(wide, "Chile", 1970, 1975, "B")
        vals = pd.to_numeric(wide["Chile"], errors="coerce")
        wide.loc[years.le(1975), "Chile"] = vals[years.le(1975)] / 1000
        wide = _mitchell_convert_units(wide, "Chile", 1983, 2010, "B")
    wide = _mitchell_convert_units(wide, "Colombia", 1975, 2010, "B")
    wide = _mitchell_convert_units(wide, "Ecuador", 1983, 2000, "B")
    if "Ecuador" in wide.columns:
        vals = pd.to_numeric(wide["Ecuador"], errors="coerce")
        wide.loc[years.le(2000), "Ecuador"] = vals[years.le(2000)] / 25000
    if "Paraguay" in wide.columns:
        vals = pd.to_numeric(wide["Paraguay"], errors="coerce")
        wide.loc[years.le(1942), "Paraguay"] = vals[years.le(1942)] / 100
        wide = _mitchell_convert_units(wide, "Paraguay", 1975, 2010, "B")
    if "Peru" in wide.columns:
        vals = pd.to_numeric(wide["Peru"], errors="coerce")
        wide.loc[years.le(1929), "Peru"] = vals[years.le(1929)] / 1000
        vals = pd.to_numeric(wide["Peru"], errors="coerce")
        wide.loc[years.le(1989), "Peru"] = vals[years.le(1989)] / (10 ** 6)
        vals = pd.to_numeric(wide["Peru"], errors="coerce")
        wide.loc[years.le(1954), "Peru"] = vals[years.le(1954)] / 1000
    if "Uruguay" in wide.columns:
        vals = pd.to_numeric(wide["Uruguay"], errors="coerce")
        wide.loc[years.le(1982), "Uruguay"] = vals[years.le(1982)] / 1000
        vals = pd.to_numeric(wide["Uruguay"], errors="coerce")
        wide.loc[years.le(1969), "Uruguay"] = vals[years.le(1969)] / 1000
    if "Venezuela" in wide.columns:
        vals = pd.to_numeric(wide["Venezuela"], errors="coerce")
        wide.loc[:, "Venezuela"] = vals / (10 ** 8)
        vals = pd.to_numeric(wide["Venezuela"], errors="coerce")
        wide.loc[years.le(2000), "Venezuela"] = vals[years.le(2000)] / 1000

    out = _mitchell_reshape(wide, "M0")
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[out["countryname"].eq("Bolivia"), "M0"] = values[out["countryname"].eq("Bolivia")] / 1_000_000
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "M0"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Latam_banknotes.dta")
    return out


def _mitchell_africa_banknotes(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Africa_banknotes")

    sheet2 = _mitchell_import_columns(path, 2)
    sheet2 = _mitchell_destring(_mitchell_drop_blank_year(sheet2))
    if {"CapeofGoodHope", "Natal"}.issubset(sheet2.columns):
        cape = pd.to_numeric(sheet2["CapeofGoodHope"], errors="coerce")
        natal = pd.to_numeric(sheet2["Natal"], errors="coerce")
        sheet2["SouthAfrica"] = cape.add(natal, fill_value=np.nan)
        sheet2["SouthAfrica"] = sheet2["SouthAfrica"].combine_first(cape)
        sheet2 = sheet2.drop(columns=["CapeofGoodHope", "Natal"])
    sheet2 = _mitchell_convert_units(sheet2, "SouthAfrica", 1876, 1909, "Th")
    master: pd.DataFrame | None = _mitchell_reshape(sheet2, "M0")

    sheet3 = _mitchell_import_columns(path, 3)
    sheet3 = _mitchell_destring(_mitchell_drop_blank_year(sheet3))
    master = _mitchell_append(master, _mitchell_reshape(sheet3, "M0"))

    sheet4 = _mitchell_import_columns(path, 4)
    sheet4 = _mitchell_destring(_mitchell_drop_blank_year(sheet4))
    sheet4 = _mitchell_convert_units(sheet4, "SouthAfrica", 1910, 1911, "Th")
    master = _mitchell_append(master, _mitchell_reshape(sheet4, "M0"))

    sheet5 = _mitchell_import_columns(path, 5)
    sheet5 = _mitchell_drop_columns(sheet5, ["FrenchEquatorialAfrica"])
    if "Mozambique" in sheet5.columns:
        sheet5.loc[sheet5["Mozambique"].astype("string").eq("thousand million meticais"), "Mozambique"] = ""
    if "Sudan" in sheet5.columns:
        sheet5.loc[sheet5["Sudan"].astype("string").eq("thousand million dinars"), "Sudan"] = ""
    sheet5 = _mitchell_destring(_mitchell_drop_blank_year(sheet5))
    master = _mitchell_append(master, _mitchell_reshape(sheet5, "M0"))

    wide = master.pivot(index="year", columns="countryname", values="M0").reset_index()
    wide.columns.name = None
    wide = _mitchell_convert_units(wide, "Algeria", 1945, 1960, "B")
    wide = _mitchell_convert_units(wide, "Algeria", 1975, 2010, "B")
    for country in ["Benin", "BurkinaFaso", "CentralAfricanRepublic", "Chad", "Congo", "Gabon", "IvoryCoast", "Madagascar", "Mali", "Niger", "Senegal", "Tunisia"]:
        wide = _mitchell_convert_units(wide, country, 1950, 2010, "B")
    wide = _mitchell_convert_units(wide, "Cameroon", 1961, 2010, "B")
    wide = _mitchell_convert_units(wide, "Ghana", 1985, 2010, "B")
    wide = _mitchell_convert_units(wide, "Morocco", 1945, 1956, "B")
    wide = _mitchell_convert_units(wide, "Uganda", 1966, 1974, "Th")
    wide = _mitchell_convert_units(wide, "Uganda", 2001, 2010, "B")
    wide = _mitchell_convert_units(wide, "SouthAfrica", 1983, 2010, "B")
    wide = _mitchell_convert_currency(wide, "Morocco", 1956, 10)

    years = pd.to_numeric(wide["year"], errors="coerce")
    def _wide_apply(col: str, mask, op: str, factor: float) -> None:
        if col not in wide.columns:
            return
        vals = pd.to_numeric(wide[col], errors="coerce")
        wide.loc[mask, col] = vals[mask] * factor if op == "*" else vals[mask] / factor

    _wide_apply("Morocco", years.le(1956), "/", 100)
    _wide_apply("Nigeria", years.le(1972), "*", 2)
    _wide_apply("SierraLeone", years.le(1963), "*", 2)
    _wide_apply("Ghana", years.le(1964), "/", 0.417)
    _wide_apply("Madagascar", years.le(2000), "/", 5)
    _wide_apply("SouthAfrica", years.le(1959), "*", 2)
    _wide_apply("SouthAfrica", years.ge(1983), "/", 1000)
    _wide_apply("Algeria", years.le(1960), "/", 100)
    _wide_apply("Angola", years.le(1974), "/", 1_000_000)
    _wide_apply("Sudan", years.ge(1994), "*", 1000)
    _wide_apply("Sudan", years.lt(1994), "/", 10)
    _wide_apply("Tanzania", years.lt(1983), "/", 1000)

    out = _mitchell_reshape(wide, "M0")
    years = pd.to_numeric(out["year"], errors="coerce")
    country = out["countryname"]
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Madagascar") & years.between(1950, 1959), "M0"] = values[country.eq("Madagascar") & years.between(1950, 1959)] / 1000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("SierraLeone"), "M0"] = values[country.eq("SierraLeone")] / 1000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Tanzania"), "M0"] = values[country.eq("Tanzania")] * 1000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Togo") & years.ge(1970), "M0"] = values[country.eq("Togo") & years.ge(1970)] * 1000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Tunisia"), "M0"] = values[country.eq("Tunisia")] / 1000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Kenya") & years.le(1965), "M0"] = values[country.eq("Kenya") & years.le(1965)] * 5
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Zaire") & years.le(1967), "M0"] = values[country.eq("Zaire") & years.le(1967)] / 1000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Zaire") & years.le(1979), "M0"] = values[country.eq("Zaire") & years.le(1979)] / 3000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Zaire") & years.le(1977), "M0"] = values[country.eq("Zaire") & years.le(1977)] / 1000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Zaire") & years.le(1993), "M0"] = values[country.eq("Zaire") & years.le(1993)] / 1000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Zaire") & years.le(1991), "M0"] = values[country.eq("Zaire") & years.le(1991)] / 100000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Mozambique") & years.ge(1994), "M0"] = values[country.eq("Mozambique") & years.ge(1994)] * 1000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Zambia") & years.ge(1988), "M0"] = values[country.eq("Zambia") & years.ge(1988)] * 1000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Zimbabwe") & years.le(2000), "M0"] = values[country.eq("Zimbabwe") & years.le(2000)] / 1000
    values = pd.to_numeric(out["M0"], errors="coerce")
    out.loc[country.eq("Zimbabwe") & years.le(1963), "M0"] = values[country.eq("Zimbabwe") & years.le(1963)] / 1000
    out["year"] = years.astype("int32")
    out = out[["countryname", "year", "M0"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Africa_banknotes.dta")
    return out


def _mitchell_oceania_banknotes(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Oceania_banknotes")

    frame = _mitchell_import_columns(path, 2)
    frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
    frame = _mitchell_convert_currency(frame, "NewZealand", 1965, 2)
    if "NewZealand" in frame.columns:
        years = pd.to_numeric(frame["year"], errors="coerce")
        vals = pd.to_numeric(frame["NewZealand"], errors="coerce")
        frame.loc[years.le(1909), "NewZealand"] = vals[years.le(1909)] / 1000
    out = _mitchell_reshape(frame, "M0")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out = out[["countryname", "year", "M0"]].sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(out, temp_dir / "Oceania_banknotes.dta")
    return out


def _mitchell_partial_m0_final(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    parts = [
        _mitchell_africa_banknotes(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_americas_banknotes(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_europe_banknotes(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_asia_banknotes(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_latam_banknotes(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_oceania_banknotes(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
    ]
    return _mitchell_finalize_value_parts(parts, "M0", data_helper_dir=data_helper_dir, euro_cutoff_year=1998)


def _mitchell_finalize_value_parts(
    parts: list[pd.DataFrame],
    value_col: str,
    *,
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    euro_cutoff_year: int,
) -> pd.DataFrame:
    helper_dir = _resolve(data_helper_dir)
    master = pd.concat(parts, ignore_index=True, sort=False)
    master = master.sort_values(["countryname", "year"]).drop_duplicates(["countryname", "year"], keep="last").reset_index(drop=True)
    master = _mitchell_standardize_countrynames(master)
    lookup = _country_name_lookup(helper_dir)
    master["ISO3"] = master["countryname"].map(lookup)
    master = master.loc[master["ISO3"].notna(), ["ISO3", "year", value_col]].copy()
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int32")
    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].drop_duplicates().copy()
    master = master.merge(eur_fx, on="ISO3", how="left", indicator=True)
    euro_mask = master["_merge"].eq("both") & master["ISO3"].ne("CYP") & master["year"].le(euro_cutoff_year)
    cyp_mask = master["_merge"].eq("both") & master["ISO3"].eq("CYP")
    master.loc[euro_mask, value_col] = pd.to_numeric(master.loc[euro_mask, value_col], errors="coerce") / pd.to_numeric(
        master.loc[euro_mask, "EUR_irrevocable_FX"], errors="coerce"
    )
    master.loc[cyp_mask, value_col] = pd.to_numeric(master.loc[cyp_mask, value_col], errors="coerce") / pd.to_numeric(
        master.loc[cyp_mask, "EUR_irrevocable_FX"], errors="coerce"
    )
    master = master.drop(columns=["EUR_irrevocable_FX", "_merge"])
    master = master.sort_values(["ISO3", "year"]).drop_duplicates(["ISO3", "year"], keep="last")
    master = master.loc[~master["ISO3"].isin(["YUG", "SRB", "ZMB"])].copy()
    return master.sort_values(["ISO3", "year"]).reset_index(drop=True)


def _mitchell_partial_govexp_final(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    parts = [
        _mitchell_africa_govexp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_americas_govexp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_asia_govexp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_australia_govexp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_canada_govexp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_europe_govexp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_latam_govexp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_oceania_govexp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
    ]
    return _mitchell_finalize_value_parts(parts, "govexp", data_helper_dir=data_helper_dir, euro_cutoff_year=1993)


def _mitchell_partial_govrev_final(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    parts = [
        _mitchell_africa_govrev(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_americas_govrev(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_asia_govrev(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_australia_govrev(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_canada_govrev(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_europe_govrev(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_latam_govrev(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_oceania_govrev(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
    ]
    return _mitchell_finalize_value_parts(parts, "govrev", data_helper_dir=data_helper_dir, euro_cutoff_year=1993)

def _mitchell_partial_ngdp_final(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    helper_dir = _resolve(data_helper_dir)
    parts = [
        _mitchell_africa_ngdp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_americas_ngdp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_latam_ngdp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_asia_ngdp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_europe_ngdp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_oceania_ngdp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
    ]
    master: pd.DataFrame | None = None
    for part in parts:
        master = _mitchell_merge_values(master, part)
    assert master is not None
    master = _mitchell_standardize_countrynames(master)
    lookup = _country_name_lookup(helper_dir)
    master["ISO3"] = master["countryname"].map(lookup)
    master = master.loc[master["ISO3"].notna(), ["ISO3", "year", "nGDP_LCU"]].copy()
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int32")
    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].drop_duplicates().copy()
    master = master.merge(eur_fx, on="ISO3", how="left", indicator=True)
    euro_mask = master["_merge"].eq("both") & master["ISO3"].ne("CYP") & master["year"].le(1998)
    cyp_mask = master["_merge"].eq("both") & master["ISO3"].eq("CYP")
    master.loc[euro_mask, "nGDP_LCU"] = pd.to_numeric(master.loc[euro_mask, "nGDP_LCU"], errors="coerce") / pd.to_numeric(
        master.loc[euro_mask, "EUR_irrevocable_FX"], errors="coerce"
    )
    master.loc[cyp_mask, "nGDP_LCU"] = pd.to_numeric(master.loc[cyp_mask, "nGDP_LCU"], errors="coerce") / pd.to_numeric(
        master.loc[cyp_mask, "EUR_irrevocable_FX"], errors="coerce"
    )
    master = master.drop(columns=["EUR_irrevocable_FX", "_merge"])
    master = master.loc[~master["ISO3"].isin(["YUG", "SRB", "ZMB"])].copy()
    return master.sort_values(["ISO3", "year"]).reset_index(drop=True)


def _mitchell_merge_values(master: pd.DataFrame | None, current: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    merge_keys = keys or ["countryname", "year"]
    if master is None or master.empty:
        return current.sort_values(merge_keys).reset_index(drop=True)
    current_idx = current.set_index(merge_keys)
    master_idx = master.set_index(merge_keys)
    union_index = current_idx.index.union(master_idx.index)
    out = current_idx.reindex(union_index).copy()
    master_idx = master_idx.reindex(union_index)
    current_present = pd.Series(True, index=current_idx.index).reindex(union_index, fill_value=False)
    for col in master_idx.columns:
        if col in out.columns:
            out[col] = out[col].where(current_present, master_idx[col])
        else:
            out[col] = master_idx[col]
    return out.reset_index().sort_values(merge_keys).reset_index(drop=True)


def _mitchell_africa_bop(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path1 = _mitchell_workbook_path(raw_dir, "Africa_BoP_A")
    path2 = _mitchell_workbook_path(raw_dir, "Africa_BoP_B")

    part1 = _mitchell_import_columns_first(path1, 2)
    part1 = _mitchell_fill_header_rows(part1, 2)
    part1 = _mitchell_keep_by_header(part1, header_row=2, predicate=lambda value: value == "OCB")
    part1 = _mitchell_rename_from_row(part1, 0)
    part1 = _mitchell_destring(_mitchell_drop_blank_year(part1))
    part1 = _mitchell_convert_currency(part1, "Zambia", 1950, 2)
    master = _mitchell_reshape(part1, "CA")

    part2 = _mitchell_import_columns_first(path1, 3)
    part2 = _mitchell_fill_header_rows(part2, 2)
    part2 = _mitchell_keep_columns(part2, ["G"])
    part2 = part2.rename(columns={"G": "SouthAfrica"})
    part2 = _mitchell_destring(_mitchell_drop_blank_year(part2))
    part2 = _mitchell_convert_currency(part2, "SouthAfrica", 1950, 2)
    master = _mitchell_merge_values(master, _mitchell_reshape(part2, "CA"))

    part3 = _mitchell_import_columns_first(path2, 2)
    part3 = _mitchell_fill_header_rows(part3, 2)
    part3 = _mitchell_keep_by_header(
        part3,
        header_row=1,
        predicate=lambda value: value.replace(" ", "") == "OCB",
        normalizer=lambda value: value.replace(" ", ""),
    )
    part3 = _mitchell_rename_from_row(part3, 0)
    part3 = _mitchell_destring(_mitchell_drop_blank_year(part3))
    master = _mitchell_merge_values(master, _mitchell_reshape(part3, "CA_USD"))
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int32")
    master = master.sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(master, temp_dir / "Africa_BoP.dta")
    return master


def _mitchell_americas_bop(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path1 = _mitchell_workbook_path(raw_dir, "Americas_BoP_A")
    path2 = _mitchell_workbook_path(raw_dir, "Americas_BoP_B")

    part1 = _mitchell_import_columns_first(path1, 2)
    part1 = _mitchell_fill_header_rows(part1, 2)
    part1 = _mitchell_keep_by_header(part1, header_row=2, predicate=lambda value: value == "OCB")
    part1 = _mitchell_rename_from_row(part1, 0)
    part1 = part1.rename(
        columns={"O": "CostaRica", "AE": "DominicanRepublic", "AM": "ElSalvador"}
    )
    part1 = _mitchell_destring(_mitchell_drop_blank_year(part1))
    master = _mitchell_reshape(part1, "CA")

    part2 = _mitchell_import_columns_first(path2, 2)
    part2 = _mitchell_fill_header_rows(part2, 2)
    part2 = _mitchell_keep_by_header(part2, header_row=1, predicate=lambda value: value == "OCB")
    part2 = _mitchell_rename_from_row(part2, 0)
    part2 = part2.rename(columns={"CQ": "TrinidadandTobago", "TrinidadTobago": "TrinidadandTobago"})
    part2 = _mitchell_destring(_mitchell_drop_blank_year(part2))
    master = _mitchell_merge_values(master, _mitchell_reshape(part2, "CA_USD"))
    mexico_mask = master["countryname"].eq("Mexico") & pd.to_numeric(master["CA_USD"], errors="coerce").notna()
    master.loc[mexico_mask, "CA"] = pd.to_numeric(master.loc[mexico_mask, "CA_USD"], errors="coerce")
    master.loc[mexico_mask, "CA_USD"] = np.nan
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int32")
    master = master.sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(master, temp_dir / "Americas_BoP.dta")
    return master


def _mitchell_latam_bop(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path1 = _mitchell_workbook_path(raw_dir, "Latam_BoP_A")
    path2 = _mitchell_workbook_path(raw_dir, "Latam_BoP_B")

    # Part A: local-currency OCB, with a subset explicitly flagged as USD.
    part1 = _mitchell_import_columns_first(path1, 2)
    part1 = _mitchell_fill_header_rows(part1, 2)
    part1 = _mitchell_keep_by_header(part1, header_row=3, predicate=lambda value: value == "OCB")
    part1 = _mitchell_drop_rows(part1, [1])
    if len(part1) > 1:
        for col in [c for c in part1.columns if c != "year"]:
            if pd.notna(part1.at[1, col]):
                part1.at[1, col] = str(part1.at[1, col]).lower()
    part1 = _mitchell_rename_from_row(part1, 0)
    part1 = _mitchell_drop_blank_year(part1)
    year_text = part1["year"].astype("string").str.strip()
    # The reference source normalizes year spans like 1921-2/1921-22 and 1924-5/1924-25.
    year_text = year_text.str.replace(r"^1921[^0-9]*2$", "1922", regex=True)
    year_text = year_text.str.replace(r"^1921[^0-9]*22$", "1922", regex=True)
    year_text = year_text.str.replace(r"^1924[^0-9]*5$", "1925", regex=True)
    year_text = year_text.str.replace(r"^1924[^0-9]*25$", "1925", regex=True)
    part1["year"] = year_text
    part1 = _mitchell_destring(part1)
    part1 = _mitchell_convert_currency(part1, "Argentina", 1950, _pow10_literal(-13))
    part1 = _mitchell_convert_currency(part1, "Brazil", 1950, _pow10_literal(-15))
    part1 = _mitchell_convert_currency(part1, "Colombia", 1950, _pow10_literal(-9))
    part1 = _mitchell_convert_currency(part1, "Peru", 1950, _pow10_literal(-9))
    if "Chile" in part1.columns:
        years = pd.to_numeric(part1["year"], errors="coerce")
        part1.loc[years.le(1948), "Chile"] = np.nan
    temp_master = _mitchell_reshape(part1, "CA")
    usd_countries = ["Bolivia", "Chile", "Ecuador", "Paraguay", "Uruguay", "Venezuela"]
    usd_mask = temp_master["countryname"].isin(usd_countries)
    temp_master["CA_USD"] = np.where(usd_mask, pd.to_numeric(temp_master["CA"], errors="coerce"), np.nan)
    temp_master.loc[usd_mask, "CA"] = np.nan

    # Part B: OCB in USD directly.
    part2 = _mitchell_import_columns_first(path2, 2)
    part2 = _mitchell_fill_header_rows(part2, 2)
    part2 = _mitchell_keep_by_header(part2, header_row=2, predicate=lambda value: value == "OCB")
    part2 = _mitchell_drop_rows(part2, [1])
    if len(part2) > 0:
        for col in [c for c in part2.columns if c != "year"]:
            if pd.notna(part2.at[0, col]):
                part2.at[0, col] = str(part2.at[0, col]).replace(" ", "")
    part2 = _mitchell_rename_from_row(part2, 0)
    part2 = _mitchell_drop_blank_year(part2)
    part2 = _mitchell_destring(part2)
    part2_long = _mitchell_reshape(part2, "CA_USD")
    master = _mitchell_merge_values(temp_master, part2_long)
    ven_mask = master["countryname"].eq("Venezuela")
    master.loc[ven_mask, "CA"] = pd.to_numeric(master.loc[ven_mask, "CA"], errors="coerce") / (_pow10_literal(11))

    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int32")
    master = master.sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(master, temp_dir / "Latam_BoP.dta")
    return master


def _mitchell_asia_bop(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path1 = _mitchell_workbook_path(raw_dir, "Asia_BoP_A")
    path2 = _mitchell_workbook_path(raw_dir, "Asia_BoP_B")

    part1 = _mitchell_import_columns_first(path1, 2)
    part1 = _mitchell_fill_header_rows(part1, 2)
    part1 = _mitchell_keep_by_header(part1, header_row=2, predicate=lambda value: value == "OCB")
    part1 = _mitchell_rename_from_row(part1, 0)
    part1 = part1.rename(columns={"CA": "SriLanka"})
    part1 = _mitchell_destring(_mitchell_drop_blank_year(part1))
    ca = _mitchell_reshape(part1, "CA")
    tur_mask = ca["countryname"].eq("Turkey")
    twn_mask = ca["countryname"].eq("Taiwan")
    ca.loc[tur_mask, "CA"] = pd.to_numeric(ca.loc[tur_mask, "CA"], errors="coerce") * (10 ** -6)
    ca.loc[twn_mask, "CA"] = pd.to_numeric(ca.loc[twn_mask, "CA"], errors="coerce") / 40000
    master = ca

    part2 = _mitchell_import_columns_first(path2, 2)
    part2 = _mitchell_fill_header_rows(part2, 2)
    part2 = _mitchell_keep_by_header(part2, header_row=1, predicate=lambda value: value == "OCB")
    part2 = _mitchell_rename_from_row(part2, 0)
    if "Kuwait" in part2.columns:
        kuwait = part2["Kuwait"].astype("string")
        part2.loc[kuwait.eq("14,03"), "Kuwait"] = "14364"
    part2 = _mitchell_destring(_mitchell_drop_blank_year(part2))
    if "Japan" in part2.columns:
        years = pd.to_numeric(part2["year"], errors="coerce")
        part2.loc[years.ge(1970), "Japan"] = pd.to_numeric(part2.loc[years.ge(1970), "Japan"], errors="coerce") * 1000
    master = _mitchell_merge_values(master, _mitchell_reshape(part2, "CA_USD"))
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int32")
    master = master.sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(master, temp_dir / "Asia_BoP.dta")
    return master


def _mitchell_oceania_bop(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path1 = _mitchell_workbook_path(raw_dir, "Oceania_BoP_A")
    path2 = _mitchell_workbook_path(raw_dir, "Oceania_BoP_B")

    part1 = _mitchell_import_columns(path1, 2)
    part1 = _mitchell_keep_columns(part1, ["Australia"])
    part1 = _mitchell_destring(_mitchell_drop_blank_year(part1))
    part1 = _mitchell_convert_currency(part1, "Australia", 1950, 2)
    master = _mitchell_reshape(part1, "CA")

    part2 = _mitchell_import_columns_first(path2, 2)
    part2 = _mitchell_fill_header_rows(part2, 2)
    part2 = _mitchell_keep_by_header(part2, header_row=1, predicate=lambda value: value == "OCB")
    part2 = _mitchell_rename_from_row(part2, 0)
    part2 = part2.rename(columns={"W": "NewZealand"})
    part2 = _mitchell_destring(_mitchell_drop_blank_year(part2))
    master = _mitchell_merge_values(master, _mitchell_reshape(part2, "CA_USD"))
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int32")
    master = master.sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(master, temp_dir / "Oceania_BoP.dta")
    return master


def _mitchell_europe_bop(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Europe_BoP")

    sheet2 = _mitchell_import_columns_first(path, 2)
    sheet2 = _mitchell_fill_header_rows(sheet2, 2)
    sheet2 = _mitchell_keep_by_header(sheet2, header_row=2, predicate=lambda value: value == "OCB")
    sheet2 = _mitchell_rename_from_row(sheet2, 0)
    sheet2 = sheet2.rename(columns={"AI": "UnitedKingdom"})
    sheet2 = _mitchell_destring(_mitchell_drop_blank_year(sheet2))
    sheet2 = _mitchell_convert_currency(sheet2, "France", 1914, 1 / 100)
    master = _mitchell_reshape(sheet2, "CA")

    sheet3 = _mitchell_import_columns_first(path, 3)
    sheet3 = _mitchell_fill_header_rows(sheet3, 2)
    sheet3 = _mitchell_keep_by_header(sheet3, header_row=2, predicate=lambda value: value == "OCB")
    sheet3 = _mitchell_rename_from_row(sheet3, 0)
    sheet3 = sheet3.rename(columns={"BD": "Ireland", "CS": "UnitedKingdom"})
    if "Ireland" not in sheet3.columns and "SouthernIreland" in sheet3.columns:
        sheet3 = sheet3.rename(columns={"SouthernIreland": "Ireland"})
    sheet3 = _mitchell_destring(_mitchell_drop_blank_year(sheet3))
    sheet3 = _mitchell_convert_currency(sheet3, "France", 1944, 1 / 100)
    sheet3 = _mitchell_convert_currency(sheet3, "Finland", 1948, 1 / 100)
    part3 = _mitchell_reshape(sheet3, "CA")
    usd_mask = (
        (part3["countryname"].isin(["France", "Greece"]) & pd.to_numeric(part3["year"], errors="coerce").ge(1945))
        | part3["countryname"].eq("Spain")
    )
    part3["CA_USD"] = np.where(usd_mask, pd.to_numeric(part3["CA"], errors="coerce"), np.nan)
    part3.loc[usd_mask, "CA"] = np.nan
    master = _mitchell_merge_values(master, part3)

    sheet4 = _mitchell_import_columns_first(path, 4)
    sheet4 = _mitchell_fill_header_rows(sheet4, 2)
    sheet4 = _mitchell_keep_by_header(sheet4, header_row=2, predicate=lambda value: value == "OCB")
    sheet4 = _mitchell_rename_from_row(sheet4, 0)
    sheet4 = sheet4.rename(columns={"AO": "Germany", "BC": "Ireland", "DG": "UnitedKingdom"})
    if "Germany" not in sheet4.columns and "WestGermany" in sheet4.columns:
        sheet4 = sheet4.rename(columns={"WestGermany": "Germany"})
    if "Ireland" not in sheet4.columns and "SouthernIreland" in sheet4.columns:
        sheet4 = sheet4.rename(columns={"SouthernIreland": "Ireland"})
    sheet4 = _mitchell_destring(_mitchell_drop_blank_year(sheet4))
    if "Germany" in sheet4.columns:
        sheet4["Germany"] = pd.to_numeric(sheet4["Germany"], errors="coerce") * 1000
    master = _mitchell_merge_values(master, _mitchell_reshape(sheet4, "CA_USD"))

    for sheet_name in (5, 6):
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 2)
        frame = _mitchell_keep_by_header(frame, header_row=1, predicate=lambda value: value == "OCB")
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        master = _mitchell_merge_values(master, _mitchell_reshape(frame, "CA_USD"))

    years = pd.to_numeric(master["year"], errors="coerce")
    germany_mask = master["countryname"].eq("Germany") & years.le(1913)
    master.loc[germany_mask, "CA"] = pd.to_numeric(master.loc[germany_mask, "CA"], errors="coerce") / (10 ** 12)
    bulgaria_mask = master["countryname"].eq("Bulgaria")
    master.loc[bulgaria_mask, "CA"] = pd.to_numeric(master.loc[bulgaria_mask, "CA"], errors="coerce") / 1_000_000
    master["year"] = years.astype("int32")
    master = master.sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(master, temp_dir / "Europe_BoP.dta")
    return master


_MITCHELL_COUNTRY_RENAMES = {
    "BurkinaFaso": "Burkina Faso",
    "CentralAfricanRepublic": "Central African Republic",
    "Zaire": "Democratic Republic of the Congo",
    "Congo": "Republic of the Congo",
    "CostaRica": "Costa Rica",
    "DominicanRepublic": "Dominican Republic",
    "EastGermany": "German Democratic Republic",
    "ElSalvador": "El Salvador",
    "HongKong": "Hong Kong",
    "IvoryCoast": "Ivory Coast",
    "NewZealand": "New Zealand",
    "Papua-NewGuinea": "Papua New Guinea",
    "PapuaNewGuinea": "Papua New Guinea",
    "PuertoRico": "Puerto Rico",
    "Russia": "Russian Federation",
    "RussiaUSSR": "Russian Federation",
    "Runion": "Réunion",
    "SaudiArabia": "Saudi Arabia",
    "SerbiaYugoslavia": "Serbia",
    "SierraLeone": "Sierra Leone",
    "SouthAfrica": "South Africa",
    "SouthKorea": "South Korea",
    "SriLanka": "Sri Lanka",
    "TrinidadandTobago": "Trinidad and Tobago",
    "USA": "United States",
    "USSR": "Soviet Union",
    "UnitedKingdom": "United Kingdom",
    "UnitedArabEmirates": "United Arab Emirates",
    "NorthKorea": "North Korea",
    "CzechRepublic": "Czech Republic",
}


def _mitchell_standardize_countrynames(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "countryname" not in out.columns:
        return out
    out["countryname"] = out["countryname"].astype(str).replace(_MITCHELL_COUNTRY_RENAMES)
    out = out.loc[~out["countryname"].isin(["Hawaii", "Korea", "FrenchEquatorialAfrica", "FrenchWestAfrica", "NIreland"])].copy()
    return out


def _mitchell_partial_bop_final(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    helper_dir = _resolve(data_helper_dir)
    parts = [
        _mitchell_africa_bop(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_americas_bop(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_latam_bop(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_asia_bop(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_europe_bop(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_oceania_bop(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
    ]
    master = pd.concat(parts, ignore_index=True, sort=False)
    master = _mitchell_standardize_countrynames(master)
    lookup = _country_name_lookup(helper_dir)
    master["ISO3"] = master["countryname"].map(lookup)
    master = master.loc[master["ISO3"].notna()].copy()
    master = master.drop(columns=["countryname"])
    master = master[["ISO3", "year", "CA", "CA_USD"]].copy()
    master["year"] = pd.to_numeric(master["year"], errors="coerce")
    master["CA"] = pd.to_numeric(master["CA"], errors="coerce")
    master["CA_USD"] = pd.to_numeric(master["CA_USD"], errors="coerce")

    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].drop_duplicates().copy()
    master = master.merge(eur_fx, on="ISO3", how="left", indicator=True)
    euro_mask = master["_merge"].eq("both") & master["ISO3"].ne("CYP") & master["year"].le(1993)
    master.loc[euro_mask, "CA"] = pd.to_numeric(master.loc[euro_mask, "CA"], errors="coerce") / pd.to_numeric(master.loc[euro_mask, "EUR_irrevocable_FX"], errors="coerce")
    cyp_mask = master["_merge"].eq("both") & master["ISO3"].eq("CYP")
    master.loc[cyp_mask, "CA"] = pd.to_numeric(master.loc[cyp_mask, "CA"], errors="coerce") / pd.to_numeric(master.loc[cyp_mask, "EUR_irrevocable_FX"], errors="coerce")
    master = master.drop(columns=["EUR_irrevocable_FX", "_merge"])

    null_ca = master["ISO3"].isin(["ECU", "IRQ", "MEX", "SLE", "SDN", "TWN", "VEN", "ZMB"])
    master.loc[null_ca, "CA"] = np.nan
    master.loc[null_ca, "CA_USD"] = np.nan
    master = master.loc[~master["ISO3"].isin(["YUG", "SRB", "ZMB"])].copy()
    master["year"] = master["year"].astype("int32")
    return master.sort_values(["ISO3", "year"]).reset_index(drop=True)


def _mitchell_oceania_trade(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Oceania_trade")

    master: pd.DataFrame | None = None

    def _sheet_imports(sheet_name: int) -> pd.DataFrame:
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 2)
        frame = _mitchell_keep_by_header(
            frame,
            header_row=1,
            predicate=lambda value: value == "imports",
            normalizer=lambda value: value.replace(" ", "").strip().lower(),
        )
        frame = _mitchell_rename_from_row(frame, 0)
        if sheet_name == 4:
            frame = frame.rename(columns={"WesternSamoa": "Samoa"})
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 2:
            for country in ["NewZealand", "Hawaii"]:
                frame = _mitchell_convert_units(frame, country, 1825, 1874, "Th")
        elif sheet_name == 3:
            for country in ["Fiji", "FrenchPolynesia", "Samoa"]:
                frame = _mitchell_convert_units(frame, country, 1875, 1919, "Th")
            frame = _mitchell_convert_units(frame, "NewZealand", 1875, 1904, "Th")
            frame = _mitchell_convert_units(frame, "NewCaledonia", 1875, 1914, "Th")
        elif sheet_name == 4:
            frame = _mitchell_convert_units(frame, "Fiji", 1920, 1949, "Th")
            frame = _mitchell_convert_units(frame, "Samoa", 1920, 1959, "Th")
        return _mitchell_reshape(frame, "imports")

    def _sheet_exports(sheet_name: int) -> pd.DataFrame:
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 2)
        frame = _mitchell_keep_by_header(
            frame,
            header_row=1,
            predicate=lambda value: value == "exports",
            normalizer=lambda value: value.replace(" ", "").strip().lower(),
        )
        frame = _mitchell_rename_from_row(frame, 0)
        if sheet_name == 4:
            frame = frame.rename(columns={"WesternSamoa": "Samoa"})
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if sheet_name == 2:
            for country in ["NewZealand", "Hawaii"]:
                frame = _mitchell_convert_units(frame, country, 1825, 1874, "Th")
        elif sheet_name == 3:
            for country in ["Fiji", "FrenchPolynesia", "Samoa"]:
                frame = _mitchell_convert_units(frame, country, 1875, 1919, "Th")
            frame = _mitchell_convert_units(frame, "NewZealand", 1875, 1904, "Th")
            frame = _mitchell_convert_units(frame, "NewCaledonia", 1875, 1914, "Th")
        elif sheet_name == 4:
            frame = _mitchell_convert_units(frame, "Fiji", 1920, 1949, "Th")
            frame = _mitchell_convert_units(frame, "Samoa", 1920, 1959, "Th")
        return _mitchell_reshape(frame, "exports")

    for sheet_name in (2, 3, 4, 5):
        master = _mitchell_merge_values(master, _sheet_imports(sheet_name))
        master = _mitchell_merge_values(master, _sheet_exports(sheet_name))

    years = pd.to_numeric(master["year"], errors="coerce")
    drop_mask = (
        (master["countryname"].eq("Samoa") & years.ge(1993))
        | (master["countryname"].isin(["Australia", "Fiji", "FrenchPolynesia", "NewCaledonia", "NewZealand", "PapuaNewGuinea"]) & years.ge(1997))
    )
    master = master.loc[~drop_mask].copy()
    years = pd.to_numeric(master["year"], errors="coerce")

    for country, end_year in [("Australia", 1959), ("NewZealand", 1959), ("Fiji", 1964)]:
        mask = master["countryname"].eq(country) & years.le(end_year)
        for col in ["imports", "exports"]:
            master.loc[mask, col] = pd.to_numeric(master.loc[mask, col], errors="coerce") * 2
    for country in ["FrenchPolynesia", "NewCaledonia"]:
        mask = master["countryname"].eq(country) & years.le(1949)
        for col in ["imports", "exports"]:
            master.loc[mask, col] = pd.to_numeric(master.loc[mask, col], errors="coerce") / 5.5
    samoa_pre1915 = master["countryname"].eq("Samoa") & years.le(1914)
    samoa_pre1960 = master["countryname"].eq("Samoa") & years.le(1959)
    for col in ["imports", "exports"]:
        master.loc[samoa_pre1915, col] = pd.to_numeric(master.loc[samoa_pre1915, col], errors="coerce") / 25.4377
        master.loc[samoa_pre1960, col] = pd.to_numeric(master.loc[samoa_pre1960, col], errors="coerce") * 2

    master["year"] = years.astype("int32")
    master = master.sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(master, temp_dir / "Oceania_trade.dta")
    return master


def _mitchell_americas_trade(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Americas_trade")

    master: pd.DataFrame | None = None

    def _sheet_trade(sheet_name: int, kind: str) -> pd.DataFrame:
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 2)
        frame = _mitchell_keep_by_header(
            frame,
            header_row=1,
            predicate=lambda value: value == kind,
            normalizer=lambda value: value.strip().lower(),
        )
        frame = _mitchell_rename_from_row(frame, 0)
        if sheet_name in (5, 6) and "TrinidadTobago" in frame.columns:
            frame = frame.rename(columns={"TrinidadTobago": "Trinidad"})
        if sheet_name == 5:
            frame = frame.rename(columns={"AH": "Trinidad", "AI": "Trinidad"})
        if sheet_name == 6:
            frame = frame.rename(columns={"AF": "Trinidad", "AG": "Trinidad"})
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))

        if sheet_name == 2:
            for country in ["Barbados", "Jamaica", "Mexico", "Trinidad"]:
                frame = _mitchell_convert_units(frame, country, 1790, 1839, "Th")
        elif sheet_name == 3:
            for country in ["Barbados", "ElSalvador", "Guatemala", "Jamaica", "Nicaragua", "Trinidad"]:
                frame = _mitchell_convert_units(frame, country, 1840, 1884, "Th")
            frame = _mitchell_convert_units(frame, "Mexico", 1840, 1871, "Th")
        elif sheet_name == 4:
            for country in ["Barbados", "Jamaica", "Honduras", "Trinidad"]:
                frame = _mitchell_convert_units(frame, country, 1885, 1929, "Th")
            frame = _mitchell_convert_units(frame, "ElSalvador", 1885, 1900, "Th")
            frame = _mitchell_convert_units(frame, "Guatemala", 1885, 1919, "Th")
            frame = _mitchell_convert_units(frame, "Nicaragua", 1885, 1924, "Th")
        elif sheet_name == 5:
            frame = _mitchell_convert_units(frame, "Barbados", 1930, 1939, "Th")
            frame = _mitchell_convert_units(frame, "Honduras", 1930, 1938, "Th")
            frame = _mitchell_convert_units(frame, "Jamaica", 1930, 1949, "Th")
            frame = _mitchell_convert_units(frame, "Trinidad", 1930, 1944, "Th")
        elif sheet_name == 6:
            frame = _mitchell_convert_units(frame, "Mexico", 1980, 2010, "B")

        return _mitchell_reshape(frame, kind)

    for sheet_name in (2, 3, 4, 5, 6):
        master = _mitchell_merge_values(master, _sheet_trade(sheet_name, "imports"))
        master = _mitchell_merge_values(master, _sheet_trade(sheet_name, "exports"))

    master.loc[master["countryname"].isin(["Trinidad", "TrinidadTobago"]), "countryname"] = "TrinidadandTobago"
    years = pd.to_numeric(master["year"], errors="coerce")
    master["exports_USD"] = pd.Series(np.nan, index=master.index, dtype="float64")
    master["imports_USD"] = pd.Series(np.nan, index=master.index, dtype="float64")
    usd_rules = [
        ("DominicanRepublic", 1989),
        ("Nicaragua", 1988),
        ("Honduras", 1989),
        ("Mexico", 1994),
        ("Jamaica", 1998),
        ("CostaRica", 1938),
        ("Guatemala", 1975),
        ("TrinidadandTobago", 1998),
    ]
    for country, start_year in usd_rules:
        mask = master["countryname"].eq(country) & years.ge(start_year)
        master.loc[mask, "exports_USD"] = pd.to_numeric(master.loc[mask, "exports"], errors="coerce").to_numpy(dtype="float64")
        master.loc[mask, "imports_USD"] = pd.to_numeric(master.loc[mask, "imports"], errors="coerce").to_numpy(dtype="float64")
    usa_mask = master["countryname"].eq("USA")
    master.loc[usa_mask, "exports_USD"] = pd.to_numeric(master.loc[usa_mask, "exports"], errors="coerce").to_numpy(dtype="float64")
    master.loc[usa_mask, "imports_USD"] = pd.to_numeric(master.loc[usa_mask, "imports"], errors="coerce").to_numpy(dtype="float64")
    master.loc[master["exports_USD"].notna(), "exports"] = np.nan
    master.loc[master["imports_USD"].notna(), "imports"] = np.nan

    jamaica_mask = master["countryname"].eq("Jamaica") & years.le(1949)
    nicaragua_10 = master["countryname"].eq("Nicaragua") & years.le(1973)
    guadeloupe_mask = master["countryname"].eq("Guadeloupe") & years.le(1957)
    martinique_mask = master["countryname"].eq("Martinique") & years.le(1957)
    mexico_mask = master["countryname"].eq("Mexico") & years.le(1993)
    nicaragua_5000 = master["countryname"].eq("Nicaragua") & years.le(1990)
    nicaragua_1912 = master["countryname"].eq("Nicaragua") & years.le(1912)
    salvador_mask = master["countryname"].eq("ElSalvador") & years.le(1999)
    for col in ["imports", "exports"]:
        master.loc[jamaica_mask, col] = pd.to_numeric(master.loc[jamaica_mask, col], errors="coerce") * 2
        master.loc[nicaragua_10, col] = pd.to_numeric(master.loc[nicaragua_10, col], errors="coerce") * 10
        master.loc[guadeloupe_mask, col] = pd.to_numeric(master.loc[guadeloupe_mask, col], errors="coerce") / 100
        master.loc[martinique_mask, col] = pd.to_numeric(master.loc[martinique_mask, col], errors="coerce") / 100
        master.loc[mexico_mask, col] = pd.to_numeric(master.loc[mexico_mask, col], errors="coerce") / 1000
        master.loc[nicaragua_5000, col] = pd.to_numeric(master.loc[nicaragua_5000, col], errors="coerce") / 5000
        master.loc[master["countryname"].eq("Nicaragua"), col] = pd.to_numeric(master.loc[master["countryname"].eq("Nicaragua"), col], errors="coerce") / 1_000_000
        master.loc[nicaragua_1912, col] = pd.to_numeric(master.loc[nicaragua_1912, col], errors="coerce") / 12.5
        master.loc[salvador_mask, col] = pd.to_numeric(master.loc[salvador_mask, col], errors="coerce") / 8.5

    master["year"] = years.astype("int32")
    master = master.sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(master, temp_dir / "Americas_trade.dta")
    return master


def _mitchell_europe_trade(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Europe_trade")

    master: pd.DataFrame | None = None

    def _base_sheet(sheet_name: int) -> pd.DataFrame:
        frame = _mitchell_import_columns_first(path, sheet_name)
        return _mitchell_fill_header_rows(frame, 2)

    sheet2_i = _base_sheet(2)
    sheet2_i = _mitchell_keep_by_header(sheet2_i, header_row=2, predicate=lambda value: value == "I")
    for col in ["P", "B"]:
        if col in sheet2_i.columns:
            sheet2_i.at[0, col] = col
    sheet2_i = _mitchell_rename_from_row(sheet2_i, 0).rename(columns={"P": "UnitedKingdom", "B": "AustriaHungary"})
    sheet2_i = _mitchell_destring(_mitchell_drop_blank_year(sheet2_i))
    master = _mitchell_merge_values(master, _mitchell_reshape(sheet2_i, "imports"))

    sheet2_e = _base_sheet(2)
    sheet2_e = _mitchell_keep_by_header(sheet2_e, header_row=2, predicate=lambda value: value != "I")
    for col in ["Q", "R", "C"]:
        if col in sheet2_e.columns:
            sheet2_e.at[0, col] = col
    sheet2_e = _mitchell_rename_from_row(sheet2_e, 0).rename(columns={"Q": "UnitedKingdom", "R": "UnitedKingdom_R", "C": "AustriaHungary"})
    if "Russia" in sheet2_e.columns:
        sheet2_e.loc[sheet2_e["Russia"].astype("string").eq("93/4"), "Russia"] = "93.4"
    sheet2_e = _mitchell_destring(_mitchell_drop_blank_year(sheet2_e))
    if {"UnitedKingdom", "UnitedKingdom_R"}.issubset(sheet2_e.columns):
        sheet2_e["UnitedKingdom"] = pd.to_numeric(sheet2_e["UnitedKingdom"], errors="coerce") + pd.to_numeric(sheet2_e["UnitedKingdom_R"], errors="coerce")
        sheet2_e = sheet2_e.drop(columns=["UnitedKingdom_R"])
    master = _mitchell_merge_values(master, _mitchell_reshape(sheet2_e, "exports"))

    sheet3_i = _base_sheet(3)
    sheet3_i = _mitchell_keep_by_header(sheet3_i, header_row=2, predicate=lambda value: value == "I")
    for col in ["AJ", "B"]:
        if col in sheet3_i.columns:
            sheet3_i.at[0, col] = col
    sheet3_i = _mitchell_rename_from_row(sheet3_i, 0).rename(columns={"AJ": "UnitedKingdom", "B": "AustriaHungary"})
    sheet3_i = _mitchell_destring(_mitchell_drop_blank_year(sheet3_i))
    master = _mitchell_merge_values(master, _mitchell_reshape(sheet3_i, "imports"))

    sheet3_e = _base_sheet(3)
    sheet3_e = _mitchell_keep_by_header(sheet3_e, header_row=2, predicate=lambda value: value != "I")
    for col in ["AK", "AL", "C"]:
        if col in sheet3_e.columns:
            sheet3_e.at[0, col] = col
    sheet3_e = _mitchell_rename_from_row(sheet3_e, 0).rename(columns={"AK": "UnitedKingdom", "AL": "UnitedKingdom_R", "C": "AustriaHungary"})
    sheet3_e = _mitchell_destring(_mitchell_drop_blank_year(sheet3_e))
    if {"UnitedKingdom", "UnitedKingdom_R"}.issubset(sheet3_e.columns):
        sheet3_e["UnitedKingdom"] = pd.to_numeric(sheet3_e["UnitedKingdom"], errors="coerce") + pd.to_numeric(sheet3_e["UnitedKingdom_R"], errors="coerce")
        sheet3_e = sheet3_e.drop(columns=["UnitedKingdom_R"])
    master = _mitchell_merge_values(master, _mitchell_reshape(sheet3_e, "exports"))

    sheet4_i = _base_sheet(4)
    if "U" in sheet4_i.columns:
        sheet4_i["EastGermany"] = sheet4_i["U"]
    imports_cols_4 = [
        col
        for col in sheet4_i.columns
        if col not in {"year", "EastGermany"} and str(sheet4_i.at[2, col] if 2 < len(sheet4_i) else "") == "I"
    ]
    sheet4_i = sheet4_i[["year", "EastGermany"] + imports_cols_4].copy()
    for col in ["D", "AN", "AV", "AY"]:
        if col in sheet4_i.columns:
            sheet4_i.at[0, col] = col
    renamed_i = {}
    for col in [c for c in sheet4_i.columns if c not in {"year", "EastGermany"}]:
        newname = _mitchell_sanitize_name(sheet4_i.at[0, col])
        if newname:
            renamed_i[col] = newname
    sheet4_i = sheet4_i.rename(columns=renamed_i).rename(columns={"D": "Austria", "AN": "Russia", "AV": "UnitedKingdom", "AY": "Yugoslavia"})
    sheet4_i = _mitchell_destring(_mitchell_drop_blank_year(sheet4_i))
    year4_i = pd.to_numeric(sheet4_i["year"], errors="coerce")
    if "Austria" in sheet4_i.columns:
        sheet4_i.loc[year4_i.le(1919), "Austria"] = np.nan
    sheet4_i.loc[year4_i.le(1947), "EastGermany"] = np.nan
    for country in ["Greece", "Italy"]:
        sheet4_i = _mitchell_convert_units(sheet4_i, country, 1946, 1949, "B")
    master = _mitchell_merge_values(master, _mitchell_reshape(sheet4_i, "imports"))

    sheet4_e = _base_sheet(4)
    if "T" in sheet4_e.columns:
        sheet4_e["EastGermany"] = sheet4_e["T"]
    exports_cols_4 = [
        col
        for col in sheet4_e.columns
        if col not in {"year", "EastGermany", "S", "U"} and str(sheet4_e.at[2, col] if 2 < len(sheet4_e) else "") != "I"
    ]
    sheet4_e = sheet4_e[["year", "EastGermany"] + exports_cols_4].copy()
    for col in ["E", "AO", "AW", "AX", "AZ"]:
        if col in sheet4_e.columns:
            sheet4_e.at[0, col] = col
    renamed_e = {}
    for col in [c for c in sheet4_e.columns if c not in {"year", "EastGermany"}]:
        newname = _mitchell_sanitize_name(sheet4_e.at[0, col])
        if newname:
            renamed_e[col] = newname
    sheet4_e = sheet4_e.rename(columns=renamed_e).rename(columns={"E": "Austria", "AO": "Russia", "AW": "UnitedKingdom", "AX": "UnitedKingdom_R", "AZ": "Yugoslavia"})
    sheet4_e = _mitchell_destring(_mitchell_drop_blank_year(sheet4_e))
    year4_e = pd.to_numeric(sheet4_e["year"], errors="coerce")
    if "Austria" in sheet4_e.columns:
        sheet4_e.loc[year4_e.le(1919), "Austria"] = np.nan
    sheet4_e.loc[year4_e.le(1947), "EastGermany"] = np.nan
    for country in ["Greece", "Italy"]:
        sheet4_e = _mitchell_convert_units(sheet4_e, country, 1946, 1949, "B")
    if {"UnitedKingdom", "UnitedKingdom_R"}.issubset(sheet4_e.columns):
        sheet4_e["UnitedKingdom"] = pd.to_numeric(sheet4_e["UnitedKingdom"], errors="coerce") + pd.to_numeric(sheet4_e["UnitedKingdom_R"], errors="coerce")
        sheet4_e = sheet4_e.drop(columns=["UnitedKingdom_R"])
    master = _mitchell_merge_values(master, _mitchell_reshape(sheet4_e, "exports"))

    sheet5_i = _base_sheet(5)
    sheet5_i = _mitchell_keep_by_header(sheet5_i, header_row=2, predicate=lambda value: value == "I")
    for col in ["R", "T", "AN", "AV"]:
        if col in sheet5_i.columns:
            sheet5_i.at[0, col] = col
    sheet5_i = _mitchell_rename_from_row(sheet5_i, 0).rename(columns={"R": "EastGermany", "T": "Germany", "AN": "Russia", "AV": "UnitedKingdom"})
    if "EastGermany" in sheet5_i.columns:
        sheet5_i.loc[sheet5_i["year"].astype("string").eq("1990"), "EastGermany"] = ""
    sheet5_i = _mitchell_destring(_mitchell_drop_blank_year(sheet5_i))
    years5_i = pd.to_numeric(sheet5_i["year"], errors="coerce")
    if "Czechoslovakia" in sheet5_i.columns:
        sheet5_i["Czech"] = np.where(years5_i.ge(1994), pd.to_numeric(sheet5_i["Czechoslovakia"], errors="coerce"), np.nan)
        sheet5_i.loc[years5_i.ge(1994), "Czechoslovakia"] = np.nan
    for country in [c for c in sheet5_i.columns if c not in {"year", "Italy"}]:
        if country != "year":
            sheet5_i = _mitchell_convert_units(sheet5_i, country, 1950, 1993, "B")
    sheet5_i = _mitchell_convert_units(sheet5_i, "Italy", 1950, 1993, "Tri")
    master = _mitchell_merge_values(master, _mitchell_reshape(sheet5_i, "imports"))

    sheet5_e = _base_sheet(5)
    sheet5_e = _mitchell_keep_by_header(sheet5_e, header_row=2, predicate=lambda value: value != "I")
    for col in ["S", "U", "AO", "AW", "AX"]:
        if col in sheet5_e.columns:
            sheet5_e.at[0, col] = col
    sheet5_e = _mitchell_rename_from_row(sheet5_e, 0).rename(columns={"S": "EastGermany", "U": "Germany", "AO": "Russia", "AW": "UnitedKingdom", "AX": "UnitedKingdom_R"})
    sheet5_e = _mitchell_destring(_mitchell_drop_blank_year(sheet5_e))
    years5_e = pd.to_numeric(sheet5_e["year"], errors="coerce")
    if {"UnitedKingdom", "UnitedKingdom_R"}.issubset(sheet5_e.columns):
        uk_r = pd.to_numeric(sheet5_e["UnitedKingdom_R"], errors="coerce")
        sheet5_e["UnitedKingdom"] = np.where(
            uk_r.notna(),
            pd.to_numeric(sheet5_e["UnitedKingdom"], errors="coerce") + uk_r,
            pd.to_numeric(sheet5_e["UnitedKingdom"], errors="coerce"),
        )
        sheet5_e = sheet5_e.drop(columns=["UnitedKingdom_R"])
    if "Czechoslovakia" in sheet5_e.columns:
        sheet5_e["Czech"] = np.where(years5_e.ge(1994), pd.to_numeric(sheet5_e["Czechoslovakia"], errors="coerce"), np.nan)
        sheet5_e.loc[years5_e.ge(1994), "Czechoslovakia"] = np.nan
    for country in [c for c in sheet5_e.columns if c not in {"year", "Italy"}]:
        if country != "year":
            sheet5_e = _mitchell_convert_units(sheet5_e, country, 1950, 1993, "B")
    sheet5_e = _mitchell_convert_units(sheet5_e, "Italy", 1950, 1993, "Tri")
    master = _mitchell_merge_values(master, _mitchell_reshape(sheet5_e, "exports"))

    years = pd.to_numeric(master["year"], errors="coerce")
    for country in ["France", "Finland"]:
        end_year = 1958 if country == "France" else 1945
        mask = master["countryname"].eq(country) & years.le(end_year)
        for col in ["imports", "exports"]:
            master.loc[mask, col] = pd.to_numeric(master.loc[mask, col], errors="coerce") / 100
    greece_mask = master["countryname"].eq("Greece") & years.le(1953)
    for col in ["imports", "exports"]:
        master.loc[greece_mask, col] = pd.to_numeric(master.loc[greece_mask, col], errors="coerce") / 1000

    usd_mask = years.ge(1994)
    export_values = pd.to_numeric(master["exports"], errors="coerce")
    master["exports_USD"] = np.where(usd_mask, export_values, np.nan)
    master["imports_USD"] = np.where(usd_mask, export_values, np.nan)
    master.loc[usd_mask, "imports"] = np.nan
    master.loc[usd_mask, "exports"] = np.nan
    italy_mask = master["countryname"].eq("Italy") & years.between(1950, 1969)
    ireland_mask = master["countryname"].eq("Ireland") & years.ge(1950)
    poland_mask = master["countryname"].eq("Poland")
    bulgaria_early = master["countryname"].eq("Bulgaria") & years.le(1948)
    bulgaria_late = master["countryname"].eq("Bulgaria") & years.ge(1952)
    romania_all = master["countryname"].eq("Romania")
    romania_early = romania_all & years.le(1946)
    germany_early = master["countryname"].eq("Germany") & years.le(1913)
    germany_gap = master["countryname"].eq("Germany") & years.between(1920, 1923)
    for col in ["imports", "exports"]:
        master.loc[italy_mask, col] = pd.to_numeric(master.loc[italy_mask, col], errors="coerce") / 1000
        master.loc[ireland_mask, col] = pd.to_numeric(master.loc[ireland_mask, col], errors="coerce") / 1000
        master.loc[poland_mask, col] = pd.to_numeric(master.loc[poland_mask, col], errors="coerce") / 1000
        master.loc[bulgaria_early, col] = pd.to_numeric(master.loc[bulgaria_early, col], errors="coerce") / 1_000_000
        master.loc[bulgaria_late, col] = pd.to_numeric(master.loc[bulgaria_late, col], errors="coerce") / 1000
        master.loc[romania_all, col] = pd.to_numeric(master.loc[romania_all, col], errors="coerce") / 10000
        master.loc[romania_early, col] = pd.to_numeric(master.loc[romania_early, col], errors="coerce") / 20000
        master.loc[germany_early, col] = pd.to_numeric(master.loc[germany_early, col], errors="coerce") / (10 ** 12)
        master.loc[germany_gap, col] = np.nan

    master["year"] = years.astype("int32")
    master = master.sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(master, temp_dir / "Europe_trade.dta")
    return master


def _mitchell_canada_trade(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Canada_trade")

    def _sheet(kind: str) -> pd.DataFrame:
        frame = _mitchell_import_columns_first(path, 2)
        frame = _mitchell_fill_header_rows(frame, 2)
        frame = _mitchell_keep_by_header(
            frame,
            header_row=1,
            predicate=lambda value: value == kind,
            normalizer=lambda value: value.strip().lower(),
        )
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        frame["Canada"] = 0.0
        for col in [c for c in frame.columns if c not in {"year", "Canada"}]:
            values = pd.to_numeric(frame[col], errors="coerce").fillna(0)
            frame[col] = values
            frame["Canada"] = pd.to_numeric(frame["Canada"], errors="coerce") + values
        frame = _mitchell_convert_units(frame, "Canada", 1830, 1867, "Th")
        frame = frame[["year", "Canada"]].copy()
        return _mitchell_reshape(frame, kind)

    master = _mitchell_merge_values(None, _sheet("imports"))
    master = _mitchell_merge_values(master, _sheet("exports"))
    for col in ["imports", "exports"]:
        master[col] = pd.to_numeric(master[col], errors="coerce")
        master.loc[master[col].eq(0), col] = np.nan
        master[col] = master[col] * 4.86
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int32")
    master = master.sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(master, temp_dir / "Canada_trade.dta")
    return master


def _mitchell_latam_trade(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Latam_trade")

    def _sheet(kind: str, sheet_name: int) -> pd.DataFrame:
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 2)
        frame = _mitchell_keep_by_header(
            frame,
            header_row=1,
            predicate=lambda value: value == kind,
            normalizer=lambda value: value.replace(" ", "").strip().lower(),
        )
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))

        if sheet_name == 2:
            frame = _mitchell_convert_units(frame, "Guyana", 1821, 1854, "Th")
        elif sheet_name == 3:
            frame = _mitchell_convert_units(frame, "Guyana", 1855, 1894, "Th")
        elif sheet_name == 4:
            frame = _mitchell_convert_units(frame, "Guyana", 1895, 1934, "Th")
            frame = _mitchell_convert_units(frame, "Paraguay", 1895, 1909, "Th")
        return _mitchell_reshape(frame, kind)

    master: pd.DataFrame | None = None
    for sheet_name in (2, 3, 4, 5, 6):
        master = _mitchell_merge_values(master, _sheet("imports", sheet_name))
        master = _mitchell_merge_values(master, _sheet("exports", sheet_name))

    assert master is not None
    years = pd.to_numeric(master["year"], errors="coerce")
    for col in ["imports", "exports"]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")

    guyana_3545 = master["countryname"].eq("Guyana") & years.between(1935, 1945)
    master.loc[guyana_3545, "imports"] = master.loc[guyana_3545, "imports"] / 1000
    master.loc[guyana_3545, "exports"] = master.loc[guyana_3545, "exports"] / 1000
    guyana_pre46 = master["countryname"].eq("Guyana") & years.le(1945)
    master.loc[guyana_pre46, "imports"] = master.loc[guyana_pre46, "imports"] * 4.5
    master.loc[guyana_pre46, "exports"] = master.loc[guyana_pre46, "exports"] * 4.5

    venezuela = master["countryname"].eq("Venezuela")
    ecuador = master["countryname"].eq("Ecuador")
    master.loc[venezuela, "imports"] = master.loc[venezuela, "imports"] / (10 ** 11)
    master.loc[venezuela, "exports"] = master.loc[venezuela, "exports"] / (10 ** 11)
    venezuela_pre75 = venezuela & years.le(1974)
    master.loc[venezuela_pre75, "imports"] = master.loc[venezuela_pre75, "imports"] / 1000
    master.loc[venezuela_pre75, "exports"] = master.loc[venezuela_pre75, "exports"] / 1000
    master.loc[ecuador, "imports"] = master.loc[ecuador, "imports"] / 1000
    master.loc[ecuador, "exports"] = master.loc[ecuador, "exports"] / 1000

    master["imports_USD"] = pd.Series(np.nan, index=master.index, dtype="float64")
    master["exports_USD"] = pd.Series(np.nan, index=master.index, dtype="float64")
    usd_rules = [
        ("Uruguay", lambda y: y > 1930),
        ("Argentina", lambda y: y > 1948),
        ("Bolivia", lambda y: y >= 1936),
        ("Brazil", lambda y: y > 1948),
        ("Chile", lambda y: y >= 1967),
        ("Colombia", lambda y: y > 1948),
        ("Ecuador", lambda y: y >= 1950),
        ("Paraguay", lambda y: y >= 1950),
        ("Peru", lambda y: y > 1952),
        ("Guyana", lambda y: y >= 1998),
        ("Suriname", lambda y: y >= 1996),
        ("Venezuela", lambda y: y >= 1998),
    ]
    for country, predicate in usd_rules:
        mask = master["countryname"].eq(country) & predicate(years)
        master.loc[mask, "imports_USD"] = pd.to_numeric(master.loc[mask, "imports"], errors="coerce").astype("float64")
        master.loc[mask, "exports_USD"] = pd.to_numeric(master.loc[mask, "exports"], errors="coerce").astype("float64")

    master = master.loc[~(master["countryname"].eq("NetherlandsAntilles") & years.gt(1997))].copy()
    years = pd.to_numeric(master["year"], errors="coerce")
    master.loc[master["imports_USD"].notna(), "imports"] = np.nan
    master.loc[master["exports_USD"].notna(), "exports"] = np.nan

    uruguay = master["countryname"].eq("Uruguay")
    peru = master["countryname"].eq("Peru")
    argentina = master["countryname"].eq("Argentina")
    bolivia = master["countryname"].eq("Bolivia")
    brazil = master["countryname"].eq("Brazil")
    chile = master["countryname"].eq("Chile")
    paraguay_pre1895 = master["countryname"].eq("Paraguay") & years.le(1894)
    suriname = master["countryname"].eq("Suriname")
    master.loc[uruguay, "imports"] = master.loc[uruguay, "imports"] / 1_000_000
    master.loc[uruguay, "exports"] = master.loc[uruguay, "exports"] / 1_000_000
    master.loc[peru, "imports"] = master.loc[peru, "imports"] / 1_000_000_000
    master.loc[peru, "exports"] = master.loc[peru, "exports"] / 1_000_000_000
    master.loc[argentina, "imports"] = master.loc[argentina, "imports"] * (10 ** -13)
    master.loc[argentina, "exports"] = master.loc[argentina, "exports"] * (10 ** -13)
    master.loc[brazil, "imports"] = master.loc[brazil, "imports"] * 2.750e-15
    master.loc[brazil, "exports"] = master.loc[brazil, "exports"] * 2.750e-15
    master.loc[bolivia, "imports"] = master.loc[bolivia, "imports"] * (10 ** -9)
    master.loc[bolivia, "exports"] = master.loc[bolivia, "exports"] * (10 ** -9)
    master.loc[chile, "imports"] = master.loc[chile, "imports"] * (10 ** -3)
    master.loc[chile, "exports"] = master.loc[chile, "exports"] * (10 ** -3)
    master.loc[paraguay_pre1895, "imports"] = master.loc[paraguay_pre1895, "imports"] * (10 ** -3)
    master.loc[paraguay_pre1895, "exports"] = master.loc[paraguay_pre1895, "exports"] * (10 ** -3)
    master.loc[suriname, "imports"] = master.loc[suriname, "imports"] * (10 ** -3)
    master.loc[suriname, "exports"] = master.loc[suriname, "exports"] * (10 ** -3)

    master["year"] = years.astype("int32")
    master = master.sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(master, temp_dir / "Latam_trade.dta")
    return master


def _mitchell_asia_trade(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Asia_trade")

    def _sheet(kind: str, sheet_name: int) -> pd.DataFrame:
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 2)
        frame = _mitchell_keep_by_header(
            frame,
            header_row=1,
            predicate=lambda value: value == kind,
            normalizer=lambda value: value.replace(" ", "").strip().lower(),
        )
        frame = _mitchell_rename_from_row(frame, 0)
        if sheet_name in (2, 3, 4):
            frame = frame.iloc[1:].reset_index(drop=True)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))

        if sheet_name == 2:
            for country in ["Cyprus", "Sabah", "Sarawak"]:
                frame = _mitchell_convert_units(frame, country, 1860, 1904, "Th")
            frame = _mitchell_convert_units(frame, "Japan", 1860, 1889, "Th")
        elif sheet_name == 3:
            for country in ["Brunei", "Cyprus", "Sabah", "Sarawak", "SouthYemen"]:
                frame = _mitchell_convert_units(frame, country, 1905, 1944, "Th")
        elif sheet_name == 4:
            if kind == "imports":
                frame = _mitchell_convert_units(frame, "Brunei", 1945, 1995, "Th")
            frame = _mitchell_convert_units(frame, "China", 1945, 1995, "B")
        elif sheet_name == 5:
            frame = _mitchell_convert_units(frame, "Lebanon", 1945, 1974, "Th")
            for country in ["India", "HongKong"]:
                frame = _mitchell_convert_units(frame, country, 1980, 1996, "B")
            frame = _mitchell_convert_units(frame, "Iran", 1965, 1995, "B")
            frame = _mitchell_convert_units(frame, "Japan", 1950, 1996, "B")
            frame = _mitchell_convert_units(frame, "Lebanon", 1950, 1973, "B")
            for country in ["SaudiArabia", "Thailand"]:
                frame = _mitchell_convert_units(frame, country, 1975, 1996, "B")
            frame = _mitchell_convert_units(frame, "Turkey", 1975, 1993, "B")
            if kind == "exports" and "A" in frame.columns:
                frame = frame.drop(columns=["A"], errors="ignore")
        return _mitchell_reshape(frame, kind)

    master: pd.DataFrame | None = None
    for sheet_name in (2, 3, 4, 5):
        master = _mitchell_merge_values(master, _sheet("imports", sheet_name))
        master = _mitchell_merge_values(master, _sheet("exports", sheet_name))

    assert master is not None
    for col in ["imports", "exports"]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
    years = pd.to_numeric(master["year"], errors="coerce")
    master["imports_USD"] = pd.Series(np.nan, index=master.index, dtype="float64")
    master["exports_USD"] = pd.Series(np.nan, index=master.index, dtype="float64")

    def _assign_usd(country_list: list[str], year_mask) -> None:
        mask = master["countryname"].isin(country_list) & year_mask
        master.loc[mask, "imports_USD"] = master.loc[mask, "imports"].astype("float64")
        master.loc[mask, "exports_USD"] = master.loc[mask, "exports"].astype("float64")

    _assign_usd(
        ["Afghanistan", "Bahrain", "Bangladesh", "Cambodia", "China", "Cyprus", "HongKong", "India", "Japan"],
        years.ge(1997),
    )
    _assign_usd(
        ["Kuwait", "Malaya", "Oman", "Pakistan", "SaudiArabia", "Singapore", "SriLanka", "Syria", "Thailand"],
        years.ge(1997),
    )
    _assign_usd(["Jordan", "Yemen", "Macau"], years.ge(1997))
    _assign_usd(["Brunei", "Iran", "Qatar"], years.ge(1996))
    _assign_usd(["Turkey", "UnitedArabEmirates"], years.ge(1994))
    _assign_usd(["Nepal"], years.ge(1994))
    _assign_usd(["Vietnam"], years.between(1959, 1974) | years.ge(1980))
    _assign_usd(["Lebanon", "Laos", "Indonesia"], years.ge(1975))
    _assign_usd(["Philippines"], years.ge(1950))
    _assign_usd(["Israel"], years.ge(1948))
    _assign_usd(["SouthKorea"], years.ge(1945))

    afg = master["countryname"].eq("Afghanistan")
    tur = master["countryname"].eq("Turkey")
    tai = master["countryname"].eq("Taiwan")
    laos = master["countryname"].eq("Laos")
    for col in ["imports", "exports"]:
        master.loc[afg, col] = master.loc[afg, col] * (10 ** -3)
        master.loc[tur, col] = master.loc[tur, col] / 1_000_000
        master.loc[tai & years.ge(1970), col] = master.loc[tai & years.ge(1970), col] * 1000
        master.loc[tai & years.lt(1945), col] = master.loc[tai & years.lt(1945), col] * (10 ** -4)
        master.loc[tai & years.lt(1945), col] = master.loc[tai & years.lt(1945), col] / 4
        master.loc[laos, col] = master.loc[laos, col] * (10 ** -3)

    master.loc[master["imports_USD"].notna(), "imports"] = np.nan
    master.loc[master["exports_USD"].notna(), "exports"] = np.nan
    master["year"] = years.astype("int32")
    master = master.sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(master, temp_dir / "Asia_trade.dta")
    return master


def _mitchell_africa_trade(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    temp_dir = _resolve(data_temp_dir) / "MITCHELL"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = _mitchell_workbook_path(raw_dir, "Africa_trade")

    def _sheet(kind: str, sheet_name: int) -> pd.DataFrame:
        frame = _mitchell_import_columns_first(path, sheet_name)
        frame = _mitchell_fill_header_rows(frame, 2)
        if sheet_name in (2, 3):
            frame = _mitchell_keep_by_header(
                frame,
                header_row=1,
                predicate=lambda value: value == kind,
                normalizer=lambda value: value.replace(" ", "").strip().lower(),
            )
        else:
            frame = _mitchell_keep_by_header(
                frame,
                header_row=1,
                predicate=lambda value: kind in value,
                normalizer=lambda value: value.replace(" ", "").strip().lower(),
            )
        if sheet_name == 2:
            frame = frame.drop(columns=["F"] if kind == "imports" else ["G"], errors="ignore")
        frame = _mitchell_rename_from_row(frame, 0)
        frame = _mitchell_destring(_mitchell_drop_blank_year(frame))
        if kind == "exports" and sheet_name == 3:
            frame = frame.dropna(axis=1, how="all")

        if sheet_name == 2:
            for country in ["Gambia", "Ghana"]:
                frame = _mitchell_convert_units(frame, country, 1831, 1869, "Th")
        elif sheet_name == 3:
            if "CapeofGoodHope" in frame.columns or "Natal" in frame.columns:
                cape = pd.to_numeric(frame.get("CapeofGoodHope"), errors="coerce")
                natal = pd.to_numeric(frame.get("Natal"), errors="coerce")
                south = cape + natal
                south = south.where(south.notna(), cape)
                frame["SouthAfrica"] = south
                frame = frame.drop(columns=[c for c in ["CapeofGoodHope", "Natal"] if c in frame.columns], errors="ignore")
            for country in ["Nigeria", "SierraLeone", "SouthAfrica"]:
                frame = _mitchell_convert_units(frame, country, 1826, 1869, "Th")
        elif sheet_name == 4:
            for col in ["CapeofGoodHope", "Natal", "OrangeFreeState", "Transvaal"]:
                if col in frame.columns:
                    frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
            south = None
            for col in ["CapeofGoodHope", "Natal", "OrangeFreeState", "Transvaal"]:
                if col in frame.columns:
                    south = frame[col] if south is None else pd.to_numeric(south, errors="coerce") + pd.to_numeric(frame[col], errors="coerce")
            if south is not None:
                frame["SouthAfrica"] = pd.to_numeric(south, errors="coerce")
                frame = frame.drop(columns=[c for c in ["CapeofGoodHope", "Natal", "OrangeFreeState", "Transvaal"] if c in frame.columns], errors="ignore")
            for country in ["Gambia", "Ghana", "Kenya", "Nigeria", "SierraLeone", "SouthAfrica", "Uganda", "Zambia", "Zanzibar", "Zimbabwe"]:
                frame = _mitchell_convert_units(frame, country, 1870, 1909, "Th")
        elif sheet_name == 5:
            for country in ["BritishSomaliland", "Kenya", "SierraLeone", "Uganda", "Zambia", "Zanzibar"]:
                frame = _mitchell_convert_units(frame, country, 1910, 1949, "Th")
            for country in ["Ghana", "Nigeria"]:
                frame = _mitchell_convert_units(frame, country, 1910, 1919, "Th")
        return _mitchell_reshape(frame, kind)

    master: pd.DataFrame | None = None
    for sheet_name in (2, 3, 4, 5, 6):
        master = _mitchell_merge_values(master, _sheet("imports", sheet_name))
        master = _mitchell_merge_values(master, _sheet("exports", sheet_name))

    assert master is not None
    years = pd.to_numeric(master["year"], errors="coerce")
    for col in ["imports", "exports"]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")

    master["exports_USD"] = pd.Series(np.nan, index=master.index, dtype="float64")
    master["imports_USD"] = pd.Series(np.nan, index=master.index, dtype="float64")

    usd_rules = [
        ("Algeria", years.ge(1997)),
        ("Benin", years.ge(1997)),
        ("BurkinaFaso", years.ge(1995)),
        ("Burundi", years.ge(1997)),
        ("Cameroon", years.ge(1995)),
        ("CentralAfricanRepublic", years.ge(1997)),
        ("Egypt", years.ge(1997)),
        ("Ethiopia", years.ge(1997)),
        ("Gabon", years.ge(1997)),
        ("Gambia", years.ge(1997)),
        ("Ghana", years.ge(1997)),
        ("Guinea", years.ge(1988)),
        ("IvoryCoast", years.ge(1997)),
        ("Kenya", years.ge(1997)),
        ("Liberia", pd.Series(True, index=master.index)),
        ("Libya", years.ge(1997)),
        ("Madagascar", years.ge(1997)),
        ("Malawi", years.ge(1994)),
        ("Mali", years.ge(1996)),
        ("Mauritania", years.ge(1997)),
        ("Mauritius", years.ge(1997)),
        ("Morocco", years.ge(1997)),
        ("Mozambique", years.ge(1994)),
        ("Nigeria", years.ge(1995)),
        ("Rwanda", years.ge(1995)),
        ("Senegal", years.ge(1997)),
        ("Sierra Leone", years.ge(1997)),
        ("SouthAfrica", years.ge(1997)),
        ("Sudan", years.ge(1995)),
        ("Tanzania", years.ge(2004)),
        ("Togo", years.ge(1997)),
        ("Tunisia", years.ge(1997)),
        ("Uganda", years.ge(1997)),
        ("Zaire", years.ge(1994)),
        ("Zambia", years.ge(1993)),
    ]
    for country, mask in usd_rules:
        cmask = master["countryname"].eq(country) & mask
        master.loc[cmask, "exports_USD"] = master.loc[cmask, "exports"].astype("float64")
        master.loc[cmask, "imports_USD"] = master.loc[cmask, "imports"].astype("float64")

    master.loc[master["imports_USD"].notna(), "imports"] = np.nan
    master.loc[master["exports_USD"].notna(), "exports"] = np.nan

    def _apply(mask, factor, op="/"):
        if op == "/":
            master.loc[mask, "imports"] = master.loc[mask, "imports"] / factor
            master.loc[mask, "exports"] = master.loc[mask, "exports"] / factor
        else:
            master.loc[mask, "imports"] = master.loc[mask, "imports"] * factor
            master.loc[mask, "exports"] = master.loc[mask, "exports"] * factor

    _apply(master["countryname"].eq("Tunisia") & years.le(1949), 1000, "/")
    _apply(master["countryname"].eq("Sudan") & years.le(1998), 1000, "/")
    _apply(master["countryname"].eq("Nigeria") & years.le(1972), 0.5, "/")
    _apply(master["countryname"].eq("SierraLeone") & years.le(1964), 0.5, "/")
    _apply(master["countryname"].eq("Ghana") & years.le(1964), 0.417, "/")
    _apply(master["countryname"].eq("Madagascar") & years.le(2000), 5, "/")
    _apply(master["countryname"].eq("SouthAfrica") & years.le(1959), 0.5, "/")
    _apply(master["countryname"].eq("Angola"), 1_000_000_000, "/")
    _apply(master["countryname"].eq("Algeria") & years.le(1957), 100, "/")
    _apply(master["countryname"].eq("Cameroon") & years.le(1919), 3.3538549, "*")
    _apply(master["countryname"].eq("Gambia") & years.le(1969), 200, "/")
    _apply(master["countryname"].eq("Morocco") & years.le(1945), 100, "/")
    for country in ["Cameroon", "Congo", "Gabon", "Madagascar", "Niger", "Senegal"]:
        _apply(master["countryname"].eq(country) & years.ge(1980), 1000, "*")
    _apply(master["countryname"].eq("IvoryCoast") & years.ge(1971), 1000, "*")
    _apply(master["countryname"].eq("SierraLeone") & years.between(1950, 1952), 1000, "/")
    _apply(master["countryname"].eq("Mozambique"), 1000, "/")
    _apply(master["countryname"].eq("Ghana"), 10000, "/")
    _apply(master["countryname"].eq("Guinea") & years.le(1963), 10, "/")
    _apply(master["countryname"].eq("Mauritania"), 10, "/")
    _apply(master["countryname"].eq("Zimbabwe") & years.le(1944), 1000, "/")
    _apply(master["countryname"].eq("Gambia") & years.le(1909), 1000, "*")
    _apply(master["countryname"].eq("Zambia"), 1000, "/")
    _apply(master["countryname"].eq("Malawi") & years.le(1970), 500, "/")
    _apply(master["countryname"].eq("Zaire") & years.le(1959), 1000, "/")
    _apply(master["countryname"].eq("Zaire") & years.le(1993), 30000, "/")
    _apply(master["countryname"].eq("Zaire") & years.le(1990), 1_000_000, "/")
    _apply(master["countryname"].eq("Zaire") & years.le(1989), 10, "/")
    _apply(master["countryname"].eq("Uganda") & years.le(1986), 100, "/")
    lib_mask = master["countryname"].eq("Liberia") & years.ge(1974)
    master.loc[lib_mask, "imports_USD"] = np.nan
    master.loc[lib_mask, "exports_USD"] = np.nan

    master["year"] = years.astype("int32")
    master = master.sort_values(["countryname", "year"]).reset_index(drop=True)
    _save_dta(master, temp_dir / "Africa_trade.dta")
    return master


def _mitchell_partial_trade_final(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    parts = [
        _mitchell_americas_trade(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_canada_trade(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_latam_trade(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_asia_trade(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_africa_trade(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_europe_trade(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_oceania_trade(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
    ]
    master: pd.DataFrame | None = None
    for part in parts:
        master = _mitchell_merge_values(master, part)
    assert master is not None
    master = _mitchell_standardize_countrynames(master)
    lookup = _country_name_lookup(helper_dir)
    master["ISO3"] = master["countryname"].map(lookup)
    master = master.loc[master["ISO3"].notna()].copy()
    master = master.drop(columns=["countryname"])
    master = master[["ISO3", "year", "imports", "exports", "imports_USD", "exports_USD"]].copy()
    master["year"] = pd.to_numeric(master["year"], errors="coerce")
    for col in ["imports", "exports", "imports_USD", "exports_USD"]:
        master[col] = pd.to_numeric(master[col], errors="coerce")
    for col in ["imports_USD", "exports_USD"]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float32")

    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].drop_duplicates().copy()
    master = master.merge(eur_fx, on="ISO3", how="left", indicator=True)
    euro_mask = master["_merge"].eq("both") & master["ISO3"].ne("CYP") & master["year"].le(1993)
    cyp_mask = master["_merge"].eq("both") & master["ISO3"].eq("CYP")
    for col in ["imports", "exports"]:
        master.loc[euro_mask, col] = pd.to_numeric(master.loc[euro_mask, col], errors="coerce") / pd.to_numeric(master.loc[euro_mask, "EUR_irrevocable_FX"], errors="coerce")
        master.loc[cyp_mask, col] = pd.to_numeric(master.loc[cyp_mask, col], errors="coerce") / pd.to_numeric(master.loc[cyp_mask, "EUR_irrevocable_FX"], errors="coerce")
    master = master.drop(columns=["EUR_irrevocable_FX", "_merge"])

    bis = _load_dta(clean_dir / "aggregators" / "BIS" / "BIS_USDfx.dta")[["ISO3", "year", "BIS_USDfx"]].copy()
    bis["year"] = pd.to_numeric(bis["year"], errors="coerce")
    master = master.merge(bis, on=["ISO3", "year"], how="left")
    imports_from_usd = master["imports"].isna() & master["imports_USD"].notna() & master["BIS_USDfx"].notna()
    exports_from_usd = master["exports"].isna() & master["exports_USD"].notna() & master["BIS_USDfx"].notna()
    master.loc[imports_from_usd, "imports"] = (
        pd.to_numeric(master.loc[imports_from_usd, "imports_USD"], errors="coerce").astype("float32").astype("float64")
        * pd.to_numeric(master.loc[imports_from_usd, "BIS_USDfx"], errors="coerce")
    ).to_numpy(dtype="float64")
    master.loc[exports_from_usd, "exports"] = (
        pd.to_numeric(master.loc[exports_from_usd, "exports_USD"], errors="coerce").astype("float32").astype("float64")
        * pd.to_numeric(master.loc[exports_from_usd, "BIS_USDfx"], errors="coerce")
    ).to_numpy(dtype="float64")
    master = master.drop(columns=["BIS_USDfx"])

    cod_mask = master["ISO3"].eq("COD") & master["year"].ge(1989)
    idn_mask = master["ISO3"].eq("IDN") & master["year"].le(1974)
    master.loc[cod_mask | idn_mask, ["imports", "exports"]] = np.nan
    master = master.loc[~master["ISO3"].isin(["YUG", "SRB", "ZMB"])].copy()
    master["year"] = master["year"].astype("int32")
    return master.sort_values(["ISO3", "year"]).reset_index(drop=True)


def _mitchell_partial_rgdp_final(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    master: pd.DataFrame | None = None
    parts = [
        _mitchell_africa_rgdp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_americas_rgdp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_asia_rgdp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_europe_rgdp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_latam_rgdp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_oceania_rgdp(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
    ]
    for part in parts:
        master = _mitchell_merge_values(master, part)
    assert master is not None
    out = _mitchell_finalize_country_frame(
        master,
        ["rGDP_LCU"],
        data_helper_dir=data_helper_dir,
        euro_cutoff_year=1998,
    )
    years = pd.to_numeric(out["year"], errors="coerce")
    iso = out["ISO3"]
    values = pd.to_numeric(out["rGDP_LCU"], errors="coerce")
    out.loc[iso.eq("RUS") & years.between(1928, 1955), "rGDP_LCU"] = values[iso.eq("RUS") & years.between(1928, 1955)] * 0.21130085764592288
    values = pd.to_numeric(out["rGDP_LCU"], errors="coerce")
    out.loc[iso.eq("RUS") & years.between(1956, 1967), "rGDP_LCU"] = values[iso.eq("RUS") & years.between(1956, 1967)] * 0.9748953974895398
    return out.sort_values(["ISO3", "year"]).reset_index(drop=True)


def _mitchell_partial_cpi_final(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    master: pd.DataFrame | None = None
    parts = [
        _mitchell_africa_cpi(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_americas_cpi(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_asia_cpi(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_europe_cpi(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_latam_cpi(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
        _mitchell_oceania_cpi(data_raw_dir=data_raw_dir, data_temp_dir=data_temp_dir),
    ]
    for part in parts:
        master = _mitchell_append(master, part)
    assert master is not None
    out = _mitchell_finalize_country_frame(
        master,
        ["CPI", "infl"],
        data_helper_dir=data_helper_dir,
    )
    return out.sort_values(["ISO3", "year"]).reset_index(drop=True)


MITCHELL_FINAL_COLUMNS = [
    "ISO3",
    "year",
    "Mitchell_exports",
    "Mitchell_imports",
    "Mitchell_rGDP",
    "Mitchell_nGDP",
    "Mitchell_M2",
    "Mitchell_M1",
    "Mitchell_govtax",
    "Mitchell_govrev",
    "Mitchell_govexp",
    "Mitchell_finv",
    "Mitchell_M0",
    "Mitchell_CPI",
    "Mitchell_CA_USD",
    "Mitchell_CA",
    "Mitchell_imports_USD",
    "Mitchell_exports_USD",
    "Mitchell_inv",
    "Mitchell_imports_GDP",
    "Mitchell_exports_GDP",
    "Mitchell_govexp_GDP",
    "Mitchell_govrev_GDP",
    "Mitchell_govtax_GDP",
    "Mitchell_finv_GDP",
    "Mitchell_inv_GDP",
    "Mitchell_govdef_GDP",
    "Mitchell_infl",
]


MITCHELL_DTYPE_MAP = {
    "year": "int16",
    "Mitchell_exports": "float64",
    "Mitchell_imports": "float64",
    "Mitchell_rGDP": "float64",
    "Mitchell_nGDP": "float64",
    "Mitchell_M2": "float64",
    "Mitchell_M1": "float64",
    "Mitchell_govtax": "float64",
    "Mitchell_govrev": "float64",
    "Mitchell_govexp": "float64",
    "Mitchell_finv": "float64",
    "Mitchell_M0": "float64",
    "Mitchell_CPI": "float64",
    "Mitchell_CA_USD": "float64",
    "Mitchell_CA": "float64",
    "Mitchell_imports_USD": "float32",
    "Mitchell_exports_USD": "float32",
    "Mitchell_inv": "float64",
    "Mitchell_imports_GDP": "float32",
    "Mitchell_exports_GDP": "float32",
    "Mitchell_govexp_GDP": "float32",
    "Mitchell_govrev_GDP": "float32",
    "Mitchell_govtax_GDP": "float32",
    "Mitchell_finv_GDP": "float32",
    "Mitchell_inv_GDP": "float32",
    "Mitchell_govdef_GDP": "float32",
    "Mitchell_infl": "float32",
}


def _mitchell_finalize_country_frame(
    frame: pd.DataFrame,
    value_cols: list[str],
    *,
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    euro_cutoff_year: int | None = None,
) -> pd.DataFrame:
    helper_dir = _resolve(data_helper_dir)
    out = _mitchell_standardize_countrynames(frame)
    lookup = _country_name_lookup(helper_dir)
    out["ISO3"] = out["countryname"].map(lookup)
    keep_cols = ["ISO3", "year"] + [col for col in value_cols if col in out.columns]
    out = out.loc[out["ISO3"].notna(), keep_cols].copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    for col in value_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if euro_cutoff_year is not None:
        eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].drop_duplicates().copy()
        out = out.merge(eur_fx, on="ISO3", how="left", indicator=True)
        euro_mask = out["_merge"].eq("both") & out["ISO3"].ne("CYP") & out["year"].le(euro_cutoff_year)
        cyp_mask = out["_merge"].eq("both") & out["ISO3"].eq("CYP")
        for col in value_cols:
            if col in out.columns:
                out.loc[euro_mask, col] = pd.to_numeric(out.loc[euro_mask, col], errors="coerce") / pd.to_numeric(
                    out.loc[euro_mask, "EUR_irrevocable_FX"], errors="coerce"
                )
                out.loc[cyp_mask, col] = pd.to_numeric(out.loc[cyp_mask, col], errors="coerce") / pd.to_numeric(
                    out.loc[cyp_mask, "EUR_irrevocable_FX"], errors="coerce"
                )
        out = out.drop(columns=["EUR_irrevocable_FX", "_merge"])
    out = out.sort_values(["ISO3", "year"]).drop_duplicates(["ISO3", "year"], keep="last")
    out = out.loc[~out["ISO3"].isin(["YUG", "SRB", "ZMB"])].copy()
    return out.sort_values(["ISO3", "year"]).reset_index(drop=True)


def _mitchell_partial_final_assembly(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    helper_dir = _resolve(data_helper_dir)
    pieces = [
        _mitchell_partial_bop_final(data_raw_dir=data_raw_dir, data_helper_dir=helper_dir, data_temp_dir=data_temp_dir),
        _mitchell_partial_ngdp_final(data_raw_dir=data_raw_dir, data_helper_dir=helper_dir, data_temp_dir=data_temp_dir),
        _mitchell_partial_govexp_final(data_raw_dir=data_raw_dir, data_helper_dir=helper_dir, data_temp_dir=data_temp_dir),
        _mitchell_partial_govrev_final(data_raw_dir=data_raw_dir, data_helper_dir=helper_dir, data_temp_dir=data_temp_dir),
        _mitchell_partial_govtax_final(data_raw_dir=data_raw_dir, data_helper_dir=helper_dir, data_temp_dir=data_temp_dir),
        _mitchell_partial_m0_final(data_raw_dir=data_raw_dir, data_helper_dir=helper_dir, data_temp_dir=data_temp_dir),
        _mitchell_partial_money_final(data_raw_dir=data_raw_dir, data_helper_dir=helper_dir, data_temp_dir=data_temp_dir),
        _mitchell_partial_finv_inv_final(data_raw_dir=data_raw_dir, data_helper_dir=helper_dir, data_temp_dir=data_temp_dir),
        _mitchell_partial_stocks_final(data_raw_dir=data_raw_dir, data_helper_dir=helper_dir, data_temp_dir=data_temp_dir),
        _mitchell_partial_trade_final(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=helper_dir, data_temp_dir=data_temp_dir),
        _mitchell_partial_rgdp_final(data_raw_dir=data_raw_dir, data_helper_dir=helper_dir, data_temp_dir=data_temp_dir),
        _mitchell_partial_cpi_final(data_raw_dir=data_raw_dir, data_helper_dir=helper_dir, data_temp_dir=data_temp_dir),
    ]

    master: pd.DataFrame | None = None
    for piece in pieces:
        master = piece if master is None else master.merge(piece, on=["ISO3", "year"], how="outer")
    assert master is not None

    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    for col in ["nGDP_LCU", "rGDP_LCU", "govexp", "govrev", "CPI", "CA", "CA_USD", "finv", "inv", "stocks", "imports", "exports", "imports_USD", "exports_USD"]:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce")
    if "govtax" in master.columns:
        master["govtax"] = pd.to_numeric(master["govtax"], errors="coerce")
    if "CPI" in master.columns:
        eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].drop_duplicates().copy()
        master = master.merge(eur_fx, on="ISO3", how="left", indicator=True)
        cyp_mask = master["_merge"].eq("both") & master["ISO3"].eq("CYP")
        master.loc[cyp_mask, "CPI"] = pd.to_numeric(master.loc[cyp_mask, "CPI"], errors="coerce") / pd.to_numeric(
            master.loc[cyp_mask, "EUR_irrevocable_FX"], errors="coerce"
        )
        master = master.drop(columns=["EUR_irrevocable_FX", "_merge"])
    tur_mask = master["ISO3"].eq("TUR")
    master.loc[tur_mask, ["govrev", "govexp"]] = np.nan
    master.loc[master["ISO3"].isin(["IDN", "TUR", "ARG"]), "govtax"] = np.nan
    if {"finv", "stocks", "inv"}.issubset(master.columns):
        fill_mask = master["stocks"].notna() & master["inv"].isna()
        master.loc[fill_mask, "inv"] = (
            pd.to_numeric(master.loc[fill_mask, "finv"], errors="coerce")
            + pd.to_numeric(master.loc[fill_mask, "stocks"], errors="coerce")
        ).to_numpy(dtype="float64")
    master.loc[master["ISO3"].eq("RUS"), "inv"] = np.nan
    master.loc[master["ISO3"].eq("POL"), "inv"] = np.nan

    master["Mitchell_govexp_GDP"] = (pd.to_numeric(master["govexp"], errors="coerce") / pd.to_numeric(master["nGDP_LCU"], errors="coerce")) * 100
    master["Mitchell_govrev_GDP"] = (pd.to_numeric(master["govrev"], errors="coerce") / pd.to_numeric(master["nGDP_LCU"], errors="coerce")) * 100
    master["Mitchell_govtax_GDP"] = (pd.to_numeric(master["govtax"], errors="coerce") / pd.to_numeric(master["nGDP_LCU"], errors="coerce")) * 100
    master["Mitchell_finv_GDP"] = (pd.to_numeric(master["finv"], errors="coerce") / pd.to_numeric(master["nGDP_LCU"], errors="coerce")) * 100
    master["Mitchell_inv_GDP"] = (pd.to_numeric(master["inv"], errors="coerce") / pd.to_numeric(master["nGDP_LCU"], errors="coerce")) * 100
    master["Mitchell_imports_GDP"] = (pd.to_numeric(master["imports"], errors="coerce") / pd.to_numeric(master["nGDP_LCU"], errors="coerce")) * 100
    master["Mitchell_exports_GDP"] = (pd.to_numeric(master["exports"], errors="coerce") / pd.to_numeric(master["nGDP_LCU"], errors="coerce")) * 100
    master["Mitchell_govexp_GDP"] = pd.to_numeric(master["Mitchell_govexp_GDP"], errors="coerce").astype("float32")
    master["Mitchell_govrev_GDP"] = pd.to_numeric(master["Mitchell_govrev_GDP"], errors="coerce").astype("float32")
    master["Mitchell_govtax_GDP"] = pd.to_numeric(master["Mitchell_govtax_GDP"], errors="coerce").astype("float32")
    master["Mitchell_finv_GDP"] = pd.to_numeric(master["Mitchell_finv_GDP"], errors="coerce").astype("float32")
    master["Mitchell_inv_GDP"] = pd.to_numeric(master["Mitchell_inv_GDP"], errors="coerce").astype("float32")
    master["Mitchell_imports_GDP"] = pd.to_numeric(master["Mitchell_imports_GDP"], errors="coerce").astype("float32")
    master["Mitchell_exports_GDP"] = pd.to_numeric(master["Mitchell_exports_GDP"], errors="coerce").astype("float32")
    nic_mask = master["ISO3"].eq("NIC")
    master.loc[nic_mask, ["Mitchell_govrev_GDP", "Mitchell_govexp_GDP"]] = np.nan
    grc_mask = master["ISO3"].eq("GRC") & master["year"].le(1940)
    master.loc[grc_mask, "Mitchell_govtax_GDP"] = np.nan
    master.loc[master["ISO3"].eq("POL") & master["year"].between(1982, 1990), ["Mitchell_imports_GDP", "Mitchell_exports_GDP"]] = np.nan
    master.loc[master["ISO3"].eq("GRC") & master["year"].le(1940), ["Mitchell_imports_GDP", "Mitchell_exports_GDP"]] = np.nan
    master.loc[master["ISO3"].eq("BRA") & master["year"].le(1948), ["Mitchell_imports_GDP", "Mitchell_exports_GDP"]] = np.nan
    master.loc[master["ISO3"].eq("TUR") & master["year"].between(1994, 1998), ["Mitchell_imports_GDP", "Mitchell_exports_GDP"]] = np.nan
    master.loc[master["ISO3"].eq("CHL") & master["year"].le(1955), ["Mitchell_imports_GDP", "Mitchell_exports_GDP"]] = np.nan
    master.loc[master["ISO3"].isin(["ECU", "IRQ", "MEX", "SLE", "SDN", "TWN", "VEN", "ZMB"]), ["CA", "CA_USD"]] = np.nan
    master["Mitchell_govdef_GDP"] = master["Mitchell_govexp_GDP"] - master["Mitchell_govrev_GDP"]

    master = master.rename(
        columns={
            "imports": "Mitchell_imports",
            "exports": "Mitchell_exports",
            "imports_USD": "Mitchell_imports_USD",
            "exports_USD": "Mitchell_exports_USD",
            "rGDP_LCU": "Mitchell_rGDP",
            "nGDP_LCU": "Mitchell_nGDP",
            "govexp": "Mitchell_govexp",
            "govrev": "Mitchell_govrev",
            "govtax": "Mitchell_govtax",
            "finv": "Mitchell_finv",
            "inv": "Mitchell_inv",
            "M0": "Mitchell_M0",
            "M1": "Mitchell_M1",
            "M2": "Mitchell_M2",
            "CPI": "Mitchell_CPI",
            "CA": "Mitchell_CA",
            "CA_USD": "Mitchell_CA_USD",
        }
    )
    master = master.sort_values(["ISO3", "year"]).reset_index(drop=True)
    prev_cpi = _lag_if_consecutive_year(master, "Mitchell_CPI")
    master["Mitchell_infl"] = np.where(
        prev_cpi.notna(),
        (
            pd.to_numeric(master["Mitchell_CPI"], errors="coerce")
            - pd.to_numeric(prev_cpi, errors="coerce")
        )
        / pd.to_numeric(prev_cpi, errors="coerce")
        * 100,
        np.nan,
    )

    for col in MITCHELL_FINAL_COLUMNS:
        if col not in master.columns:
            master[col] = np.nan
    master = master[MITCHELL_FINAL_COLUMNS].copy()
    master = _coerce_numeric_dtypes(master, MITCHELL_DTYPE_MAP)
    master = master.sort_values(["ISO3", "year"]).reset_index(drop=True)
    if master.duplicated(["ISO3", "year"]).any():
        raise ValueError("MITCHELL contains duplicate ISO3-year keys after processing.")
    return master
