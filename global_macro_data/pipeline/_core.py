from __future__ import annotations

from datetime import date
from functools import wraps
import math
from pathlib import Path
import re
import shutil
from typing import Sequence

import numpy as np
import pandas as pd

from .. import helpers as sh
from ..bundled_specs import get_pipeline_specs


REPO_ROOT = Path(__file__).resolve().parents[2]
ZERO_AS_MISSING_VARS = {"nGDP", "cons", "cons_GDP"}

def _resolve(path: str | Path) -> Path:
    return sh._resolve_path(path)


def _key_sort(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    return df.sort_values(list(keys)).reset_index(drop=True)


def _save_dta(df: pd.DataFrame, path: Path) -> None:
    sh.write_dta(df, path)


def _load_dta(path: Path) -> pd.DataFrame:
    return pd.read_dta(path, convert_categoricals=False)


def _sync_tree(
    source_root: Path,
    target_root: Path,
    *,
    overwrite: bool,
) -> list[Path]:
    copied: list[Path] = []
    for source_path in source_root.rglob("*"):
        if not source_path.is_file():
            continue
        target_path = target_root / source_path.relative_to(source_root)
        if overwrite or not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            copied.append(target_path)
    return copied










def _chainlinked_path(varname: str, data_final_dir: Path | str = sh.DATA_FINAL_DIR) -> Path:
    return _resolve(data_final_dir) / f"chainlinked_{varname}.dta"


def _require_blank_panel(data_temp_dir: Path | str = sh.DATA_TEMP_DIR) -> Path:
    path = _resolve(data_temp_dir) / "blank_panel.dta"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing blank panel at {path}. Run make_blank_panel() before executing merge/combine stages."
        )
    return path


def _require_clean_data_wide(data_final_dir: Path | str = sh.DATA_FINAL_DIR) -> Path:
    path = _resolve(data_final_dir) / "clean_data_wide.dta"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing merged clean dataset at {path}. Run merge_clean_data() before calling this stage."
        )
    return path


def _iter_clean_dta_files(clean_dir: Path | str) -> list[Path]:
    root = _resolve(clean_dir)
    preferred_roots = [root / "aggregators", root / "country_level"]
    if any(path.exists() for path in preferred_roots):
        files: list[Path] = []
        for path in preferred_roots:
            if path.exists():
                files.extend(path.rglob("*.dta"))
        return sorted(files, key=lambda p: p.as_posix().lower())
    return sorted(root.rglob("*.dta"), key=lambda p: p.as_posix().lower())


def _bundled_combine_entries() -> list[dict[str, object]]:
    entries = get_pipeline_specs().get("combine", [])
    return [entry for entry in entries if isinstance(entry, dict)]


def _bundled_combine_entry_by_target(varname: str) -> dict[str, object] | None:
    target = str(varname)
    for entry in _bundled_combine_entries():
        if str(entry.get("target", "")) == target:
            return entry
    return None


def _bundled_combine_entry_by_path(path: Path) -> dict[str, object] | None:
    name = path.name
    for entry in _bundled_combine_entries():
        if str(entry.get("path", "")) == name:
            return entry
    return None


def _combine_relative_paths() -> list[Path]:
    bundled = _bundled_combine_entries()
    if bundled:
        return [Path(str(entry.get("path", ""))) for entry in bundled if str(entry.get("path", "")).strip()]

    raise ValueError("Bundled combine specs are unavailable.")


def _merge_keep13(
    master: pd.DataFrame,
    using: pd.DataFrame,
    *,
    keys: Sequence[str] = ("ISO3", "year"),
    keepus: Sequence[str] | None = None,
) -> pd.DataFrame:
    keep_cols = list(keys) + ([] if keepus is None else [col for col in keepus if col in using.columns])
    using_trim = using.loc[:, keep_cols].copy()
    out = master.merge(using_trim, on=list(keys), how="left", sort=False)
    return out


def _merge_keep3(
    master: pd.DataFrame,
    using: pd.DataFrame,
    *,
    keys: Sequence[str] = ("ISO3", "year"),
    keepus: Sequence[str] | None = None,
) -> pd.DataFrame:
    keep_cols = list(keys) + ([] if keepus is None else [col for col in keepus if col in using.columns])
    using_trim = using.loc[:, keep_cols].copy()
    out = master.merge(using_trim, on=list(keys), how="inner", sort=False)
    return out


def _merge_keep123(
    master: pd.DataFrame,
    using: pd.DataFrame,
    *,
    keys: Sequence[str] = ("ISO3", "year"),
    keepus: Sequence[str] | None = None,
) -> pd.DataFrame:
    keep_cols = list(keys) + ([] if keepus is None else [col for col in keepus if col in using.columns])
    using_trim = using.loc[:, keep_cols].copy()
    out = master.merge(using_trim, on=list(keys), how="outer", sort=False)
    return _key_sort(out, keys)


def _find_dta_by_fragment(fragment: str | Path, root: Path) -> Path:
    expanded = Path(sh._expand_path_macros(fragment))
    candidates = []
    if expanded.suffix.lower() == ".dta":
        candidates.append(expanded)
    else:
        candidates.append(expanded.with_suffix(".dta"))
        candidates.append(expanded)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    stem = expanded.stem if expanded.suffix else expanded.name
    aliases = {"Madisson": "Maddison", "MAD": "Maddison"}
    search_names = [stem]
    if stem in aliases:
        search_names.append(aliases[stem])

    by_name = {p.stem.lower(): p for p in root.rglob("*.dta")}
    for name in search_names:
        match = by_name.get(name.lower())
        if match is not None:
            return match

    raise FileNotFoundError(f"Could not resolve dataset '{fragment}' under {root}")


def _source_dtype_map(
    varname: str,
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> dict[str, object]:
    clean_dir = _resolve(data_clean_dir)
    suffix = f"_{varname}"
    dtype_map: dict[str, object] = {}
    for path in _iter_clean_dta_files(clean_dir):
        try:
            sample = next(pd.read_dta(path, convert_categoricals=False, chunksize=1))
        except Exception:
            continue
        for col in sample.columns:
            if not col.endswith(suffix):
                continue
            dtype = sample[col].dtype
            if col in dtype_map:
                dtype_map[col] = np.result_type(dtype_map[col], dtype)
            else:
                dtype_map[col] = dtype
    return dtype_map


def _restore_column_dtypes(df: pd.DataFrame, dtype_map: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    for col, dtype in dtype_map.items():
        if col not in out.columns:
            continue
        try:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(dtype)
        except Exception:
            continue
    return out


def _build_blank_merge_input(
    varname: str,
    *,
    extra_keep_cols: Sequence[str] = (),
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
) -> pd.DataFrame:
    clean_dir = _resolve(data_clean_dir)
    blank_panel_path = _require_blank_panel(data_temp_dir)
    master = pd.read_dta(blank_panel_path, convert_categoricals=False)
    suffix = f"_{varname}"
    dtype_map = _source_dtype_map(varname, data_clean_dir=clean_dir)
    extra_dtype_map: dict[str, object] = {}

    for path in _iter_clean_dta_files(clean_dir):
        try:
            using = pd.read_dta(path, convert_categoricals=False)
        except Exception:
            continue
        keep_cols = ["ISO3", "year"] + [
            col for col in using.columns if col.endswith(suffix) or col in extra_keep_cols
        ]
        if len(keep_cols) == 2:
            continue
        for col in keep_cols:
            if col in {"ISO3", "year"}:
                continue
            if col not in extra_dtype_map:
                extra_dtype_map[col] = using[col].dtype
            else:
                extra_dtype_map[col] = np.result_type(extra_dtype_map[col], using[col].dtype)
        master = _merge_update_1to1(master, using[keep_cols].copy(), keys=["ISO3", "year"], error_label=path.stem)

    master = _restore_column_dtypes(master, dtype_map | extra_dtype_map)
    return master


def _merge_update_1to1(
    master: pd.DataFrame,
    using: pd.DataFrame,
    *,
    keys: Sequence[str],
    error_label: str,
) -> pd.DataFrame:
    master_idx = master.set_index(list(keys), drop=False)
    using_idx = using.set_index(list(keys), drop=False)
    missing_keys = using_idx.index.difference(master_idx.index)
    if len(missing_keys) > 0:
        missing_frame = using_idx.loc[missing_keys].reset_index(drop=True)
        countries = sorted({str(v) for v in missing_frame["ISO3"].dropna().tolist()}) if "ISO3" in keys else []
        years = sorted({str(int(v)) for v in pd.to_numeric(missing_frame["year"], errors="coerce").dropna().tolist()}) if "year" in keys else []
        if countries:
            sh._emit("Cannot merge because the following countries are not in the master list:")
            sh._emit(" ".join(countries))
        if years:
            sh._emit("Cannot merge because the following years are not in the master list:")
            sh._emit(" ".join(years))
        raise sh.PipelineRuntimeError(f"Merge failed for {error_label}", code=198)

    for col in using.columns:
        if col in keys:
            continue
        if col not in master_idx.columns:
            master_idx[col] = pd.NA

        aligned = using_idx[col].reindex(master_idx.index)
        master_missing = master_idx[col].map(sh._is_missing_scalar)
        using_nonmissing = ~aligned.map(sh._is_missing_scalar)
        fill_mask = master_missing & using_nonmissing
        if fill_mask.any():
            master_idx.loc[fill_mask, col] = aligned.loc[fill_mask]

    return _key_sort(master_idx.reset_index(drop=True), keys)






def _parse_splice_spec(varname: str) -> dict[str, str | int]:
    bundled = _bundled_combine_entry_by_target(varname)
    if bundled is not None and isinstance(bundled.get("spec"), dict):
        spec = dict(bundled["spec"])
        return {
            "priority": str(spec["priority"]),
            "method": str(spec["method"]),
            "base_year": int(spec["base_year"]),
        }

    raise ValueError(f"Could not parse splice spec for {varname}")


def _parse_note_sources(varname: str) -> list[tuple[str, str]]:
    bundled = _bundled_combine_entry_by_target(varname)
    if bundled is not None:
        notes = bundled.get("notes", [])
        out: list[tuple[str, str]] = []
        for row in notes if isinstance(notes, list) else []:
            if isinstance(row, (list, tuple)) and len(row) == 2:
                out.append((str(row[0]), str(row[1])))
        return out

    return []


def _extract_generate_var(path: Path) -> str | None:
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"generate\(([^)]+)\)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    bundled = _bundled_combine_entry_by_path(path)
    if bundled is not None:
        target = str(bundled.get("target", "")).strip()
        return target or None
    return None


def _parse_documentation_spec() -> tuple[list[dict[str, object]], list[str]]:
    bundled = get_pipeline_specs()
    if bundled:
        blocks = [dict(block) for block in bundled.get("documentation_blocks", []) if isinstance(block, dict)]
        local_vars = [str(var) for var in bundled.get("documentation_local_vars", [])]
        if blocks and local_vars:
            return blocks, local_vars

    raise ValueError("Documentation specs are unavailable from bundled specs.")


def _parse_gmdmakedoc_options(options: str) -> dict[str, object]:
    parsed: dict[str, object] = {
        "log": False,
        "ylabel": None,
        "transformation": None,
        "graphformat": None,
    }
    if not options:
        return parsed

    parsed["log"] = bool(re.search(r"\blog\b", options))
    match = re.search(r'ylabel\("([^"]*)"\)', options)
    if match:
        parsed["ylabel"] = match.group(1)
    match = re.search(r'transformation\("([^"]*)"\)', options)
    if match:
        parsed["transformation"] = match.group(1)
    match = re.search(r'graphformat\("([^"]*)"\)', options)
    if match:
        parsed["graphformat"] = match.group(1)
    return parsed


def _load_clean_data_wide(
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
) -> pd.DataFrame:
    path = _require_clean_data_wide(data_final_dir)
    return pd.read_dta(path, convert_categoricals=False)


def _dta_varlist(path: Path) -> list[str]:
    chunk = next(pd.read_dta(path, convert_categoricals=False, chunksize=1))
    return chunk.columns.tolist()


def _build_splice_input(
    varname: str,
    *,
    extra_keep_cols: Sequence[str] = (),
    prefer_clean_wide: bool = True,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
) -> pd.DataFrame:
    clean_dir = _resolve(data_clean_dir)
    suffix = f"_{varname}"
    extra_keep = [str(col) for col in extra_keep_cols]
    final_dir = _resolve(data_final_dir)
    clean_wide_path = final_dir / "clean_data_wide.dta"
    if prefer_clean_wide and clean_wide_path.exists():
        clean_wide = pd.read_dta(clean_wide_path, convert_categoricals=False)
        keep_cols = ["ISO3", "year"] + [
            col for col in clean_wide.columns if col.endswith(suffix) or col in extra_keep
        ]
        if len(keep_cols) > 2:
            return _key_sort(clean_wide[keep_cols].copy(), ["ISO3", "year"])

    dtype_map = _source_dtype_map(varname, data_clean_dir=clean_dir)
    matched_files: list[Path] = []

    for path in _iter_clean_dta_files(clean_dir):
        try:
            varlist = _dta_varlist(path)
        except Exception:
            continue
        if any(col.endswith(suffix) or col in extra_keep for col in varlist):
            matched_files.append(path)

    if not matched_files:
        return pd.DataFrame(columns=["ISO3", "year"])

    pieces: list[pd.DataFrame] = []
    for file in matched_files:
        using = pd.read_dta(file, convert_categoricals=False)
        keep_cols = ["ISO3", "year"] + [col for col in using.columns if col.endswith(suffix) or col in extra_keep]
        pieces.append(using[keep_cols].copy())

    master = pieces[0].set_index(["ISO3", "year"])
    for piece in pieces[1:]:
        piece_idx = piece.set_index(["ISO3", "year"])
        master = master.reindex(master.index.union(piece_idx.index))
        for col in piece_idx.columns:
            if col not in master.columns:
                master[col] = pd.NA
            aligned = piece_idx[col].reindex(master.index)
            master_missing = master[col].map(sh._is_missing_scalar)
            using_nonmissing = ~aligned.map(sh._is_missing_scalar)
            fill_mask = master_missing & using_nonmissing
            if fill_mask.any():
                master.loc[fill_mask, col] = aligned.loc[fill_mask]

    master = master.reset_index()
    idcm_cols = [col for col in master.columns if col.startswith("IDCM_")]
    if idcm_cols:
        master = master.drop(columns=idcm_cols)

    source_cols = [col for col in master.columns if col.endswith(suffix) or col in extra_keep]
    if varname in ZERO_AS_MISSING_VARS:
        for col in [col for col in source_cols if col.endswith(suffix)]:
            numeric = pd.to_numeric(master[col], errors="coerce")
            master.loc[numeric.eq(0), col] = pd.NA
    any_nonmiss = master[source_cols].notna().any(axis=1)
    ranges = (
        master.loc[any_nonmiss, ["ISO3", "year"]]
        .groupby("ISO3", sort=False)["year"]
        .agg(["min", "max"])
        .reset_index()
    )

    panel_rows: list[dict[str, object]] = []
    for iso3, min_year, max_year in ranges.itertuples(index=False):
        for year in range(int(min_year), int(max_year) + 1):
            panel_rows.append({"ISO3": iso3, "year": year})

    panel = pd.DataFrame(panel_rows)
    master = panel.merge(master, on=["ISO3", "year"], how="left", sort=False)
    master = _restore_column_dtypes(master, dtype_map)

    return _key_sort(master, ["ISO3", "year"])


def _expand_country_year_panel(df: pd.DataFrame, *, keys: Sequence[str] = ("ISO3", "year")) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    key_cols = list(keys)
    if key_cols != ["ISO3", "year"]:
        return _key_sort(df.copy(), keys)

    work = _key_sort(df.copy(), ["ISO3", "year"])
    ranges = (
        work.loc[:, ["ISO3", "year"]]
        .dropna(subset=["ISO3", "year"])
        .assign(year=pd.to_numeric(work["year"], errors="coerce"))
        .dropna(subset=["year"])
        .groupby("ISO3", sort=False)["year"]
        .agg(["min", "max"])
        .reset_index()
    )
    panel_rows: list[dict[str, object]] = []
    for iso3, min_year, max_year in ranges.itertuples(index=False):
        for year in range(int(min_year), int(max_year) + 1):
            panel_rows.append({"ISO3": iso3, "year": float(year)})

    panel = pd.DataFrame(panel_rows)
    expanded = panel.merge(work, on=["ISO3", "year"], how="left", sort=False)
    return _key_sort(expanded, ["ISO3", "year"])


def _rebase_mean_local(series: pd.Series, *, mode: str = "extended") -> float:
    numeric_series = pd.to_numeric(series.loc[~series.map(sh._is_missing_scalar)], errors="coerce").dropna()
    if numeric_series.empty:
        return float("nan")

    # Rebase in the reference pipeline runs through summarize -> r(mean) -> local text rendering.
    # A compensated double-domain mean tracks the exported reference rGDP artifact
    # more closely than NumPy's vectorized float64/longdouble reductions.
    mean_value = math.fsum(float(value) for value in numeric_series.tolist()) / len(numeric_series)
    if abs(mean_value) < 0.01:
        return _render_general18(float(mean_value))
    return sh._local_numeric_value(float(mean_value))


def _render_general18(value: float | int | None) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float("nan")
    x = float(numeric)
    best: str | None = None
    for precision in range(1, 18):
        rendered = format(x, f".{precision}g")
        limit = 17 if ("e" in rendered or "E" in rendered) else 18
        if len(rendered) <= limit:
            best = rendered
        else:
            break
    if best is None:
        best = format(x, ".1g")
    return float(best)


def _rebase_ratio_local(numerator: float | int | None, denominator: float | int | None) -> float:
    num = sh._local_numeric_value(numerator)
    den = sh._local_numeric_value(denominator)
    if pd.isna(num) or pd.isna(den) or float(den) == 0.0:
        return float("nan")
    raw_ratio = float(num) / float(den)
    rendered16 = format(raw_ratio, ".16g")
    if ("e" in rendered16 or "E" in rendered16) or abs(raw_ratio) >= 1_000_000_000 or abs(raw_ratio) < 0.1:
        return _render_general18(raw_ratio)
    return sh._local_numeric_value(raw_ratio)
