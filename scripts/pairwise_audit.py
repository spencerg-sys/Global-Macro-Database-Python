from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from global_macro_data.clean_api import (
    _normalize_clean_source_name,
    _reorder_clean_sources_for_dependencies,
)


def _source_to_module_rel(source: str) -> str:
    src = _normalize_clean_source_name(source)
    if re.fullmatch(r"[A-Z]{3}_[0-9]+", src):
        return f"global_macro_data/clean/country_level/{src.lower()}.py"
    module_stem = {
        "BIT_USDfx": "bit",
        "Bruegel": "bruegel",
        "Madisson": "madisson",
        "DALLASFED_HPI": "dallasfed_hpi",
    }.get(src, src.lower())
    return f"global_macro_data/clean/aggregators/{module_stem}.py"


def _load_registry_sources(source_list_path: Path | None) -> list[str]:
    if source_list_path is None or not source_list_path.exists():
        return []

    source_list = pd.read_csv(source_list_path)
    if "source_name" not in source_list.columns:
        raise ValueError(f"Missing 'source_name' column in {source_list_path}")

    normalized: list[str] = []
    seen: set[str] = set()
    for source in source_list["source_name"].dropna().astype(str).tolist():
        src = _normalize_clean_source_name(source)
        if src and src not in seen:
            normalized.append(src)
            seen.add(src)
    return _reorder_clean_sources_for_dependencies(normalized)


def _normalize_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ISO3"] = out["ISO3"].astype(str)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out


def _compare_file(
    candidate_path: Path,
    reference_path: Path,
    *,
    abs_tol: float,
    rel_tol: float,
    drop_all_empty_rows: bool,
) -> dict[str, object]:
    try:
        candidate_df = pd.read_dta(candidate_path, convert_categoricals=False)
    except Exception as exc:  # noqa: BLE001
        return {"status": "read_error", "error": f"candidate_read: {type(exc).__name__}: {exc}"}
    try:
        reference_df = pd.read_dta(reference_path, convert_categoricals=False)
    except Exception as exc:  # noqa: BLE001
        return {"status": "read_error", "error": f"reference_read: {type(exc).__name__}: {exc}"}

    keys = ["ISO3", "year"]
    if not set(keys).issubset(candidate_df.columns) or not set(keys).issubset(reference_df.columns):
        return {
            "status": "no_keys",
            "candidate_cols": list(candidate_df.columns),
            "reference_cols": list(reference_df.columns),
        }

    common_value_cols = sorted((set(candidate_df.columns) & set(reference_df.columns)) - set(keys))
    candidate_dropped_all_empty = 0
    reference_dropped_all_empty = 0
    if drop_all_empty_rows and common_value_cols:
        candidate_empty = candidate_df[common_value_cols].isna().all(axis=1)
        reference_empty = reference_df[common_value_cols].isna().all(axis=1)
        candidate_dropped_all_empty = int(candidate_empty.sum())
        reference_dropped_all_empty = int(reference_empty.sum())
        candidate_df = candidate_df.loc[~candidate_empty].copy()
        reference_df = reference_df.loc[~reference_empty].copy()

    candidate_df = _normalize_key_frame(candidate_df)
    reference_df = _normalize_key_frame(reference_df)

    merged = candidate_df.merge(
        reference_df,
        on=keys,
        how="outer",
        suffixes=("_candidate", "_reference"),
        indicator=True,
    )
    candidate_only = int((merged["_merge"] == "left_only").sum())
    reference_only = int((merged["_merge"] == "right_only").sum())

    non_tail_cells = 0
    tail_cells = 0
    max_abs = 0.0
    diff_cols: list[str] = []
    tail_only_cols: list[str] = []

    for col in common_value_cols:
        candidate = pd.to_numeric(merged[f"{col}_candidate"], errors="coerce")
        reference = pd.to_numeric(merged[f"{col}_reference"], errors="coerce")
        raw_diff = ~((candidate.isna() & reference.isna()) | (candidate == reference))
        if not raw_diff.any():
            continue
        abs_diff = (candidate - reference).abs()
        close = np.isclose(
            candidate.to_numpy(dtype="float64"),
            reference.to_numpy(dtype="float64"),
            atol=abs_tol,
            rtol=rel_tol,
            equal_nan=True,
        )
        close = pd.Series(close, index=merged.index)
        non_tail = raw_diff & ~close
        if non_tail.any():
            diff_cols.append(col)
            non_tail_cells += int(non_tail.sum())
        else:
            tail_only_cols.append(col)
        tail_cells += int((raw_diff & close).sum())
        col_max = float(abs_diff[raw_diff].max())
        if math.isfinite(col_max):
            max_abs = max(max_abs, col_max)

    status = "ok"
    if candidate_only or reference_only or non_tail_cells:
        status = "partial"

    return {
        "status": status,
        "candidate_rows": int(len(candidate_df)),
        "reference_rows": int(len(reference_df)),
        "candidate_cols": int(candidate_df.shape[1]),
        "reference_cols": int(reference_df.shape[1]),
        "candidate_dropped_all_empty_rows": candidate_dropped_all_empty,
        "reference_dropped_all_empty_rows": reference_dropped_all_empty,
        "candidate_only_keys": candidate_only,
        "reference_only_keys": reference_only,
        "common_value_cols": int(len(common_value_cols)),
        "non_tail_cells": int(non_tail_cells),
        "tail_only_cells": int(tail_cells),
        "diff_cols": diff_cols,
        "tail_only_cols": tail_only_cols,
        "max_abs": max_abs,
    }


def _build_semantic_checklist(
    *,
    checklist_path: Path,
    run_summary: dict[str, object] | None,
    pairwise_records: list[dict[str, object]],
    source_list_path: Path | None,
) -> None:
    pairwise_by_source: dict[str, dict[str, object]] = {}
    for record in pairwise_records:
        stem = Path(str(record["file"])).stem
        pairwise_by_source[_normalize_clean_source_name(stem)] = record

    run_by_source: dict[str, dict[str, object]] = {}
    if run_summary is not None:
        for item in run_summary.get("records", []):
            source = _normalize_clean_source_name(str(item.get("source", "")))
            if source:
                run_by_source[source] = item

    observed_sources = set(pairwise_by_source) | set(run_by_source)
    ordered_sources = [source for source in _load_registry_sources(source_list_path) if source in observed_sources]
    ordered_sources.extend(sorted(observed_sources - set(ordered_sources)))

    lines: list[str] = []
    lines.append("# Semantic Checklist")
    lines.append("")
    lines.append("| Source | Python Module | Run | Compare | KeyDiff | NonTailCells |")
    lines.append("|---|---|---|---|---:|---:|")
    for source in ordered_sources:
        module_rel = _source_to_module_rel(source)
        run_item = run_by_source.get(source, {})
        run_status = str(run_item.get("status", "n/a")).upper()
        pair_item = pairwise_by_source.get(source, {})
        compare_status = str(pair_item.get("status", "missing")).upper()
        key_diff = int(pair_item.get("candidate_only_keys", 0)) + int(pair_item.get("reference_only_keys", 0))
        non_tail = int(pair_item.get("non_tail_cells", 0))
        lines.append(
            "| "
            + f"{source} | {module_rel} | {run_status} | {compare_status} | {key_diff} | {non_tail} |"
        )

    checklist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pairwise audit two clean-data directories.")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--abs-tol", type=float, default=1e-6)
    parser.add_argument("--rel-tol", type=float, default=1e-6)
    parser.add_argument("--drop-all-empty-rows", action="store_true", default=False)
    parser.add_argument("--run-summary", type=Path, default=None)
    parser.add_argument(
        "--source-list-path",
        type=Path,
        default=REPO_ROOT / "data" / "helpers" / "source_list.csv",
    )
    parser.add_argument("--checklist-output", type=Path, default=None)
    args = parser.parse_args()

    candidate_files = {
        path.relative_to(args.candidate_dir).as_posix(): path for path in sorted(args.candidate_dir.rglob("*.dta"))
    }
    reference_files = {
        path.relative_to(args.reference_dir).as_posix(): path for path in sorted(args.reference_dir.rglob("*.dta"))
    }
    relative_paths = sorted(set(candidate_files) | set(reference_files))

    records: list[dict[str, object]] = []
    for relative_path in relative_paths:
        candidate_file = candidate_files.get(relative_path)
        reference_file = reference_files.get(relative_path)
        if candidate_file is None:
            records.append({"file": relative_path, "status": "missing_candidate"})
            continue
        if reference_file is None:
            records.append({"file": relative_path, "status": "missing_reference"})
            continue
        record = _compare_file(
            candidate_file,
            reference_file,
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
            drop_all_empty_rows=args.drop_all_empty_rows,
        )
        record["file"] = relative_path
        records.append(record)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_dir": str(args.reference_dir),
        "candidate_dir": str(args.candidate_dir),
        "abs_tol": args.abs_tol,
        "rel_tol": args.rel_tol,
        "drop_all_empty_rows": args.drop_all_empty_rows,
        "total_reference_files": len(reference_files),
        "total_candidate_files": len(candidate_files),
        "total_compared_files": len(relative_paths),
        "exact_pairs": sum(1 for record in records if record.get("status") == "ok"),
        "partial_pairs": sum(1 for record in records if record.get("status") == "partial"),
        "missing_reference": sum(1 for record in records if record.get("status") == "missing_reference"),
        "missing_candidate": sum(1 for record in records if record.get("status") == "missing_candidate"),
        "no_keys": sum(1 for record in records if record.get("status") == "no_keys"),
        "read_error": sum(1 for record in records if record.get("status") == "read_error"),
        "pairs_with_key_diff": sum(
            1
            for record in records
            if int(record.get("candidate_only_keys", 0)) + int(record.get("reference_only_keys", 0)) > 0
        ),
        "pairs_with_non_tail": sum(1 for record in records if int(record.get("non_tail_cells", 0)) > 0),
    }
    out = {"summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    run_summary: dict[str, object] | None = None
    if args.run_summary is not None and args.run_summary.exists():
        run_summary = json.loads(args.run_summary.read_text(encoding="utf-8"))
    checklist_output = args.checklist_output or (args.output.parent / "semantic_checklist.md")
    _build_semantic_checklist(
        checklist_path=checklist_output,
        run_summary=run_summary,
        pairwise_records=records,
        source_list_path=args.source_list_path,
    )

    print(
        json.dumps(
            {
                "total_compared_files": summary["total_compared_files"],
                "exact_pairs": summary["exact_pairs"],
                "partial_pairs": summary["partial_pairs"],
                "missing_reference": summary["missing_reference"],
                "missing_candidate": summary["missing_candidate"],
                "pairs_with_key_diff": summary["pairs_with_key_diff"],
                "pairs_with_non_tail": summary["pairs_with_non_tail"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
