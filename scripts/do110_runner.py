from __future__ import annotations

import argparse
import json
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from global_macro_data.clean_api import (
    HISTORICAL_SOURCES_WITHOUT_DO,
    _normalize_clean_source_name,
    _reorder_clean_sources_for_dependencies,
    clean_source,
)


def _load_sources(
    *,
    source_list_path: Path,
    explicit_sources: list[str] | None,
    include_historical_sources: bool,
) -> tuple[list[str], list[str]]:
    if explicit_sources:
        raw_sources = explicit_sources
    else:
        source_list = pd.read_csv(source_list_path)
        if "source_name" not in source_list.columns:
            raise ValueError(f"Missing 'source_name' column in {source_list_path}")
        raw_sources = source_list["source_name"].dropna().astype(str).tolist()

    normalized: list[str] = []
    seen: set[str] = set()
    for source in raw_sources:
        src = _normalize_clean_source_name(source)
        if src and src not in seen:
            normalized.append(src)
            seen.add(src)

    ordered = _reorder_clean_sources_for_dependencies(normalized)
    if include_historical_sources:
        return ordered, []

    historical_skip = {
        _normalize_clean_source_name(source)
        for source in HISTORICAL_SOURCES_WITHOUT_DO
        if str(source).strip()
    }
    run_sources: list[str] = []
    skipped_historical: list[str] = []
    for source in ordered:
        if source in historical_skip:
            skipped_historical.append(source)
            continue
        run_sources.append(source)
    return run_sources, skipped_historical


def _copy_temp_prereqs(source_temp_dir: Path, target_temp_dir: Path) -> dict[str, bool]:
    target_temp_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, bool] = {}
    for name in ("blank_panel.dta", "notes.dta"):
        src = source_temp_dir / name
        dst = target_temp_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            out[name] = True
        else:
            out[name] = False
    return out


def _run_sources(
    sources: Iterable[str],
    *,
    data_raw_dir: Path,
    data_clean_dir: Path,
    data_helper_dir: Path,
    data_temp_dir: Path,
    max_attempts: int = 2,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for source in sources:
        record: dict[str, object] = {"source": source}
        attempts = max(1, int(max_attempts))
        last_error: str | None = None
        last_traceback: str | None = None
        for attempt in range(1, attempts + 1):
            try:
                frame = clean_source(
                    source,
                    data_raw_dir=data_raw_dir,
                    data_clean_dir=data_clean_dir,
                    data_helper_dir=data_helper_dir,
                    data_temp_dir=data_temp_dir,
                )
                record["status"] = "ok"
                record["shape"] = [int(frame.shape[0]), int(frame.shape[1])]
                if attempt > 1:
                    record["retry_ok_on_attempt"] = attempt
                break
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                last_traceback = traceback.format_exc()
        if record.get("status") != "ok":
            record["status"] = "fail"
            record["shape"] = None
            record["error"] = last_error
            record["traceback"] = last_traceback
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Run clean sources from the Python source registry.")
    parser.add_argument(
        "--source-list-path",
        type=Path,
        default=REPO_ROOT / "data" / "helpers" / "source_list.csv",
    )
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--data-raw-dir", type=Path, required=True)
    parser.add_argument("--data-helper-dir", type=Path, required=True)
    parser.add_argument("--data-temp-dir", "--source-temp-dir", type=Path, required=True, dest="data_temp_dir")
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--include-historical-sources", action="store_true", default=False)
    args = parser.parse_args()

    sources, skipped_historical = _load_sources(
        source_list_path=args.source_list_path,
        explicit_sources=args.sources,
        include_historical_sources=args.include_historical_sources,
    )

    clean_dir = args.audit_dir / "clean"
    temp_dir = args.audit_dir / "tempfiles"
    clean_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    prereqs = _copy_temp_prereqs(args.data_temp_dir, temp_dir)
    records = _run_sources(
        sources,
        data_raw_dir=args.data_raw_dir,
        data_clean_dir=clean_dir,
        data_helper_dir=args.data_helper_dir,
        data_temp_dir=temp_dir,
    )

    ok = sum(1 for item in records if item["status"] == "ok")
    fail = len(records) - ok
    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_list_path": str(args.source_list_path),
        "explicit_sources": args.sources,
        "include_historical_sources": args.include_historical_sources,
        "data_raw_dir": str(args.data_raw_dir),
        "data_helper_dir": str(args.data_helper_dir),
        "data_temp_dir": str(args.data_temp_dir),
        "audit_dir": str(args.audit_dir),
        "prereq_files_copied": prereqs,
        "skipped_historical_sources": skipped_historical,
        "total": len(records),
        "ok": ok,
        "fail": fail,
        "records": records,
    }

    output = args.audit_dir / "run_summary.json"
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"total": out["total"], "ok": out["ok"], "fail": out["fail"]}))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
