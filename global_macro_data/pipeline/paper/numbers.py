from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from . import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})


def build_paper_numbers(
    *,
    repo_root: Path | str = REPO_ROOT,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    numbers_dir: Path | str = sh.OUTPUT_NUMBERS_DIR,
) -> dict[str, Path]:
    root = _resolve(repo_root)
    outputs: dict[str, Path] = {}

    data_final = _load_data_final(data_final_dir=data_final_dir)
    data_cols = [col for col in data_final.columns if col not in {"ISO3", "year", "countryname"}]
    outputs["number_variables_final"] = sh.data_export(
        len(data_cols),
        "number_variables_final",
        numbers_dir=numbers_dir,
        whole=True,
    )

    non_ratio_cols = [col for col in data_cols if not str(col).endswith("_GDP")]
    outputs["number_variables"] = sh.data_export(
        len(non_ratio_cols),
        "number_variables",
        numbers_dir=numbers_dir,
        whole=True,
    )

    clean_counts = _clean_module_counts(repo_root=root)
    download_counts = _download_module_counts(repo_root=root)

    number_aggregators = clean_counts["aggregators"]
    number_countryspecific = clean_counts["country_level"]
    outputs["number_aggregators"] = sh.data_export(number_aggregators, "number_aggregators", numbers_dir=numbers_dir, whole=True)
    outputs["number_countryspecific"] = sh.data_export(number_countryspecific, "number_countryspecific", numbers_dir=numbers_dir, whole=True)
    outputs["number_sources"] = sh.data_export(number_aggregators + number_countryspecific, "number_sources", numbers_dir=numbers_dir, whole=True)

    number_sources_current_aggr = download_counts["aggregators"]
    number_sources_historical_aggr = number_aggregators - number_sources_current_aggr
    outputs["number_sources_current_aggr"] = sh.data_export(
        number_sources_current_aggr,
        "number_sources_current_aggr",
        numbers_dir=numbers_dir,
        whole=True,
    )
    outputs["number_sources_historical_aggr"] = sh.data_export(
        number_sources_historical_aggr,
        "number_sources_historical_aggr",
        numbers_dir=numbers_dir,
        whole=True,
    )

    number_current_countryspecific = download_counts["country_level"]
    number_historical_countryspec = number_countryspecific - number_current_countryspecific
    outputs["number_current_countryspecific"] = sh.data_export(
        number_current_countryspecific,
        "number_current_countryspecific",
        numbers_dir=numbers_dir,
        whole=True,
    )
    outputs["number_historical_countryspec"] = sh.data_export(
        number_historical_countryspec,
        "number_historical_countryspec",
        numbers_dir=numbers_dir,
        whole=True,
    )
    outputs["number_current"] = sh.data_export(
        number_sources_current_aggr + number_current_countryspecific,
        "number_current",
        numbers_dir=numbers_dir,
        whole=True,
    )
    outputs["number_historical"] = sh.data_export(
        number_sources_historical_aggr + number_historical_countryspec,
        "number_historical",
        numbers_dir=numbers_dir,
        whole=True,
    )

    work = data_final.copy()
    value_cols = [col for col in work.columns if col not in {"ISO3", "year", "countryname"}]
    work = work.loc[work[value_cols].notna().any(axis=1)].copy()
    current_year = date.today().year

    outputs["year_start"] = sh.data_export(int(pd.to_numeric(work["year"], errors="coerce").min()), "year_start", numbers_dir=numbers_dir, round="1")
    outputs["year_end_forecasts"] = sh.data_export(int(pd.to_numeric(work["year"], errors="coerce").max()), "year_end_forecasts", numbers_dir=numbers_dir, round="1")
    year_end = int(pd.to_numeric(work.loc[pd.to_numeric(work["year"], errors="coerce").lt(current_year), "year"], errors="coerce").max())
    outputs["year_end"] = sh.data_export(year_end, "year_end", numbers_dir=numbers_dir, round="1")
    outputs["number_countries"] = sh.data_export(
        int(work["ISO3"].astype(str).nunique()),
        "number_countries",
        numbers_dir=numbers_dir,
        whole=True,
    )
    return outputs


__all__ = ["build_paper_numbers"]
