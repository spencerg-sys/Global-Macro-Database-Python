from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def download_wdi(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)
    _ = helper_dir

    country_docs = _world_bank_get_all("country", timeout=timeout)
    countries = pd.DataFrame(
        [
            {
                "countrycode": doc.get("id", ""),
                "countryname": _normalize_wb_text(doc.get("name", "")),
                "region": doc.get("region", {}).get("id", ""),
                "regionname": _normalize_wb_text(doc.get("region", {}).get("value", "")),
                "adminregion": doc.get("adminregion", {}).get("id", ""),
                "adminregionname": _normalize_wb_text(doc.get("adminregion", {}).get("value", "")),
                "incomelevel": doc.get("incomeLevel", {}).get("id", ""),
                "incomelevelname": _normalize_wb_text(doc.get("incomeLevel", {}).get("value", "")),
                "lendingtype": doc.get("lendingType", {}).get("id", ""),
                "lendingtypename": _normalize_wb_text(doc.get("lendingType", {}).get("value", "")),
            }
            for doc in country_docs
        ]
    )

    indicator_rows: list[dict[str, object]] = []
    for indicator in WDI_INDICATORS:
        docs = _world_bank_get_all(f"country/all/indicator/{indicator}", timeout=timeout)
        for doc in docs:
            indicator_rows.append(
                {
                    "countrycode": doc.get("countryiso3code", ""),
                    "indicatorcode": doc.get("indicator", {}).get("id", ""),
                    "indicatorname": _normalize_wb_text(doc.get("indicator", {}).get("value", "")),
                    "year": int(str(doc.get("date", ""))),
                    "value": pd.to_numeric(pd.Series([doc.get("value")]), errors="coerce").iloc[0],
                }
            )

    indicator_df = pd.DataFrame(indicator_rows)
    key_cols = ["countrycode", "indicatorcode", "indicatorname"]
    all_keys = indicator_df[key_cols].drop_duplicates().reset_index(drop=True)
    wide_values = (
        indicator_df.pivot_table(
            index=key_cols,
            columns="year",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide = all_keys.merge(wide_values, on=key_cols, how="left")
    wide.columns.name = None

    year_columns = sorted([col for col in wide.columns if isinstance(col, int)])
    wide = wide.rename(columns={year: f"yr{year}" for year in year_columns})
    wide = wide.merge(countries, on="countrycode", how="left")

    ordered_years = [f"yr{year}" for year in year_columns]
    ordered_cols = WDI_RAW_COLUMNS + ordered_years
    wide = wide[
        [
            "countrycode",
            "countryname",
            "region",
            "regionname",
            "adminregion",
            "adminregionname",
            "incomelevel",
            "incomelevelname",
            "lendingtype",
            "lendingtypename",
            "indicatorname",
            "indicatorcode",
        ]
        + ordered_years
    ]

    for col in ordered_years:
        wide[col] = pd.to_numeric(wide[col], errors="coerce")

    base_path = raw_dir / "aggregators" / "WB" / "WDI"
    gmdsavedate_result = sh.gmdsavedate("WDI", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    _ = gmdsavedate_result
    return sh.savedelta(wide[ordered_cols], str(base_path), id_columns=["countrycode", "indicatorcode"])
__all__ = ["download_wdi"]
