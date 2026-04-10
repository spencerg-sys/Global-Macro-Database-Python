from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def combine_rgdp(
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
) -> pd.DataFrame:
    from ...clean_api import clean_source
    from .combine_splice_variable import combine_splice_variable

    clean_dir = _resolve(data_clean_dir)
    final_dir = _resolve(data_final_dir)
    # Match rGDP.do semantics: rerun required dependency steps each time.
    combine_splice_variable(
        "pop",
        data_clean_dir=clean_dir,
        data_temp_dir=data_temp_dir,
        data_final_dir=final_dir,
    )
    clean_source(
        "WDI",
        data_raw_dir=REPO_ROOT / "data" / "raw",
        data_clean_dir=clean_dir,
        data_helper_dir=REPO_ROOT / "data" / "helpers",
        data_temp_dir=data_temp_dir,
    )

    mad = _load_dta(_find_dta_by_fragment("Madisson", clean_dir))
    pop_df = _load_dta(_chainlinked_path("pop", final_dir))
    wdi = _load_dta(_find_dta_by_fragment("WDI", clean_dir))
    mad = _merge_keep13(mad, pop_df, keepus=["pop", "source"])
    mad = mad.loc[mad["source"].astype(str) != "MAD"].copy()
    mad = mad.drop(columns=["source"], errors="ignore")
    mad = _merge_keep13(mad, wdi, keepus=["WDI_rGDP"])
    # The reference pipeline evaluates the generate expression in double precision, then stores
    # the new variable as float. Casting both inputs to float32 first loses an
    # extra ULP and perturbs the later rebasing ratios.
    mad["MAD_rGDP"] = (
        pd.to_numeric(mad["MAD_rGDP_pc_USD"], errors="coerce").astype("float64")
        * pd.to_numeric(mad["pop"], errors="coerce").astype("float64")
    ).astype("float32")

    mad_spliced = sh.splice(
        mad,
        priority="WDI MAD",
        generate="rGDP",
        varname="rGDP",
        base_year=2000,
        method="chainlink",
        save="NO",
        data_final_dir=final_dir,
    )
    mad_spliced = _key_sort(mad_spliced, ["ISO3", "year"])
    mad_spliced["gap"] = mad_spliced.groupby("ISO3")["year"].diff().where(pd.to_numeric(mad_spliced["rGDP"], errors="coerce").notna())
    next_gap = mad_spliced.groupby("ISO3")["gap"].shift(-1)
    keep_mask = mad_spliced["gap"].eq(1) & next_gap.eq(1)
    keep_mask = keep_mask | pd.to_numeric(mad_spliced["year"], errors="coerce").eq(2022)
    mad_spliced = mad_spliced.loc[keep_mask, ["ISO3", "year", "rGDP", "pop"]].copy()
    mad_spliced = mad_spliced.rename(columns={"rGDP": "MAD_rGDP"})

    rgdp_panel = _build_splice_input(
        "rGDP",
        extra_keep_cols=["BARRO_rGDP_pc"],
        prefer_clean_wide=False,
        data_clean_dir=clean_dir,
        data_temp_dir=data_temp_dir,
        data_final_dir=final_dir,
    )
    rgdp_panel = _merge_keep123(
        mad_spliced,
        rgdp_panel,
        keepus=[col for col in rgdp_panel.columns if col not in {"ISO3", "year"}],
    )
    rgdp_panel = _expand_country_year_panel(rgdp_panel)
    rgdp_panel["BARRO_rGDP"] = (
        pd.to_numeric(rgdp_panel.get("BARRO_rGDP_pc"), errors="coerce").astype("float64")
        * pd.to_numeric(rgdp_panel.get("pop"), errors="coerce").astype("float64")
    ).astype("float32")
    rgdp_panel = rgdp_panel.drop(columns=["pop"], errors="ignore")

    keep_cols = ["ISO3", "year"] + [col for col in rgdp_panel.columns if col.endswith("rGDP")]
    rgdp_panel = rgdp_panel.loc[:, keep_cols].copy()

    countries = sorted(rgdp_panel["ISO3"].dropna().astype(str).unique().tolist())
    value_cols = [col for col in rgdp_panel.columns if col not in {"ISO3", "year", "IMF_WEO_rGDP"}]
    value_dtypes = {col: rgdp_panel[col].dtype for col in value_cols}
    for country in countries:
        country_mask = rgdp_panel["ISO3"].astype(str) == country
        for col in value_cols:
            var_series = pd.to_numeric(rgdp_panel.loc[country_mask, col], errors="coerce")
            if var_series.notna().sum() == 0:
                continue

            anchor_col = "IMF_WEO_rGDP"
            overlap_weo = country_mask & pd.to_numeric(rgdp_panel.get("IMF_WEO_rGDP"), errors="coerce").notna() & pd.to_numeric(rgdp_panel[col], errors="coerce").notna()
            if overlap_weo.sum() > 0:
                overlap_years = pd.to_numeric(rgdp_panel.loc[overlap_weo, "year"], errors="coerce")
                min_year = float(overlap_years.min())
                max_year = float(overlap_years.max())
                range_mask = country_mask & pd.to_numeric(rgdp_panel["year"], errors="coerce").between(min_year, max_year)
            else:
                sh._emit(f"No overlapping between {col} and WEO")
                anchor_col = "BORDO_rGDP"
                overlap_bordo = country_mask & pd.to_numeric(rgdp_panel.get("BORDO_rGDP"), errors="coerce").notna() & pd.to_numeric(rgdp_panel[col], errors="coerce").notna()
                overlap_years = pd.to_numeric(rgdp_panel.loc[overlap_bordo, "year"], errors="coerce")
                min_year = float(overlap_years.min()) if not overlap_years.empty else float("nan")
                max_year = float(overlap_years.max()) if not overlap_years.empty else float("nan")
                range_mask = country_mask & pd.to_numeric(rgdp_panel["year"], errors="coerce").between(min_year, max_year)

            first_mean = _rebase_mean_local(rgdp_panel.loc[range_mask, anchor_col])
            base_mean = _rebase_mean_local(rgdp_panel.loc[range_mask, col])
            ratio = _rebase_ratio_local(first_mean, base_mean)
            updated = pd.to_numeric(rgdp_panel.loc[country_mask, col], errors="coerce").astype("float64") * ratio
            target_dtype = value_dtypes.get(col)
            if target_dtype is not None:
                try:
                    updated = updated.astype(target_dtype)
                except Exception:
                    pass
            rgdp_panel.loc[country_mask, col] = updated.to_numpy()

    for col, dtype in value_dtypes.items():
        if col in rgdp_panel.columns:
            rgdp_panel[col] = pd.to_numeric(rgdp_panel[col], errors="coerce").astype(dtype)

    spec = _parse_splice_spec("rGDP")
    result = sh.splice(
        rgdp_panel,
        priority=str(spec["priority"]),
        generate="rGDP",
        varname="rGDP",
        base_year=int(spec["base_year"]),
        method=str(spec["method"]),
        data_final_dir=final_dir,
    )
    return result
__all__ = ["combine_rgdp"]
