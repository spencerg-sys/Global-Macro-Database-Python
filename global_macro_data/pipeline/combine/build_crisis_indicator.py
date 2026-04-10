from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def build_crisis_indicator(
    crisis: str,
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
) -> pd.DataFrame:
    clean_dir = _resolve(data_clean_dir)
    final_dir = _resolve(data_final_dir)
    blank_panel_path = _require_blank_panel(data_temp_dir)
    df = pd.read_dta(blank_panel_path, convert_categoricals=False)

    sources = {
        "BankingCrisis": ["RR", "LV", "JST", "BVX"],
        "CurrencyCrisis": ["RR", "LV"],
        "SovDebtCrisis": ["RR", "LV"],
    }[crisis]

    for source in sources:
        path = _find_dta_by_fragment(source, clean_dir)
        using = pd.read_dta(path, convert_categoricals=False)
        if crisis == "BankingCrisis" and source == "JST":
            keep = [col for col in ["ISO3", "year", "JST_crisisB"] if col in using.columns]
            using = using[keep]
        df = _merge_update_1to1(df, using, keys=["ISO3", "year"], error_label=source)

    if crisis == "SovDebtCrisis":
        df["LV_SovDebtCrisis"] = pd.concat(
            [pd.to_numeric(df.get("LV_crisisSD1"), errors="coerce"), pd.to_numeric(df.get("LV_crisisSD2"), errors="coerce")],
            axis=1,
        ).max(axis=1, skipna=True)
        df["RR_SovDebtCrisis"] = pd.concat(
            [
                pd.to_numeric(df.get("RR_crisisDD"), errors="coerce"),
                pd.to_numeric(df.get("RR_crisisED1"), errors="coerce"),
                pd.to_numeric(df.get("RR_crisisED2"), errors="coerce"),
            ],
            axis=1,
        ).max(axis=1, skipna=True)
        priority = ["LV", "RR"]
        target_column = "SovDebtCrisis"
        source_map = {"LV": "LV_SovDebtCrisis", "RR": "RR_SovDebtCrisis"}
    elif crisis == "CurrencyCrisis":
        priority = ["LV", "RR"]
        target_column = "CurrencyCrisis"
        source_map = {"LV": "LV_crisisC", "RR": "RR_crisisC"}
    else:
        priority = ["BVX", "LV", "JST", "RR"]
        target_column = "BankingCrisis"
        source_map = {"BVX": "BVX_crisisB", "LV": "LV_crisisB", "JST": "JST_crisisB", "RR": "RR_crisisB"}

    df[target_column] = pd.NA
    for source in priority:
        col = source_map[source]
        if col in df.columns:
            mask = df[target_column].isna() & df[col].notna()
            df.loc[mask, target_column] = df.loc[mask, col]

    df = _key_sort(df, ["ISO3", "year"])
    # Mirror `xtset id year` lag behavior from Stata: lags are only valid when
    # years are consecutive within each country panel.
    years = pd.to_numeric(df["year"], errors="coerce")
    target = pd.to_numeric(df[target_column], errors="coerce")
    lag_masks: list[pd.Series] = []
    for lag in (1, 2, 3):
        lag_target = df.groupby("ISO3")[target_column].shift(lag)
        lag_year = df.groupby("ISO3")["year"].shift(lag)
        lag_target = pd.to_numeric(lag_target, errors="coerce")
        lag_year = pd.to_numeric(lag_year, errors="coerce")
        consecutive = years.sub(lag_year).eq(lag)
        lag_masks.append(lag_target.eq(1) & consecutive)

    suppress_mask = lag_masks[0] | lag_masks[1] | lag_masks[2]
    target = target.mask(suppress_mask, 0)
    df[target_column] = target.astype("float32")

    out = df[["ISO3", "year", target_column]].copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("float64")
    out[target_column] = pd.to_numeric(out[target_column], errors="coerce").astype("float32")
    final_dir.mkdir(parents=True, exist_ok=True)
    _save_dta(out, final_dir / f"{target_column}.dta")
    return out
__all__ = ["build_crisis_indicator"]
