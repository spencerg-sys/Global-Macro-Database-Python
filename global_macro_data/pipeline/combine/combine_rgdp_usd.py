from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def combine_rgdp_usd(
    *,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
) -> pd.DataFrame:
    from .combine_rgdp import combine_rgdp
    from .combine_splice_variable import combine_splice_variable
    from .combine_usdfx import combine_usdfx

    final_dir = _resolve(data_final_dir)
    if not _chainlinked_path("rGDP", final_dir).exists():
        combine_rgdp(
            data_clean_dir=data_clean_dir,
            data_temp_dir=data_temp_dir,
            data_final_dir=final_dir,
        )
    if not _chainlinked_path("USDfx", final_dir).exists():
        combine_usdfx(data_clean_dir=data_clean_dir, data_final_dir=final_dir)
    if not _chainlinked_path("nGDP", final_dir).exists():
        combine_splice_variable("nGDP", data_clean_dir=data_clean_dir, data_final_dir=final_dir)

    df = _load_dta(_chainlinked_path("rGDP", final_dir))[["ISO3", "year", "rGDP"]].copy()
    df = _merge_keep3(df, _load_dta(_chainlinked_path("USDfx", final_dir)), keepus=["USDfx"])
    df = _merge_keep3(df, _load_dta(_chainlinked_path("nGDP", final_dir)), keepus=["nGDP"])
    df = _key_sort(df, ["ISO3", "year"])

    r = pd.to_numeric(df["rGDP"], errors="coerce").astype("float32")
    lag = df.groupby("ISO3")["rGDP"].shift(1).pipe(pd.to_numeric, errors="coerce").astype("float32")
    lead = df.groupby("ISO3")["rGDP"].shift(-1).pipe(pd.to_numeric, errors="coerce").astype("float32")

    df["rgdp_growth_forward"] = ((r.astype("float64") / lag.astype("float64")) - 1.0).astype("float32")
    df.loc[pd.to_numeric(df["year"], errors="coerce") <= 2015, "rgdp_growth_forward"] = np.float32(np.nan)
    df["rgdp_growth_back"] = ((lead.astype("float64") / r.astype("float64")) - 1.0).astype("float32")
    df.loc[pd.to_numeric(df["year"], errors="coerce") >= 2015, "rgdp_growth_back"] = np.float32(np.nan)

    ngdp_num = pd.to_numeric(df["nGDP"], errors="coerce").astype("float32")
    usdfx_num = pd.to_numeric(df["USDfx"], errors="coerce").astype("float32")
    df["nGDP_USD"] = (ngdp_num.astype("float64") / usdfx_num.astype("float64")).astype("float32")
    df["base_2015"] = df["nGDP_USD"].where(pd.to_numeric(df["year"], errors="coerce").eq(2015))
    df["gdp_2015_usd"] = df.groupby("ISO3")["base_2015"].transform("max").astype("float32")
    df = df.drop(columns=["base_2015"])
    df["rGDP_USD"] = np.full(len(df), np.nan, dtype="float32")
    df.loc[pd.to_numeric(df["year"], errors="coerce").eq(2015), "rGDP_USD"] = df.loc[pd.to_numeric(df["year"], errors="coerce").eq(2015), "gdp_2015_usd"]

    for current_year in range(2015, 2024):
        prev_values = df.groupby("ISO3")["rGDP_USD"].shift(1)
        mask = pd.to_numeric(df["year"], errors="coerce").eq(current_year + 1)
        df.loc[mask, "rGDP_USD"] = (
            pd.to_numeric(prev_values.loc[mask], errors="coerce").astype("float64")
            * (1.0 + pd.to_numeric(df.loc[mask, "rgdp_growth_forward"], errors="coerce").astype("float64"))
        ).astype("float32")

    for current_year in range(2015, 1789, -1):
        next_values = df.groupby("ISO3")["rGDP_USD"].shift(-1)
        mask = pd.to_numeric(df["year"], errors="coerce").eq(current_year - 1)
        df.loc[mask, "rGDP_USD"] = (
            pd.to_numeric(next_values.loc[mask], errors="coerce").astype("float64")
            / (1.0 + pd.to_numeric(df.loc[mask, "rgdp_growth_back"], errors="coerce").astype("float64"))
        ).astype("float32")

    out = df[["ISO3", "year", "rGDP_USD"]].copy()
    _save_dta(out, final_dir / "chainlinked_rGDP_USD.dta")
    return out
__all__ = ["combine_rgdp_usd"]
