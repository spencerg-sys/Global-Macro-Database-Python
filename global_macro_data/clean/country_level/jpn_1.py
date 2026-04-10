from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_jpn_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)

    def _extract_year(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series.astype(str).str.extract(r"(\d{4})", expand=False), errors="coerce")

    def _keep_last_by_year(df: pd.DataFrame, value_cols: list[str], drop_missing: bool = False) -> pd.DataFrame:
        out = df.copy()
        out["year"] = _extract_year(out["date"])
        if drop_missing:
            mask = pd.Series(False, index=out.index)
            for col in value_cols:
                mask = mask | pd.to_numeric(out[col], errors="coerce").notna()
            out = out.loc[mask].copy()
        keep = ["year"] + value_cols
        out = out[keep].copy()
        for col in value_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.sort_values("year", kind="mergesort").groupby("year", sort=False).tail(1).copy()
        return out

    ltrate = _keep_last_by_year(pd.read_excel(raw_dir / "country_level" / "JPN_1a.xlsx", sheet_name="ltrate"), ["ltrate"])
    strate_a = _keep_last_by_year(pd.read_excel(raw_dir / "country_level" / "JPN_1a.xlsx", sheet_name="strate"), ["strate"])
    master1 = ltrate.merge(strate_a, on="year", how="outer")

    cbrate = _keep_last_by_year(
        pd.read_excel(raw_dir / "country_level" / "JPN_1b.xlsx", sheet_name="cbrate"),
        ["cbrate"],
        drop_missing=True,
    )
    strate_b_raw = pd.read_excel(raw_dir / "country_level" / "JPN_1b.xlsx", sheet_name="strate")
    strate_b_raw["strate"] = (
        (pd.to_numeric(strate_b_raw["High"], errors="coerce") + pd.to_numeric(strate_b_raw["Low"], errors="coerce")) / 2
    ).astype("float32").astype("float64")
    strate_b = _keep_last_by_year(strate_b_raw, ["strate"])
    cpi = _keep_last_by_year(pd.read_excel(raw_dir / "country_level" / "JPN_1b.xlsx", sheet_name="CPI"), ["CPI"])
    m0 = _keep_last_by_year(
        pd.read_excel(raw_dir / "country_level" / "JPN_1b.xlsx", sheet_name="M0"),
        ["M0"],
        drop_missing=True,
    )
    m0["M0"] = pd.to_numeric(m0["M0"], errors="coerce") / 1000
    master2 = cbrate.merge(strate_b, on="year", how="outer")
    master2 = master2.merge(cpi, on="year", how="outer")
    master2 = master2.merge(m0, on="year", how="outer")
    master2["ISO3"] = "JPN"
    master2 = _sort_keys(master2[["ISO3", "year", "M0", "CPI", "strate", "cbrate"]], keys=("ISO3", "year"))
    prev_cpi = pd.to_numeric(master2["CPI"], errors="coerce").groupby(master2["ISO3"]).shift(1)
    prev_year = pd.to_numeric(master2["year"], errors="coerce").groupby(master2["ISO3"]).shift(1)
    year_num = pd.to_numeric(master2["year"], errors="coerce")
    infl = (pd.to_numeric(master2["CPI"], errors="coerce") - prev_cpi) / prev_cpi * 100
    master2["infl"] = infl.where(prev_cpi.notna() & year_num.eq(prev_year + 1)).astype("float32")

    master1["ISO3"] = "JPN"
    master1 = master1[["ISO3", "year", "ltrate", "strate"]].copy()
    merged = master2.merge(master1, on=["ISO3", "year"], how="outer", suffixes=("", "_using"))
    if "strate_using" in merged.columns:
        merged["strate"] = pd.to_numeric(merged["strate"], errors="coerce").where(
            pd.to_numeric(merged["strate"], errors="coerce").notna(),
            pd.to_numeric(merged["strate_using"], errors="coerce"),
        )
        merged = merged.drop(columns=["strate_using"], errors="ignore")
    merged = merged.rename(columns={col: f"CS1_{col}" for col in merged.columns if col not in {"ISO3", "year"}})
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce").astype("int16")
    for col in ["CS1_M0", "CS1_CPI", "CS1_strate", "CS1_cbrate", "CS1_ltrate"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float64")
    merged["CS1_infl"] = pd.to_numeric(merged["CS1_infl"], errors="coerce").astype("float32")
    merged = merged[["ISO3", "year", "CS1_M0", "CS1_CPI", "CS1_strate", "CS1_cbrate", "CS1_infl", "CS1_ltrate"]].copy()
    for col in ["CS1_M0", "CS1_CPI", "CS1_strate", "CS1_cbrate", "CS1_ltrate"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float64")
    merged["CS1_infl"] = pd.to_numeric(merged["CS1_infl"], errors="coerce").astype("float32")
    merged = _sort_keys(merged)
    out_path = clean_dir / "country_level" / "JPN_1.dta"
    _save_dta(merged, out_path)
    return merged
__all__ = ["clean_jpn_1"]
