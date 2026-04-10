from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_bvx(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)

    add = _load_dta(raw_dir / "aggregators" / "BVX" / "bvx_crisis_final.dta")
    add = add.loc[add["year"].notna(), ["ISO3", "year", "revised", "panic"]].copy()
    add["BVX_crisisB"] = np.where(add["revised"].notna(), 1.0, 0.0).astype("float32")
    add = add.rename(columns={"panic": "BVX_panic"})
    add = add[["ISO3", "year", "BVX_crisisB", "BVX_panic"]].copy()

    df = _load_dta(raw_dir / "aggregators" / "BVX" / "bvx_annual_regdata_final.dta")
    df = df[["ISO3", "year", "C_B30", "C_N30", "JC", "RC", "PANIC_ind", "PANIC_finer", "bankfailure_narrative"]].copy()
    df = df.rename(
        columns={
            "C_B30": "BVX_crash_bank",
            "C_N30": "BVX_crash_nonfin",
            "JC": "BVX_narr",
            "RC": "BVX_crisisB",
            "PANIC_ind": "BVX_panic",
            "PANIC_finer": "BVX_panic_finer",
            "bankfailure_narrative": "BVX_bfail",
        }
    )
    csk_mask = (df["ISO3"].astype(str) == "CZE") & (pd.to_numeric(df["year"], errors="coerce") < 1992)
    df.loc[csk_mask, "ISO3"] = "CSK"

    merged = df.merge(add, on=["ISO3", "year"], how="outer", suffixes=("", "_using"))
    for col in ["BVX_crisisB", "BVX_panic"]:
        using_col = f"{col}_using"
        if using_col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[using_col])
            merged = merged.drop(columns=[using_col], errors="ignore")

    merged["year"] = pd.to_numeric(merged["year"], errors="coerce").astype("float64")
    for col in ["BVX_crash_bank", "BVX_crash_nonfin", "BVX_narr", "BVX_crisisB", "BVX_panic_finer", "BVX_bfail"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float32")
    merged["BVX_panic"] = pd.to_numeric(merged["BVX_panic"], errors="coerce").astype("float64")
    merged = _sort_keys(merged[["ISO3", "year", "BVX_crash_bank", "BVX_crash_nonfin", "BVX_narr", "BVX_crisisB", "BVX_panic", "BVX_panic_finer", "BVX_bfail"]])
    out_path = clean_dir / "aggregators" / "BVX" / "BVX.dta"
    _save_dta(merged, out_path)
    return merged
__all__ = ["clean_bvx"]
