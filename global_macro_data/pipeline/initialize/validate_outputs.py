from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def validate_outputs(
    df: pd.DataFrame | None = None,
    *,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
) -> dict[str, pd.DataFrame]:
    if df is None:
        df = _load_clean_data_wide(data_final_dir=data_final_dir)

    work = _key_sort(df.copy(), ["ISO3", "year"])
    issues: dict[str, pd.DataFrame] = {}
    for col in [c for c in work.columns if c not in {"ISO3", "year"}]:
        series = pd.to_numeric(work[col], errors="coerce")
        ratio = series / work.groupby("ISO3")[col].shift(1).pipe(pd.to_numeric, errors="coerce")
        valid = ratio.notna()
        if not valid.any():
            sh._emit(f"Note: No valid ratios calculated for {col}. Skipping checks.")
            continue
        max_ratio = ratio[valid].max()
        min_ratio = ratio[valid].min()
        if pd.notna(max_ratio) and pd.notna(min_ratio) and (max_ratio > 1000 or abs(min_ratio) < 1 / 1000):
            flagged = work.loc[(ratio > 1000) | ((ratio.abs() < 1 / 1000) & ratio.ne(0)), ["ISO3", "year", col]].copy()
            if not flagged.empty:
                issues[col] = flagged
                sh._emit(f"Values for {col} differ by a factor of 1,000 or more:")
                sh._emit(flagged.to_string(index=False))
    return issues
__all__ = ["validate_outputs"]
