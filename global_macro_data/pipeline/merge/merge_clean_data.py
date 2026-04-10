from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def merge_clean_data(
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
) -> pd.DataFrame:
    clean_dir = _resolve(data_clean_dir)
    final_dir = _resolve(data_final_dir)
    blank_panel_path = _require_blank_panel(data_temp_dir)
    master = pd.read_dta(blank_panel_path, convert_categoricals=False)
    files = [
        p
        for p in _iter_clean_dta_files(clean_dir)
        if "MITCHELL" not in str(p).upper() or "INDIVIDUAL_FILES" not in str(p).upper()
    ]

    for file in files:
        printname = str(file).replace(str(clean_dir) + "\\", "").replace(str(clean_dir) + "/", "")
        sh._emit(f"Merging file {printname}")
        using = pd.read_dta(file, convert_categoricals=False)
        master = _merge_update_1to1(master, using, keys=["ISO3", "year"], error_label=printname)

    value_cols = [col for col in master.columns if col not in {"ISO3", "year"}]
    anydata = master[value_cols].notna().sum(axis=1)
    master["minyear"] = master["year"].where(anydata > 0)
    master["minyear"] = master.groupby("ISO3")["minyear"].transform("min")
    master = master.loc[master["year"] >= master["minyear"]].copy()
    master = master.drop(columns=["minyear"])

    idcm_cols = [col for col in master.columns if col.startswith("IDCM")]
    if idcm_cols:
        master = master.drop(columns=idcm_cols)

    zero_exclude_patterns = ["ISO3", "year", "BVX*", "RR*", "LV*", "*ltrate", "*strate", "*cbrate", "*infl"]
    zero_replace_cols: list[str] = []
    for col in master.columns:
        if any(Path(col).match(pattern) for pattern in zero_exclude_patterns):
            continue
        zero_replace_cols.append(col)

    for col in zero_replace_cols:
        numeric = pd.to_numeric(master[col], errors="coerce")
        master.loc[numeric.eq(0), col] = pd.NA

    final_dir.mkdir(parents=True, exist_ok=True)
    out_path = final_dir / "clean_data_wide.dta"
    master = _key_sort(master, ["ISO3", "year"])
    _save_dta(master, out_path)
    return master
__all__ = ["merge_clean_data"]
