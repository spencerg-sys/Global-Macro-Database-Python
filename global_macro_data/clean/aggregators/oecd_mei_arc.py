from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_oecd_mei_arc(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    path = raw_dir / "aggregators" / "OECD" / "OECD_MEI_ARC.xlsx"

    def _merge_current_master(current: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
        overlap = [col for col in current.columns if col not in {"ISO3", "year"} and col in master.columns]
        merged = current.merge(master, on=["ISO3", "year"], how="outer", suffixes=("_master", "_using"))
        for col in overlap:
            merged[col] = pd.to_numeric(merged[f"{col}_master"], errors="coerce").where(
                pd.to_numeric(merged[f"{col}_master"], errors="coerce").notna(),
                pd.to_numeric(merged[f"{col}_using"], errors="coerce"),
            )
            merged = merged.drop(columns=[f"{col}_master", f"{col}_using"], errors="ignore")
        return merged

    master = _read_excel_compat(path, sheet_name="Sheet1")
    for sheet in ["Sheet2", "Sheet3"]:
        current = _read_excel_compat(path, sheet_name=sheet)
        master = _merge_current_master(current, master)

    master = master.rename(columns={col: f"OECD_MEI_ARC_{col}" for col in master.columns if col not in {"ISO3", "year"}})
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    for col in ["OECD_MEI_ARC_ltrate", "OECD_MEI_ARC_strate", "OECD_MEI_ARC_cbrate"]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
    master = master[["ISO3", "year", "OECD_MEI_ARC_ltrate", "OECD_MEI_ARC_strate", "OECD_MEI_ARC_cbrate"]].copy()
    master = _sort_keys(master)
    out_path = clean_dir / "aggregators" / "OECD" / "OECD_MEI_ARC.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_oecd_mei_arc"]
