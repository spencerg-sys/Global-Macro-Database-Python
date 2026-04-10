from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_lbr_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    path = raw_dir / "country_level" / "LBR_1.xlsx"

    def _read_lbr_sheet(sheet: str, value_name: str, *, nrows: int | None = None) -> pd.DataFrame:
        frame = _read_excel_compat(path, sheet_name=sheet, nrows=nrows)
        frame = frame.iloc[:, :2].copy()
        frame.columns = ["year", value_name]
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
        frame[value_name] = pd.to_numeric(frame[value_name], errors="coerce")
        frame = frame.loc[frame["year"].notna(), ["year", value_name]].copy()
        return frame

    master = _read_lbr_sheet("1. Population", "pop")
    master["pop"] = pd.to_numeric(master["pop"], errors="coerce") / 1_000_000

    for sheet, value_name in [
        ("3. Public Revenue", "govrev"),
        ("4. Public Spending", "govexp"),
        ("6. Total exports", "exports"),
        ("7. Total imports ", "imports"),
    ]:
        current = _read_lbr_sheet(sheet, value_name)
        current[value_name] = pd.to_numeric(current[value_name], errors="coerce") / 1_000_000
        master = current.merge(master, on="year", how="outer")

    rgdp_pc = _read_lbr_sheet("11. GDP per capita", "rGDP_pc_USD", nrows=136)
    master = rgdp_pc.merge(master, on="year", how="outer")
    master["rGDP_USD"] = pd.to_numeric(master["rGDP_pc_USD"], errors="coerce") * pd.to_numeric(master["pop"], errors="coerce")
    master["ISO3"] = "LBR"
    master = master.rename(columns={col: f"CS1_{col}" for col in master.columns if col not in {"ISO3", "year"}})
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    master["CS1_rGDP_pc_USD"] = pd.to_numeric(master["CS1_rGDP_pc_USD"], errors="coerce").astype("int16")
    for col in ["CS1_imports", "CS1_exports", "CS1_govexp", "CS1_govrev", "CS1_pop"]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
    master["CS1_rGDP_USD"] = pd.to_numeric(master["CS1_rGDP_USD"], errors="coerce").astype("float32")
    master = master[
        [
            "ISO3",
            "year",
            "CS1_rGDP_pc_USD",
            "CS1_imports",
            "CS1_exports",
            "CS1_govexp",
            "CS1_govrev",
            "CS1_pop",
            "CS1_rGDP_USD",
        ]
    ].copy()
    master = _sort_keys(master)
    out_path = clean_dir / "country_level" / "LBR_1.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_lbr_1"]
