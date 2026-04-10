from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_bg(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    from .wdi import clean_wdi

    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    path = raw_dir / "aggregators" / "BG" / "BG.xlsx"

    wdi_path = clean_dir / "aggregators" / "WB" / "WDI.dta"
    if not wdi_path.exists():
        clean_wdi(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=helper_dir)

    df = _read_excel_compat(path, header=None, usecols="A:I", skiprows=1, nrows=125)
    df.columns = ["year", "ZAF", "ZWE", "GHA", "NGA", "KEN", "UGA", "ZMB", "MWI"]
    df = df.melt(id_vars="year", var_name="ISO3", value_name="rGDP_pc_USD")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["rGDP_pc_USD"] = pd.to_numeric(df["rGDP_pc_USD"], errors="coerce")
    df = df.loc[df["year"].notna(), ["ISO3", "year", "rGDP_pc_USD"]].copy()

    wdi = _load_dta(wdi_path)[["ISO3", "year", "WDI_rGDP"]].copy()
    merged = df.merge(wdi, on=["ISO3", "year"], how="left")
    merged = merged.rename(columns={"rGDP_pc_USD": "BG_rGDP"})
    spliced = sh.splice(merged, priority="WDI BG", generate="rGDP", varname="rGDP", method="chainlink", base_year=2000, save="NO")
    out = spliced[["ISO3", "year", "BG_rGDP", "rGDP"]].copy()
    out = out.rename(columns={"BG_rGDP": "BG_rGDP_pc_USD", "rGDP": "BG_rGDP"})
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int16")
    out["BG_rGDP_pc_USD"] = pd.to_numeric(out["BG_rGDP_pc_USD"], errors="coerce").astype("float64")
    out["BG_rGDP"] = pd.to_numeric(out["BG_rGDP"], errors="coerce").astype("float32")
    out = _sort_keys(out)
    out_path = clean_dir / "aggregators" / "BG" / "BG.dta"
    _save_dta(out, out_path)
    return out
__all__ = ["clean_bg"]
