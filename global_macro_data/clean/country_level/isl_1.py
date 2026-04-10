from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_isl_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = pd.read_excel(raw_dir / "country_level" / "ISL_1.xlsx", header=None)
    df = df.iloc[3:].reset_index(drop=True).copy()
    df["varname"] = ""
    df.loc[1, "varname"] = "CS1_nGDP"
    df.loc[2, "varname"] = "CS1_rGDP_pc_index"
    df.loc[8, "varname"] = "CS1_pop"
    df = df.loc[df["varname"] != ""].copy()

    value_cols = [col for col in df.columns if col not in {0, 1, "varname"}]
    year_map = {col: str(1870 + idx) for idx, col in enumerate(value_cols)}
    out = df.drop(columns=[0, 1], errors="ignore").rename(columns=year_map)
    out["ISO3"] = "ISL"
    out = out.melt(id_vars=["ISO3", "varname"], var_name="year", value_name="value")
    out = out.pivot(index=["ISO3", "year"], columns="varname", values="value").reset_index()
    out.columns.name = None
    out = out.rename(columns={"CS1_rGDP_pc_index": "CS1_deflator"})
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out["CS1_nGDP"] = pd.to_numeric(out["CS1_nGDP"], errors="coerce").astype("float64")
    out["CS1_deflator"] = pd.to_numeric(out["CS1_deflator"], errors="coerce").astype("float64")
    out["CS1_pop"] = (pd.to_numeric(out["CS1_pop"], errors="coerce") / 1_000_000).astype("float64")
    out = out[["ISO3", "year", "CS1_nGDP", "CS1_pop", "CS1_deflator"]].copy()
    out = _sort_keys(out)
    out_path = clean_dir / "country_level" / "ISL_1.dta"
    _save_dta(out, out_path)
    return out
__all__ = ["clean_isl_1"]
