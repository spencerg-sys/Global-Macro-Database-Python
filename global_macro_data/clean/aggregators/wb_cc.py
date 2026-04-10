from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_wb_cc(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    path = raw_dir / "aggregators" / "WB" / "WB_inflation.xlsx"

    def _year_label(value: object) -> str:
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(number):
            text = "" if pd.isna(value) else str(value).strip()
            return text.replace(".0", "")
        return str(int(number))

    def _load_wb_cc_sheet(sheet: str, prefix: str, keep_rows: int, drop_cols: list[int]) -> pd.DataFrame:
        df = _read_excel_compat(path, sheet_name=sheet, header=None)
        df = df.loc[:, df.notna().any(axis=0)].copy()
        df = df.drop(columns=drop_cols, errors="ignore").iloc[:keep_rows].copy()
        df.iloc[:, 0] = df.iloc[:, 0].astype("string").str.slice(0, 3)
        headers = ["ISO3"] + [f"{prefix}{_year_label(v)}" for v in df.iloc[0, 1:].tolist()]
        df.columns = headers
        df = df.iloc[1:].reset_index(drop=True).copy()
        df = df.melt(id_vars=["ISO3"], var_name="year_token", value_name=prefix)
        df["year"] = pd.to_numeric(df["year_token"].astype("string").str.removeprefix(prefix), errors="coerce")
        df[prefix] = pd.to_numeric(df[prefix], errors="coerce")
        return df[["ISO3", "year", prefix]].copy()

    infl = _load_wb_cc_sheet("hcpi_a", "infl", 204, [1, 2, 3, 4, 59]).rename(columns={"infl": "WB_CC_infl"})
    deflator = _load_wb_cc_sheet("def_a", "deflator", 197, [1, 2, 3, 4]).rename(columns={"deflator": "WB_CC_deflator"})
    out = deflator.merge(infl, on=["ISO3", "year"], how="outer")
    out = out.loc[out["ISO3"].astype(str) != "XXK"].copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int32")
    out["WB_CC_deflator"] = pd.to_numeric(out["WB_CC_deflator"], errors="coerce").astype("float64")
    out["WB_CC_infl"] = pd.to_numeric(out["WB_CC_infl"], errors="coerce").astype("float64")
    out = out[["ISO3", "year", "WB_CC_deflator", "WB_CC_infl"]].copy()
    out = _sort_keys(out)
    out_path = clean_dir / "aggregators" / "WB" / "WB_CC.dta"
    _save_dta(out, out_path)
    return out
__all__ = ["clean_wb_cc"]
