from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_mar_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)

    def _parse_monetary_table(path: Path, sheet_name: str) -> pd.DataFrame:
        df = _read_excel_compat(path, sheet_name=sheet_name, header=None)
        labels = df.iloc[:, 0].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
        keep_labels = {"Circulation fiduciaire": "M0", "M1": "M1", "M2": "M2", "M3": "M3"}
        date_row_idx = int(pd.to_datetime(df.iloc[:, 1], format="mixed", errors="coerce").notna().idxmax())
        keep_mask = labels.isin(keep_labels.keys())
        subset = pd.concat([df.loc[[date_row_idx]], df.loc[keep_mask]]).copy()

        dates = pd.to_datetime(subset.iloc[0, 1:], format="mixed", errors="coerce")
        values = subset.iloc[1:, 1:].copy()
        values.index = labels.loc[keep_mask].map(keep_labels).tolist()
        values.columns = dates
        values = values.T
        values.index.name = "datem"
        values = values.reset_index()
        values.columns.name = None
        for col in ["M0", "M1", "M2", "M3"]:
            values[col] = values[col].map(lambda value: np.nan if pd.isna(value) else float(format(float(value), ".16g")))
        values["year"] = pd.to_numeric(pd.to_datetime(values["datem"], errors="coerce").dt.year, errors="coerce")
        values["month"] = pd.to_numeric(pd.to_datetime(values["datem"], errors="coerce").dt.month, errors="coerce")
        return values[["year", "month", "M0", "M1", "M2", "M3"]].copy()

    mar1_book = pd.ExcelFile(raw_dir / "country_level" / "MAR_1.xls")
    mar1_sheet = [s for s in mar1_book.sheet_names if "Agr" in s][0]
    part1 = _parse_monetary_table(raw_dir / "country_level" / "MAR_1.xls", mar1_sheet)
    part2 = _parse_monetary_table(raw_dir / "country_level" / "MAR_2.xlsx", "Feuil1")

    out = pd.concat([part2, part1], ignore_index=True)
    out = out.sort_values(["year", "month"], kind="mergesort").groupby("year", sort=False).tail(1).copy()
    out = out.drop(columns=["month"], errors="ignore")
    out["ISO3"] = "MAR"
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("float32")
    for col in ["M0", "M1", "M2", "M3"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    out = out.rename(columns={col: f"CS1_{col}" for col in ["M0", "M1", "M2", "M3"]})
    out = out[["ISO3", "year", "CS1_M0", "CS1_M1", "CS1_M2", "CS1_M3"]].copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("float32")
    for col in ["CS1_M0", "CS1_M1", "CS1_M2", "CS1_M3"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    out = _sort_keys(out, keys=("ISO3", "year"))
    out_path = clean_dir / "country_level" / "MAR_1.dta"
    _save_dta(out, out_path)
    return out
__all__ = ["clean_mar_1"]
