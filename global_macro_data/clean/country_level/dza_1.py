from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_dza_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    path = raw_dir / "country_level" / "DZA_1.xlsx"
    sheet = pd.ExcelFile(path).sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet, header=None, dtype=str)

    labels = df.iloc[:, 0].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    mapping = {
        "MONNAIE ET QUASI-MONNAIE": "CS1_M2",
        "Monnaie": "CS1_M1",
        "Circulation fiduciaire H/BA": "CS1_M0",
    }
    mapped = labels.replace(mapping)
    date_row_candidates = pd.to_datetime(df.iloc[:, 1], format="mixed", errors="coerce")
    date_row_idx = int(date_row_candidates.notna().idxmax())
    keep_mask = mapped.isin({"CS1_M0", "CS1_M1", "CS1_M2"})
    subset = pd.concat([df.loc[[date_row_idx]], df.loc[keep_mask]]).copy()

    dates = pd.to_datetime(subset.iloc[0, 1:], format="mixed", errors="coerce")
    values = subset.iloc[1:, 1:].copy()
    values.index = mapped.loc[keep_mask].tolist()
    values.columns = dates
    values = values.T
    values.index.name = "datem"
    values = values.reset_index()
    values.columns.name = None
    for col in ["CS1_M0", "CS1_M1", "CS1_M2"]:
        values[col] = values[col].map(
            lambda value: np.nan if pd.isna(value) else float(format(float(str(value).strip()), ".16g"))
        )
    values["year"] = pd.to_numeric(pd.to_datetime(values["datem"], errors="coerce").dt.year, errors="coerce")
    values["month"] = pd.to_numeric(pd.to_datetime(values["datem"], errors="coerce").dt.month, errors="coerce")
    values = values.sort_values(["year", "month"], kind="mergesort").groupby("year", sort=False).tail(1).copy()
    values = values.drop(columns=["datem", "month"], errors="ignore")
    values["ISO3"] = "DZA"
    values["year"] = pd.to_numeric(values["year"], errors="coerce").astype("int16")
    for col in ["CS1_M0", "CS1_M1", "CS1_M2"]:
        values[col] = pd.to_numeric(values[col], errors="coerce").astype("float64")
    values = values[["ISO3", "year", "CS1_M0", "CS1_M1", "CS1_M2"]].copy()
    for col in ["CS1_M0", "CS1_M1", "CS1_M2"]:
        values[col] = pd.to_numeric(values[col], errors="coerce").astype("float64")
    values = _sort_keys(values)
    out_path = clean_dir / "country_level" / "DZA_1.dta"
    _save_dta(values, out_path)
    return values
__all__ = ["clean_dza_1"]
