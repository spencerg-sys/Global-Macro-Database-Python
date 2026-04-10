from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_flora(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    path = raw_dir / "aggregators" / "FLORA" / "Flora_expenditure_series_Europe.xlsx"

    def _flora_value(value: object) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if text == "":
            return np.nan
        try:
            return float(format(float(text), ".16g"))
        except ValueError:
            return np.nan

    sheet_configs = [
        ("Austria", "AUT", "AK", 5, 53, None),
        ("Belgium", "BEL", "AL", 5, 141, None),
        ("Denmark", "DNK", "AK", 5, 107, None),
        ("Finland", "FIN", "AL", 5, 94, None),
        ("France", "FRA", "AM", 5, 154, None),
        ("Germany", "DEU", "AK", 5, 126, None),
        (" Italy", "ITA", "AK", 5, 114, "italy"),
        ("Netherlands", "NLD", "AL", 5, 126, None),
        ("Norway", "NOR", "AK", 6, 126, None),
        ("Sweden", "SWE", "AL", 6, 126, None),
        ("Switzerland", "CHE", "AL", 5, 38, None),
        (" UK", "GBR", "AL", 5, 186, "uk"),
    ]

    pieces: list[pd.DataFrame] = []
    for sheet_name, iso3, value_col, drop_head, keep_n, special in sheet_configs:
        frame = _read_excel_compat(path, sheet_name=sheet_name, header=None, dtype=str)
        frame = frame.iloc[:, [_excel_column_to_index("A"), _excel_column_to_index(value_col)]].copy().reset_index(drop=True)

        if special == "italy":
            fix_mask = frame.iloc[:, 0].astype("string").eq("1973") & (
                frame.iloc[:, 1].isna() | frame.iloc[:, 1].astype("string").str.strip().eq("")
            )
            frame.loc[fix_mask, frame.columns[0]] = "1974"

        frame = frame.iloc[drop_head:].reset_index(drop=True)
        values = frame.iloc[:, 1].map(_flora_value)

        if special == "uk":
            source_years = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
            years = pd.Series(np.nan, index=frame.index, dtype="float64")
            for i in range(keep_n):
                j = i + 151
                if j < len(source_years):
                    years.iloc[i] = source_years.iloc[j]
        else:
            years = pd.to_numeric(frame.iloc[:, 0], errors="coerce")

        out = pd.DataFrame(
            {
                "ISO3": iso3,
                "year": years.iloc[:keep_n].to_numpy(),
                "FLORA_govexp_GDP": values.iloc[:keep_n].to_numpy(),
            }
        )
        out.loc[pd.to_numeric(out["FLORA_govexp_GDP"], errors="coerce").eq(0), "FLORA_govexp_GDP"] = np.nan
        pieces.append(out)

    result = pd.concat(pieces, ignore_index=True)
    result = result[["ISO3", "year", "FLORA_govexp_GDP"]].copy()
    result["ISO3"] = result["ISO3"].astype("object")
    result["year"] = pd.to_numeric(result["year"], errors="coerce").astype("float32")
    result["FLORA_govexp_GDP"] = pd.to_numeric(result["FLORA_govexp_GDP"], errors="coerce").astype("float64")
    result["ISO3"] = result["ISO3"].astype("object")
    result["year"] = pd.to_numeric(result["year"], errors="coerce").astype("float32")
    result["FLORA_govexp_GDP"] = pd.to_numeric(result["FLORA_govexp_GDP"], errors="coerce").astype("float64")
    result = _sort_keys(result)
    out_path = clean_dir / "aggregators" / "FLORA" / "Flora.dta"
    _save_dta(result, out_path)
    return result
__all__ = ["clean_flora"]
