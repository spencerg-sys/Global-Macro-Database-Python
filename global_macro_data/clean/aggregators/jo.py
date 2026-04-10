from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_jo(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    input_dir = raw_dir / "aggregators" / "JO"

    country_map = {
        "AUSTRALIA": "AUS",
        "CANADA": "CAN",
        "DENMARK": "DNK",
        "FINLAND": "FIN",
        "FRANCE": "FRA",
        "GERMANY": "DEU",
        "ITALY": "ITA",
        "JAPAN": "JPN",
        "NORWAY": "NOR",
        "RUSSIA": "RUS",
        "SWEDEN": "SWE",
        "U_K": "GBR",
        "USA": "USA",
    }

    def _jo_value(value: object) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if text in {"", "<NA>", "nan", "None"}:
            return np.nan
        try:
            return float(format(float(text), ".16g"))
        except ValueError:
            return np.nan

    frames: list[pd.DataFrame] = []
    for path in sorted(input_dir.glob("*.xls")):
        current = _read_excel_compat(path, sheet_name="Final")
        if current.empty:
            continue
        first_var = str(current.columns[0])
        keep = current.iloc[:, :5].copy()
        keep.columns = [first_var, "B", "C", "D", "E"]
        keep["Country"] = first_var
        keep[first_var] = pd.to_numeric(keep[first_var], errors="coerce")
        keep = keep.loc[keep[first_var].notna(), ["Country", first_var, "B", "C", "D", "E"]].copy()
        keep = keep.rename(columns={first_var: "year", "Country": "ISO3", "B": "JO_nGDP", "C": "JO_finv", "D": "JO_stock_change", "E": "JO_CA"})
        for col in ["JO_nGDP", "JO_finv", "JO_stock_change", "JO_CA"]:
            keep[col] = keep[col].map(_jo_value)
        frames.append(keep)

    if not frames:
        raise ValueError("No JO raw files found")

    df = pd.concat(frames, ignore_index=True)
    df["ISO3"] = df["ISO3"].astype("string").map(country_map)
    for col in ["JO_nGDP", "JO_finv", "JO_stock_change", "JO_CA"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["JO_nGDP", "JO_finv", "JO_stock_change", "JO_CA"]:
        df.loc[df["ISO3"].eq("FRA"), col] = pd.to_numeric(df.loc[df["ISO3"].eq("FRA"), col], errors="coerce") / 100
        df.loc[df["ISO3"].eq("FIN"), col] = pd.to_numeric(df.loc[df["ISO3"].eq("FIN"), col], errors="coerce") / 100000
        df.loc[df["ISO3"].eq("AUS"), col] = pd.to_numeric(df.loc[df["ISO3"].eq("AUS"), col], errors="coerce") * 2
        df.loc[df["ISO3"].eq("DEU") & pd.to_numeric(df["year"], errors="coerce").le(1913), col] = (
            pd.to_numeric(df.loc[df["ISO3"].eq("DEU") & pd.to_numeric(df["year"], errors="coerce").le(1913), col], errors="coerce") / (10**12)
        )

    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].copy()
    df = df.merge(eur_fx, on="ISO3", how="left")
    fx_mask = df["EUR_irrevocable_FX"].notna()
    for col in ["JO_nGDP", "JO_finv", "JO_stock_change", "JO_CA"]:
        df.loc[fx_mask, col] = pd.to_numeric(df.loc[fx_mask, col], errors="coerce") / pd.to_numeric(df.loc[fx_mask, "EUR_irrevocable_FX"], errors="coerce")
    df = df.drop(columns=["EUR_irrevocable_FX"], errors="ignore")

    df["JO_CA_GDP"] = pd.to_numeric(df["JO_CA"], errors="coerce") / pd.to_numeric(df["JO_nGDP"], errors="coerce") * 100
    df["JO_inv"] = (
        pd.to_numeric(df["JO_finv"], errors="coerce")
        + pd.to_numeric(df["JO_stock_change"], errors="coerce")
    ).astype("float32")
    df.loc[pd.to_numeric(df["JO_inv"], errors="coerce") < 0, "JO_inv"] = pd.NA
    df["JO_finv_GDP"] = pd.to_numeric(df["JO_finv"], errors="coerce") / pd.to_numeric(df["JO_nGDP"], errors="coerce") * 100
    df["JO_inv_GDP"] = pd.to_numeric(df["JO_inv"], errors="coerce") / pd.to_numeric(df["JO_nGDP"], errors="coerce") * 100

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    for col in ["JO_nGDP", "JO_finv", "JO_CA"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in ["JO_CA_GDP", "JO_inv", "JO_finv_GDP", "JO_inv_GDP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df = df[["ISO3", "year", "JO_nGDP", "JO_finv", "JO_CA", "JO_CA_GDP", "JO_inv", "JO_finv_GDP", "JO_inv_GDP"]].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    for col in ["JO_nGDP", "JO_finv", "JO_CA"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in ["JO_CA_GDP", "JO_inv", "JO_finv_GDP", "JO_inv_GDP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df = _sort_keys(df)
    out_path = clean_dir / "aggregators" / "JO" / "JO.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_jo"]
