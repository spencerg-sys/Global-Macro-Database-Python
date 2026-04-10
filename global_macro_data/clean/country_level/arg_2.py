from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_arg_2(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    path = raw_dir / "country_level" / "ARG_2.xlsx"

    def _arg2_usdfx(value: object) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if text == "":
            return np.nan
        return float(format(float(text), ".7g"))

    def _arg2_strate(value: object) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if text == "":
            return np.nan
        return float(format(float(text), ".16g"))

    df = pd.read_excel(path, sheet_name="8.7 Series historicas", header=None, usecols=[0, 4, 8, 9, 10, 11, 35], dtype=str)
    df = df.iloc[9:].copy()
    df.columns = ["date", "USDfx", "M0", "M1", "M2", "M3", "strate"]
    df["datem"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = pd.to_numeric(df["datem"].dt.year, errors="coerce")
    df["USDfx"] = df["USDfx"].map(_arg2_usdfx)
    df["strate"] = df["strate"].map(_arg2_strate)
    for col in ["M0", "M1", "M2", "M3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    sums = df[["year", "M0", "M1", "M2", "M3"]].copy()
    for col in ["M0", "M1", "M2", "M3"]:
        sums[col] = pd.to_numeric(sums[col], errors="coerce").fillna(0)
    sums = sums.groupby("year", as_index=False)[["M0", "M1", "M2", "M3"]].sum()

    eoy = df.loc[df["datem"].dt.month.eq(12), ["year", "USDfx", "strate"]].copy()
    eoy = eoy.sort_values("year", kind="mergesort").groupby("year", sort=False).tail(1).copy()

    out = eoy.merge(sums, on="year", how="left")
    out["ISO3"] = "ARG"
    out = out.rename(columns={col: f"CS2_{col}" for col in out.columns if col not in {"ISO3", "year"}})
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int16")
    out["CS2_USDfx"] = pd.to_numeric(out["CS2_USDfx"], errors="coerce").astype("float64")
    out["CS2_strate"] = pd.to_numeric(out["CS2_strate"], errors="coerce").astype("float64")
    for col in ["CS2_M0", "CS2_M1", "CS2_M2", "CS2_M3"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    out = out[["ISO3", "year", "CS2_USDfx", "CS2_strate", "CS2_M0", "CS2_M1", "CS2_M2", "CS2_M3"]].copy()
    out["CS2_USDfx"] = pd.to_numeric(out["CS2_USDfx"], errors="coerce").astype("float64")
    out["CS2_strate"] = pd.to_numeric(out["CS2_strate"], errors="coerce").astype("float64")
    for col in ["CS2_M0", "CS2_M1", "CS2_M2", "CS2_M3"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    out = _sort_keys(out)
    out_path = clean_dir / "country_level" / "ARG_2.dta"
    _save_dta(out, out_path)
    return out
__all__ = ["clean_arg_2"]
