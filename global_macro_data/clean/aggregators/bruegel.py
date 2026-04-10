from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_bruegel(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    df = _read_excel_compat(raw_dir / "aggregators" / "Bruegel" / "Bruegel_reer.xls", sheet_name="REER_ANNUAL_65")
    df = df.rename(columns={df.columns[0]: "year"})
    df = df.loc[df["year"].astype("string").str.strip().ne(""), ["year"] + [col for col in df.columns if str(col).startswith("REER_65_")]].copy()
    long = df.melt(id_vars=["year"], var_name="country", value_name="BRUEGEL_REER")
    long["ISO2"] = long["country"].astype("string").str.removeprefix("REER_65_")
    long = long.drop(columns=["country"])
    long = long.loc[~long["ISO2"].eq("EA")].copy()
    long.loc[long["ISO2"].eq("SQ"), "ISO2"] = "RS"

    countrylist = _load_dta(helper_dir / "countrylist.dta")[["ISO2", "ISO3"]].dropna().drop_duplicates().copy()
    long = long.merge(countrylist, on="ISO2", how="left")
    long = long.loc[long["ISO3"].notna(), ["ISO3", "year", "BRUEGEL_REER"]].copy()
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long["BRUEGEL_REER"] = pd.to_numeric(long["BRUEGEL_REER"], errors="coerce")
    base = long.loc[long["year"].eq(2010), ["ISO3", "BRUEGEL_REER"]].rename(columns={"BRUEGEL_REER": "REER_2010"})
    base = base.groupby("ISO3", as_index=False)["REER_2010"].mean()
    # The reference egen target is stored as float before the division.
    base["REER_2010"] = base["REER_2010"].astype("float32")
    long = long.merge(base, on="ISO3", how="left")
    long["BRUEGEL_REER"] = (
        pd.to_numeric(long["BRUEGEL_REER"], errors="coerce")
        * 100
        / pd.to_numeric(long["REER_2010"], errors="coerce").astype("float64")
    )
    long = long.drop(columns=["REER_2010"], errors="ignore")
    long.loc[long["ISO3"].eq("ARG") & long["year"].eq(2010), "BRUEGEL_REER"] = 100
    long.loc[long["ISO3"].eq("TKM") & long["year"].eq(2010), "BRUEGEL_REER"] = 100
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("int16")
    long["BRUEGEL_REER"] = pd.to_numeric(long["BRUEGEL_REER"], errors="coerce").astype("float32")
    long = long[["ISO3", "year", "BRUEGEL_REER"]].copy()
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("int16")
    long["BRUEGEL_REER"] = pd.to_numeric(long["BRUEGEL_REER"], errors="coerce").astype("float32")
    long = _sort_keys(long)
    out_path = clean_dir / "aggregators" / "Bruegel" / "Bruegel.dta"
    _save_dta(long, out_path)
    return long
__all__ = ["clean_bruegel"]
