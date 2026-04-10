from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_imf_gdd(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    df = _load_dta(raw_dir / "aggregators" / "IMF" / "Global Debt Database.dta")
    mapping = _kountry_imfn_to_iso3(helper_dir)
    df = df.merge(mapping, left_on="ifscode", right_on="IFS", how="left")
    df = df.loc[df["ISO3"].notna(), ["ISO3", "year", "cg", "ngdp"]].copy()
    df = df.rename(columns={"cg": "govdebt_GDP", "ngdp": "nGDP"})
    df.loc[df["ISO3"].astype(str) != "VEN", "nGDP"] = pd.to_numeric(df.loc[df["ISO3"].astype(str) != "VEN", "nGDP"], errors="coerce") * 1000
    df.loc[df["ISO3"].astype(str) == "VEN", "nGDP"] = pd.to_numeric(df.loc[df["ISO3"].astype(str) == "VEN", "nGDP"], errors="coerce") / 1000
    mask = (df["ISO3"].astype(str) == "AFG") & (pd.to_numeric(df["year"], errors="coerce") <= 1993)
    df.loc[mask, "nGDP"] = pd.to_numeric(df.loc[mask, "nGDP"], errors="coerce") / 1000
    df.loc[df["ISO3"].astype(str) == "HRV", "nGDP"] = pd.to_numeric(df.loc[df["ISO3"].astype(str) == "HRV", "nGDP"], errors="coerce") / 7.5345
    df = df.rename(columns={col: f"IMF_GDD_{col}" for col in df.columns if col not in {"ISO3", "year"}})
    df = _sort_keys(df)
    if df.duplicated(["ISO3", "year"]).any():
        raise sh.PipelineRuntimeError("ISO3 year do not uniquely identify observations", code=459)
    out_path = clean_dir / "aggregators" / "IMF" / "IMF_GDD.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_imf_gdd"]
