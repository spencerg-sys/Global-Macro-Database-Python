from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_imf_fpp(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    df = _load_dta(raw_dir / "aggregators" / "IMF" / "IMF_FPP.dta")
    mapping = _kountry_imfn_to_iso3(helper_dir)
    df = df.merge(mapping, left_on="ifscode", right_on="IFS", how="left")
    df = df.loc[df["ISO3"].notna(), ["ISO3", "year", "revenue", "expenditure", "debt"]].copy()
    df = df.rename(columns={"revenue": "govrev_GDP", "expenditure": "govexp_GDP", "debt": "govdebt_GDP"})
    df["govdef_GDP"] = pd.to_numeric(df["govrev_GDP"], errors="coerce") - pd.to_numeric(df["govexp_GDP"], errors="coerce")
    df["govdef_GDP"] = pd.to_numeric(df["govdef_GDP"], errors="coerce").astype("float32")
    df = df.rename(columns={col: f"IMF_FPP_{col}" for col in df.columns if col not in {"ISO3", "year"}})
    df = _sort_keys(df)
    if df.duplicated(["ISO3", "year"]).any():
        raise sh.PipelineRuntimeError("ISO3 year do not uniquely identify observations", code=459)
    out_path = clean_dir / "aggregators" / "IMF" / "IMF_FPP.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_imf_fpp"]
