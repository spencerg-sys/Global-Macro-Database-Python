from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_oecd_qna(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    df = _load_dta(raw_dir / "aggregators" / "OECD" / "OECD_QNA.dta")
    df = df[["period", "value", "location", "subject"]].copy()
    df = df.rename(columns={"period": "year", "location": "ISO3", "value": "OECD_QNA_"})

    df.loc[df["subject"].astype(str) == "B1_GS1", "subject"] = "nGDP"
    df.loc[df["subject"].astype(str) == "B8GS1", "subject"] = "sav"
    df.loc[df["subject"].astype(str) == "P3", "subject"] = "cons"
    df.loc[df["subject"].astype(str) == "P5", "subject"] = "inv"
    df.loc[df["subject"].astype(str) == "P51", "subject"] = "finv"

    df = df.drop_duplicates(subset=["ISO3", "year", "subject"], keep="first")
    countrylist = _load_dta(helper_dir / "countrylist.dta")[["ISO3"]].copy()
    df = df.merge(countrylist.assign(_keep=1), on="ISO3", how="inner").drop(columns=["_keep"])

    wide = df.pivot(index=["ISO3", "year"], columns="subject", values="OECD_QNA_").reset_index()
    wide.columns.name = None

    ind_idn_mask = wide["ISO3"].astype(str).isin(["IND", "IDN"])
    for col in [c for c in ["cons", "inv", "finv"] if c in wide.columns]:
        wide.loc[ind_idn_mask, col] = pd.to_numeric(wide.loc[ind_idn_mask, col], errors="coerce") * 1000

    wide = wide.rename(columns={col: f"OECD_QNA_{col}" for col in wide.columns if col not in {"ISO3", "year"}})
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int16")
    wide = _sort_keys(wide)
    wide = wide[["ISO3", "year"] + [col for col in wide.columns if col not in {"ISO3", "year"}]]

    out_path = clean_dir / "aggregators" / "OECD" / "OECD_QNA.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_oecd_qna"]
