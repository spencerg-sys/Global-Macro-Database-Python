from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_oecd_kei(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    df = _load_dta(raw_dir / "aggregators" / "OECD" / "OECD_KEI.dta")
    df = df[["period", "value", "frequency", "location", "subject", "measure", "series_name"]].copy()
    df = df.rename(columns={"period": "year", "location": "ISO3", "value": "OECD_KEI_"})

    countrylist = _load_dta(helper_dir / "countrylist.dta")[["ISO3"]].copy()
    df = df.merge(countrylist.assign(_keep=1), on="ISO3", how="inner")
    df = df.drop(columns=["_keep"])

    df.loc[df["subject"].astype(str) == "B6BLTT02", "subject"] = "CA_GDP"
    df.loc[(df["subject"].astype(str) == "CPALTT01") & (df["measure"].astype(str) == "GP"), "subject"] = "infl"
    df.loc[(df["subject"].astype(str) == "CPALTT01") & (df["measure"].astype(str) == "ST"), "subject"] = "CPI"
    df.loc[df["subject"].astype(str) == "IR3TIB01", "subject"] = "strate"
    df.loc[df["subject"].astype(str) == "IRLTLT01", "subject"] = "ltrate"
    df.loc[df["subject"].astype(str) == "LRHUTTTT", "subject"] = "unemp"
    df = df.loc[df["subject"].astype(str) != "CPALTT01"].copy()
    df = df.drop(columns=["series_name", "frequency", "measure"], errors="ignore")

    wide = df.pivot_table(index=["ISO3", "year"], columns="subject", values="OECD_KEI_", aggfunc="first").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={col: f"OECD_KEI_{col}" for col in wide.columns if col not in {"ISO3", "year"}})
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int16")
    wide = _sort_keys(wide)

    out_path = clean_dir / "aggregators" / "OECD" / "OECD_KEI.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_oecd_kei"]
