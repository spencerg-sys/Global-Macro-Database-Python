from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_oecd_hpi(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)

    df = _load_dta(raw_dir / "aggregators" / "OECD" / "OECD_HPI.dta")
    df = df.rename(columns={"cou": "ISO3", "period": "year", "value": "OECD_"})
    df = df.loc[~df["ISO3"].astype(str).isin(["EA", "EA17", "OECD"])].copy()
    df = df[["ISO3", "year", "OECD_", "ind"]].copy()
    wide = df.pivot_table(index=["ISO3", "year"], columns="ind", values="OECD_", aggfunc="first").reset_index()
    wide.columns.name = None
    if "RHP" in wide.columns:
        wide = wide.rename(columns={"RHP": "OECD_rHPI"})
    if "HPI" in wide.columns:
        wide = wide.rename(columns={"HPI": "OECD_HPI"})
    keep_cols = ["ISO3", "year"] + [col for col in ["OECD_HPI", "OECD_rHPI"] if col in wide.columns]
    wide = wide[keep_cols]
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int16")
    wide = _sort_keys(wide)

    out_path = clean_dir / "aggregators" / "OECD" / "OECD_HPI.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_oecd_hpi"]
