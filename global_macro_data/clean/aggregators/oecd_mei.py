from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_oecd_mei(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)

    df = _load_dta(raw_dir / "aggregators" / "OECD" / "OECD_MEI.dta")
    df = df[["period", "value", "subject", "location"]].copy()
    wide = df.pivot_table(index=["period", "location"], columns="subject", values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={
            "period": "year",
            "location": "ISO3",
            "CCRETT01": "REER",
            "IRLTLT01": "ltrate",
            "IRSTCB01": "cbrate",
            "MABMM301": "M3",
            "MANMM101": "M1",
        }
    )
    wide = wide.loc[wide["ISO3"].astype(str) != "EA19"].copy()

    if "M1" in wide.columns:
        mask = ~wide["ISO3"].astype(str).isin(["GBR", "ISR"])
        wide.loc[mask, "M1"] = pd.to_numeric(wide.loc[mask, "M1"], errors="coerce") * 1000
        idn_mask = wide["ISO3"].astype(str) == "IDN"
        wide.loc[idn_mask, "M1"] = pd.to_numeric(wide.loc[idn_mask, "M1"], errors="coerce") * 1000
        jpn_mask = wide["ISO3"].astype(str) == "JPN"
        wide.loc[jpn_mask, "M1"] = pd.to_numeric(wide.loc[jpn_mask, "M1"], errors="coerce") / 10

    if "M3" in wide.columns:
        mask = ~wide["ISO3"].astype(str).isin(["GBR", "ISR"])
        wide.loc[mask, "M3"] = pd.to_numeric(wide.loc[mask, "M3"], errors="coerce") * 1000
        idn_mask = wide["ISO3"].astype(str) == "IDN"
        wide.loc[idn_mask, "M3"] = pd.to_numeric(wide.loc[idn_mask, "M3"], errors="coerce") * 1000
        jpn_mask = wide["ISO3"].astype(str) == "JPN"
        wide.loc[jpn_mask, "M3"] = pd.to_numeric(wide.loc[jpn_mask, "M3"], errors="coerce") / 10

    value_cols = [col for col in wide.columns if col not in {"ISO3", "year"}]
    wide = wide.rename(columns={col: f"OECD_MEI_{col}" for col in value_cols})
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int16")
    wide = _sort_keys(wide)
    wide = wide[["ISO3", "year"] + [col for col in wide.columns if col not in {"ISO3", "year"}]]

    out_path = clean_dir / "aggregators" / "OECD" / "OECD_MEI.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_oecd_mei"]
