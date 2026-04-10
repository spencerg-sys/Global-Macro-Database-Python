from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_bit(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)

    df = pd.read_csv(raw_dir / "aggregators" / "BIT" / "BIT_USDfx.csv")
    df.columns = [str(col).strip().lower().replace(" ", "") for col in df.columns]
    df["rateconvention"] = df["rateconvention"].replace(
        {
            "Dollars amount per 1 units of foreign currency": "USDfx_i",
            "Foreign currency amount for 1 Dollar": "USDfx",
            "Liras amount per 1 units of foreign currency": "LIRfx",
        }
    )
    usd_mask = df["rateconvention"].eq("USDfx_i")
    # Match the reference float materialization before the reciprocal conversion.
    rate_float = pd.to_numeric(df.loc[usd_mask, "rate"], errors="coerce").astype("float32")
    df.loc[usd_mask, "rate"] = 1 / rate_float.astype("float64")
    df.loc[usd_mask, "rateconvention"] = "USDfx"
    df = df.drop(columns=["isocode", "uiccode"], errors="ignore")

    wide = df.pivot(index=["currency", "referencedate(cet)"], columns="rateconvention", values="rate").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={col: f"BIT_{col}" for col in wide.columns if col not in {"currency", "referencedate(cet)"}})
    wide["ISO3"] = pd.Series(pd.NA, index=wide.index, dtype="object")
    wide.loc[wide["currency"].eq("Austrian Shilling"), "ISO3"] = "AUT"
    wide.loc[wide["currency"].eq("CFP Franc"), "ISO3"] = "PYF"
    wide.loc[wide["currency"].eq("DDR Mark"), "ISO3"] = "DDR"
    wide.loc[wide["currency"].eq("Falkland Pound"), "ISO3"] = "FLK"
    wide.loc[wide["currency"].eq("Gibraltar Pound"), "ISO3"] = "GIB"
    wide.loc[wide["currency"].eq("Greek Drachma"), "ISO3"] = "GRC"
    wide.loc[wide["currency"].eq("Italian Lira"), "ISO3"] = "ITA"
    wide.loc[wide["currency"].eq("Lek"), "ISO3"] = "ALB"
    wide.loc[wide["currency"].eq("North Korean Won"), "ISO3"] = "PRK"
    wide.loc[wide["currency"].eq("Ruble"), "ISO3"] = "SUN"
    wide.loc[wide["currency"].eq("St. Helena Pound"), "ISO3"] = "SHN"
    wide = wide.drop(columns=["currency", "BIT_LIRfx"], errors="ignore")
    wide = wide.rename(columns={"referencedate(cet)": "year"})
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int16")
    wide["BIT_USDfx"] = pd.to_numeric(wide["BIT_USDfx"], errors="coerce").astype("float32")
    wide = wide[["ISO3", "year", "BIT_USDfx"]].copy()
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int16")
    wide["BIT_USDfx"] = pd.to_numeric(wide["BIT_USDfx"], errors="coerce").astype("float32")
    wide = _sort_keys(wide)
    out_path = clean_dir / "aggregators" / "BIT" / "BIT_USDfx.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_bit"]
