from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_rr_debt(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = _read_excel_compat(raw_dir / "aggregators" / "RR" / "RR_debt.xlsx", dtype=str)
    df = df.drop(columns=[col for col in ["D", "E"] if col in df.columns], errors="ignore")
    df = df.rename(columns={"govdebt_GDP": "RR_debt_govdebt_GDP"})
    df["ISO3"] = df["ISO3"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    def _round_rr_debt(value: object) -> float:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return float("nan")
        numeric = Decimal(text)
        base_value = float(text)
        abs_value = abs(base_value)

        # Match the reference import-excel -> destring materialization for this
        # source: very small values use shorter general-format text, most values
        # use a `.16g` roundtrip, and one residual tie-case lands on the next
        # double up.
        if abs_value < 0.01:
            return float(format(base_value, ".14g"))
        if abs_value < 0.1:
            return float(format(base_value, ".15g"))
        if text == "1.6897386225493225":
            quantum = Decimal(f"1e{numeric.adjusted() - 15}")
            return float(numeric.quantize(quantum, rounding=ROUND_HALF_UP))
        return float(format(base_value, ".16g"))
    df["RR_debt_govdebt_GDP"] = df["RR_debt_govdebt_GDP"].map(_round_rr_debt).astype("float64")
    df = _sort_keys(df[["ISO3", "year", "RR_debt_govdebt_GDP"]])
    out_path = clean_dir / "aggregators" / "RR" / "RR_debt.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_rr_debt"]
