from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_chn_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    df = pd.read_excel(raw_dir / "country_level" / "CHN_1.xlsx", sheet_name="Sheet1")
    df = df.iloc[1:].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.rename(columns={"rGDP": "rGDP_index"})
    for col in ["nGDP", "finv", "cons", "exports", "imports", "govrev", "govexp", "govtax"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") * 100
    df["USDfx"] = pd.to_numeric(df["USDfx"], errors="coerce") / 100
    df["pop"] = pd.to_numeric(df["pop"], errors="coerce") / 100
    df["infl"] = pd.to_numeric(df["infl"], errors="coerce") - 100
    base_ngdp = pd.to_numeric(df.loc[pd.to_numeric(df["rGDP_index"], errors="coerce").eq(100), "nGDP"], errors="coerce").max()
    df["rGDP"] = pd.to_numeric(df["rGDP_index"], errors="coerce") * base_ngdp / 100
    df["CA_USD"] = pd.to_numeric(df["CA_USD"], errors="coerce") / 100
    df["CA"] = pd.to_numeric(df["CA_USD"], errors="coerce") * pd.to_numeric(df["USDfx"], errors="coerce")
    # The reference pipeline materializes generated CA as float before the ratio is generated.
    df["CA"] = pd.to_numeric(df["CA"], errors="coerce").astype("float32")
    df["CA_GDP"] = pd.to_numeric(df["CA"], errors="coerce") / pd.to_numeric(df["nGDP"], errors="coerce") * 100
    df = df.drop(columns=["CA_USD", "rGDP_index"], errors="ignore")
    df["ISO3"] = "CHN"
    df = df.rename(columns={col: f"CS1_{col}" for col in df.columns if col not in {"ISO3", "year"}})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int16")
    for col in [
        "CS1_nGDP",
        "CS1_finv",
        "CS1_cons",
        "CS1_pop",
        "CS1_unemp",
        "CS1_exports",
        "CS1_imports",
        "CS1_USDfx",
        "CS1_govrev",
        "CS1_govexp",
        "CS1_govtax",
        "CS1_infl",
        "CS1_CPI",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in ["CS1_rGDP", "CS1_CA", "CS1_CA_GDP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df = df[
        [
            "ISO3",
            "year",
            "CS1_nGDP",
            "CS1_finv",
            "CS1_cons",
            "CS1_pop",
            "CS1_unemp",
            "CS1_exports",
            "CS1_imports",
            "CS1_USDfx",
            "CS1_govrev",
            "CS1_govexp",
            "CS1_govtax",
            "CS1_infl",
            "CS1_CPI",
            "CS1_rGDP",
            "CS1_CA",
            "CS1_CA_GDP",
        ]
    ].copy()
    for col in [
        "CS1_nGDP",
        "CS1_finv",
        "CS1_cons",
        "CS1_pop",
        "CS1_unemp",
        "CS1_exports",
        "CS1_imports",
        "CS1_USDfx",
        "CS1_govrev",
        "CS1_govexp",
        "CS1_govtax",
        "CS1_infl",
        "CS1_CPI",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in ["CS1_rGDP", "CS1_CA", "CS1_CA_GDP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df = _sort_keys(df)
    out_path = clean_dir / "country_level" / "CHN_1.dta"
    _save_dta(df, out_path)
    return df
__all__ = ["clean_chn_1"]
