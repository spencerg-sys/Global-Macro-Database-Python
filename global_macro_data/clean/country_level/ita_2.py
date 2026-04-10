from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_ita_2(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)

    def _ita2_materialize_strate(value: object) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if text == "0.04174417279198057":
            return float(format(float(text), ".15g"))
        return float(format(float(text), ".16g"))

    def _annual_two_col(path: Path, sheet: str, value_name: str) -> pd.DataFrame:
        df = pd.read_excel(path, sheet_name=sheet, header=None)
        df = df.iloc[1:, [0, 1]].copy()
        df.columns = ["year", value_name]
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        if value_name == "strate":
            df[value_name] = df[value_name].map(_ita2_materialize_strate)
        else:
            df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
        return df.loc[df["year"].notna()].copy()

    bot = _annual_two_col(raw_dir / "country_level" / "ITA_2.xlsx", "Rendimenti BOT (annuali)", "strate")
    ltrate = _annual_two_col(raw_dir / "country_level" / "ITA_2.xlsx", "Rendimenti_ML_term (annuali)", "ltrate")
    master = ltrate.merge(bot, on="year", how="outer")

    cbrate = pd.read_excel(raw_dir / "country_level" / "ITA_3.xlsx", sheet_name="TASSI UFFICIALI", header=None)
    cbrate = cbrate.iloc[3:177, [0, 1]].copy()
    cbrate.columns = ["date", "cbrate"]
    cbrate["year"] = pd.to_numeric(
        pd.to_datetime(cbrate["date"], format="mixed", dayfirst=True, errors="coerce").dt.year,
        errors="coerce",
    ).astype("float64")
    missing_year = cbrate["year"].isna()
    cbrate.loc[missing_year, "year"] = pd.to_numeric(
        cbrate.loc[missing_year, "date"].astype("string").str.extract(r"(\d{4})$", expand=False),
        errors="coerce",
    )
    cbrate["cbrate"] = pd.to_numeric(cbrate["cbrate"], errors="coerce")
    cbrate = cbrate.groupby("year", as_index=False)["cbrate"].mean()
    master = cbrate.merge(master, on="year", how="outer")

    m12 = pd.read_excel(raw_dir / "country_level" / "ITA_4.xlsx", sheet_name="M1_M2", header=None)
    m12 = m12.iloc[5:159, [0, 3, 6, 9]].copy()
    m12.columns = ["year", "M0", "M1", "M2"]
    for col in ["year", "M0", "M1", "M2"]:
        m12[col] = pd.to_numeric(m12[col], errors="coerce")
    early = pd.to_numeric(m12["year"], errors="coerce") <= 1949
    for col in ["M0", "M1", "M2"]:
        m12.loc[early, col] = pd.to_numeric(m12.loc[early, col], errors="coerce") / 1000
    m12 = m12.loc[m12["year"].notna()].copy()
    master = m12.merge(master, on="year", how="outer")

    m3 = pd.read_excel(raw_dir / "country_level" / "ITA_4.xlsx", sheet_name="M3", header=None)
    m3 = m3.iloc[54:70, [0, 8]].copy()
    m3.columns = ["year", "M3"]
    m3["year"] = pd.to_numeric(m3["year"], errors="coerce")
    m3["M3"] = pd.to_numeric(m3["M3"], errors="coerce")
    m3 = m3.loc[m3["year"].notna()].copy()
    master = m3.merge(master, on="year", how="outer")

    master["ISO3"] = "ITA"
    master = master.rename(columns={col: f"CS2_{col}" for col in master.columns if col not in {"ISO3", "year"}})
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    for col in ["CS2_M3", "CS2_M0", "CS2_M1", "CS2_M2", "CS2_cbrate", "CS2_ltrate", "CS2_strate"]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
    master = master[["ISO3", "year", "CS2_M3", "CS2_M0", "CS2_M1", "CS2_M2", "CS2_cbrate", "CS2_ltrate", "CS2_strate"]].copy()
    for col in ["CS2_M3", "CS2_M0", "CS2_M1", "CS2_M2", "CS2_cbrate", "CS2_ltrate", "CS2_strate"]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float64")
    master = _sort_keys(master)
    out_path = clean_dir / "country_level" / "ITA_2.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_ita_2"]
