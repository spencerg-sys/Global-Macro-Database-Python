from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_lv(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    df = _read_excel_compat(raw_dir / "aggregators" / "LV" / "41308_2020_107_MOESM1_ESM.xlsx", sheet_name="Crisis Years")
    df = df.iloc[1:].copy()
    df = df.rename(
        columns={
            "Country": "country_name",
            "Systemic Banking Crisis (starting date)": "LV_crisisByears",
            "Currency Crisis": "LV_crisisCyears",
            "Sovereign Debt Crisis (year)": "LV_crisisSD1years",
            "Sovereign Debt Restructuring (year)": "LV_crisisSD2years",
        }
    )
    df["country_name"] = df["country_name"].astype(str).str.strip()
    lookup = _country_name_lookup(helper_dir)
    manual_iso = {
        "Central African Rep.": "CAF",
        "China, P.R.": "CHN",
        "China, P.R.: Hong Kong": "HKG",
        "Congo, Dem. Rep. of": "COD",
        "Congo, Rep. of": "COG",
        "Côte d’Ivoire": "CIV",
        "Gambia, The": "GMB",
        "Iran, I.R. of": "IRN",
        "Korea": "KOR",
        "Kyrgyz Republic": "KGZ",
        "Lao People's Dem. Rep.": "LAO",
        "Russia": "RUS",
        "Serbia, Republic of": "SRB",
        "Slovak Republic": "SVK",
        "St. Kitts and Nevis": "KNA",
        "Swaziland": "SWZ",
        "Syrian Arab Republic": "SYR",
        "São Tomé and Principe": "STP",
        "Yugoslavia, SFR": "",
    }
    df["ISO3"] = df["country_name"].map(manual_iso)
    missing = df["ISO3"].isna()
    df.loc[missing, "ISO3"] = df.loc[missing, "country_name"].map(lookup)
    df = df.loc[df["ISO3"].fillna("").astype(str) != ""].copy()

    def parse_years(value: object) -> list[int]:
        if pd.isna(value):
            return []
        text = str(value).strip()
        if not text or text.lower() == "n.a.":
            return []
        years: list[int] = []
        for token in text.split(","):
            token = token.strip()
            if not token:
                continue
            parsed = pd.to_numeric(token, errors="coerce")
            if pd.isna(parsed):
                continue
            years.append(int(parsed))
        return years

    year_cols = ["LV_crisisByears", "LV_crisisCyears", "LV_crisisSD1years", "LV_crisisSD2years"]
    for col in year_cols:
        df[col] = df[col].map(parse_years)
    active = df[year_cols].map(len).sum(axis=1) > 0
    df = df.loc[active].copy()

    countries = sorted(df["ISO3"].dropna().astype(str).unique())
    panel = pd.MultiIndex.from_product([countries, range(1970, 2018)], names=["ISO3", "year"]).to_frame(index=False)
    panel["year"] = panel["year"].astype("float32")
    for col in ["LV_crisisB", "LV_crisisC", "LV_crisisSD1", "LV_crisisSD2"]:
        panel[col] = np.float32(0)

    rename_lists = {
        "LV_crisisByears": "LV_crisisB",
        "LV_crisisCyears": "LV_crisisC",
        "LV_crisisSD1years": "LV_crisisSD1",
        "LV_crisisSD2years": "LV_crisisSD2",
    }
    for _, row in df.iterrows():
        iso3 = str(row["ISO3"])
        for src_col, out_col in rename_lists.items():
            years = row[src_col]
            if not isinstance(years, list):
                continue
            if years:
                mask = (panel["ISO3"] == iso3) & panel["year"].isin([float(y) for y in years])
                panel.loc[mask, out_col] = np.float32(1)

    panel = _sort_keys(panel[["ISO3", "year", "LV_crisisB", "LV_crisisC", "LV_crisisSD1", "LV_crisisSD2"]])
    out_path = clean_dir / "aggregators" / "LV" / "LV.dta"
    _save_dta(panel, out_path)
    return panel
__all__ = ["clean_lv"]
