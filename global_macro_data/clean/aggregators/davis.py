from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_davis(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    raw_base = raw_dir / "aggregators" / "Davis"

    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].drop_duplicates().copy()

    def _load(sheet_file: str) -> pd.DataFrame:
        return _read_excel_compat(raw_base / sheet_file, sheet_name="nominal GDP", header=None, dtype=str)

    def _melt(frame: pd.DataFrame) -> pd.DataFrame:
        long = frame.melt(id_vars=["year"], var_name="variable", value_name="Davis_nGDP")
        long["ISO3"] = long["variable"].astype("string").str.replace("Davis_nGDP", "", regex=False)
        long = long[["ISO3", "year", "Davis_nGDP"]].copy()
        long["year"] = pd.to_numeric(long["year"], errors="coerce")
        long["Davis_nGDP"] = pd.to_numeric(long["Davis_nGDP"], errors="coerce")
        return long

    def _convert_eur(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.merge(eur_fx, on="ISO3", how="left")
        mask = out["EUR_irrevocable_FX"].notna()
        out.loc[mask, "Davis_nGDP"] = (
            pd.to_numeric(out.loc[mask, "Davis_nGDP"], errors="coerce")
            / pd.to_numeric(out.loc[mask, "EUR_irrevocable_FX"], errors="coerce")
        )
        return out.drop(columns=["EUR_irrevocable_FX"], errors="ignore")

    africa = _load("Nominal_GDP_Africa.xlsx").iloc[3:].reset_index(drop=True)
    africa = africa.iloc[:, [c for c in africa.columns if str(africa.iloc[1, c]) != "World Bank"]].copy()
    africa_drop = {_excel_column_to_index(col) for col in ["V", "AT", "BL", "CG", "CH", "CL", "CN"]}
    africa = africa[[c for c in africa.columns if c not in africa_drop]].copy()
    africa = africa.iloc[3:].reset_index(drop=True)
    africa_map = {
        "Year": "year",
        "Algeria": "DZA",
        "Benin": "BEN",
        "Burkina Faso": "BFA",
        "Burundi": "BDI",
        "Cameroon": "CMR",
        "Central African Rep.": "CAF",
        "Chad": "TCD",
        "Congo, Dem. Rep.": "COD",
        "Egypt": "EGY",
        "Ethiopia": "ETH",
        "Gabon": "GAB",
        "Ghana": "GHA",
        "Ivory Coast": "CIV",
        "Kenya": "KEN",
        "Lesotho": "LSO",
        "Liberia": "LBR",
        "Libya": "LBY",
        "Madagascar": "MDG",
        "Malawi": "MWI",
        "Mali": "MLI",
        "Mauritania": "MRT",
        "Mauritius": "MUS",
        "Morocco": "MAR",
        "Mozambique": "MOZ",
        "Nigeria": "NGA",
        "Niger": "NER",
        "Rwanda": "RWA",
        "Senegal": "SEN",
        "Sierra Leone": "SLE",
        "South Africa": "ZAF",
        "Sudan": "SDN",
        "Tanzania": "TZA",
        "Togo": "TGO",
        "Tunisia": "TUN",
        "Uganda": "UGA",
        "Zambia": "ZMB",
        "Zimbabwe": "ZWE",
    }
    africa.columns = [
        "year" if africa_map.get(str(africa.iloc[0, i]), str(africa.iloc[0, i])) == "year" else f"Davis_nGDP{africa_map.get(str(africa.iloc[0, i]), str(africa.iloc[0, i]))}"
        for i in range(len(africa.columns))
    ]
    africa = africa.iloc[1:].reset_index(drop=True)
    africa = _melt(africa)
    africa_mul = {"DZA", "BEN", "BFA", "BDI", "CAF", "CMR", "CIV", "COD", "GAB", "KEN", "MDG", "MLI", "MRT", "MAR", "NGA", "NER", "RWA", "SEN", "TZA", "TCD", "TGO"}
    africa.loc[africa["ISO3"].isin(africa_mul), "Davis_nGDP"] = pd.to_numeric(africa.loc[africa["ISO3"].isin(africa_mul), "Davis_nGDP"], errors="coerce") * 1000
    africa.loc[africa["ISO3"].eq("MWI"), "Davis_nGDP"] = pd.to_numeric(africa.loc[africa["ISO3"].eq("MWI"), "Davis_nGDP"], errors="coerce") / 1000
    africa.loc[africa["ISO3"].eq("GHA"), "Davis_nGDP"] = pd.to_numeric(africa.loc[africa["ISO3"].eq("GHA"), "Davis_nGDP"], errors="coerce") / 1000

    west = _load("Nominal_GDP_West_Europe.xlsx").iloc[:, [_excel_column_to_index(col) for col in ["A", "B", "D", "G", "P"]]].copy()
    west = west.iloc[7:].reset_index(drop=True)
    west.columns = ["year", "Davis_nGDPAUT", "Davis_nGDPBEL", "Davis_nGDPDNK", "Davis_nGDPDEU"]
    west = west.iloc[2:].reset_index(drop=True)
    west = _melt(west)
    west = _convert_eur(west)

    east = _load("Nominal_GDP_East_Europe.xlsx").iloc[:, [_excel_column_to_index(col) for col in ["A", "E", "N", "R", "Z"]]].copy()
    east = east.iloc[7:].reset_index(drop=True)
    east.columns = ["year", "Davis_nGDPBGR", "Davis_nGDPGRC", "Davis_nGDPHUN", "Davis_nGDPPOL"]
    east = east.iloc[2:].reset_index(drop=True)
    east = _melt(east)
    east.loc[east["ISO3"].eq("POL"), "Davis_nGDP"] = pd.to_numeric(east.loc[east["ISO3"].eq("POL"), "Davis_nGDP"], errors="coerce") * 1000
    east = _convert_eur(east)

    india = _load("Nominal_GDP_MidEast_SAsia.xlsx").iloc[57:, [_excel_column_to_index(col) for col in ["A", "L"]]].reset_index(drop=True)
    india.columns = ["year", "Davis_nGDP"]
    india["ISO3"] = "IND"
    india["year"] = pd.to_numeric(india["year"], errors="coerce")
    india["Davis_nGDP"] = pd.to_numeric(india["Davis_nGDP"], errors="coerce")
    india = india[["ISO3", "year", "Davis_nGDP"]].copy()

    asia = _load("Nominal_GDP_East_Asia_Oceania.xlsx").iloc[:, [_excel_column_to_index(col) for col in ["A", "I", "AK"]]].copy()
    asia = asia.iloc[8:].reset_index(drop=True)
    asia.columns = ["year", "Davis_nGDPCHN", "Davis_nGDPNZL"]
    asia = _melt(asia)

    americas = _load("Nominal_GDP_Americas.xlsx").iloc[:, [_excel_column_to_index(col) for col in ["A", "B", "K", "R", "S", "W", "AD", "BH", "BW", "CQ"]]].copy()
    americas.columns = ["year", "Davis_nGDPARG", "Davis_nGDPBRA", "Davis_nGDPCAN", "Davis_nGDPCHL", "Davis_nGDPCOL", "Davis_nGDPCRI", "Davis_nGDPMEX", "Davis_nGDPPER", "Davis_nGDPURY"]
    americas = americas.iloc[1:].reset_index(drop=True)
    americas = americas.loc[americas[[c for c in americas.columns if c != "year"]].replace({"": pd.NA}).notna().any(axis=1)].reset_index(drop=True)
    americas = americas.iloc[4:].reset_index(drop=True)
    americas = _melt(americas)

    master = pd.concat([africa, west, east, india, asia, americas], ignore_index=True)
    master["year"] = pd.to_numeric(master["year"], errors="coerce")
    master["Davis_nGDP"] = pd.to_numeric(master["Davis_nGDP"], errors="coerce")

    master.loc[master["ISO3"].eq("MWI"), "Davis_nGDP"] = pd.to_numeric(master.loc[master["ISO3"].eq("MWI"), "Davis_nGDP"], errors="coerce") * 1000
    master.loc[master["ISO3"].eq("ZMB"), "Davis_nGDP"] = pd.to_numeric(master.loc[master["ISO3"].eq("ZMB"), "Davis_nGDP"], errors="coerce") / 1000
    master.loc[master["ISO3"].eq("GHA"), "Davis_nGDP"] = pd.to_numeric(master.loc[master["ISO3"].eq("GHA"), "Davis_nGDP"], errors="coerce") / 10
    master.loc[master["ISO3"].eq("CRI"), "Davis_nGDP"] = pd.to_numeric(master.loc[master["ISO3"].eq("CRI"), "Davis_nGDP"], errors="coerce") / 1000
    master.loc[master["ISO3"].eq("SDN"), "Davis_nGDP"] = pd.to_numeric(master.loc[master["ISO3"].eq("SDN"), "Davis_nGDP"], errors="coerce") / 1000
    master.loc[master["ISO3"].eq("DEU") & master["year"].le(1923), "Davis_nGDP"] = pd.to_numeric(master.loc[master["ISO3"].eq("DEU") & master["year"].le(1923), "Davis_nGDP"], errors="coerce") / (10**12)
    master.loc[master["ISO3"].eq("POL"), "Davis_nGDP"] = pd.to_numeric(master.loc[master["ISO3"].eq("POL"), "Davis_nGDP"], errors="coerce") / (10**4)
    master.loc[master["ISO3"].eq("COD"), "Davis_nGDP"] = np.nan

    master["ISO3"] = master["ISO3"].astype("object")
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    master["Davis_nGDP"] = pd.to_numeric(master["Davis_nGDP"], errors="coerce").astype("float64")
    master = _sort_keys(master[["ISO3", "year", "Davis_nGDP"]].copy())
    out_path = clean_dir / "aggregators" / "Davis" / "Davis.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_davis"]
