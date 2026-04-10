from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_barro(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    from .wdi import clean_wdi

    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    clean_wdi(data_raw_dir=raw_dir, data_clean_dir=clean_dir, data_helper_dir=helper_dir)

    lookup = _country_name_lookup(helper_dir)
    name_fix = {
        "Korea": "South Korea",
        "NewZealand": "New Zealand",
        "Russia": "Russian Federation",
        "S. Africa": "South Africa",
        "SAfrica": "South Africa",
        "SriLanka": "Sri Lanka",
        "UnitedKingdom": "United Kingdom",
        "UnitedStates": "United States",
    }

    def parse_sheet(sheet: str, value_name: str) -> pd.DataFrame:
        df = _read_excel_compat(raw_dir / "aggregators" / "BARRO" / "GDP_Barro-Ursua.xls", sheet_name=sheet)
        df = df.iloc[1:].copy()
        year_col = df.columns[0]
        df = df.rename(columns={year_col: "year"})
        for col in df.columns:
            if col == "year":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        long = df.melt(id_vars=["year"], var_name="countryname", value_name=value_name)
        long["countryname"] = long["countryname"].astype(str).replace(name_fix)
        long["ISO3"] = long["countryname"].map(lookup)
        long = long.loc[long["ISO3"].notna(), ["ISO3", "year", value_name]].copy()
        long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("int16")
        return long

    gdp = parse_sheet("GDP", "BARRO_rGDP_pc_index")
    cons = parse_sheet("C", "BARRO_rcons_pc_index")
    master = cons.merge(gdp, on=["ISO3", "year"], how="left")

    wdi = _load_dta(clean_dir / "aggregators" / "WB" / "WDI.dta")[["ISO3", "year", "WDI_rGDP_pc", "WDI_rcons"]].copy()
    master = master.merge(wdi[["ISO3", "year", "WDI_rGDP_pc"]], on=["ISO3", "year"], how="left")
    master = master.rename(columns={"BARRO_rGDP_pc_index": "BARRO_rGDP_pc"})
    spliced = sh.splice(master, priority="WDI BARRO", generate="rGDP_pc", varname="rGDP_pc", method="chainlink", base_year=2006, save="NO")
    master = spliced[["ISO3", "year", "BARRO_rcons_pc_index", "rGDP_pc"]].rename(columns={"rGDP_pc": "BARRO_rGDP_pc"})
    master = master.merge(wdi[["ISO3", "year", "WDI_rcons"]], on=["ISO3", "year"], how="left")
    master = master.rename(columns={"BARRO_rcons_pc_index": "BARRO_rcons"})
    spliced = sh.splice(master, priority="WDI BARRO", generate="rcons", varname="rcons", method="chainlink", base_year=2006, save="NO")
    out = spliced[["ISO3", "year", "BARRO_rGDP_pc", "rcons"]].rename(columns={"rcons": "BARRO_rcons"}).copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int16")
    out["BARRO_rGDP_pc"] = pd.to_numeric(out["BARRO_rGDP_pc"], errors="coerce").astype("float32")
    out["BARRO_rcons"] = pd.to_numeric(out["BARRO_rcons"], errors="coerce").astype("float32")
    out = _sort_keys(out[["ISO3", "year", "BARRO_rGDP_pc", "BARRO_rcons"]])
    out_path = clean_dir / "aggregators" / "BARRO" / "BARRO.dta"
    _save_dta(out, out_path)
    return out
__all__ = ["clean_barro"]
