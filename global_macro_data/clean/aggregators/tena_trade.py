from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_tena_trade(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    from .tena_usdfx import clean_tena_usdfx

    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    tena_dir = raw_dir / "aggregators" / "Tena" / "trade"

    def _num(series: pd.Series) -> pd.Series:
        text = series.astype("string").str.replace("\n", " ", regex=False).str.strip()
        text = text.str.replace(",", "", regex=False).replace(
            {
                "": pd.NA,
                "<NA>": pd.NA,
                "...": pd.NA,
                "бк": pd.NA,
                "бн": pd.NA,
            }
        )
        return pd.to_numeric(text, errors="coerce")

    def _norm_name(value: object) -> str:
        return str(value).replace("\n", " ").strip() if pd.notna(value) else ""

    def _aggregate_components(
        df: pd.DataFrame,
        *,
        value_name: str,
        target: str,
        sources: list[str],
        zero_to_missing: bool,
    ) -> pd.DataFrame:
        subset = df.loc[df["ISO3"].isin(sources), ["year", value_name]].copy()
        if subset.empty:
            return df
        # The reference `destring` path defaults to float storage, so downstream `gen a+b+c`
        # inherits float32 rounding rather than Python's float64 summation.
        grouped = (
            subset.assign(
                _value=pd.to_numeric(subset[value_name], errors="coerce")
                .astype("float32")
                .fillna(np.float32(0))
            )
            .groupby("year", as_index=False)["_value"]
            .sum()
            .rename(columns={"_value": value_name})
        )
        grouped[value_name] = pd.to_numeric(grouped[value_name], errors="coerce").astype("float32").astype("float64")
        grouped["ISO3"] = target
        if zero_to_missing:
            grouped.loc[pd.to_numeric(grouped[value_name], errors="coerce").eq(0), value_name] = np.nan
        grouped = grouped[["ISO3", "year", value_name]]
        base = df.loc[~df["ISO3"].isin(sources)].copy()
        return pd.concat([base, grouped], ignore_index=True, sort=False)

    def _parse_dual_file(
        path: Path,
        *,
        name_row: int,
        split_col: int,
        mapping: dict[str, str],
        imports_mapping: dict[str, str] | None = None,
        exports_mapping: dict[str, str] | None = None,
        direct_zero_missing: set[str] | None = None,
        aggregate_rules: list[tuple[str, list[str], bool]] | None = None,
    ) -> pd.DataFrame:
        sheet = _read_excel_compat(path, sheet_name="Current prices, current borders", header=None, dtype=str)
        direct_zero_missing = direct_zero_missing or set()
        aggregate_rules = aggregate_rules or []
        imports_mapping = imports_mapping or mapping
        exports_mapping = exports_mapping or mapping

        def _block(start: int, end: int, value_name: str, mapping_use: dict[str, str]) -> pd.DataFrame:
            names = [_norm_name(value) for value in sheet.iloc[name_row, start:end].tolist()]
            frame = sheet.iloc[6:, [0] + list(range(start, end))].copy()
            frame.columns = ["year"] + names
            keep_cols = ["year"] + [name for name in names if name in mapping_use]
            frame = frame[keep_cols].copy()
            frame["year"] = _num(frame["year"])
            for col in keep_cols[1:]:
                frame[col] = _num(frame[col])
            if keep_cols[1:]:
                frame = frame.loc[frame[keep_cols[1:]].notna().any(axis=1)].copy()
            long = frame.melt(id_vars=["year"], var_name="countryname", value_name=value_name)
            long["ISO3"] = long["countryname"].map(mapping_use)
            long = long.loc[long["ISO3"].notna() & long["year"].notna(), ["ISO3", "year", value_name]].copy()
            for iso3 in direct_zero_missing:
                mask = long["ISO3"].eq(iso3) & pd.to_numeric(long[value_name], errors="coerce").eq(0)
                long.loc[mask, value_name] = np.nan
            for target, sources, zero_to_missing in aggregate_rules:
                long = _aggregate_components(long, value_name=value_name, target=target, sources=sources, zero_to_missing=zero_to_missing)
            return long

        imports = _block(1, split_col, "Tena_imports_USD", imports_mapping)
        exports = _block(split_col, sheet.shape[1], "Tena_exports_USD", exports_mapping)
        out = imports.merge(exports, on=["ISO3", "year"], how="outer")
        return _sort_keys(out)

    africa_map = {
        "Algeria": "DZA",
        "Angola (Portuguess Africa)": "AGO",
        "Belgium Congo (Zaire)": "COD",
        "British East Africa (Kenia & Uganda)": "KEN",
        "British Somaliland": "SOM_1",
        "Cabo Verde (Portuguese Africa)": "CPV",
        "Camerun": "CMR",
        "Egypt": "EGY",
        "Eritrea": "ERI",
        "Ethiopia": "ETH",
        "French Somalia": "DJI",
        "Gambia": "GMB",
        "German South West Africa": "NAM",
        "Ghana-Gold Coast": "GHA",
        "Guinea Bissau (Portuguese Africa)": "GNB",
        "Italia Somalia": "SOM_2",
        "Italian Libia Cyrenaica": "LBY",
        "Liberia": "LBR",
        "Madagascar": "MDG",
        "Malawi": "MWI",
        "Marocco": "MAR",
        "Mauritius": "MUS",
        "Mozambique (Portuguese Africa)": "MOZ",
        "Nigeria": "NGA",
        "Rodhesia": "ZWE",
        "Rwanda and Burundi": "RWA",
        "S.Tome e Principe (Portuguess Africa)": "STP",
        "Seychelles": "SYC",
        "Sierra Leone": "SLE",
        "South Africa": "ZAF",
        "Sudan (Anglo-Egyptian Sudan)": "SDN",
        "Tanganica (German East Africa)": "TZA_1",
        "Togo (German West Africa)": "TGO",
        "Tunisia": "TUN",
        "Zanzibar Isl.": "TZA_2",
    }
    europe_map = {
        "Albania": "ALB",
        "Austria": "AUT",
        "Belgium": "BEL",
        "Bulgaria": "BGR",
        "Crete": "GRC_1",
        "Cyprus": "CYP",
        "Czechoslowakia": "CSK",
        "Denmark": "DNK",
        "Estonia": "EST",
        "Finland": "FIN",
        "France": "FRA",
        "Germany/Zollverein": "DEU",
        "Hungary": "HUN",
        "Iceland": "ISL",
        "Ionian islands": "GRC_2",
        "Ireland": "IRL",
        "Italy": "ITA",
        "Latvia": "LVA",
        "Lithuania": "LTU",
        "Netherlands": "NLD",
        "Norway": "NOR",
        "Poland": "POL",
        "Portugal": "PRT",
        "Romania": "ROU",
        "Russia/USSR": "RUS",
        "Serbia/Yugoslavia": "YUG",
        "Spain": "ESP",
        "Sweden": "SWE",
        "Switzerland": "CHE",
        "United Kingdom": "GBR",
    }
    asia_map = {
        "Afghanistan": "AFG",
        "British Malaya": "MYS_1",
        "Brunei": "BRN",
        "Ceylon (Sri Lanka)": "LKA",
        "China": "CHN",
        "Dutch East Indies (Indonesia)": "IDN",
        "Formosa (Taiwan)": "TWN",
        "French India": "IND_2",
        "French Indochina": "VNM",
        "India": "IND_1",
        "Iraq": "IRQ",
        "Japan": "JPN",
        "Korea": "KOR",
        "Nepal": "NPL",
        "North Yemen": "YEM",
        "Ottoman Empire/Turkey": "TUR",
        "Palestine": "PSE",
        "Persia (Iran)": "IRN",
        "Philippines": "PHL",
        "Portuguese India": "IND_3",
        "Sabah (British Borneo)": "MYS_3",
        "Sarawak": "MYS_2",
        "Saudi Arabia": "SAU",
        "Siam (Thailand)": "THA",
    }
    america_map = {
        "Argentina": "ARG",
        "Bahamas": "BHS",
        "Barbados": "BRB",
        "Bermuda": "BMU",
        "Bolivia": "BOL",
        "Brasil": "BRA",
        "British Guiana": "GUY",
        "British Honduras (Belize)": "BLZ",
        "Canada": "CAN",
        "Chile": "CHL",
        "Colombia": "COL",
        "Costa Rica": "CRI",
        "Cuba": "CUB",
        "Danish Virgin Island": "VIR",
        "Dominican Republic": "DOM",
        "Ecuador": "ECU",
        "El Salvador": "SLV",
        "Falkland Islands": "FLK",
        "French Guiana (French Colonies)": "GUF",
        "Granada (Winward Island)": "GRD",
        "Guadalupe (French Colonies)": "GLP",
        "Guatemala": "GTM",
        "Haiti": "HTI",
        "Honduras": "HND",
        "Jamaica": "JAM",
        "Martinique (French Colonies)": "MTQ",
        "Mexico": "MEX",
        "Nicaragua": "NIC",
        "Panama": "PAN",
        "Paraguay": "PRY",
        "Peru": "PER",
        "Puerto Rico": "PRI",
        "St. Barthelemy (Norvegian Colonies)": "BLM",
        "St.Lucia (Winward Island)": "LCA",
        "St.Pierre e Michelon": "SPM",
        "St. Vicente (Winward Island)": "VCT",
        "Surinam (Duch Guayana)": "SUR",
        "Trinidad & Tobago (Winward Island)": "TTO",
        "Turk Island": "TCA",
        "United states": "USA",
        "Uruguay": "URY",
        "Venezuela": "VEN",
    }
    america_imports_map = {k: v for k, v in america_map.items() if k != "Danish Virgin Island"}
    america_imports_map["New Foundland"] = "VIR"

    africa = _parse_dual_file(
        tena_dir / "africa_1817_1938.xlsx",
        name_row=5,
        split_col=50,
        mapping=africa_map,
        direct_zero_missing={"COD", "TGO"},
        aggregate_rules=[("TZA", ["TZA_1", "TZA_2"], True), ("SOM", ["SOM_1", "SOM_2"], True)],
    )
    europe = _parse_dual_file(
        tena_dir / "europe_1800_1938.xlsx",
        name_row=3,
        split_col=43,
        mapping=europe_map,
        aggregate_rules=[("GRC", ["GRC_1", "GRC_2"], False)],
    )
    if "Tena_exports_USD" in europe.columns:
        export_zero = europe["ISO3"].eq("GRC") & pd.to_numeric(europe["Tena_exports_USD"], errors="coerce").eq(0)
        europe.loc[export_zero, "Tena_exports_USD"] = np.nan
    asia = _parse_dual_file(
        tena_dir / "asia_1800_1938.xlsx",
        name_row=3,
        split_col=57,
        mapping=asia_map,
        aggregate_rules=[("MYS", ["MYS_1", "MYS_2", "MYS_3"], True), ("IND", ["IND_1", "IND_2", "IND_3"], False)],
    )
    america = _parse_dual_file(
        tena_dir / "america_1800_1938.xlsx",
        name_row=2,
        split_col=48,
        mapping=america_map,
        imports_mapping=america_imports_map,
        exports_mapping=america_map,
        direct_zero_missing={"PRI"},
    )

    def _single_country_trade(path: Path, iso3: str) -> pd.DataFrame:
        sheet = _read_excel_compat(path, sheet_name="Current prices", header=None, dtype=str)
        frame = sheet.iloc[6:, [0, 1, 2]].copy()
        frame.columns = ["year", "Tena_imports_USD", "Tena_exports_USD"]
        frame["year"] = _num(frame["year"])
        frame["Tena_imports_USD"] = _num(frame["Tena_imports_USD"])
        frame["Tena_exports_USD"] = _num(frame["Tena_exports_USD"])
        frame["ISO3"] = iso3
        frame = frame.loc[frame["year"].notna(), ["ISO3", "year", "Tena_imports_USD", "Tena_exports_USD"]].copy()
        return _sort_keys(frame)

    australia = _single_country_trade(tena_dir / "australia_1826_1938.xlsx", "AUS")
    new_zealand = _single_country_trade(tena_dir / "new_zealand_1826_1938.xlsx", "NZL")

    master = pd.concat([africa, europe, asia, america, australia, new_zealand], ignore_index=True, sort=False)
    master = _sort_keys(master)

    tena_fx = clean_tena_usdfx(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    master = master.merge(tena_fx, on=["ISO3", "year"], how="outer")
    master["Tena_imports"] = pd.to_numeric(master["Tena_imports_USD"], errors="coerce") * pd.to_numeric(master["Tena_USDfx"], errors="coerce")
    master["Tena_exports"] = pd.to_numeric(master["Tena_exports_USD"], errors="coerce") * pd.to_numeric(master["Tena_USDfx"], errors="coerce")
    master = master.drop(columns=["Tena_USDfx"], errors="ignore")

    master["ISO3"] = master["ISO3"].astype(str)
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    master["Tena_imports_USD"] = pd.to_numeric(master["Tena_imports_USD"], errors="coerce").astype("float64")
    master["Tena_exports_USD"] = pd.to_numeric(master["Tena_exports_USD"], errors="coerce").astype("float64")
    master["Tena_imports"] = pd.to_numeric(master["Tena_imports"], errors="coerce").astype("float32")
    master["Tena_exports"] = pd.to_numeric(master["Tena_exports"], errors="coerce").astype("float32")
    master = master[["ISO3", "year", "Tena_imports_USD", "Tena_exports_USD", "Tena_imports", "Tena_exports"]].copy()
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    master["Tena_imports_USD"] = pd.to_numeric(master["Tena_imports_USD"], errors="coerce").astype("float64")
    master["Tena_exports_USD"] = pd.to_numeric(master["Tena_exports_USD"], errors="coerce").astype("float64")
    master["Tena_imports"] = pd.to_numeric(master["Tena_imports"], errors="coerce").astype("float32")
    master["Tena_exports"] = pd.to_numeric(master["Tena_exports"], errors="coerce").astype("float32")
    master = _sort_keys(master)

    out_path = clean_dir / "aggregators" / "Tena" / "trade" / "Tena_trade.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_tena_trade"]
