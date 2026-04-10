from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_tena_pop(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    base_dir = raw_dir / "aggregators" / "Tena" / "pop"

    def _tena_pop_value(value: object) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if text in {"", "<NA>", "nan", "None"}:
            return np.nan
        try:
            return float(format(float(text), ".16g"))
        except (TypeError, ValueError):
            return np.nan

    def _tena_pop_sig_value(value: object, significant_digits: int) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if text in {"", "<NA>", "nan", "None"}:
            return np.nan
        try:
            numeric = Decimal(text)
            if numeric.is_zero():
                return 0.0
            quantum = Decimal(f"1e{numeric.adjusted() - (significant_digits - 1)}")
            return float(numeric.quantize(quantum, rounding=ROUND_HALF_UP))
        except Exception:
            return np.nan

    def _process_sheet(
        filename: str,
        sheet_name: str,
        mapping: dict[str, str],
        drop_head: int,
        drop_tail: int,
        combine_groups: dict[str, list[str]] | None = None,
        parser_overrides: dict[str, dict[int, int]] | None = None,
    ) -> pd.DataFrame:
        df = _read_excel_compat(base_dir / filename, sheet_name=sheet_name, header=None, dtype=str)
        keep_map = {"A": "year", **mapping}
        keep_idx = [_excel_column_to_index(col) for col in keep_map]
        rename_map = {_excel_column_to_index(col): name for col, name in keep_map.items()}
        df = df.iloc[:, keep_idx].rename(columns=rename_map).copy()
        if drop_head:
            df = df.iloc[drop_head:].copy()
        if drop_tail:
            df = df.iloc[:-drop_tail].copy()
        raw_df = df.copy()
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        raw_df["year"] = pd.to_numeric(raw_df["year"], errors="coerce")
        for col in [col for col in df.columns if col != "year"]:
            df[col] = df[col].map(_tena_pop_value)
            if parser_overrides and col in parser_overrides:
                for year_value, significant_digits in parser_overrides[col].items():
                    mask = df["year"].eq(year_value)
                    if mask.any():
                        df.loc[mask, col] = raw_df.loc[mask, col].map(
                            lambda v: _tena_pop_sig_value(v, significant_digits)
                        )
        if combine_groups:
            for target, pieces in combine_groups.items():
                total = sum(pd.to_numeric(df[piece], errors="coerce").fillna(0.0) for piece in pieces)
                total = pd.to_numeric(total, errors="coerce").astype("float32").astype("float64")
                df[target] = total.mask(total.eq(0))
                df = df.drop(columns=pieces)
        value_cols = [col for col in df.columns if col != "year"]
        df = df.rename(columns={col: f"Tena_pop{col}" for col in value_cols})
        long = df.melt(id_vars="year", var_name="ISO3", value_name="Tena_pop")
        long["ISO3"] = long["ISO3"].astype("string").str.removeprefix("Tena_pop")
        return long[["ISO3", "year", "Tena_pop"]].copy()

    parts = [
        _process_sheet(
            "africa_1800-1938_FTWPHD_2023_v01.xlsx",
            "AFRICA",
            {
                "B": "DZA",
                "C": "AGO",
                "D": "LSO",
                "E": "BWA",
                "F": "COD",
                "H": "CPV",
                "I": "CMR",
                "L": "EGY",
                "M": "GNQ",
                "N": "ERI",
                "O": "ETH",
                "Q": "DJI",
                "S": "GMB",
                "T": "TZA_1",
                "U": "NAM",
                "V": "TGO",
                "W": "GHA",
                "X": "GNB",
                "Y": "LBY",
                "AA": "LBR",
                "AB": "MDG",
                "AC": "MWI",
                "AD": "MUS",
                "AE": "MAR",
                "AF": "MOZ",
                "AG": "NGA",
                "AH": "ZMB",
                "AL": "STP",
                "AM": "SYC",
                "AN": "SLE",
                "AO": "ZAF",
                "AP": "ZWE",
                "AQ": "ESH",
                "AR": "SDN",
                "AS": "SWZ",
                "AT": "TUN",
                "AU": "TZA_2",
            },
            4,
            2,
            {"TZA": ["TZA_1", "TZA_2"]},
            {"EGY": {1862: 16}},
        ),
        _process_sheet(
            "america_1800-1938_FTWPHD_2023_v01.xlsx",
            "AMERICA",
            {
                "B": "ARG",
                "C": "BHS",
                "D": "BRB",
                "E": "BMU",
                "F": "BOL",
                "G": "BRA",
                "H": "GUY",
                "I": "BLZ",
                "J": "CAN",
                "K": "CHL",
                "L": "COL",
                "M": "CRI",
                "N": "CUB",
                "O": "VIR",
                "P": "DOM",
                "R": "ECU",
                "S": "SLV",
                "T": "FLK",
                "U": "GUF",
                "V": "GRD",
                "W": "GLP",
                "X": "GTM",
                "Y": "HTI",
                "Z": "HND",
                "AA": "JAM",
                "AC": "MTQ",
                "AD": "MEX",
                "AF": "NIC",
                "AG": "PAN",
                "AH": "PRY",
                "AI": "PER",
                "AJ": "PRI",
                "AK": "BLM",
                "AL": "LCA",
                "AM": "SPM",
                "AN": "VCT",
                "AO": "SUR",
                "AP": "TTO",
                "AQ": "TCA",
                "AR": "USA",
                "AS": "URY",
                "AT": "VEN",
            },
            2,
            2,
            None,
            {"FLK": {1800: 15, 1803: 15, 1804: 15, 1805: 15, 1806: 15, 1807: 15, 1808: 15, 1809: 15, 1810: 15, 1811: 15}},
        ),
        _process_sheet(
            "asia_1800-1938_FTWPHD_2023_v01.xlsx",
            "ASIA",
            {
                "B": "YEM_1",
                "C": "AFG",
                "D": "SAU",
                "E": "BHR",
                "F": "BTN",
                "G": "MYS",
                "I": "BRN",
                "J": "LKA",
                "K": "CHN",
                "L": "IDN",
                "M": "TLS",
                "P": "HKG",
                "Q": "IND",
                "R": "IRQ",
                "S": "JPN",
                "T": "UZB",
                "U": "KOR",
                "V": "KWT",
                "W": "MAC",
                "Y": "MNG",
                "Z": "NPL",
                "AA": "YEM_2",
                "AB": "OMN",
                "AC": "TUR",
                "AD": "PSE",
                "AE": "IRN",
                "AF": "PHL",
                "AH": "QAT",
                "AJ": "THA",
                "AL": "ARE",
            },
            2,
            2,
            {"YEM": ["YEM_1", "YEM_2"]},
        ),
        _process_sheet(
            "europe_1800-1938_FTWPHD_2023_v01.xlsx",
            "EUROPA",
            {
                "B": "ALB",
                "C": "AND",
                "D": "AUT",
                "F": "BEL",
                "G": "BGR",
                "H": "GRC_1",
                "I": "CYP",
                "K": "DNK",
                "L": "GRC_2",
                "M": "EST",
                "N": "FIN",
                "O": "FRA",
                "P": "DEU",
                "Q": "GIB",
                "R": "GRC_3",
                "S": "HUN",
                "T": "ISL",
                "U": "GRC_4",
                "V": "IRL",
                "W": "ITA",
                "X": "LVA",
                "Y": "LTU",
                "Z": "LUX",
                "AA": "MLT",
                "AB": "MCO",
                "AC": "MNE",
                "AD": "NLD",
                "AE": "NOR",
                "AG": "POL",
                "AH": "PRT",
                "AI": "ROU",
                "AJ": "RUS",
                "AL": "ESP",
                "AM": "SWE",
                "AN": "CHE",
                "AO": "GBR",
            },
            2,
            2,
            {"GRC": ["GRC_1", "GRC_2", "GRC_3", "GRC_4"]},
            {"DEU": {1848: 16}},
        ),
        _process_sheet(
            "oceania_1800-1938_FTWPHD_2023_v01.xlsx",
            "OCEANIA",
            {
                "B": "AUS",
                "C": "Hawaii",
                "D": "NZL",
                "F": "FSM",
            },
            2,
            2,
            None,
            {"NZL": {1915: 16}},
        ),
    ]

    out = pd.concat(parts, ignore_index=True)
    out.loc[out["ISO3"].eq("Hawaii"), "ISO3"] = "USA"
    out["Tena_pop"] = pd.to_numeric(out["Tena_pop"], errors="coerce")
    out["total_pop"] = _group_sum_float(
        out,
        group_cols=["ISO3", "year"],
        value_col="Tena_pop",
    )
    out.loc[out["ISO3"].eq("USA"), "Tena_pop"] = out.loc[out["ISO3"].eq("USA"), "total_pop"]
    out = out.drop(columns=["total_pop"]).sort_values(["ISO3", "year"]).reset_index(drop=True)
    out["count"] = out.groupby(["ISO3", "year"]).cumcount() + 1
    out = out.loc[~(out["ISO3"].eq("USA") & out["count"].gt(1))].drop(columns=["count"])
    out["Tena_pop"] = pd.to_numeric(out["Tena_pop"], errors="coerce") / 1000
    out.loc[out["ISO3"].eq("BLM") & out["Tena_pop"].eq(0), "Tena_pop"] = np.nan
    out = out[["ISO3", "year", "Tena_pop"]].copy()
    out["ISO3"] = out["ISO3"].astype("object")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int16")
    out["Tena_pop"] = pd.to_numeric(out["Tena_pop"], errors="coerce").astype("float64")
    out["ISO3"] = out["ISO3"].astype("object")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int16")
    out["Tena_pop"] = pd.to_numeric(out["Tena_pop"], errors="coerce").astype("float64")
    out = _sort_keys(out)
    out_path = clean_dir / "aggregators" / "Tena" / "pop" / "Tena_pop.dta"
    _save_dta(out, out_path)
    return out
__all__ = ["clean_tena_pop"]
