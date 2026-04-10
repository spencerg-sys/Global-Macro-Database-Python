from __future__ import annotations

from datetime import date
from functools import wraps
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .. import helpers as sh


REPO_ROOT = Path(__file__).resolve().parents[2]
DBNOMICS_SERIES_API = "https://api.db.nomics.world/v22/series"
WORLD_BANK_API = "https://api.worldbank.org/v2"
REQUEST_TIMEOUT = 120
HTTP = requests.Session()

WDI_INDICATORS = [
    "NY.GDP.MKTP.CD",
    "NY.GDP.MKTP.CN",
    "NY.GDP.MKTP.KD",
    "NY.GDP.MKTP.KN",
    "NY.GDP.PCAP.KN",
    "FP.CPI.TOTL",
    "FP.CPI.TOTL.ZG",
    "SP.POP.TOTL",
    "NE.GDI.TOTL.CN",
    "NE.GDI.FTOT.CN",
    "NY.GNS.ICTR.CN",
    "NE.CON.TOTL.CN",
    "NE.CON.TOTL.KN",
    "NE.EXP.GNFS.CN",
    "NE.EXP.GNFS.CD",
    "NE.IMP.GNFS.CN",
    "NE.IMP.GNFS.CD",
    "PA.NUS.FCRF",
    "PX.REX.REER",
    "BN.CAB.XOKA.GD.ZS",
    "GC.TAX.TOTL.CN",
    "GC.DOD.TOTL.CN",
    "GC.XPN.TOTL.CN",
    "GC.TAX.TOTL.GD.ZS",
    "GC.REV.XGRT.GD.ZS",
    "FM.LBL.BMNY.CN",
]

IMF_INITIAL_SUBJECTS = ["NGDP"]
IMF_SUBJECTS = [
    "NGDP_R",
    "NGDPRPC",
    "NID_NGDP",
    "PCPI",
    "TM_RPCH",
    "TX_RPCH",
    "LUR",
    "LP",
    "GGR_NGDP",
    "GGX_NGDP",
    "GGXCNL_NGDP",
    "GGSB_NPGDP",
    "GGXWDG_NGDP",
    "BCA_NGDPD",
]

OECD_MEI_SUBJECT_SPECS = [
    {"SUBJECT": ["MABMM301"], "MEASURE": ["STSA"], "FREQUENCY": ["A"]},
    {"SUBJECT": ["MANMM101"], "MEASURE": ["STSA"], "FREQUENCY": ["A"]},
    {"SUBJECT": ["CCRETT01"], "FREQUENCY": ["A"]},
    {"SUBJECT": ["IRLTLT01"], "FREQUENCY": ["A"]},
    {"SUBJECT": ["IRSTCB01"], "FREQUENCY": ["A"]},
]

OECD_EO_VARIABLES = [
    "GDP",
    "GDPV",
    "CPIH",
    "POP",
    "IT",
    "CP",
    "CG",
    "XGS",
    "XGSD",
    "MGS",
    "MGSD",
    "EXCH",
    "EXCHER",
    "CBGDPR",
    "GGFLQ",
    "IRS",
    "IRCB",
    "UNR",
    "CPIH_YTYPCT",
    "ITISK",
]

OECD_NAAG_INDICATORS = ["TES13S", "TRS13S", "D2D5D91RS13S", "B9S13S"]

OECD_KEI_SUBJECT_SPECS = [
    {"FREQUENCY": ["A"], "SUBJECT": ["CPALTT01"]},
    {"FREQUENCY": ["A"], "SUBJECT": ["IR3TIB01"]},
    {"FREQUENCY": ["A"], "SUBJECT": ["IRLTLT01"]},
    {"FREQUENCY": ["A"], "SUBJECT": ["B6BLTT02"]},
    {"FREQUENCY": ["A"], "SUBJECT": ["LRHUTTTT"]},
]

OECD_QNA_SUBJECT_SPECS = [
    {"FREQUENCY": ["A"], "SUBJECT": ["P5"], "MEASURE": ["CQR"]},
    {"FREQUENCY": ["A"], "SUBJECT": ["P51"], "MEASURE": ["CQR"]},
    {"FREQUENCY": ["A"], "SUBJECT": ["P3"], "MEASURE": ["CQR"]},
    {"FREQUENCY": ["A"], "SUBJECT": ["B1_GS1"], "MEASURE": ["CQR"]},
]

EUS_DOWNLOAD_SPECS = [
    ("ei_cphi_m", {"unit": ["HICP2015"], "indic": ["CP-HI00"]}),
    ("ei_cphi_m", {"unit": ["RT12"], "indic": ["CP-HI00"]}),
    ("ei_hppi_q", {"unit": ["I15_NSA"]}),
    ("nama_10_gdp", {"na_item": ["B1GQ"], "unit": ["CP_MNAC"]}),
    ("nama_10_gdp", {"na_item": ["B1GQ"], "unit": ["CLV10_MNAC"]}),
    ("nama_10_gdp", {"na_item": ["P3"], "unit": ["CP_MNAC"]}),
    ("nama_10_gdp", {"na_item": ["P5G"], "unit": ["CP_MNAC"]}),
    ("nama_10_gdp", {"na_item": ["P51G"], "unit": ["CP_MNAC"]}),
    ("nama_10_gdp", {"na_item": ["P7"], "unit": ["CP_MNAC"]}),
    ("nama_10_gdp", {"na_item": ["P6"], "unit": ["CP_MNAC"]}),
    ("gov_10a_main", {"na_item": ["TR"], "unit": ["MIO_NAC"], "sector": ["S1311"]}),
    ("gov_10a_main", {"na_item": ["TE"], "unit": ["MIO_NAC"], "sector": ["S1311"]}),
    ("gov_10a_taxag", {"na_item": ["D2_D5_D91_D61_M_D995"], "unit": ["MIO_NAC"], "sector": ["S1311"]}),
    ("gov_10a_main", {"na_item": ["B9"], "unit": ["PC_GDP"], "sector": ["S1311"]}),
    ("demo_gind", {"indic_de": ["JAN"]}),
    ("ei_mfir_m", {"indic": ["MF-3MI-RT"]}),
    ("tipsun20", {"age": ["Y15-74"]}),
    ("tipser13", None),
]

AMECO_DATASETS = [
    ("XUNRQ-1", None),
    ("ISN", {"unit": ["-"]}),
    ("ILN", {"unit": ["-"]}),
    ("UXGS", None),
    ("UMGS", None),
    ("UCNT", None),
    ("UIGT", None),
    ("UITT", None),
    ("UVGD", None),
    ("OVGD", None),
    ("NPTD", None),
    ("NUTN", None),
    ("ZCPIN", None),
]

IMF_MFS_INDICATORS = [
    "14____XDC",
    "FMA_XDC",
    "34____XDC",
    "FMBCC_XDC",
    "35L___XDC",
    "FMB_XDC",
    "FITB_PA",
    "FPOLM_PA",
    "FIGB_PA",
]

IMF_GFS_SPECS = [
    {"CLASSIFICATION": ["GNLB__Z"], "REF_SECTOR": ["S1311B"], "UNIT_MEASURE": ["XDC_R_B1GQ"]},
    {"CLASSIFICATION": ["G11__Z"], "REF_SECTOR": ["S1311B"], "UNIT_MEASURE": ["XDC"]},
    {"CLASSIFICATION": ["G2M__Z"], "REF_SECTOR": ["S1311B"], "UNIT_MEASURE": ["XDC"]},
    {"CLASSIFICATION": ["G1__Z"], "REF_SECTOR": ["S1311B"], "UNIT_MEASURE": ["XDC"]},
    {"CLASSIFICATION": ["GNLB__Z"], "REF_SECTOR": ["S1311"], "UNIT_MEASURE": ["XDC_R_B1GQ"]},
    {"CLASSIFICATION": ["G11__Z"], "REF_SECTOR": ["S1311"], "UNIT_MEASURE": ["XDC"]},
    {"CLASSIFICATION": ["G2M__Z"], "REF_SECTOR": ["S1311"], "UNIT_MEASURE": ["XDC"]},
    {"CLASSIFICATION": ["G1__Z"], "REF_SECTOR": ["S1311"], "UNIT_MEASURE": ["XDC"]},
]

IMF_IFS_INDICATORS = [
    "NGDP_R_XDC",
    "NC_R_XDC",
    "NGDP_XDC",
    "ENDA_XDC_USD_RATE",
    "EREER_IX",
    "FPOLM_PA",
    "BCAXF_BP6_USD",
    "LP_PE_NUM",
    "LUR_PT",
    "BGS_BP6_USD",
    "NFI_XDC",
    "NI_XDC",
    "NM_XDC",
    "NX_XDC",
    "NC_XDC",
    "PCPI_IX",
    "PCPI_PC_CP_A_PT",
    "EDNE_USD_XDC_RATE",
]

AFDB_INDICATORS = [
    "FM.LBL.MONY.CN",
    "NY.GDP.MKTP.CN",
    "FM.LBL.MQMY.CN",
    "GC.REV.TAX.GD.CN",
    "GC.REV.TOTL.GD.ZS",
    "GC.REV.TOTL.GD.CN",
    "GC.XPN.TOTL.GD.ZS",
    "GC.XPN.TOTL.GD.CN",
    "GC.BAL.CASH.GD.ZS",
    "GC.BAL.CASH.GD.CN",
    "SL.TLF.15UP.UEM",
    "LM.POP.EPP.TOT",
]

FRANC_ZONE_INDICATORS = [
    "gdp_FCFA",
    "gdp_KMF",
    "investment",
    "money_FCFA",
    "money_KMF",
    "price_index_percent",
    "budget_balance_percent",
]

BCEAO_DATASETS = ["PIBN", "PIBC", "IMECO", "IHPC", "BDP4", "DPE", "SIM", "AM_A", "TOFE"]

IDCM_DOWNLOAD_SPECS = [
    {"FREQ": ["A"], "STO": ["B1GQ"], "PRICES": ["V"], "UNIT_MEASURE": ["XDC"]},
    {"FREQ": ["A"], "STO": ["B1GQ"], "PRICES": ["Q"], "UNIT_MEASURE": ["XDC"]},
    {"FREQ": ["A"], "STO": ["B9"], "UNIT_MEASURE": ["XDC"]},
    {"FREQ": ["A"], "STO": ["EMP"], "REF_SECTOR": ["S1"]},
    {"FREQ": ["A"], "STO": ["P51C"], "REF_SECTOR": ["S1"]},
    {"FREQ": ["A"], "STO": ["B8N"], "REF_SECTOR": ["S1"]},
    {"FREQ": ["A"], "STO": ["P3"], "REF_SECTOR": ["S1"], "PRICES": ["V"], "UNIT_MEASURE": ["XDC"]},
    {"FREQ": ["A"], "STO": ["P5"], "REF_SECTOR": ["S1"], "PRICES": ["V"], "UNIT_MEASURE": ["XDC"]},
    {"FREQ": ["A"], "STO": ["P51G"], "REF_SECTOR": ["S1"], "PRICES": ["V"], "UNIT_MEASURE": ["XDC"]},
    {"FREQ": ["A"], "STO": ["P6"], "REF_SECTOR": ["S1"], "PRICES": ["V"], "UNIT_MEASURE": ["XDC"]},
    {"FREQ": ["A"], "STO": ["P7"], "REF_SECTOR": ["S1"], "PRICES": ["V"], "UNIT_MEASURE": ["XDC"]},
]

UN_GDP_URL = "https://data.un.org/_Docs/SYB/CSV/SYB66_230_202310_GDP%20and%20GDP%20Per%20Capita.csv"
UN_POP_URL = "https://data.un.org/_Docs/SYB/CSV/SYB66_1_202310_Population,%20Surface%20Area%20and%20Density.csv"

WDI_RAW_COLUMNS = [
    "countrycode",
    "countryname",
    "region",
    "regionname",
    "adminregion",
    "adminregionname",
    "incomelevel",
    "incomelevelname",
    "lendingtype",
    "lendingtypename",
    "indicatorname",
    "indicatorcode",
]

IMF_RAW_COLUMNS = [
    "period",
    "value",
    "dataset_code",
    "dataset_name",
    "unit",
    "weo_country",
    "weo_subject",
    "provider_code",
    "series_code",
    "series_name",
]

BIS_RAW_COLUMNS = [
    "period",
    "period_start_day",
    "value",
    "obs_conf",
    "obs_status",
    "frequency",
    "dataset_code",
    "dataset_name",
    "freq",
    "ref_area",
    "unit_measure",
    "indexed_at",
    "provider_code",
    "series_code",
    "series_name",
    "series_num",
]

OECD_MEI_RAW_COLUMNS = [
    "period",
    "period_start_day",
    "value",
    "frequency",
    "dataset_code",
    "dataset_name",
    "FREQUENCY",
    "location",
    "measure",
    "subject",
    "indexed_at",
    "provider_code",
    "series_code",
    "series_name",
    "series_num",
]

OECD_HPI_RAW_COLUMNS = [
    "period",
    "period_start_day",
    "value",
    "frequency",
    "dataset_code",
    "dataset_name",
    "cou",
    "FREQUENCY",
    "ind",
    "indexed_at",
    "provider_code",
    "series_code",
    "series_name",
    "series_num",
]

OECD_EO_RAW_COLUMNS = [
    "period",
    "period_start_day",
    "value",
    "frequency",
    "dataset_code",
    "dataset_name",
    "indicator",
    "location",
    "indexed_at",
    "provider_code",
    "series_code",
    "series_name",
    "series_num",
]

OECD_KEI_RAW_COLUMNS = [
    "period",
    "period_start_day",
    "value",
    "frequency",
    "dataset_code",
    "dataset_name",
    "FREQUENCY",
    "location",
    "measure",
    "subject",
    "indexed_at",
    "provider_code",
    "series_code",
    "series_name",
    "series_num",
]

OECD_QNA_RAW_COLUMNS = [
    "period",
    "period_start_day",
    "value",
    "frequency",
    "dataset_code",
    "dataset_name",
    "FREQUENCY",
    "location",
    "measure",
    "subject",
    "indexed_at",
    "provider_code",
    "series_code",
    "series_name",
    "series_num",
]

OECD_REV_RAW_COLUMNS = [
    "period",
    "period_start_day",
    "value",
    "frequency",
    "dataset_code",
    "dataset_name",
    "cou",
    "gov",
    "tax",
    "var",
    "indexed_at",
    "provider_code",
    "series_code",
    "series_name",
    "series_num",
]

EUS_RAW_COLUMNS = [
    "period",
    "value",
    "dataset_name",
    "geo",
    "series_code",
    "series_name",
]

AMECO_RAW_COLUMNS = [
    "period",
    "period_start_day",
    "value",
    "frequency",
    "dataset_code",
    "dataset_name",
    "freq",
    "geo",
    "unit",
    "indexed_at",
    "provider_code",
    "series_code",
    "series_name",
    "series_num",
]

IMF_MFS_RAW_COLUMNS = [
    "period",
    "period_start_day",
    "value",
    "frequency",
    "dataset_code",
    "dataset_name",
    "freq",
    "indicator",
    "ref_area",
    "indexed_at",
    "provider_code",
    "series_code",
    "series_name",
    "series_num",
]

IMF_GFS_RAW_COLUMNS = [
    "period",
    "period_start_day",
    "value",
    "BASES_OF_RECORDING_CASH_NON_CASH",
    "BASES_OF_RECORDING_GROSSNET",
    "NATURE_OF_DATA",
    "VALUATION",
    "_frequency",
    "dataset_code",
    "dataset_name",
    "CLASSIFICATION",
    "FREQ",
    "REF_AREA",
    "REF_SECTOR",
    "UNIT_MEASURE",
    "indexed_at",
    "provider_code",
    "series_code",
    "series_name",
    "series_num",
    "OBS_STATUS",
]

IMF_IFS_RAW_COLUMNS = ["period", "value", "indicator", "ref_area", "series_name"]
UN_RAW_COLUMNS = ["regioncountryarea", "country", "year", "series", "value", "footnotes", "source"]
AFDB_RAW_COLUMNS = [
    "period",
    "period_start_day",
    "value",
    "_frequency",
    "dataset_code",
    "dataset_name",
    "country",
    "frequency",
    "indicator",
    "scale",
    "units",
    "indexed_at",
    "provider_code",
    "series_code",
    "series_name",
    "series_num",
]
FRANC_ZONE_RAW_COLUMNS = [
    "period",
    "period_start_day",
    "value",
    "frequency",
    "dataset_code",
    "dataset_name",
    "country",
    "freq",
    "indicator",
    "indexed_at",
    "provider_code",
    "series_code",
    "series_name",
    "series_num",
    "_frequency",
]
BCEAO_RAW_COLUMNS = [
    "period",
    "period_start_day",
    "value",
    "frequency",
    "dataset_code",
    "dataset_name",
    "country",
    "label",
    "indexed_at",
    "provider_code",
    "series_code",
    "series_name",
    "series_num",
]
IDCM_RAW_COLUMNS = [
    "period",
    "period_start_day",
    "value",
    "frequency",
    "dataset_code",
    "dataset_name",
    "accounting_entry",
    "activity",
    "adjustment",
    "counterpart_area",
    "counterpart_sector",
    "expenditure",
    "freq",
    "instr_asset",
    "prices",
    "ref_area",
    "ref_sector",
    "sto",
    "transformation",
    "unit_measure",
    "indexed_at",
    "provider_code",
    "series_code",
    "series_name",
    "series_num",
]


def _resolve(path: str | Path) -> Path:
    return sh._resolve_path(path)


def _read_raw_artifact(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".dta":
        return pd.read_dta(path, convert_categoricals=False)
    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            try:
                return pd.read_csv(path, encoding="latin-1")
            except Exception:
                return pd.DataFrame({"file_path": [str(path)]})
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except Exception:
            try:
                from ..clean._core import _read_excel_compat

                return _read_excel_compat(path)
            except Exception:
                return pd.DataFrame({"file_path": [str(path)]})
    return pd.DataFrame({"file_path": [str(path)]})


def _normalize_wb_text(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("&", "and").replace("  ", " ").strip()


def _request_json(url: str, *, params: dict[str, object] | None = None, timeout: int = REQUEST_TIMEOUT) -> object:
    response = HTTP.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _dbnomics_series_url(provider_code: str, dataset_code: str) -> str:
    return f"{DBNOMICS_SERIES_API}/{provider_code}/{dataset_code}"


def _dbnomics_dataset_exists(provider_code: str, dataset_code: str, *, timeout: int = REQUEST_TIMEOUT) -> bool:
    response = HTTP.get(
        _dbnomics_series_url(provider_code, dataset_code),
        params={"observations": 1, "offset": 0},
        timeout=timeout,
    )
    return response.status_code == 200


def _dbnomics_fetch_docs(
    provider_code: str,
    dataset_code: str,
    *,
    dimensions: dict[str, list[str]] | None = None,
    timeout: int = REQUEST_TIMEOUT,
) -> list[dict[str, object]]:
    docs: list[dict[str, object]] = []
    offset = 0
    limit = 1000

    while True:
        params: dict[str, object] = {"observations": 1, "offset": offset, "limit": limit}
        if dimensions is not None:
            params["dimensions"] = json.dumps(dimensions)

        data = _request_json(_dbnomics_series_url(provider_code, dataset_code), params=params, timeout=timeout)
        series = data["series"]
        page_docs = list(series.get("docs", []))
        docs.extend(page_docs)

        offset = int(series.get("offset", 0)) + int(series.get("limit", 0))
        if offset >= int(series.get("num_found", 0)):
            return docs


def _dbnomics_dataset_dimension_codes(
    provider_code: str,
    dataset_code: str,
    dimension: str,
    *,
    timeout: int = REQUEST_TIMEOUT,
) -> list[str]:
    data = _request_json(
        f"https://api.db.nomics.world/v22/datasets/{provider_code}/{dataset_code}",
        params={"limit": 1},
        timeout=timeout,
    )
    docs = list(data.get("datasets", {}).get("docs", []))
    if not docs:
        return []
    values = docs[0].get("dimensions_values_labels", {}).get(dimension, {})
    if not isinstance(values, dict):
        return []
    return [str(code) for code in values.keys()]


def _string_value(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    return str(value)


def _flatten_imf_docs(docs: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for doc in docs:
        dimensions = doc.get("dimensions", {})
        periods = list(doc.get("period", []))
        values = list(doc.get("value", []))
        for period, value in zip(periods, values):
            rows.append(
                {
                    "period": int(str(period)),
                    "value": _string_value(value),
                    "dataset_code": doc.get("dataset_code", ""),
                    "dataset_name": doc.get("dataset_name", ""),
                    "unit": dimensions.get("unit", ""),
                    "weo_country": dimensions.get("weo-country", ""),
                    "weo_subject": dimensions.get("weo-subject", ""),
                    "provider_code": doc.get("provider_code", ""),
                    "series_code": doc.get("series_code", ""),
                    "series_name": doc.get("series_name", ""),
                }
            )
    out = pd.DataFrame(rows, columns=IMF_RAW_COLUMNS)
    if not out.empty:
        out["period"] = out["period"].astype("int16")
    return out


def _flatten_bis_docs(docs: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for series_num, doc in enumerate(docs):
        dimensions = doc.get("dimensions", {})
        periods = list(doc.get("period", []))
        period_start_days = list(doc.get("period_start_day", []))
        values = list(doc.get("value", []))
        obs_attrs = {key: value for key, value in doc.get("observations_attributes", [])}

        def expand_attr(name: str) -> list[str]:
            value = obs_attrs.get(name, "")
            if isinstance(value, list):
                return [str(item) for item in value]
            return [str(value)] * len(periods)

        obs_conf_values = expand_attr("OBS_CONF")
        obs_status_values = expand_attr("OBS_STATUS")

        for idx, (period, period_start_day, value) in enumerate(zip(periods, period_start_days, values)):
            rows.append(
                {
                    "period": int(str(period)),
                    "period_start_day": str(period_start_day),
                    "value": _string_value(value),
                    "obs_conf": obs_conf_values[idx] if idx < len(obs_conf_values) else "",
                    "obs_status": obs_status_values[idx] if idx < len(obs_status_values) else "",
                    "frequency": doc.get("@frequency", ""),
                    "dataset_code": doc.get("dataset_code", ""),
                    "dataset_name": doc.get("dataset_name", ""),
                    "freq": dimensions.get("FREQ", ""),
                    "ref_area": dimensions.get("REF_AREA", ""),
                    "unit_measure": int(str(dimensions.get("UNIT_MEASURE", ""))),
                    "indexed_at": doc.get("indexed_at", ""),
                    "provider_code": doc.get("provider_code", ""),
                    "series_code": doc.get("series_code", ""),
                    "series_name": doc.get("series_name", ""),
                    "series_num": series_num,
                }
            )
    out = pd.DataFrame(rows, columns=BIS_RAW_COLUMNS)
    if not out.empty:
        out["period"] = out["period"].astype("int16")
        out["unit_measure"] = out["unit_measure"].astype("int16")
        out["series_num"] = pd.to_numeric(out["series_num"], downcast="integer")
    return out


def _flatten_bis_generic_docs(
    docs: list[dict[str, object]],
    *,
    columns: list[str],
    extra_dimensions: dict[str, str],
    period_as_int: bool,
    value_as_string: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for series_num, doc in enumerate(docs):
        dimensions = doc.get("dimensions", {})
        obs_attrs = {key: value for key, value in doc.get("observations_attributes", [])}
        periods = list(doc.get("period", []))
        period_start_days = list(doc.get("period_start_day", []))
        values = list(doc.get("value", []))

        def expand_attr(name: str) -> list[str]:
            value = obs_attrs.get(name, "")
            if isinstance(value, list):
                return [str(item) for item in value]
            return [str(value)] * len(periods)

        obs_conf_values = expand_attr("OBS_CONF")
        obs_status_values = expand_attr("OBS_STATUS")

        for idx, (period, period_start_day, value) in enumerate(zip(periods, period_start_days, values)):
            row: dict[str, object] = {
                "period": int(str(period)) if period_as_int else str(period),
                "period_start_day": str(period_start_day),
                "value": _string_value(value) if value_as_string else pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0],
                "obs_conf": obs_conf_values[idx] if idx < len(obs_conf_values) else "",
                "obs_status": obs_status_values[idx] if idx < len(obs_status_values) else "",
                "frequency": doc.get("@frequency", ""),
                "dataset_code": doc.get("dataset_code", ""),
                "dataset_name": doc.get("dataset_name", ""),
                "freq": dimensions.get("FREQ", ""),
                "ref_area": dimensions.get("REF_AREA", ""),
                "indexed_at": doc.get("indexed_at", ""),
                "provider_code": doc.get("provider_code", ""),
                "series_code": doc.get("series_code", ""),
                "series_name": doc.get("series_name", ""),
                "series_num": series_num,
            }
            for source_name, target_name in extra_dimensions.items():
                row[target_name] = dimensions.get(source_name, "")
            rows.append(row)

    out = pd.DataFrame(rows, columns=columns)
    if not out.empty:
        if period_as_int:
            out["period"] = out["period"].astype("int16")
        if "series_num" in out.columns:
            out["series_num"] = pd.to_numeric(out["series_num"], downcast="integer")
    return out


def _flatten_oecd_generic_docs(
    docs: list[dict[str, object]],
    *,
    columns: list[str],
    extra_dimensions: dict[str, str],
    period_as_int: bool,
    value_as_string: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for series_num, doc in enumerate(docs):
        dimensions = doc.get("dimensions", {})
        periods = list(doc.get("period", []))
        period_start_days = list(doc.get("period_start_day", []))
        values = list(doc.get("value", []))
        for period, period_start_day, value in zip(periods, period_start_days, values):
            row: dict[str, object] = {
                "period": int(str(period)) if period_as_int else str(period),
                "period_start_day": str(period_start_day),
                "value": _string_value(value) if value_as_string else pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0],
                "frequency": doc.get("@frequency", ""),
                "dataset_code": doc.get("dataset_code", ""),
                "dataset_name": doc.get("dataset_name", ""),
                "indexed_at": doc.get("indexed_at", ""),
                "provider_code": doc.get("provider_code", ""),
                "series_code": doc.get("series_code", ""),
                "series_name": doc.get("series_name", ""),
                "series_num": series_num,
            }
            for source_name, target_name in extra_dimensions.items():
                row[target_name] = dimensions.get(source_name, "")
            rows.append(row)

    out = pd.DataFrame(rows, columns=columns)
    if not out.empty:
        if period_as_int:
            out["period"] = out["period"].astype("int16")
        if "series_num" in out.columns:
            out["series_num"] = pd.to_numeric(out["series_num"], downcast="integer")
    return out


def _prepend_frame(current: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    if current.empty:
        return master.reset_index(drop=True)
    if master.empty:
        return current.reset_index(drop=True)
    return pd.concat([current, master], ignore_index=True)


def _trim_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda value: value.strip() if isinstance(value, str) else value)
    return out


def _read_un_csv(url: str, *, timeout: int = REQUEST_TIMEOUT) -> pd.DataFrame:
    response = HTTP.get(url, timeout=timeout)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.content.decode("latin1")), skiprows=1)
    rename_map: dict[str, str] = {}
    for col in df.columns:
        text = str(col)
        if text.startswith("Unnamed:"):
            rename_map[col] = "country"
        else:
            rename_map[col] = "".join(ch for ch in text.lower() if ch.isalnum() or ch == "_")
    df = df.rename(columns=rename_map)
    df = df[UN_RAW_COLUMNS].copy()
    df["regioncountryarea"] = pd.to_numeric(df["regioncountryarea"], errors="coerce").astype("Int64")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df


def _flatten_afdb_docs(docs: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered_countries = sorted(str(doc.get("dimensions", {}).get("country", "")) for doc in docs)
    series_num_map = {
        country: idx
        for idx, country in enumerate(ordered_countries)
    }
    for doc in docs:
        dimensions = doc.get("dimensions", {})
        periods = list(doc.get("period", []))
        period_start_days = list(doc.get("period_start_day", []))
        values = list(doc.get("value", []))
        country = str(dimensions.get("country", ""))
        series_num = series_num_map.get(country, 0)
        for period, period_start_day, value in zip(periods, period_start_days, values):
            rows.append(
                {
                    "period": int(str(period)),
                    "period_start_day": str(period_start_day),
                    "value": pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0],
                    "_frequency": doc.get("@frequency", ""),
                    "dataset_code": doc.get("dataset_code", ""),
                    "dataset_name": doc.get("dataset_name", ""),
                    "country": country,
                    "frequency": dimensions.get("frequency", ""),
                    "indicator": dimensions.get("indicator", ""),
                    "scale": int(str(dimensions.get("scale", "0") or "0")),
                    "units": dimensions.get("units", ""),
                    "indexed_at": doc.get("indexed_at", ""),
                    "provider_code": doc.get("provider_code", ""),
                    "series_code": doc.get("series_code", ""),
                    "series_name": doc.get("series_name", ""),
                    "series_num": series_num,
                }
            )
    out = pd.DataFrame(rows, columns=AFDB_RAW_COLUMNS)
    if not out.empty:
        out["period"] = out["period"].astype("int16")
        out["scale"] = pd.to_numeric(out["scale"], downcast="integer")
    return out


def _flatten_franc_zone_docs(docs: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered_countries = sorted(str(doc.get("dimensions", {}).get("country", "")) for doc in docs)
    series_num_map = {
        country: idx
        for idx, country in enumerate(ordered_countries)
    }
    for doc in docs:
        dimensions = doc.get("dimensions", {})
        periods = list(doc.get("period", []))
        period_start_days = list(doc.get("period_start_day", []))
        values = list(doc.get("value", []))
        indicator = dimensions.get("indicator", "")
        is_kmf = indicator in {"gdp_KMF", "money_KMF"}
        country = str(dimensions.get("country", ""))
        series_num = np.nan if is_kmf else series_num_map.get(country, 0)
        for period, period_start_day, value in zip(periods, period_start_days, values):
            rows.append(
                {
                    "period": int(str(period)),
                    "period_start_day": str(period_start_day),
                    "value": _string_value(value),
                    "frequency": "" if is_kmf else doc.get("@frequency", ""),
                    "dataset_code": doc.get("dataset_code", ""),
                    "dataset_name": doc.get("dataset_name", ""),
                    "country": country,
                    "freq": dimensions.get("freq", ""),
                    "indicator": indicator,
                    "indexed_at": doc.get("indexed_at", ""),
                    "provider_code": doc.get("provider_code", ""),
                    "series_code": doc.get("series_code", ""),
                    "series_name": doc.get("series_name", ""),
                    "series_num": series_num,
                    "_frequency": doc.get("@frequency", "") if is_kmf else "",
                }
            )
    out = pd.DataFrame(rows, columns=FRANC_ZONE_RAW_COLUMNS)
    if not out.empty:
        out["period"] = out["period"].astype("int16")
    return out


def _flatten_bceao_docs(docs: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for doc_idx, doc in enumerate(docs):
        dimensions = doc.get("dimensions", {})
        periods = list(doc.get("period", []))
        period_start_days = list(doc.get("period_start_day", []))
        values = list(doc.get("value", []))
        for period, period_start_day, value in zip(periods, period_start_days, values):
            rows.append(
                {
                    "period": int(str(period)),
                    "period_start_day": str(period_start_day),
                    "value": _string_value(value),
                    "frequency": doc.get("@frequency", ""),
                    "dataset_code": doc.get("dataset_code", ""),
                    "dataset_name": doc.get("dataset_name", ""),
                    "country": dimensions.get("country", ""),
                    "label": dimensions.get("label", ""),
                    "indexed_at": doc.get("indexed_at", ""),
                    "provider_code": doc.get("provider_code", ""),
                    "series_code": doc.get("series_code", ""),
                    "series_name": doc.get("series_name", ""),
                    "series_num": doc_idx,
                }
            )
    out = pd.DataFrame(rows, columns=BCEAO_RAW_COLUMNS)
    if not out.empty:
        out["period"] = out["period"].astype("int16")
        out["series_num"] = pd.to_numeric(out["series_num"], downcast="integer")
    return out


def _flatten_idcm_docs(docs: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for series_num, doc in enumerate(docs):
        dimensions = doc.get("dimensions", {})
        periods = list(doc.get("period", []))
        period_start_days = list(doc.get("period_start_day", []))
        values = list(doc.get("value", []))
        for period, period_start_day, value in zip(periods, period_start_days, values):
            rows.append(
                {
                    "period": int(str(period)),
                    "period_start_day": str(period_start_day),
                    "value": pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0],
                    "frequency": doc.get("@frequency", ""),
                    "dataset_code": doc.get("dataset_code", ""),
                    "dataset_name": doc.get("dataset_name", ""),
                    "accounting_entry": dimensions.get("ACCOUNTING_ENTRY", ""),
                    "activity": dimensions.get("ACTIVITY", ""),
                    "adjustment": dimensions.get("ADJUSTMENT", ""),
                    "counterpart_area": dimensions.get("COUNTERPART_AREA", ""),
                    "counterpart_sector": dimensions.get("COUNTERPART_SECTOR", ""),
                    "expenditure": dimensions.get("EXPENDITURE", ""),
                    "freq": dimensions.get("FREQ", ""),
                    "instr_asset": dimensions.get("INSTR_ASSET", ""),
                    "prices": dimensions.get("PRICES", ""),
                    "ref_area": dimensions.get("REF_AREA", ""),
                    "ref_sector": dimensions.get("REF_SECTOR", ""),
                    "sto": dimensions.get("STO", ""),
                    "transformation": dimensions.get("TRANSFORMATION", ""),
                    "unit_measure": dimensions.get("UNIT_MEASURE", ""),
                    "indexed_at": doc.get("indexed_at", ""),
                    "provider_code": doc.get("provider_code", ""),
                    "series_code": doc.get("series_code", ""),
                    "series_name": doc.get("series_name", ""),
                    "series_num": series_num,
                }
            )
    out = pd.DataFrame(rows, columns=IDCM_RAW_COLUMNS)
    if not out.empty:
        out["period"] = out["period"].astype("int16")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out["series_num"] = pd.to_numeric(out["series_num"], downcast="integer")
    return out


def _flatten_imf_generic_docs(
    docs: list[dict[str, object]],
    *,
    columns: list[str],
    extra_dimensions: dict[str, str],
    period_as_int: bool,
    value_as_string: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for series_num, doc in enumerate(docs):
        dimensions = doc.get("dimensions", {})
        periods = list(doc.get("period", []))
        period_start_days = list(doc.get("period_start_day", []))
        values = list(doc.get("value", []))
        for period, period_start_day, value in zip(periods, period_start_days, values):
            row: dict[str, object] = {
                "period": int(str(period)) if period_as_int else str(period),
                "period_start_day": str(period_start_day),
                "value": _string_value(value) if value_as_string else pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0],
                "frequency": doc.get("@frequency", ""),
                "dataset_code": doc.get("dataset_code", ""),
                "dataset_name": doc.get("dataset_name", ""),
                "indexed_at": doc.get("indexed_at", ""),
                "provider_code": doc.get("provider_code", ""),
                "series_code": doc.get("series_code", ""),
                "series_name": doc.get("series_name", ""),
                "series_num": series_num,
            }
            for source_name, target_name in extra_dimensions.items():
                row[target_name] = dimensions.get(source_name, "")
            rows.append(row)

    out = pd.DataFrame(rows, columns=columns)
    if not out.empty:
        if period_as_int:
            out["period"] = out["period"].astype("int16")
        out["series_num"] = pd.to_numeric(out["series_num"], downcast="integer")
    return out


def _flatten_imf_gfs_docs(docs: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for series_num, doc in enumerate(docs):
        dimensions = doc.get("dimensions", {})
        periods = list(doc.get("period", []))
        period_start_days = list(doc.get("period_start_day", []))
        values = list(doc.get("value", []))
        obs_attrs = {key: value for key, value in doc.get("observations_attributes", [])}

        def expand_attr(name: str) -> list[str]:
            value = obs_attrs.get(name, "")
            if isinstance(value, list):
                return [str(item) for item in value]
            return [str(value)] * len(periods)

        cash_non_cash = expand_attr("BASES_OF_RECORDING_CASH_NON_CASH")
        grossnet = expand_attr("BASES_OF_RECORDING_GROSSNET")
        nature = expand_attr("NATURE_OF_DATA")
        valuation = expand_attr("VALUATION")
        obs_status = expand_attr("OBS_STATUS")

        for idx, (period, period_start_day, value) in enumerate(zip(periods, period_start_days, values)):
            rows.append(
                {
                    "period": str(period),
                    "period_start_day": str(period_start_day),
                    "value": _string_value(value),
                    "BASES_OF_RECORDING_CASH_NON_CASH": cash_non_cash[idx] if idx < len(cash_non_cash) else "",
                    "BASES_OF_RECORDING_GROSSNET": grossnet[idx] if idx < len(grossnet) else "",
                    "NATURE_OF_DATA": nature[idx] if idx < len(nature) else "",
                    "VALUATION": valuation[idx] if idx < len(valuation) else "",
                    "_frequency": doc.get("@frequency", ""),
                    "dataset_code": doc.get("dataset_code", ""),
                    "dataset_name": doc.get("dataset_name", ""),
                    "CLASSIFICATION": dimensions.get("CLASSIFICATION", ""),
                    "FREQ": dimensions.get("FREQ", ""),
                    "REF_AREA": dimensions.get("REF_AREA", ""),
                    "REF_SECTOR": dimensions.get("REF_SECTOR", ""),
                    "UNIT_MEASURE": dimensions.get("UNIT_MEASURE", ""),
                    "indexed_at": doc.get("indexed_at", ""),
                    "provider_code": doc.get("provider_code", ""),
                    "series_code": doc.get("series_code", ""),
                    "series_name": doc.get("series_name", ""),
                    "series_num": series_num,
                    "OBS_STATUS": obs_status[idx] if idx < len(obs_status) else "",
                }
            )

    out = pd.DataFrame(rows, columns=IMF_GFS_RAW_COLUMNS)
    if not out.empty:
        out["series_num"] = pd.to_numeric(out["series_num"], downcast="integer")
    return out


def _flatten_imf_ifs_docs(docs: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for doc in docs:
        dimensions = doc.get("dimensions", {})
        periods = list(doc.get("period", []))
        values = list(doc.get("value", []))
        for period, value in zip(periods, values):
            rows.append(
                {
                    "period": int(str(period)),
                    "value": pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0],
                    "indicator": dimensions.get("INDICATOR", ""),
                    "ref_area": dimensions.get("REF_AREA", ""),
                    "series_name": doc.get("series_name", ""),
                }
            )
    out = pd.DataFrame(rows, columns=IMF_IFS_RAW_COLUMNS)
    if not out.empty:
        out["period"] = out["period"].astype("int16")
    return out


def _world_bank_get_all(path: str, *, params: dict[str, object] | None = None, timeout: int = REQUEST_TIMEOUT) -> list[dict[str, object]]:
    request_params = {"format": "json", "per_page": 20000}
    if params is not None:
        request_params.update(params)

    first_page = _request_json(f"{WORLD_BANK_API}/{path.lstrip('/')}", params=request_params, timeout=timeout)
    if not isinstance(first_page, list) or len(first_page) < 2:
        return []

    meta = first_page[0]
    docs = list(first_page[1])
    pages = int(meta.get("pages", 1))
    for page in range(2, pages + 1):
        request_params["page"] = page
        page_payload = _request_json(f"{WORLD_BANK_API}/{path.lstrip('/')}", params=request_params, timeout=timeout)
        if isinstance(page_payload, list) and len(page_payload) >= 2:
            docs.extend(page_payload[1])
    return docs


def resolve_imf_weo_dataset_code(*, current_year: int | None = None, timeout: int = REQUEST_TIMEOUT) -> str:
    thisyear = date.today().year if current_year is None else int(current_year)
    lastyear = thisyear - 1

    current_october = f"WEO:{thisyear}-10"
    if _dbnomics_dataset_exists("IMF", current_october, timeout=timeout):
        return current_october

    current_april = f"WEO:{thisyear}-04"
    if _dbnomics_dataset_exists("IMF", current_april, timeout=timeout):
        return current_april

    return f"WEO:{lastyear}-10"
