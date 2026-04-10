from __future__ import annotations

from pathlib import Path

import pandas as pd

from .download import _core as _core
from .download._core import (
    REQUEST_TIMEOUT,
    REPO_ROOT,
    resolve_imf_weo_dataset_code,
)
from .download.aggregators import (
    download_afdb,
    download_ameco,
    download_bceao,
    download_bis_cbrate,
    download_bis_cpi,
    download_bis_hpi,
    download_bis_reer,
    download_bis_usdfx,
    download_eus,
    download_fred,
    download_franc_zone,
    download_idcm,
    download_imf_gfs,
    download_imf_ifs,
    download_imf_mfs,
    download_imf_weo,
    download_oecd_eo,
    download_oecd_hpi,
    download_oecd_kei,
    download_oecd_mei,
    download_oecd_qna,
    download_oecd_rev,
    download_un,
    download_wdi
)
from .download.country_level import (
    download_esp_1,
    download_fra_1,
    download_idn_1,
    download_ita_3,
    download_pol_1,
    download_sau_1,
    download_tur_1,
    download_zaf_1,
)

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

DEFAULT_DOWNLOAD_SOURCES: tuple[str, ...] = (
    "AFDB",
    "AMECO",
    "BCEAO",
    "BIS_cbrate",
    "BIS_CPI",
    "BIS_HPI",
    "BIS_REER",
    "BIS_USDfx",
    "EUS",
    "FRANC_ZONE",
    "FRED",
    "IDCM",
    "IMF_GFS",
    "IMF_IFS",
    "IMF_MFS",
    "IMF_WEO",
    "OECD_EO",
    "OECD_HPI",
    "OECD_KEI",
    "OECD_MEI",
    "OECD_QNA",
    "OECD_REV",
    "UN",
    "WDI",
    "ESP_1",
    "FRA_1",
    "IDN_1",
    "ITA_3",
    "POL_1",
    "SAU_1",
    "TUR_1",
    "ZAF_1",
)


def get_default_download_sources() -> list[str]:
    return list(DEFAULT_DOWNLOAD_SOURCES)


def download_source(
    source: str,
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    current_year: int | None = None,
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    src = str(source)
    if src == "WDI":
        return download_wdi(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "IMF_WEO":
        return download_imf_weo(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            current_year=current_year,
            timeout=timeout,
        )
    if src == "BIS_CPI":
        return download_bis_cpi(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "BIS_USDfx":
        return download_bis_usdfx(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "BIS_REER":
        return download_bis_reer(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "BIS_cbrate":
        return download_bis_cbrate(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "BIS_HPI":
        return download_bis_hpi(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "EUS":
        return download_eus(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "AMECO":
        return download_ameco(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "OECD_MEI":
        return download_oecd_mei(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "OECD_HPI":
        return download_oecd_hpi(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "OECD_EO":
        return download_oecd_eo(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "OECD_KEI":
        return download_oecd_kei(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "OECD_QNA":
        return download_oecd_qna(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "OECD_REV":
        return download_oecd_rev(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "IMF_MFS":
        return download_imf_mfs(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "IMF_GFS":
        return download_imf_gfs(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "IMF_IFS":
        return download_imf_ifs(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "UN":
        return download_un(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "AFDB":
        return download_afdb(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "FRANC_ZONE":
        return download_franc_zone(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "BCEAO":
        return download_bceao(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "IDCM":
        return download_idcm(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src in {"FRED", "USA_1"}:
        return download_fred(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "ESP_1":
        return download_esp_1(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "FRA_1":
        return download_fra_1(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "IDN_1":
        return download_idn_1(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "ITA_3":
        return download_ita_3(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "POL_1":
        return download_pol_1(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "SAU_1":
        return download_sau_1(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "TUR_1":
        return download_tur_1(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    if src == "ZAF_1":
        return download_zaf_1(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            timeout=timeout,
        )
    raise ValueError(f"Unsupported download source: {src}")


def download_all_sources(
    sources: list[str] | tuple[str, ...] | None = None,
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    current_year: int | None = None,
    timeout: int = REQUEST_TIMEOUT,
) -> list[tuple[str, tuple[int, int]]]:
    ordered_sources = get_default_download_sources() if sources is None else [str(source) for source in sources]
    completed: list[tuple[str, tuple[int, int]]] = []
    for source in ordered_sources:
        df = download_source(
            source,
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
            current_year=current_year,
            timeout=timeout,
        )
        completed.append((source, tuple(df.shape)))
    return completed


__all__ = [
    "DEFAULT_DOWNLOAD_SOURCES",
    "download_afdb",
    "download_all_sources",
    "download_ameco",
    "download_bis_hpi",
    "download_bis_cbrate",
    "download_bis_cpi",
    "download_bis_reer",
    "download_bis_usdfx",
    "download_bceao",
    "download_eus",
    "download_esp_1",
    "download_fra_1",
    "download_fred",
    "download_franc_zone",
    "download_idn_1",
    "download_idcm",
    "download_imf_gfs",
    "download_imf_ifs",
    "download_imf_mfs",
    "download_imf_weo",
    "download_ita_3",
    "download_oecd_kei",
    "download_oecd_eo",
    "download_oecd_hpi",
    "download_oecd_mei",
    "download_oecd_qna",
    "download_oecd_rev",
    "download_pol_1",
    "download_sau_1",
    "download_tur_1",
    "download_un",
    "download_source",
    "download_wdi",
    "download_zaf_1",
    "get_default_download_sources",
    "resolve_imf_weo_dataset_code",
]
