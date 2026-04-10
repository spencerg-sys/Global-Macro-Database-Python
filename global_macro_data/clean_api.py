from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Sequence

import pandas as pd

from . import helpers as sh
from .clean import _core as _core
from .clean._core import (
    REPO_ROOT,
    _resolve,
)
from .clean.aggregators import (
    clean_adb,
    clean_afdb,
    clean_afristat,
    clean_ahstat,
    clean_ameco,
    clean_amf,
    clean_barro,
    clean_bceao,
    clean_bg,
    clean_bis_cbrate,
    clean_bis_cpi,
    clean_bis_hpi,
    clean_bis_reer,
    clean_bis_usdfx,
    clean_bit,
    clean_bordo,
    clean_bruegel,
    clean_bvx,
    clean_cepac,
    clean_clio,
    clean_dallasfed_hpi,
    clean_davis,
    clean_eus,
    clean_fao,
    clean_flora,
    clean_franc_zone,
    clean_fz,
    clean_gapminder,
    clean_gna,
    clean_grimm,
    clean_hfs,
    clean_homer_sylla,
    clean_idcm,
    clean_ihd,
    clean_ilo,
    clean_imf_fpp,
    clean_imf_gdd,
    clean_imf_gfs,
    clean_imf_hdd,
    clean_imf_ifs,
    clean_imf_mfs,
    clean_imf_weo,
    clean_jerven,
    clean_jo,
    clean_jst,
    clean_lund,
    clean_lv,
    clean_madisson,
    clean_md,
    clean_mitchell,
    clean_moxlad,
    clean_mw,
    clean_nbs,
    clean_oecd_eo,
    clean_oecd_hpi,
    clean_oecd_kei,
    clean_oecd_mei,
    clean_oecd_mei_arc,
    clean_oecd_qna,
    clean_oecd_rev,
    clean_pwt,
    clean_rr,
    clean_rr_debt,
    clean_schmelzing,
    clean_tena_pop,
    clean_tena_trade,
    clean_tena_usdfx,
    clean_th_id,
    clean_un,
    clean_wb_cc,
    clean_wdi,
    clean_wdi_arc
)
from .clean.country_level import (
    clean_arg_1,
    clean_arg_2,
    clean_aus_1,
    clean_aus_2,
    clean_aut_1,
    clean_bra_1,
    clean_can_1,
    clean_che_1,
    clean_che_2,
    clean_chn_1,
    clean_dnk_1,
    clean_dza_1,
    clean_esp_1,
    clean_esp_2,
    clean_fra_1,
    clean_fra_2,
    clean_gbr_1,
    clean_idn_1,
    clean_isl_1,
    clean_isl_2,
    clean_ita_1,
    clean_ita_2,
    clean_ita_3,
    clean_jpn_1,
    clean_kor_1,
    clean_lbr_1,
    clean_mar_1,
    clean_nor_1,
    clean_nor_2,
    clean_pol_1,
    clean_prt_1,
    clean_sau_1,
    clean_swe_1,
    clean_tur_1,
    clean_twn_1,
    clean_usa_1,
    clean_usa_2,
    clean_zaf_1
)

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

HISTORICAL_SOURCES_WITHOUT_DO = (
    "ARG_3",
    "CAN_2",
    "ECLAC",
    "FRA_3",
    "IRL_1",
    "KOR_2",
    "TWN_2",
    "UN_trade",
)

_CLEAN_SOURCE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "MITCHELL": ("BIS_USDfx",),
}
_SPECIAL_CLEAN_SOURCE_NAMES: dict[str, str] = {
    "bis_cbrate": "BIS_cbrate",
    "bis_usdfx": "BIS_USDfx",
    "bit": "BIT_USDfx",
    "bruegel": "Bruegel",
    "davis": "Davis",
    "dallasfed_hpi": "DALLASFED_HPI",
    "gapminder": "Gapminder",
    "grimm": "Grimm",
    "homer_sylla": "Homer_Sylla",
    "madisson": "Madisson",
    "rr_debt": "RR_debt",
    "schmelzing": "Schmelzing",
    "tena_pop": "Tena_pop",
    "tena_trade": "Tena_trade",
    "tena_usdfx": "Tena_USDfx",
}


def _normalize_clean_source_name(source: str) -> str:
    src = str(source).strip()
    if not src:
        return src
    match = re.fullmatch(r"CS([123])_([A-Z]{3})", src, flags=re.IGNORECASE)
    if match:
        return f"{match.group(2).upper()}_{match.group(1)}"
    special_name = _SPECIAL_CLEAN_SOURCE_NAMES.get(src.lower())
    if special_name is not None:
        return special_name
    alias_map = {
        "bis_reer": "BIS_REER",
        "bit": "BIT_USDfx",
        "barro": "BARRO",
        "bordo": "BORDO",
        "bruegel": "Bruegel",
        "bruegel_reer": "Bruegel",
        "davis": "Davis",
        "dallasfed": "DALLASFED_HPI",
        "dallasfed_hpi": "DALLASFED_HPI",
        "mad": "Madisson",
        "madison": "Madisson",
        "maddison": "Madisson",
        "madisson": "Madisson",
        "moxlad": "MOXLAD",
        "rr_debt": "RR_debt",
        "lund": "LUND",
        "tena": "Tena_trade",
        "tena_pop": "Tena_pop",
        "tena_trade": "Tena_trade",
        "tena_usdfx": "Tena_USDfx",
    }
    return alias_map.get(src.lower(), src)


def _reorder_clean_sources_for_dependencies(sources: Sequence[str]) -> list[str]:
    ordered = list(sources)
    index = {src: i for i, src in enumerate(ordered)}
    for source, required in _CLEAN_SOURCE_DEPENDENCIES.items():
        if source not in index:
            continue
        source_idx = index[source]
        for dep in required:
            dep_norm = _normalize_clean_source_name(dep)
            dep_idx = index.get(dep_norm)
            if dep_idx is None or dep_idx < source_idx:
                continue
            item = ordered.pop(dep_idx)
            source_idx = ordered.index(source)
            ordered.insert(source_idx, item)
            index = {src: i for i, src in enumerate(ordered)}
    return ordered


def _list_python_modules(path: Path) -> list[Path]:
    return sorted(
        [file for file in path.glob("*.py") if file.name not in {"__init__.py", "_core.py"}],
        key=lambda item: item.name.lower(),
    )


def _clean_source_name_from_module(stem: str) -> str:
    lower = stem.lower()
    if re.fullmatch(r"[a-z]{3}_\d+", lower):
        iso3, version = lower.split("_", 1)
        return f"{iso3.upper()}_{version}"
    return _SPECIAL_CLEAN_SOURCE_NAMES.get(lower, stem.upper())


def get_default_clean_sources() -> list[str]:
    clean_root = Path(__file__).resolve().parent / "clean"
    aggregators = [_clean_source_name_from_module(path.stem) for path in _list_python_modules(clean_root / "aggregators")]
    country_level = [_clean_source_name_from_module(path.stem) for path in _list_python_modules(clean_root / "country_level")]
    return _reorder_clean_sources_for_dependencies(aggregators + country_level)


def rebuild_clean_sources(
    sources: Sequence[str] | None = None,
    *,
    source_list_path: Path | str = REPO_ROOT / "data" / "helpers" / "source_list.csv",
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    skip_unsupported: bool = False,
    skip_historical_without_do: bool = False,
    historical_without_do: Sequence[str] = HISTORICAL_SOURCES_WITHOUT_DO,
    run_summary_path: Path | str | None = None,
) -> dict[str, object]:
    if sources is None:
        default_source_list_path = REPO_ROOT / "data" / "helpers" / "source_list.csv"
        resolved_source_list_path = _resolve(source_list_path)
        if resolved_source_list_path != default_source_list_path and resolved_source_list_path.exists():
            source_list = pd.read_csv(resolved_source_list_path)
            raw_sources = source_list["source_name"].dropna().astype(str).tolist()
        else:
            raw_sources = get_default_clean_sources()
    else:
        raw_sources = [str(source) for source in sources]

    normalized_sources: list[str] = []
    seen: set[str] = set()
    for source in raw_sources:
        normalized = _normalize_clean_source_name(source)
        if normalized and normalized not in seen:
            normalized_sources.append(normalized)
            seen.add(normalized)

    normalized_sources = _reorder_clean_sources_for_dependencies(normalized_sources)
    historical_skip = {
        _normalize_clean_source_name(source)
        for source in historical_without_do
        if str(source).strip()
    }

    _require_notes_dataset(data_temp_dir=data_temp_dir)

    completed: list[tuple[str, tuple[int, int]]] = []
    skipped: list[str] = []
    skipped_historical_without_do: list[str] = []
    for source in normalized_sources:
        if skip_historical_without_do and source in historical_skip:
            skipped_historical_without_do.append(source)
            continue
        try:
            df = clean_source(
                source,
                data_raw_dir=data_raw_dir,
                data_clean_dir=data_clean_dir,
                data_helper_dir=data_helper_dir,
                data_temp_dir=data_temp_dir,
            )
        except ValueError as exc:
            if skip_unsupported and "Unsupported clean source" in str(exc):
                skipped.append(source)
                continue
            raise
        completed.append((source, tuple(df.shape)))

    out: dict[str, object] = {
        "completed": completed,
        "skipped": skipped,
        "skipped_historical_without_do": skipped_historical_without_do,
    }
    if run_summary_path is not None:
        path = _resolve(run_summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _require_notes_dataset(
    *,
    data_temp_dir: Path | str,
) -> Path:
    temp_dir = _resolve(data_temp_dir)
    notes_path = temp_dir / "notes.dta"
    if notes_path.exists():
        return notes_path
    raise FileNotFoundError(
        f"Missing notes dataset at {notes_path}. Run the erase/initialization stage before rebuilding clean sources."
    )


def clean_source(
    source: str,
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
) -> pd.DataFrame:
    src = _normalize_clean_source_name(source)
    if src == "WDI":
        return clean_wdi(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "IMF_WEO":
        return clean_imf_weo(
            data_raw_dir=data_raw_dir,
            data_clean_dir=data_clean_dir,
            data_helper_dir=data_helper_dir,
        )
    if src == "BIS_CPI":
        return clean_bis_cpi(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "BIS_USDfx":
        return clean_bis_usdfx(
            data_raw_dir=data_raw_dir,
            data_clean_dir=data_clean_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
        )
    if src == "BIS_REER":
        return clean_bis_reer(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "BIS_cbrate":
        return clean_bis_cbrate(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "BIS_HPI":
        return clean_bis_hpi(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "AFDB":
        return clean_afdb(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "ADB":
        return clean_adb(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "AFRISTAT":
        return clean_afristat(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "EUS":
        return clean_eus(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "AHSTAT":
        return clean_ahstat(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "CEPAC":
        return clean_cepac(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "GNA":
        return clean_gna(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "MITCHELL":
        return clean_mitchell(
            data_raw_dir=data_raw_dir,
            data_clean_dir=data_clean_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
        )
    if src == "HFS":
        return clean_hfs(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "AMECO":
        return clean_ameco(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "FRANC_ZONE":
        return clean_franc_zone(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "BCEAO":
        return clean_bceao(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "IDCM":
        return clean_idcm(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "OECD_MEI":
        return clean_oecd_mei(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "OECD_MEI_ARC":
        return clean_oecd_mei_arc(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "OECD_HPI":
        return clean_oecd_hpi(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "OECD_EO":
        return clean_oecd_eo(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "OECD_KEI":
        return clean_oecd_kei(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "OECD_QNA":
        return clean_oecd_qna(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "OECD_REV":
        return clean_oecd_rev(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "UN":
        return clean_un(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "BARRO":
        return clean_barro(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "BIT_USDfx":
        return clean_bit(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src in {"FLORA", "Flora"}:
        return clean_flora(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "FZ":
        return clean_fz(
            data_raw_dir=data_raw_dir,
            data_clean_dir=data_clean_dir,
            data_helper_dir=data_helper_dir,
        )
    if src == "DALLASFED_HPI":
        return clean_dallasfed_hpi(
            data_raw_dir=data_raw_dir,
            data_clean_dir=data_clean_dir,
            data_helper_dir=data_helper_dir,
        )
    if src == "ILO":
        return clean_ilo(
            data_raw_dir=data_raw_dir,
            data_clean_dir=data_clean_dir,
            data_helper_dir=data_helper_dir,
        )
    if src == "LUND":
        return clean_lund(
            data_raw_dir=data_raw_dir,
            data_clean_dir=data_clean_dir,
            data_helper_dir=data_helper_dir,
        )
    if src == "NBS":
        return clean_nbs(
            data_raw_dir=data_raw_dir,
            data_clean_dir=data_clean_dir,
            data_helper_dir=data_helper_dir,
        )
    if src == "BG":
        return clean_bg(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "IHD":
        return clean_ihd(
            data_raw_dir=data_raw_dir,
            data_clean_dir=data_clean_dir,
            data_helper_dir=data_helper_dir,
        )
    if src == "JO":
        return clean_jo(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "MD":
        return clean_md(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "TH_ID":
        return clean_th_id(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "CLIO":
        return clean_clio(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "BORDO":
        return clean_bordo(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "BVX":
        return clean_bvx(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "FAO":
        return clean_fao(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "Gapminder":
        return clean_gapminder(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "Grimm":
        return clean_grimm(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "Homer_Sylla":
        return clean_homer_sylla(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "JST":
        return clean_jst(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "JERVEN":
        return clean_jerven(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "LV":
        return clean_lv(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "RR":
        return clean_rr(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "RR_debt":
        return clean_rr_debt(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "IMF_MFS":
        return clean_imf_mfs(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "IMF_GFS":
        return clean_imf_gfs(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "IMF_IFS":
        return clean_imf_ifs(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "AMF":
        return clean_amf(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "IMF_FPP":
        return clean_imf_fpp(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "IMF_GDD":
        return clean_imf_gdd(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "IMF_HDD":
        return clean_imf_hdd(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "ESP_1":
        return clean_esp_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "ESP_2":
        return clean_esp_2(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "FRA_1":
        return clean_fra_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "IDN_1":
        return clean_idn_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "USA_1":
        return clean_usa_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "ARG_1":
        return clean_arg_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "ARG_2":
        return clean_arg_2(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "AUT_1":
        return clean_aut_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "JPN_1":
        return clean_jpn_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "AUS_1":
        return clean_aus_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "AUS_2":
        return clean_aus_2(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "Bruegel":
        return clean_bruegel(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "BRA_1":
        return clean_bra_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "CAN_1":
        return clean_can_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "CHE_1":
        return clean_che_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "CHE_2":
        return clean_che_2(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "CHN_1":
        return clean_chn_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "DNK_1":
        return clean_dnk_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "DZA_1":
        return clean_dza_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "Davis":
        return clean_davis(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "GBR_1":
        return clean_gbr_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "FRA_2":
        return clean_fra_2(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "ISL_1":
        return clean_isl_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "ISL_2":
        return clean_isl_2(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "ITA_1":
        return clean_ita_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "ITA_2":
        return clean_ita_2(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "ITA_3":
        return clean_ita_3(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "KOR_1":
        return clean_kor_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "LBR_1":
        return clean_lbr_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "MAR_1":
        return clean_mar_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src in {"MAD", "Madisson"}:
        return clean_madisson(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "MOXLAD":
        return clean_moxlad(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "MW":
        return clean_mw(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "NOR_1":
        return clean_nor_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "NOR_2":
        return clean_nor_2(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "POL_1":
        return clean_pol_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "PRT_1":
        return clean_prt_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "PWT":
        return clean_pwt(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "SAU_1":
        return clean_sau_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "Schmelzing":
        return clean_schmelzing(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "SWE_1":
        return clean_swe_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "Tena_pop":
        return clean_tena_pop(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "Tena_USDfx":
        return clean_tena_usdfx(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "Tena_trade":
        return clean_tena_trade(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    if src == "TUR_1":
        return clean_tur_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "TWN_1":
        return clean_twn_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "USA_2":
        return clean_usa_2(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "ZAF_1":
        return clean_zaf_1(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "WB_CC":
        return clean_wb_cc(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir)
    if src == "WDI_ARC":
        return clean_wdi_arc(data_raw_dir=data_raw_dir, data_clean_dir=data_clean_dir, data_helper_dir=data_helper_dir)
    raise ValueError(f"Unsupported clean source: {src}")


def get_default_historical_skip_sources() -> list[str]:
    return [_normalize_clean_source_name(source) for source in HISTORICAL_SOURCES_WITHOUT_DO]


__all__ = [
    "HISTORICAL_SOURCES_WITHOUT_DO",
    "clean_ahstat",
    "clean_cepac",
    "clean_gna",
    "clean_mitchell",
    "clean_hfs",
    "clean_adb",
    "clean_afdb",
    "clean_afristat",
    "clean_arg_1",
    "clean_aut_1",
    "clean_aus_1",
    "clean_amf",
    "clean_bit",
    "clean_bis_cpi",
    "clean_bis_cbrate",
    "clean_bis_hpi",
    "clean_bis_reer",
    "clean_bis_usdfx",
    "clean_bceao",
    "clean_can_1",
    "clean_che_1",
    "clean_che_2",
    "clean_dnk_1",
    "clean_idcm",
    "clean_ameco",
    "clean_barro",
    "clean_bg",
    "clean_bruegel",
    "clean_dallasfed_hpi",
    "clean_davis",
    "clean_ilo",
    "clean_lund",
    "clean_nbs",
    "clean_flora",
    "clean_fz",
    "clean_jo",
    "clean_md",
    "clean_th_id",
    "clean_eus",
    "clean_clio",
    "clean_franc_zone",
    "clean_bordo",
    "clean_bvx",
    "clean_esp_1",
    "clean_esp_2",
    "clean_imf_weo",
    "clean_imf_gfs",
    "clean_imf_gdd",
    "clean_imf_hdd",
    "clean_imf_ifs",
    "clean_imf_fpp",
    "clean_imf_mfs",
    "clean_gbr_1",
    "clean_isl_1",
    "clean_isl_2",
    "clean_jpn_1",
    "clean_jst",
    "clean_lbr_1",
    "clean_lv",
    "clean_madisson",
    "clean_moxlad",
    "clean_mw",
    "clean_fra_1",
    "clean_fao",
    "clean_gapminder",
    "clean_grimm",
    "clean_homer_sylla",
    "clean_ihd",
    "clean_idn_1",
    "clean_jerven",
    "clean_oecd_eo",
    "clean_oecd_hpi",
    "clean_oecd_kei",
    "clean_oecd_mei",
    "clean_oecd_mei_arc",
    "clean_oecd_qna",
    "clean_oecd_rev",
    "clean_pol_1",
    "clean_pwt",
    "clean_rr",
    "clean_rr_debt",
    "clean_sau_1",
    "clean_schmelzing",
    "clean_tena_pop",
    "clean_tena_trade",
    "clean_tena_usdfx",
    "clean_tur_1",
    "clean_un",
    "clean_usa_1",
    "clean_wb_cc",
    "clean_wdi_arc",
    "clean_zaf_1",
    "get_default_historical_skip_sources",
    "rebuild_clean_sources",
    "clean_source",
    "clean_wdi",
    "get_default_clean_sources",
]
