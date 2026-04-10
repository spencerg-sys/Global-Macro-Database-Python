from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})


FRA_1_RAW_COLUMNS = [
    "period",
    "period_start_day",
    "value",
    "OBS_CONF",
    "OBS_STATUS",
    "_frequency",
    "dataset_code",
    "dataset_name",
    "BS_ITEM",
    "REF_AREA",
    "indexed_at",
    "provider_code",
    "series_code",
    "series_name",
]


def download_fra_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    master = pd.DataFrame(columns=FRA_1_RAW_COLUMNS)
    for aggregate in ["M10", "M20", "M30"]:
        docs = _dbnomics_fetch_docs(
            "BDF",
            "BSI1",
            dimensions={"BS_ITEM": [aggregate], "DATA_TYPE": ["1"], "REF_AREA": ["FR"]},
            timeout=timeout,
        )
        current = _flatten_bis_generic_docs(
            docs,
            columns=FRA_1_RAW_COLUMNS,
            extra_dimensions={"BS_ITEM": "BS_ITEM", "REF_AREA": "REF_AREA"},
            period_as_int=False,
        )
        current = _trim_object_columns(current)
        master = _prepend_frame(current, master)

    base_path = raw_dir / "country_level" / "FRA_1"
    sh.gmdsavedate("FRA_1", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master[FRA_1_RAW_COLUMNS], str(base_path), id_columns=["period", "REF_AREA", "series_code"])


__all__ = ["download_fra_1"]
