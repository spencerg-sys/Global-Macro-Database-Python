from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})


ITA_3_RAW_COLUMNS = [
    "period",
    "period_start_day",
    "value",
    "frequency",
    "dataset_code",
    "dataset_name",
    "indexed_at",
    "provider_code",
    "series_code",
    "series_name",
    "series_num",
]


def download_ita_3(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    docs = _dbnomics_fetch_docs("ISTAT", "92_506_DF_DCCN_PILN_1", dimensions={"EDITION": ["2024M9"]}, timeout=timeout)
    master = _flatten_oecd_generic_docs(
        docs,
        columns=ITA_3_RAW_COLUMNS,
        extra_dimensions={},
        period_as_int=True,
        value_as_string=True,
    )

    base_path = raw_dir / "country_level" / "ITA_3"
    sh.gmdsavedate("ITA_3", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master[ITA_3_RAW_COLUMNS], str(base_path), id_columns=["period", "dataset_code", "series_code"])


__all__ = ["download_ita_3"]
