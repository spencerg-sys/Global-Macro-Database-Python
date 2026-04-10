from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})


POL_1_RAW_COLUMNS = [
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
    "ISO3",
]


def download_pol_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    master = pd.DataFrame(columns=POL_1_RAW_COLUMNS)
    for dataset in ["ABOP", "AFT", "AGG", "APF", "AINV", "AMON", "ANA", "APOP", "APRI"]:
        docs = _dbnomics_fetch_docs("STATPOL", dataset, dimensions={"FREQ": ["A"]}, timeout=timeout)
        current = _flatten_oecd_generic_docs(
            docs,
            columns=POL_1_RAW_COLUMNS[:-1],
            extra_dimensions={},
            period_as_int=True,
            value_as_string=True,
        )
        current["ISO3"] = "POL"
        current = current[POL_1_RAW_COLUMNS]
        master = _prepend_frame(current, master)

    base_path = raw_dir / "country_level" / "POL_1"
    sh.gmdsavedate("POL_1", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master[POL_1_RAW_COLUMNS], str(base_path), id_columns=["period", "ISO3", "dataset_code", "series_code"])


__all__ = ["download_pol_1"]
