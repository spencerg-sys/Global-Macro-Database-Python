from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})


IDN_1_RAW_COLUMNS = [
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
    "REF_AREA",
]


def download_idn_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    master = pd.DataFrame(columns=IDN_1_RAW_COLUMNS)
    for dataset in [
        "TABEL1_1",
        "TABEL1_2",
        "TABEL1_25",
        "TABEL4_1",
        "TABEL4_2",
        "TABEL4_3",
        "TABEL4_4",
        "TABEL5_1",
        "TABEL5_40",
        "TABEL7_1",
        "TABEL7_3",
        "TABEL7_6",
        "TABEL8_1",
    ]:
        docs = _dbnomics_fetch_docs("BI", dataset, dimensions={"freq": ["A"]}, timeout=timeout)
        current = _flatten_oecd_generic_docs(
            docs,
            columns=IDN_1_RAW_COLUMNS[:-1],
            extra_dimensions={},
            period_as_int=False,
            value_as_string=True,
        )
        current["REF_AREA"] = "IDN"
        current = current[IDN_1_RAW_COLUMNS]
        master = _prepend_frame(current, master)

    base_path = raw_dir / "country_level" / "IDN_1"
    sh.gmdsavedate("IDN_1", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master[IDN_1_RAW_COLUMNS], str(base_path), id_columns=["period", "dataset_code", "series_code"])


__all__ = ["download_idn_1"]
