from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def download_oecd_eo(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    master = pd.DataFrame(columns=OECD_EO_RAW_COLUMNS)
    for variable in OECD_EO_VARIABLES:
        docs = _dbnomics_fetch_docs("OECD", "EO", dimensions={"FREQUENCY": ["A"], "VARIABLE": [variable]}, timeout=timeout)
        current = _flatten_oecd_generic_docs(
            docs,
            columns=OECD_EO_RAW_COLUMNS,
            extra_dimensions={"VARIABLE": "indicator", "LOCATION": "location"},
            period_as_int=True,
        )
        master = _prepend_frame(current, master)

    for indicator in OECD_NAAG_INDICATORS:
        docs = _dbnomics_fetch_docs("OECD", "NAAG", dimensions={"INDICATOR": [indicator]}, timeout=timeout)
        current = _flatten_oecd_generic_docs(
            docs,
            columns=OECD_EO_RAW_COLUMNS,
            extra_dimensions={"INDICATOR": "indicator", "LOCATION": "location"},
            period_as_int=True,
        )
        master = _prepend_frame(current, master)

    base_path = raw_dir / "aggregators" / "OECD" / "OECD_EO"
    sh.gmdsavedate("OECD_EO", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master[OECD_EO_RAW_COLUMNS], str(base_path), id_columns=["period", "location", "series_code", "dataset_code"])
__all__ = ["download_oecd_eo"]
