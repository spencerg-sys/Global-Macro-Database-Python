from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def download_oecd_qna(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    master = pd.DataFrame(columns=OECD_QNA_RAW_COLUMNS)
    for dims in OECD_QNA_SUBJECT_SPECS:
        docs = _dbnomics_fetch_docs("OECD", "QNA", dimensions=dims, timeout=timeout)
        current = _flatten_oecd_generic_docs(
            docs,
            columns=OECD_QNA_RAW_COLUMNS,
            extra_dimensions={"FREQUENCY": "FREQUENCY", "LOCATION": "location", "MEASURE": "measure", "SUBJECT": "subject"},
            period_as_int=True,
        )
        master = _prepend_frame(current, master)

    base_path = raw_dir / "aggregators" / "OECD" / "OECD_QNA"
    sh.gmdsavedate("OECD_QNA", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master[OECD_QNA_RAW_COLUMNS], str(base_path), id_columns=["period", "location", "series_code", "dataset_code"])
__all__ = ["download_oecd_qna"]
