from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def download_idcm(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    master = pd.DataFrame(columns=IDCM_RAW_COLUMNS)
    for dims in IDCM_DOWNLOAD_SPECS:
        docs = _dbnomics_fetch_docs("ECB", "IDCM", dimensions=dims, timeout=timeout)
        current = _flatten_idcm_docs(docs)
        master = _prepend_frame(current, master)

    if master.duplicated(["ref_area", "series_name", "period"]).any():
        raise sh.PipelineRuntimeError("ref_area series_name period do not uniquely identify observations", code=459)

    base_path = raw_dir / "aggregators" / "IDCM" / "IDCM"
    sh.gmdsavedate("IDCM", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master[IDCM_RAW_COLUMNS], str(base_path), id_columns=["period", "ref_area", "series_code", "dataset_code"])
__all__ = ["download_idcm"]
