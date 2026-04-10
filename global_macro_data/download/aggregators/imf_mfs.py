from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def download_imf_mfs(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    master = pd.DataFrame(columns=IMF_MFS_RAW_COLUMNS)
    for indicator in IMF_MFS_INDICATORS:
        docs = _dbnomics_fetch_docs("IMF", "MFS", dimensions={"INDICATOR": [indicator], "FREQ": ["A"]}, timeout=timeout)
        current = _flatten_imf_generic_docs(
            docs,
            columns=IMF_MFS_RAW_COLUMNS,
            extra_dimensions={"FREQ": "freq", "INDICATOR": "indicator", "REF_AREA": "ref_area"},
            period_as_int=True,
        )
        master = _prepend_frame(current, master)

    base_path = raw_dir / "aggregators" / "IMF" / "IMF_MFS"
    sh.gmdsavedate("IMF_MFS", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master[IMF_MFS_RAW_COLUMNS], str(base_path), id_columns=["period", "ref_area", "series_code", "dataset_code"])
__all__ = ["download_imf_mfs"]
