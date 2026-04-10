from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def download_imf_gfs(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    master = pd.DataFrame(columns=IMF_GFS_RAW_COLUMNS)
    for dims in IMF_GFS_SPECS:
        try:
            docs = _dbnomics_fetch_docs("IMF", "GFSMAB", dimensions=dims, timeout=timeout)
        except requests.RequestException:
            continue
        current = _trim_object_columns(_flatten_imf_gfs_docs(docs))
        master = _prepend_frame(current, master)

    base_path = raw_dir / "aggregators" / "IMF" / "IMF_GFS"
    sh.gmdsavedate("IMF_GFS", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master[IMF_GFS_RAW_COLUMNS], str(base_path), id_columns=["period", "REF_AREA", "series_code", "dataset_code"])
__all__ = ["download_imf_gfs"]
