from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def download_franc_zone(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    master = pd.DataFrame(columns=FRANC_ZONE_RAW_COLUMNS)
    for indicator in FRANC_ZONE_INDICATORS:
        try:
            docs = _dbnomics_fetch_docs("Franc-zone", "FRANCZONE", dimensions={"indicator": [indicator]}, timeout=timeout)
        except requests.RequestException:
            continue
        current = _trim_object_columns(_flatten_franc_zone_docs(docs))
        master = _prepend_frame(current, master)

    base_path = raw_dir / "aggregators" / "FRANC_ZONE" / "FRANC_ZONE"
    sh.gmdsavedate("FRANC_ZONE", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master[FRANC_ZONE_RAW_COLUMNS], str(base_path), id_columns=["period", "country", "series_code", "dataset_code"])
__all__ = ["download_franc_zone"]
