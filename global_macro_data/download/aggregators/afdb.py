from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def download_afdb(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    master = pd.DataFrame(columns=AFDB_RAW_COLUMNS)
    for indicator in AFDB_INDICATORS:
        docs = _dbnomics_fetch_docs("AFDB", "bbkawjf", dimensions={"indicator": [indicator]}, timeout=timeout)
        current = _flatten_afdb_docs(docs)
        master = _prepend_frame(current, master)

    base_path = raw_dir / "aggregators" / "AFDB" / "AFDB"
    sh.gmdsavedate("AFDB", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master[AFDB_RAW_COLUMNS], str(base_path), id_columns=["period", "country", "series_code"])
__all__ = ["download_afdb"]
