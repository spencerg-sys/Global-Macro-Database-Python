from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def download_bceao(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    master = pd.DataFrame(columns=BCEAO_RAW_COLUMNS)
    for dataset_code in BCEAO_DATASETS:
        docs = _dbnomics_fetch_docs("BCEAO", dataset_code, timeout=timeout)
        current = _flatten_bceao_docs(docs)
        master = _prepend_frame(current, master)

    if not master.empty:
        master = master.sort_values(["period", "country"], kind="mergesort").reset_index(drop=True)

    base_path = raw_dir / "aggregators" / "BCEAO" / "BCEAO"
    sh.gmdsavedate("BCEAO", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master[BCEAO_RAW_COLUMNS], str(base_path), id_columns=["country", "period", "series_code", "dataset_code"])
__all__ = ["download_bceao"]
