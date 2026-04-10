from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def download_ameco(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    master = pd.DataFrame(columns=AMECO_RAW_COLUMNS)
    for dataset_code, dims in AMECO_DATASETS:
        docs = _dbnomics_fetch_docs("AMECO", dataset_code, dimensions=dims, timeout=timeout)
        current = _flatten_oecd_generic_docs(
            docs,
            columns=AMECO_RAW_COLUMNS,
            extra_dimensions={"freq": "freq", "geo": "geo", "unit": "unit"},
            period_as_int=True,
            value_as_string=True,
        )
        master = _prepend_frame(current, master)

    base_path = raw_dir / "aggregators" / "AMECO" / "AMECO"
    sh.gmdsavedate("AMECO", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master[AMECO_RAW_COLUMNS], str(base_path), id_columns=["period", "period_start_day", "geo", "dataset_code", "series_code"])
__all__ = ["download_ameco"]
