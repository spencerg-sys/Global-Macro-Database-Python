from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def download_bis_cpi(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    master = pd.DataFrame(columns=BIS_RAW_COLUMNS)
    for unit_measure in ("628", "771"):
        docs = _dbnomics_fetch_docs(
            "BIS",
            "WS_LONG_CPI",
            dimensions={"FREQ": ["A"], "UNIT_MEASURE": [unit_measure]},
            timeout=timeout,
        )
        current = _flatten_bis_docs(docs)
        master = _prepend_frame(current, master)

    base_path = raw_dir / "aggregators" / "BIS" / "BIS_CPI"
    sh.gmdsavedate("BIS_CPI", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    sh.gmdsavedate("BIS_infl", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master[BIS_RAW_COLUMNS], str(base_path), id_columns=["period", "ref_area", "series_code"])
__all__ = ["download_bis_cpi"]
