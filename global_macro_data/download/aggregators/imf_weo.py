from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def download_imf_weo(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    current_year: int | None = None,
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    dataset_code = resolve_imf_weo_dataset_code(current_year=current_year, timeout=timeout)

    master = pd.DataFrame(columns=IMF_RAW_COLUMNS)
    for subject in IMF_INITIAL_SUBJECTS:
        docs = _dbnomics_fetch_docs(
            "IMF",
            dataset_code,
            dimensions={"weo-subject": [subject]},
            timeout=timeout,
        )
        master = _flatten_imf_docs(docs)
        for next_subject in IMF_SUBJECTS:
            docs = _dbnomics_fetch_docs(
                "IMF",
                dataset_code,
                dimensions={"weo-subject": [next_subject]},
                timeout=timeout,
            )
            current = _flatten_imf_docs(docs)
            master = _prepend_frame(current, master)

    if master.duplicated(["weo_country", "series_code", "period"]).any():
        raise sh.PipelineRuntimeError("weo_country series_code period do not uniquely identify observations", code=198)

    base_path = raw_dir / "aggregators" / "IMF" / "IMF_WEO"
    gmdsavedate_result = sh.gmdsavedate("IMF_WEO", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    _ = gmdsavedate_result
    return sh.savedelta(master[IMF_RAW_COLUMNS], str(base_path), id_columns=["period", "weo_country", "series_code", "dataset_code"])
__all__ = ["download_imf_weo"]
