from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def download_bis_hpi(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    docs = _dbnomics_fetch_docs(
        "BIS",
        "WS_SPP",
        dimensions={"UNIT_MEASURE": ["628"], "VALUE": ["N"]},
        timeout=timeout,
    )
    rows: list[dict[str, object]] = []
    for doc in docs:
        dimensions = doc.get("dimensions", {})
        periods = list(doc.get("period", []))
        values = list(doc.get("value", []))
        for period, value in zip(periods, values):
            rows.append(
                {
                    "period": str(period),
                    "value": _string_value(value),
                    "dataset_name": doc.get("dataset_name", ""),
                    "freq": dimensions.get("FREQ", ""),
                    "ref_area": dimensions.get("REF_AREA", ""),
                    "series_code": doc.get("series_code", ""),
                    "series_name": doc.get("series_name", ""),
                }
            )

    columns = ["period", "value", "dataset_name", "freq", "ref_area", "series_code", "series_name"]
    master = pd.DataFrame(rows, columns=columns)
    if master.duplicated(["ref_area", "series_name", "period"]).any():
        raise sh.PipelineRuntimeError("ref_area series_name period do not uniquely identify observations", code=198)

    base_path = raw_dir / "aggregators" / "BIS" / "BIS_HPI"
    sh.gmdsavedate("BIS_HPI", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master[columns], str(base_path), id_columns=["period", "ref_area", "series_code"])
__all__ = ["download_bis_hpi"]
