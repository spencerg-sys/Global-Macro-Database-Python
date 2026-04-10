from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def download_bis_usdfx(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    columns = [
        "period",
        "period_start_day",
        "value",
        "obs_conf",
        "obs_status",
        "frequency",
        "dataset_code",
        "dataset_name",
        "collection",
        "currency",
        "freq",
        "ref_area",
        "indexed_at",
        "provider_code",
        "series_code",
        "series_name",
        "series_num",
    ]
    docs = _dbnomics_fetch_docs(
        "BIS",
        "WS_XRU",
        dimensions={"FREQ": ["A"], "COLLECTION": ["A"]},
        timeout=timeout,
    )
    master = _flatten_bis_generic_docs(
        docs,
        columns=columns,
        extra_dimensions={"COLLECTION": "collection", "CURRENCY": "currency"},
        period_as_int=True,
        value_as_string=True,
    )

    base_path = raw_dir / "aggregators" / "BIS" / "BIS_USDfx"
    sh.gmdsavedate("BIS_USDfx", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master[columns], str(base_path), id_columns=["period", "ref_area", "series_code"])
__all__ = ["download_bis_usdfx"]
