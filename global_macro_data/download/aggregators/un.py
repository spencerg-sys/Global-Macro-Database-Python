from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def download_un(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    gdp = _read_un_csv(UN_GDP_URL, timeout=timeout)
    pop = _read_un_csv(UN_POP_URL, timeout=timeout)
    master = pd.concat([pop, gdp], ignore_index=True)

    if master.duplicated(["country", "series", "year"]).any():
        raise sh.PipelineRuntimeError("country series year do not uniquely identify observations", code=459)

    base_path = raw_dir / "aggregators" / "UN" / "UN"
    sh.gmdsavedate("UN", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master[UN_RAW_COLUMNS], str(base_path), id_columns=["country", "year", "series"])
__all__ = ["download_un"]
