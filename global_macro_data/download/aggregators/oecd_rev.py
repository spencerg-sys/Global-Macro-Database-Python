from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def download_oecd_rev(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    docs = _dbnomics_fetch_docs("OECD", "REV", dimensions={"VAR": ["TAXNAT"], "TAX": ["TOTALTAX"], "GOV": ["FED"]}, timeout=timeout)
    master = _flatten_oecd_generic_docs(
        docs,
        columns=OECD_REV_RAW_COLUMNS,
        extra_dimensions={"COU": "cou", "GOV": "gov", "TAX": "tax", "VAR": "var"},
        period_as_int=True,
    )

    base_path = raw_dir / "aggregators" / "OECD" / "OECD_REV"
    sh.gmdsavedate("OECD_REV", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master[OECD_REV_RAW_COLUMNS], str(base_path), id_columns=["period", "cou", "series_code", "dataset_code"])
__all__ = ["download_oecd_rev"]
