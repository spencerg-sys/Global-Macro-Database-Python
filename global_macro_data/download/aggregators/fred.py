from __future__ import annotations

from io import StringIO

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})


FRED_SERIES = [
    "GDPA",
    "GDPCA",
    "A939RX0Q048SBEA",
    "FPCPITOTLZGUSA",
    "B230RC0A052NBEA",
    "A929RC1A027NBEA",
    "W006RC1Q027SBEA",
    "W068RCQ027SBEA",
    "EXPGSA",
    "IMPGSA",
    "A124RC1A027NBEA",
    "GFDEGDQ188S",
    "RIFSPFFNA",
    "FYFSGDA188S",
    "AFRECPT",
    "BOGMBASE",
    "M1SL",
    "M2SL",
    "UNRATE",
    "USSTHPI",
    "BOGZ1FL073161113Q",
    "RIFSGFSM03NA",
]


def _fred_csv(series_id: str, *, timeout: int) -> pd.DataFrame:
    response = HTTP.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", timeout=timeout)
    response.raise_for_status()
    current = pd.read_csv(StringIO(response.text))
    current.columns = [str(col).strip() for col in current.columns]
    date_column = None
    for candidate in ("DATE", "observation_date"):
        if candidate in current.columns:
            date_column = candidate
            break
    if date_column is None or series_id not in current.columns:
        raise ValueError(f"Unexpected FRED CSV schema for {series_id}")
    current = current.rename(columns={date_column: "datestr"})
    current[series_id] = pd.to_numeric(current[series_id].replace(".", pd.NA), errors="coerce")
    return current[["datestr", series_id]]


def download_fred(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    master: pd.DataFrame | None = None
    for series_id in FRED_SERIES:
        current = _fred_csv(series_id, timeout=timeout)
        master = current if master is None else master.merge(current, on="datestr", how="outer", sort=True)

    assert master is not None
    master = master.sort_values("datestr").reset_index(drop=True)
    master["daten"] = master["datestr"]
    ordered_cols = ["datestr", "daten", *FRED_SERIES]
    master = master[ordered_cols]

    base_path = raw_dir / "country_level" / "USA_1"
    sh.gmdsavedate("USA_1", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master, str(base_path), id_columns=["datestr"])


__all__ = ["download_fred"]
