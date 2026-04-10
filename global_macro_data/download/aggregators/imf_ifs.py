from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def download_imf_ifs(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
    data_temp_dir: Path | str = REPO_ROOT / "data" / "tempfiles",
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)

    master = pd.DataFrame(columns=IMF_IFS_RAW_COLUMNS)
    try:
        countries = _dbnomics_dataset_dimension_codes("IMF", "IFS", "REF_AREA", timeout=timeout)
    except requests.RequestException:
        countries = []
    countries = [code for code in countries if not any(ch.isdigit() for ch in code)]

    for country in countries:
        for indicator in ["NGDP_R_XDC", "NC_R_XDC"]:
            try:
                docs = _dbnomics_fetch_docs(
                    "IMF",
                    "IFS",
                    dimensions={"INDICATOR": [indicator], "FREQ": ["A"], "REF_AREA": [country]},
                    timeout=timeout,
                )
            except requests.RequestException:
                continue
            current = _flatten_imf_ifs_docs(docs)
            master = _prepend_frame(current, master)

    for indicator in [ind for ind in IMF_IFS_INDICATORS if ind not in {"NGDP_R_XDC", "NC_R_XDC"}]:
        docs = _dbnomics_fetch_docs("IMF", "IFS", dimensions={"INDICATOR": [indicator], "FREQ": ["A"]}, timeout=timeout)
        current = _flatten_imf_ifs_docs(docs)
        master = _prepend_frame(current, master)

    series_name = master["series_name"].astype(str)
    master.loc[series_name.str.contains("Gross Domestic Product, Real", na=False), "indicator"] = "NGDP_R_XDC"
    master.loc[series_name.str.contains("Final Consumption Expenditure, Real", na=False), "indicator"] = "NC_R_XDC"
    master = master[IMF_IFS_RAW_COLUMNS].copy()

    base_path = raw_dir / "aggregators" / "IMF" / "IMF_IFS"
    sh.gmdsavedate("IMF_IFS", data_helper_dir=helper_dir, data_temp_dir=temp_dir)
    return sh.savedelta(master[IMF_IFS_RAW_COLUMNS], str(base_path), id_columns=["period", "ref_area", "indicator"])
__all__ = ["download_imf_ifs"]
