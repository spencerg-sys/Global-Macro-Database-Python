from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def combine_ca_gdp(
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
) -> pd.DataFrame:
    from .combine_splice_variable import combine_splice_variable
    from .combine_usdfx import combine_usdfx

    final_dir = _resolve(data_final_dir)
    clean_dir = _resolve(data_clean_dir)
    # Match the Stata workflow in CA_GDP.do: always rerun dependencies.
    combine_splice_variable(
        "nGDP",
        data_clean_dir=data_clean_dir,
        data_temp_dir=data_temp_dir,
        data_final_dir=final_dir,
    )
    combine_usdfx(
        data_clean_dir=data_clean_dir,
        data_temp_dir=data_temp_dir,
        data_final_dir=final_dir,
    )

    work = _build_blank_merge_input(
        "CA_GDP",
        extra_keep_cols=("Mitchell_CA", "Mitchell_CA_USD", "Mitchell_nGDP"),
        data_clean_dir=clean_dir,
        data_temp_dir=data_temp_dir,
        data_helper_dir=data_helper_dir,
    )
    work = _merge_keep13(work, _load_dta(_chainlinked_path("USDfx", final_dir)), keepus=["USDfx"])
    work = _merge_keep13(work, _load_dta(_chainlinked_path("nGDP", final_dir)), keepus=["nGDP"])

    mitchell_ca = pd.to_numeric(work["Mitchell_CA"], errors="coerce")
    mitchell_ca_usd = pd.to_numeric(work["Mitchell_CA_USD"], errors="coerce")
    usdfx = pd.to_numeric(work.get("USDfx"), errors="coerce")
    mitchell_ngdp = pd.to_numeric(work["Mitchell_nGDP"], errors="coerce")

    work["Mitchell_CA"] = mitchell_ca.where(mitchell_ca.notna(), mitchell_ca_usd * usdfx)
    work["Mitchell_CA_GDP"] = (pd.to_numeric(work["Mitchell_CA"], errors="coerce") / mitchell_ngdp) * 100
    work = work.drop(columns=["USDfx", "nGDP", "Mitchell_CA_USD", "Mitchell_CA"], errors="ignore")
    work.loc[work["ISO3"].astype(str) == "SLE", "Mitchell_CA_GDP"] = pd.to_numeric(
        work.loc[work["ISO3"].astype(str) == "SLE", "Mitchell_CA_GDP"], errors="coerce"
    ) * 1000
    work.loc[(work["ISO3"].astype(str) == "IDN") & (pd.to_numeric(work["year"], errors="coerce") <= 1939), "Mitchell_CA_GDP"] = pd.to_numeric(
        work.loc[(work["ISO3"].astype(str) == "IDN") & (pd.to_numeric(work["year"], errors="coerce") <= 1939), "Mitchell_CA_GDP"],
        errors="coerce",
    ).astype("float32") / np.float32(1000)
    work.loc[(work["ISO3"].astype(str) == "GHA") & pd.to_numeric(work["year"], errors="coerce").between(1955, 1956), "Mitchell_CA_GDP"] = pd.NA
    work = _restore_column_dtypes(work, _source_dtype_map("CA_GDP", data_clean_dir=clean_dir))
    work["Mitchell_CA_GDP"] = pd.to_numeric(work["Mitchell_CA_GDP"], errors="coerce").astype("float32")

    spec = _parse_splice_spec("CA_GDP")
    result = sh.splice(
        work,
        priority=str(spec["priority"]),
        generate="CA_GDP",
        varname="CA_GDP",
        base_year=int(spec["base_year"]),
        method=str(spec["method"]),
        data_final_dir=final_dir,
    )
    return result
__all__ = ["combine_ca_gdp"]
