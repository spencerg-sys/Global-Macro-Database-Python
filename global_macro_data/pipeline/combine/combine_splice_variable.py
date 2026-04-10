from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def combine_splice_variable(
    varname: str,
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
) -> pd.DataFrame:
    work = _build_splice_input(
        varname,
        data_clean_dir=data_clean_dir,
        data_temp_dir=data_temp_dir,
        data_final_dir=data_final_dir,
    )
    spec = _parse_splice_spec(varname)
    for source, note in _parse_note_sources(varname):
        sh.gmdaddnote_source(source, note, varname, data_temp_dir=data_temp_dir)

    result = sh.splice(
        work,
        priority=str(spec["priority"]),
        generate=varname,
        varname=varname,
        base_year=int(spec["base_year"]),
        method=str(spec["method"]),
        data_final_dir=_resolve(data_final_dir),
    )

    if varname == "CPI":
        result = _key_sort(result, ["ISO3", "year"])
        result["CPI_2010"] = result["CPI"].where(pd.to_numeric(result["year"], errors="coerce").eq(2010))
        result["CPI_2010_all"] = result.groupby("ISO3")["CPI_2010"].transform("mean")
        result["CPI_rebased"] = (pd.to_numeric(result["CPI"], errors="coerce") * 100) / pd.to_numeric(result["CPI_2010_all"], errors="coerce")
        result = result.drop(columns=["CPI_2010", "CPI_2010_all"])
        result["CPI_rebased"] = result["CPI_rebased"].where(result["CPI_rebased"].notna(), result["CPI"])
        result = result.drop(columns=["CPI"])
        result = result.rename(columns={"CPI_rebased": "CPI"})
        _save_dta(result, _resolve(data_final_dir) / "chainlinked_CPI.dta")

    if varname == "REER":
        reer = pd.to_numeric(result["REER"], errors="coerce")
        invalid_mask = (
            pd.to_numeric(result["year"], errors="coerce").eq(2010)
            & reer.notna()
            & reer.ne(100)
        )
        if invalid_mask.any():
            sh._fail("Error: Not all values are 100 in 2010", code=198)

    return result
__all__ = ["combine_splice_variable"]
