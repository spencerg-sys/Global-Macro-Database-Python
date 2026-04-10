from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def validate_inputs(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
) -> dict[str, object]:
    def _read_helper_csv(path: Path) -> pd.DataFrame:
        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path)

    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    clean_dir_map = {
        str(path.parent.relative_to(clean_dir)).replace("\\", "/").lower(): str(path.parent.relative_to(clean_dir)).replace("\\", "/")
        for path in clean_dir.rglob("*")
        if path.is_file()
    }
    raw_dir_map = {
        str(path.parent.relative_to(raw_dir)).replace("\\", "/").lower(): str(path.parent.relative_to(raw_dir)).replace("\\", "/")
        for path in raw_dir.rglob("*")
        if path.is_file()
    }
    clean_only = [
        clean_dir_map[key]
        for key in sorted(clean_dir_map)
        if key not in raw_dir_map and key not in {"country_level", "."}
    ]
    raw_only = [
        raw_dir_map[key]
        for key in sorted(raw_dir_map)
        if key not in clean_dir_map and key not in {".", ""}
    ]
    if clean_only:
        sh._emit("The following folders are only in the clean data:")
        sh._emit("\n".join(clean_only))
        raise sh.PipelineRuntimeError("Clean data folders without raw counterparts", code=198)
    if raw_only:
        sh._emit("The following folders are only in the raw data:")
        sh._emit("\n".join(raw_only))

    countrylist = _load_dta(helper_dir / "countrylist.dta")
    valid_iso = set(countrylist["ISO3"].dropna().astype(str))
    sources_csv = _read_helper_csv(helper_dir / "sources.csv")
    docvars_csv = _read_helper_csv(helper_dir / "docvars.csv")
    sources_csv.columns = [str(col).strip().lower() for col in sources_csv.columns]
    docvars_csv.columns = [str(col).strip().lower() for col in docvars_csv.columns]
    possible_sources = set(sources_csv["src_specific_var_name"].dropna().astype(str).str.strip())
    valid_varabbr = set(docvars_csv["codes"].dropna().astype(str).str.strip())
    for varabbr in sorted(set(sources_csv["varabbr"].dropna().astype(str).str.strip())):
        if varabbr not in valid_varabbr:
            raise sh.PipelineRuntimeError(f"Variable {varabbr} is not in the varabbr column of the docvars.csv file.", code=198)

    checked_files = 0
    for file in _iter_clean_dta_files(clean_dir):
        if file.name == "clean_data_wide.dta":
            continue
        df = _load_dta(file)
        checked_files += 1

        if df.duplicated(["ISO3", "year"]).any():
            raise sh.PipelineRuntimeError(f"ISO3 and year are not unique in the {file}", code=198)

        string_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if string_cols != ["ISO3"]:
            raise sh.PipelineRuntimeError(f"List of strings contains variables except ISO3: {' '.join(string_cols)}", code=198)

        for iso in sorted(set(df["ISO3"].dropna().astype(str))):
            if iso not in valid_iso:
                raise sh.PipelineRuntimeError(f"{iso}, located in {file}, is not in the master country list", code=198)

        dataset_vars = [col for col in df.columns if col not in {"ISO3", "year"}]
        for var in dataset_vars:
            if var not in possible_sources:
                raise sh.PipelineRuntimeError(
                    f"Variable {var} in file {file} is not in the src_specific_var_name column of the sources.csv file.",
                    code=198,
                )

    return {
        "clean_only_dirs": clean_only,
        "raw_only_dirs": raw_only,
        "checked_files": checked_files,
    }
__all__ = ["validate_inputs"]
