from __future__ import annotations

import io
import re
from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd
import requests

from . import helpers as sh

PACKAGE_VERSION = "3.0.0"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_DATA_DIR = _REPO_ROOT / "data"

_DATA_BASES = (
    "https://gmd-releases.s3.ap-southeast-2.amazonaws.com/data",
)
_TIMEOUT_SECONDS = 60
_CACHE_DIR = Path.home() / ".global_macro_data"
_ID_COLS = ("ISO3", "year", "id", "countryname")

_APA_GMD = (
    "Müller, K., Xu, C., Lehbib, M., & Chen, Z. (2025). "
    "The Global Macro Database: A New International Macroeconomic Dataset "
    "(NBER Working Paper No. 33714)."
)
_APA_PACKAGE = (
    "Lehbib, M. & Müller, K. (2025). gmd: The Easy Way to Access the "
    "World's Most Comprehensive Macroeconomic Database. Working Paper."
)

VALID_VARIABLES = [
    "CA","CA_USD","CA_GDP","cbrate","cons_USD","cons","cons_GDP","CPI","deflator",
    "exports","exports_GDP","exports_USD","finv_USD","finv","finv_GDP","HPI","imports_GDP",
    "imports","imports_USD","infl","inv_GDP","inv","inv_USD","ltrate","M0","M1","M2",
    "M3","M4","nGDP_USD","nGDP","pop","REER","rGDP","rGDP_pc_USD","rGDP_pc","rGDP_USD",
    "strate","unemp","USDfx","BankingCrisis","SovDebtCrisis","CurrencyCrisis","cgovdebt",
    "cgovdebt_GDP","cgovdef","cgovdef_GDP","cgovexp","cgovexp_GDP","cgovrev","cgovrev_GDP",
    "cgovtax","cgovtax_GDP","gen_govdebt","gen_govdebt_GDP","gen_govdef","gen_govdef_GDP",
    "gen_govexp","gen_govexp_GDP","gen_govrev","gen_govrev_GDP","gen_govtax","gen_govtax_GDP",
    "govrev_GDP","govexp_GDP","govtax_GDP","govdef_GDP","govdebt_GDP","govrev","govexp",
    "govtax","govdef","govdebt",
]


class GMDCommandError(RuntimeError):
    def __init__(self, message: str, code: int = 498, data: Optional[pd.DataFrame] = None):
        super().__init__(message)
        self.code = code
        self.data = data


def _emit(*lines: str) -> None:
    for line in lines:
        print(line)


def _fail(*lines: str, code: int = 498) -> None:
    if lines:
        _emit(*lines)
        raise GMDCommandError(lines[-1], code=code)
    raise GMDCommandError("gmd failed", code=code)


def _fetch_from(relative_path: str, bases: Sequence[str]) -> requests.Response:
    errors: List[str] = []
    for base in bases:
        url = f"{base}/{relative_path}"
        try:
            response = requests.get(url, timeout=_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(f"Unable to load '{relative_path}'. {'; '.join(errors)}")


def _fetch_first(relative_path: str) -> requests.Response:
    return _fetch_from(relative_path, _DATA_BASES)


def _fetch_primary(relative_path: str) -> requests.Response:
    return _fetch_from(relative_path, (_DATA_BASES[0],))


def _fetch_secondary(relative_path: str) -> requests.Response:
    return _fetch_from(relative_path, _DATA_BASES)


def _local_data_path(relative_path: str) -> Path:
    return _PACKAGE_DATA_DIR.joinpath(*relative_path.split("/"))


def _read_csv(resp: requests.Response, **kwargs) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(resp.text), **kwargs)


def _read_dta(resp: requests.Response) -> pd.DataFrame:
    return pd.read_dta(io.BytesIO(resp.content), convert_categoricals=False)


def _read_csv_primary(relative_path: str, **kwargs) -> pd.DataFrame:
    local_path = _local_data_path(relative_path)
    if local_path.exists():
        return pd.read_csv(local_path, **kwargs)
    return _read_csv(_fetch_primary(relative_path), **kwargs)


def _read_dta_primary(relative_path: str) -> pd.DataFrame:
    local_path = _local_data_path(relative_path)
    if local_path.exists():
        return pd.read_dta(local_path, convert_categoricals=False)
    return _read_dta(_fetch_primary(relative_path))


def _tokens(value: Union[str, Sequence[str], None]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        bits = value.split()
        return [bit.strip() for bit in bits if bit.strip()]
    out: List[str] = []
    for item in value:
        if isinstance(item, str):
            out.extend(_tokens(item))
        else:
            out.append(str(item))
    return out


def _is_fast_yes(fast: Optional[Union[str, bool]]) -> bool:
    if isinstance(fast, str):
        return fast == "yes"
    return False


def _ensure_cache_dir() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_versions() -> List[str]:
    if not _CACHE_DIR.exists():
        return []
    out: List[str] = []
    pattern = re.compile(r"^GMD_(\d{4}_\d{2})\.dta$")
    for path in _CACHE_DIR.glob("GMD_*.dta"):
        match = pattern.match(path.name)
        if match:
            out.append(match.group(1))
    return sorted(set(out), reverse=True)


def _default_local_gmd_path() -> Optional[Path]:
    dta = _CACHE_DIR / "GMD.dta"
    if dta.exists():
        return dta
    packaged = _PACKAGE_DATA_DIR / "final" / "data_final.dta"
    if packaged.exists():
        return packaged
    return None


def _read_local_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".dta":
        return pd.read_dta(path, convert_categoricals=False)
    return pd.read_csv(path)

def _versions_df() -> pd.DataFrame:
    df = _read_csv_primary("helpers/versions.csv")
    if "versions" not in df.columns:
        raise RuntimeError("Malformed versions.csv")
    parts = df["versions"].astype(str).str.extract(r"(?P<year>\d{4})_(?P<month>\d{2})")
    df = df.assign(
        _year=pd.to_numeric(parts["year"], errors="coerce"),
        _month=pd.to_numeric(parts["month"], errors="coerce"),
    ).sort_values(["_year", "_month"], ascending=[False, False])
    return df.drop(columns=["_year", "_month"]).reset_index(drop=True)


def _varlist_df() -> pd.DataFrame:
    return _read_csv_primary("helpers/varlist.csv")


def _source_list_df() -> pd.DataFrame:
    return _read_csv_primary("helpers/source_list.csv")


def _bib_df() -> pd.DataFrame:
    return _read_csv_primary("helpers/bib_dataframe.csv")


def _country_df() -> pd.DataFrame:
    return _read_dta_primary("helpers/countrylist.dta")


def _country_tokens(value: Union[str, Sequence[str], None]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        bits = value.replace(",", " ").split()
        return [bit.strip() for bit in bits if bit.strip()]
    out: List[str] = []
    for item in value:
        if isinstance(item, str):
            out.extend(_country_tokens(item))
        else:
            out.append(str(item))
    return out


def _format_bibtex_for_print(entry: str) -> str:
    text = re.sub(r",\s*([a-zA-Z0-9_]+\s*=)", r",\n  \1", entry.strip())
    return re.sub(r"\}\s*$", "\n}", text)


def _print_var_table(df: pd.DataFrame) -> None:
    table = df.copy()
    for col in ("variable", "definition", "units"):
        if col not in table.columns:
            _fail(f"{col} not found", code=111)

    varlength = int(table["variable"].astype(str).str.len().max()) + 2
    deflength = int(table["definition"].astype(str).str.len().max()) + varlength + 2

    _emit("", "Available variables:", "")
    _emit("-" * 90)
    _emit(
        f"Variable{' ' * max(varlength - len('Variable'), 1)}"
        f"Definition{' ' * max(deflength - varlength - len('Definition'), 1)}Units"
    )
    _emit("-" * 90)
    for _, row in table.iterrows():
        vname = str(row["variable"])
        vdesc = str(row["definition"])
        vunits = str(row["units"])
        left = vname + (" " * max(varlength - len(vname), 1))
        mid = " " * max(deflength - len(left) - len(vdesc), 1)
        _emit(f"{left}{vdesc}{mid}{vunits}")
    _emit("-" * 90)


def _print_country_table(df: pd.DataFrame) -> None:
    if not {"countryname", "ISO3"}.issubset(df.columns):
        _fail(
            'Unable to access country list. Please raise an issue at {browse "https://github.com/KMueller-Lab/Global-Macro-Database-Python"}.',
            code=498,
        )
    _emit("", "Available countries:", "")
    _emit("-" * 90)
    _emit("ISO3 code  Country name")
    _emit("-" * 90)
    for _, row in df[["ISO3", "countryname"]].iterrows():
        _emit(f"{str(row['ISO3']):<10}{row['countryname']}")
    _emit("-" * 90)


def _summary(
    df: pd.DataFrame,
    anything: str,
    country: str,
    selected_version: str,
    version_opt: Optional[str],
    raw: bool,
    sources: Optional[str],
    fast: Optional[Union[str, bool]],
    saved_gmd: bool,
) -> None:
    df = df.dropna(axis=1, how="all")
    n_vars = len(df.columns)
    for ident in _ID_COLS:
        if ident in df.columns:
            n_vars -= 1
    if n_vars <= 0:
        _fail(f"The database has no data on {anything} for {country}", code=498)
    if len(df) <= 0:
        return

    _emit("Global Macro Database by Müller, Xu, Lehbib, and Chen (2025)")
    _emit('Website: {browse "https://www.globalmacrodata.com"}')
    _emit("")
    _emit("When using these data, please cite:")
    _emit("Use gmd(cite='GMD') for BibTeX and gmd(print='GMD') for APA-style citation.")
    _emit("")
    _emit("When using the Python package interface, please further cite:")
    _emit("Use gmd(cite='lehbib2025gmd') for BibTeX and gmd(print='package') for APA-style citation.")
    _emit("")

    if (fast is None or str(fast).strip() == "") and (not saved_gmd) and (not raw):
        _emit(
            f"To save the data locally for faster reloading, use gmd(version='{selected_version}', fast='yes')."
        )

    if raw or (sources is not None and str(sources) != ""):
        _emit(f"Final dataset: {len(df)} observations of {n_vars} variables")
    else:
        if n_vars > 1:
            _emit(f"Final dataset: {len(df)} observations for {n_vars} variables")
        else:
            _emit(f"Final dataset: {len(df)} observations for {n_vars} variable")

    if version_opt is not None and str(version_opt) != "":
        _emit(f"Version: {version_opt}")
    else:
        _emit(f"Version: {selected_version}")


def _normalize_source_name(source: str) -> str:
    source = source.strip()
    if len(source) == 7 and source.startswith("CS"):
        # Mirror the legacy alias rewrite exactly:
        # local sources = substr("`sources'", -3, 3) + "_" + substr("`sources'", 3, 1)
        return f"{source[-3:]}_{source[2]}"
    return source


def _strip_source_prefix_cols(df: pd.DataFrame, source: str) -> List[str]:
    pref = f"{source}_"
    out: List[str] = []
    for col in df.columns:
        if col in ("ISO3", "year"):
            continue
        if col.startswith(pref):
            out.append(col[len(pref):])
        else:
            out.append(col)
    return out


def get_available_versions() -> List[str]:
    try:
        return _versions_df()["versions"].astype(str).tolist()
    except RuntimeError:
        versions = _cache_versions()
        if versions:
            return versions
        if _default_local_gmd_path() is not None:
            return ["local"]
        raise


def get_current_version() -> str:
    return get_available_versions()[0]


def list_variables() -> None:
    _print_var_table(_varlist_df())


def list_countries() -> None:
    _print_country_table(_country_df())

def gmd(
    variables: Optional[Union[str, Sequence[str]]] = None,
    country: Optional[Union[str, Sequence[str]]] = None,
    version: Optional[str] = None,
    raw: bool = False,
    iso: bool = False,
    vars: Optional[Union[bool, str]] = None,
    sources: Optional[str] = None,
    cite: Optional[str] = None,
    print_option: Optional[str] = None,
    network: Optional[str] = None,
    fast: Optional[Union[str, bool]] = None,
    **kwargs,
) -> Optional[pd.DataFrame]:
    if "print" in kwargs:
        if print_option is not None:
            raise TypeError("Specify either print_option or print, not both.")
        print_option = kwargs.pop("print")
    if kwargs:
        raise TypeError(f"Unexpected keyword argument(s): {', '.join(kwargs.keys())}")

    if iso:
        country = "list"
    if vars is True:
        vars = "list"
    elif vars is False:
        vars = None

    anything_tokens = _tokens(variables)
    anything = " ".join(anything_tokens)
    word_count = len(anything_tokens)

    if isinstance(country, str):
        country_arg = country
    elif country is None:
        country_arg = ""
    else:
        country_arg = " ".join(_tokens(country))

    if print_option is not None:
        option = str(print_option).lower()
        if option == "gmd":
            _emit(_APA_GMD)
            return None
        if option == "package":
            _emit(_APA_PACKAGE)
            return None
        _fail("Invalid option for print(). valid arguments are 'GMD' or 'package'.", code=198)

    selected_version = ""
    available_versions: List[str] = []
    internet = "Yes"
    gmd_local_path: Optional[Path] = None
    saved_gmd = False

    try:
        versions_df = _versions_df()
        selected_version = str(versions_df.loc[0, "versions"])
        latest_version = selected_version
        available_versions = sorted(set(versions_df["versions"].astype(str).tolist()))

        if "version_package" in versions_df.columns:
            package_remote = str(versions_df.loc[0, "version_package"])
            if package_remote != PACKAGE_VERSION:
                _emit("There is a new version of the package. Update with `pip install -U global-macro-data`.")

        if version == "list":
            for ver in available_versions:
                _emit(ver)
            return None

        if version is not None and str(version) != "":
            version_tokens = _tokens(str(version))
            if len(version_tokens) != 1:
                _emit(
                    f"Version must either be one specific version ({selected_version}) or current."
                )
                return None
            requested = version_tokens[0]
            if requested in available_versions:
                selected_version = requested
            elif requested == "current":
                selected_version = latest_version
                _emit(f"Current version: {selected_version}")
            else:
                _fail(
                    f"Error: Version {requested} does not exist",
                    f"Available versions: {' '.join(available_versions)}",
                    code=498,
                )

    except RuntimeError as exc:
        if isinstance(exc, GMDCommandError):
            raise
        try:
            versions_gh = _read_csv(_fetch_secondary("helpers/versions.csv"))
            if "versions" in versions_gh.columns and "version_package" in versions_gh.columns:
                parts = versions_gh["versions"].astype(str).str.extract(r"(?P<year>\d{4})_(?P<month>\d{2})")
                versions_gh = versions_gh.assign(
                    _year=pd.to_numeric(parts["year"], errors="coerce"),
                    _month=pd.to_numeric(parts["month"], errors="coerce"),
                ).sort_values(["_year", "_month"], ascending=[False, False])
                package_remote = str(versions_gh.iloc[0]["version_package"])
                if package_remote != PACKAGE_VERSION:
                    _emit("There is a new version of the package. Update with `pip install -U global-macro-data`.")
                    _emit('Please raise an issue if the update does not work at {browse "https://github.com/KMueller-Lab/Global-Macro-Database-Python"}.')
        except RuntimeError:
            pass

        internet = "NaN" if (network is not None and str(network) != "") else "No"
        _emit("Error: Unable to access version information. Check internet connection.")
        _emit("Loading local version")

        local_default = _default_local_gmd_path()
        if local_default is None:
            _fail("Local version not found", code=498)

        gmd_local_path = local_default
        saved_gmd = True
        # The cached local fallback only guarantees a generic GMD.dta, not a dated vintage.
        selected_version = ""

    if internet == "No":
        if sources in ("load", "list"):
            _fail(
                "You need access to the internet in order to fetch the sources list",
                "If you have active internet access, rerun with network='yes'.",
                code=498,
            )
        elif sources is not None and str(sources) != "":
            _fail(
                f"You need access to the internet in order to fetch the {sources} data",
                "If you have active internet access, rerun with network='yes'.",
                code=498,
            )

        if raw:
            _fail(
                "You need access to the internet in order to fetch the raw data",
                "If you have active internet access, rerun with network='yes'.",
                code=498,
            )

        if cite == "load":
            _fail(
                "You need access to the internet in order to load the sources to cite",
                "If you have active internet access, rerun with network='yes'.",
                code=498,
            )
        elif cite is not None and str(cite) != "":
            _fail(
                f"You need access to the internet in order to cite {cite}",
                "If you have active internet access, rerun with network='yes'.",
                code=498,
            )

    if cite == "load":
        try:
            return _bib_df()
        except RuntimeError:
            _fail(
                'Unable to import the list of sources to cite. Please raise an issue at {browse "https://github.com/KMueller-Lab/Global-Macro-Database-Python"}.',
                code=498,
            )

    if cite is not None and str(cite) != "":
        cite_tokens = _tokens(str(cite))
        if len(cite_tokens) != 1:
            _fail("Only one citation can be retrieved at a time", code=498)
        key = cite_tokens[0]

        bib = _bib_df()
        if "source" not in bib.columns:
            _fail("source not found", code=111)
        if "citation" not in bib.columns:
            _fail("citation not found", code=111)
        mask = bib["source"].astype(str).str.lower() == key.lower()
        if not mask.any():
            _fail(
                f"Source '{key}' does not exist.",
                "To load the list of sources to cite, use gmd(cite='load').",
                code=498,
            )
        _emit(_format_bibtex_for_print(str(bib.loc[mask, "citation"].iloc[0])))
        return None

    if sources in ("load", "list"):
        if raw:
            _emit("Note: raw option is specified, but this is implicit when using the sources option.")
        try:
            source_df = _source_list_df()
        except RuntimeError:
            _fail(
                'Unable to load source list. Please raise an issue at {browse "https://github.com/KMueller-Lab/Global-Macro-Database-Python"}.',
                code=498,
            )
        if sources == "load":
            _emit("Imported the list of sources.")
            return source_df
        for src in sorted(set(source_df["source_name"].astype(str).tolist())):
            _emit(src)
        return None

    if sources is not None and str(sources) != "":
        if raw:
            _emit("Note: raw option is specified, but this is implicit when using the sources option.")

        src_name = str(sources)
        if len(src_name) == 7 and src_name.startswith("CS"):
            src_name = _normalize_source_name(src_name)
        src_name = " ".join(src_name.split())

        src_tokens = _tokens(src_name)
        if len(src_tokens) == 0:
            _fail(
                "Invalid source name",
                "To load the list of sources, use gmd(sources='load').",
                code=498,
            )
        if len(src_tokens) > 1:
            _fail("Warning: Please specify exactly one source.", code=498)
        src_name = src_tokens[0]

        try:
            src_df = _read_dta_primary(f"clean/combined/{src_name}.dta")
            source_load_ok = True
            corrected_source = ""
        except RuntimeError:
            source_load_ok = False
            corrected_source = ""
            try:
                source_list = _source_list_df()
            except RuntimeError:
                _emit('Unable to access variable list. Please raise an issue at {browse "https://github.com/KMueller-Lab/Global-Macro-Database-Python"}.')
            else:
                mask = source_list["source_name"].astype(str).str.lower() == src_name.lower()
                if mask.any():
                    corrected_source = str(source_list.loc[mask, "source_name"].iloc[0])
                else:
                    _fail(
                        "Invalid source name",
                        "To load the list of sources, use gmd(sources='load').",
                        code=498,
                    )
            src_df = pd.DataFrame()

        if source_load_ok or corrected_source:
            if corrected_source:
                src_name = corrected_source
            try:
                src_df = _read_dta_primary(f"clean/combined/{src_name}.dta")
            except RuntimeError:
                _fail(
                    f"Unable to load data for source '{src_name}'.",
                    "Please check your internet connection or report this issue.",
                    code=498,
                )

            if anything != "":
                src_col = f"{src_name}_{anything}"
                if src_col in src_df.columns:
                    keep_cols: List[str] = [col for col in ["ISO3", "year", src_col] if col in src_df.columns]
                    if "countryname" in src_df.columns:
                        keep_cols.append("countryname")
                    if "id" in src_df.columns:
                        keep_cols.append("id")
                    out = src_df.loc[:, keep_cols].copy()

                    if country_arg != "":
                        if "ISO3" in out.columns:
                            target = country_arg.upper()
                            out = out.loc[out["ISO3"].astype(str).str.upper() == target]
                        else:
                            _emit("Country code not valid, returning data for all countries.")
                            _emit("To print the list of countries, use gmd(country='list').")
                            _emit("To load the list of countries, use gmd(country='load').")
                    return out

                avail = _strip_source_prefix_cols(src_df, src_name)
                _emit(f"This source doesn't have data on {anything}. It has data on {' '.join(avail)}.")
                return None

            return src_df

    if vars == "load":
        try:
            return _varlist_df().copy()
        except RuntimeError:
            _fail(
                'Unable to access variable list. Please raise an issue at {browse "https://github.com/KMueller-Lab/Global-Macro-Database-Python"}.',
                code=498,
            )

    if vars == "list":
        try:
            var_df = _varlist_df().copy()
        except RuntimeError:
            _fail(
                'Unable to access variable list. Please raise an issue at {browse "https://github.com/KMueller-Lab/Global-Macro-Database-Python"}.',
                code=498,
            )
        _print_var_table(var_df)
        return None

    df: Optional[pd.DataFrame] = None
    if raw:
        if word_count != 1:
            _fail("Warning: Please specify exactly one variable.", code=498)
        try:
            df = _read_csv_primary(f"distribute/{anything}_{selected_version}.csv")
        except RuntimeError:
            try:
                var_df = _varlist_df()
                # Mirror the legacy exact-variable validation after importing varlist.csv:
                # it checks whether `anything` is a column name, not a value in the variable list.
                is_valid = anything in set(var_df.columns)
            except RuntimeError:
                is_valid = False
            if not is_valid:
                _fail("Specified variable is not valid.", code=498)
            _fail("Variable does not have raw data.", code=498)
        _emit(f"Loaded raw data on {anything}")

    if isinstance(country, str) and country.lower() in {"load", "list"}:
        mode = country.lower()
        _ensure_cache_dir()
        local_country = _CACHE_DIR / "countrylist.dta"
        if local_country.exists():
            cty_df = pd.read_dta(local_country, convert_categoricals=False)
            loaded_local = True
        else:
            try:
                cty_df = _country_df().copy()
            except RuntimeError:
                if mode == "load":
                    _fail(
                        'Unable to access country list. Please raise an issue at {browse "https://github.com/KMueller-Lab/Global-Macro-Database-Python"}.',
                        code=498,
                    )
                _fail("countryname not found", code=111)
            loaded_local = False
            if _is_fast_yes(fast):
                _emit("Saving countrylist dataframe locally")
                sh.write_dta(cty_df, local_country)
            else:
                return cty_df

        if mode == "load":
            return cty_df
        if loaded_local or _is_fast_yes(fast):
            _print_country_table(cty_df)
            return None
        return cty_df

    check_id = anything.lower()
    if check_id in {"iso3", "year", "id", "countryname"}:
        _fail(
            f"{anything} is an identifying variable loaded in the dataset, specify common variables",
            "To print the list of variables, use gmd(vars='list').",
            "To load the list of variables, use gmd(vars='load').",
            code=498,
        )

    if not raw:
        _ensure_cache_dir()
        if gmd_local_path is None:
            local_version = _CACHE_DIR / f"GMD_{selected_version}.dta"
            if local_version.exists():
                saved_gmd = True
                df = pd.read_dta(local_version, convert_categoricals=False)
            elif _is_fast_yes(fast):
                try:
                    df_remote = _read_dta_primary(f"distribute/GMD_{selected_version}.dta")
                except RuntimeError:
                    _fail(
                        'Unable to load the data. Please raise an issue at {browse "https://github.com/KMueller-Lab/Global-Macro-Database-Python"}.',
                        code=498,
                    )
                sh.write_dta(df_remote, local_version)
                sh.write_dta(df_remote, _CACHE_DIR / "GMD.dta")
                _emit(f"GMD dataset loaded and saved locally in {_CACHE_DIR}.")
                df = pd.read_dta(local_version, convert_categoricals=False)
            else:
                try:
                    df = _read_dta_primary(f"distribute/GMD_{selected_version}.dta")
                except RuntimeError:
                    _fail(
                        'Unable to load the data. Please raise an issue at {browse "https://github.com/KMueller-Lab/Global-Macro-Database-Python"}.',
                        code=498,
                    )
        else:
            df = _read_local_df(gmd_local_path)

    if df is None:
        raise GMDCommandError("No data loaded", code=498)

    if anything != "" and not raw:
        invalid_vars = [var for var in anything_tokens if var not in df.columns]
        if invalid_vars:
            if len(invalid_vars) == 1:
                _emit(f"{invalid_vars[0]} is not a valid variable code")
            else:
                _emit(f"{' '.join(invalid_vars)} are not valid variable codes")
            _fail(
                "To print the list of variables, use gmd(vars='list').",
                "To load the list of variables, use gmd(vars='load').",
                code=498,
            )

        keep_cols = list(dict.fromkeys(col for col in ["ISO3", "year", "id", "countryname"] + anything_tokens if col in df.columns))
        df = df.loc[:, keep_cols].copy()
        valid_count = df[anything_tokens].notna().sum(axis=1)
        if {"ISO3", "year"}.issubset(df.columns):
            ordered = df.assign(_valid_count=valid_count).sort_values(["ISO3", "year"])
            mask = ordered.groupby("ISO3", sort=False)["_valid_count"].cumsum() > 0
            df = ordered.loc[mask].drop(columns=["_valid_count"])
        else:
            df = df.assign(_valid_count=valid_count).drop(columns=["_valid_count"])

    if country_arg != "":
        country_clean = " ".join(_country_tokens(country_arg.upper()))
        c_tokens = _country_tokens(country_clean)
        if len(c_tokens) == 1:
            iso_code = c_tokens[0]
            if "ISO3" not in df.columns or not (df["ISO3"].astype(str).str.upper() == iso_code).any():
                _fail(
                    "Country code is invalid or no data for this country in source.",
                    "To print the list of countries, use gmd(country='list').",
                    "To load the list of countries, use gmd(country='load').",
                    code=498,
                )
            df = df.loc[df["ISO3"].astype(str).str.upper() == iso_code]
        else:
            if len(c_tokens) > 0 and "ISO3" not in df.columns:
                _fail(
                    "Country code is invalid or no data for this country in source.",
                    "To print the list of countries, use gmd(country='list').",
                    "To load the list of countries, use gmd(country='load').",
                    code=498,
                )
            iso_series = df["ISO3"].astype(str).str.upper() if "ISO3" in df.columns else pd.Series("", index=df.index)
            keep_mask = pd.Series(False, index=df.index)
            invalid: List[str] = []
            for iso_code in c_tokens:
                one = iso_series == iso_code
                if one.any():
                    keep_mask = keep_mask | one
                else:
                    invalid.append(iso_code)

            if invalid:
                inv = " ".join(invalid)
                if len(invalid) == 1:
                    _emit(f"{inv} is not a valid ISO3 code")
                else:
                    _emit(f"{inv} are not valid ISO3 codes")
                _emit("To print the list of countries, use gmd(country='list').")
                _emit("To load the list of countries, use gmd(country='load').")
                df = df.loc[keep_mask]
                raise GMDCommandError("Invalid ISO3 code", code=498, data=df.dropna(axis=1, how="all"))

            df = df.loc[keep_mask]

    _summary(
        df=df,
        anything=anything,
        country=country_arg,
        selected_version=selected_version,
        version_opt=version,
        raw=raw,
        sources=sources,
        fast=fast,
        saved_gmd=saved_gmd,
    )

    return df.dropna(axis=1, how="all")
