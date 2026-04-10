from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_lund(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    path = raw_dir / "aggregators" / "LUND" / "LUND.xlsx"

    def _reshape_sheet(
        sheet_name: str,
        start_row: int,
        value_name: str,
        *,
        drop_cols: Iterable[int] = (),
        manual_names: dict[int, str],
    ) -> pd.DataFrame:
        df = pd.read_excel(path, sheet_name=sheet_name, header=None)
        df = df.iloc[start_row:].copy()
        df = df.drop(columns=list(drop_cols), errors="ignore")
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all").reset_index(drop=True)

        rename_map: dict[int, str] = {}
        first_col = int(df.columns[0])
        rename_map[first_col] = "year"
        for col in df.columns[1:]:
            col_int = int(col)
            if col_int in manual_names:
                rename_map[col_int] = manual_names[col_int]
                continue
            header_value = str(df.at[0, col]).strip()
            rename_map[col_int] = f"REER{re.sub(r'[^0-9A-Za-z]+', '', header_value)}"
        df = df.rename(columns=rename_map)
        df = df.iloc[1:].reset_index(drop=True).copy()
        df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        long = df.melt(id_vars="year", var_name="countryname", value_name=value_name)
        long["countryname"] = long["countryname"].astype(str).str.removeprefix("REER")
        long = long.loc[long["year"].notna()].copy()
        return long[["countryname", "year", value_name]]

    first = _reshape_sheet(
        "R1900",
        8,
        "first_REER",
        drop_cols=(1,),
        manual_names={10: "REERNetherlands", 17: "REERSwitzerland", 18: "REERGreatBritain"},
    )
    second = _reshape_sheet(
        "R1929",
        8,
        "second_REER",
        manual_names={4: "REERCzechoSlovakia", 13: "REERNetherlands", 20: "REERSwitzerland", 21: "REERGreatBritain"},
    )
    third = _reshape_sheet(
        "R1960",
        3,
        "third_REER",
        manual_names={
            4: "REERCzechoSlovakia",
            8: "REERGermany",
            9: "REEREastGermany",
            14: "REERNetherlands",
            21: "REERSwitzerland",
            22: "REERGreatBritain",
        },
    )
    fourth = _reshape_sheet(
        "R1999",
        1,
        "fourth_REER",
        manual_names={16: "REERNetherlands", 23: "REERSwitzerland", 24: "REERGreatBritain"},
    )

    master = first.merge(second, on=["countryname", "year"], how="outer")
    master = master.merge(third, on=["countryname", "year"], how="outer")
    master = master.merge(fourth, on=["countryname", "year"], how="outer")
    master["countryname"] = master["countryname"].astype(str).replace(
        {
            "Czechia": "Czech Republic",
            "CzechoSlovakia": "Czechoslovakia",
            "EastGermany": "Georgia",
            "GreatBritain": "United Kingdom",
            "Russia": "Russian Federation",
        }
    )

    lookup = _country_name_lookup(helper_dir)
    master["ISO3"] = master["countryname"].map(lookup)
    master = master.loc[master["ISO3"].notna()].drop(columns=["countryname"]).copy()
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    master = _sort_keys(master[["ISO3", "year", "fourth_REER", "third_REER", "second_REER", "first_REER"]])

    spliced = sh.splice(
        master,
        priority="fourth third second first",
        generate="REER",
        varname="REER",
        base_year=1999,
        method="chainlink",
        save="NO",
    )
    out = spliced[["ISO3", "year", "REER"]].rename(columns={"REER": "LUND_REER"}).copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int16")
    out["LUND_REER"] = pd.to_numeric(out["LUND_REER"], errors="coerce").astype("float32")
    out = _sort_keys(out)
    out_path = clean_dir / "aggregators" / "LUND" / "LUND.dta"
    _save_dta(out, out_path)
    return out
__all__ = ["clean_lund"]
