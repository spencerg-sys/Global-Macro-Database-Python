from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_un(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    countrylist = _load_dta(helper_dir / "countrylist.dta")[["ISOnum", "ISO3"]].copy()
    countrylist["ISOnum"] = pd.to_numeric(countrylist["ISOnum"], errors="coerce")

    def _read_un_block(path: Path) -> pd.DataFrame:
        df = pd.read_excel(path, header=None, skiprows=3)
        year_cols = [f"y_{year}" for year in range(1970, 2021)]
        df = df.iloc[:, : 4 + len(year_cols)].copy()
        df.columns = ["A", "B", "C", "D"] + year_cols
        return df

    def _read_un_real_block(path: Path) -> pd.DataFrame:
        df = pd.read_excel(path, header=None, skiprows=5, nrows=3555)
        year_cols = [f"y_{year}" for year in range(1970, 2021)]
        df = df.iloc[:, : 4 + len(year_cols)].copy()
        df.columns = ["A", "B", "C", "D"] + year_cols
        return df

    def _fill_from_country(frame: pd.DataFrame, *, source_name: str, target_name: str, cols: list[str]) -> pd.DataFrame:
        out = frame.copy()
        source = out.loc[out["B"].astype(str) == source_name, ["year"] + cols].copy()
        source = source.groupby("year", as_index=False).max(numeric_only=False)
        for col in cols:
            mapping = source.set_index("year")[col] if col in source.columns else pd.Series(dtype="float64")
            target_mask = out["B"].astype(str) == target_name
            fill_values = pd.to_numeric(out.loc[target_mask, "year"], errors="coerce").map(mapping)
            fill_values = pd.to_numeric(fill_values, errors="coerce").astype("float32").astype("float64")
            current = pd.to_numeric(out.loc[target_mask, col], errors="coerce")
            out.loc[target_mask, col] = current.where(current.notna(), fill_values)
        return out

    def _wide_with_all_keys(frame: pd.DataFrame, *, value_name: str = "value") -> pd.DataFrame:
        keys = frame[["A", "B", "C", "year"]].drop_duplicates().reset_index(drop=True)
        values = (
            frame.pivot_table(
                index=["A", "B", "C", "year"],
                columns="D",
                values=value_name,
                aggfunc="first",
            )
            .reset_index()
        )
        values.columns.name = None
        return keys.merge(values, on=["A", "B", "C", "year"], how="left")

    # Nominal GDP block
    nominal = _read_un_block(raw_dir / "aggregators" / "UN" / "UN_nGDP.xlsx")
    nominal = nominal.melt(id_vars=["A", "B", "C", "D"], var_name="year", value_name="value")
    nominal["year"] = nominal["year"].astype(str).str.replace("y_", "", regex=False).astype("int32")
    keep_map = {
        "Gross Domestic Product (GDP)": "nGDP",
        "Final consumption expenditure": "cons",
        "Gross capital formation": "inv",
        "Gross fixed capital formation (including Acquisitions less disposals of valuables)": "finv",
        "Imports of goods and services": "imports",
        "Exports of goods and services": "exports",
    }
    nominal = nominal.loc[nominal["D"].isin(keep_map)].copy()
    nominal["D"] = nominal["D"].map(keep_map)
    nominal = _wide_with_all_keys(nominal)
    nominal["A"] = pd.to_numeric(nominal["A"], errors="coerce")
    nominal = nominal.merge(countrylist.rename(columns={"ISO3": "iso3"}), left_on="A", right_on="ISOnum", how="left")

    name = nominal["B"].astype(str)
    manual_iso3 = {
        "Cura\u00e7ao": "CUW",
        "Czechoslovakia (Former)": "CSK",
        "Ethiopia": "ETH",
        "Ethiopia (Former)": "ETH",
        "Former Netherlands Antilles": "ANT",
        "Kosovo": "XKX",
        "Montenegro": "MNE",
        "Russian Federation": "RUS",
        "Serbia": "SRB",
        "Sint Maarten (Dutch part)": "SXM",
        "South Sudan": "SSD",
        "Sudan": "SDN",
        "Sudan (Former)": "SDN",
        "U.R. of Tanzania: Mainland": "TZA",
        "USSR (Former)": "SUN",
        "Yugoslavia (Former)": "YUG",
    }
    for country_name, iso3 in manual_iso3.items():
        nominal.loc[name == country_name, "iso3"] = iso3

    value_cols = [col for col in nominal.columns if col not in {"A", "B", "C", "year", "ISOnum", "iso3"}]
    nominal = _fill_from_country(nominal, source_name="Ethiopia (Former)", target_name="Ethiopia", cols=value_cols)
    nominal = nominal.loc[nominal["B"].astype(str) != "Ethiopia (Former)"].copy()
    nominal = _fill_from_country(nominal, source_name="Sudan", target_name="Sudan (Former)", cols=value_cols)
    nominal = nominal.loc[nominal["B"].astype(str) != "Sudan"].copy()

    name = nominal["B"].astype(str)
    nominal = nominal.loc[~name.isin(["Zanzibar", "Yemen Arab Republic (Former)", "Yemen Democratic (Former)", "Montenegro", "South Sudan"])].copy()
    nominal = nominal.drop(columns=["A", "B", "C", "ISOnum"], errors="ignore")
    nominal = nominal.rename(columns={"iso3": "ISO3"})
    nominal = nominal.loc[nominal["ISO3"].notna()].copy()

    for col in [c for c in nominal.columns if c not in {"ISO3", "year"}]:
        nominal[col] = pd.to_numeric(nominal[col], errors="coerce") / 1_000_000

    nominal["cons_GDP"] = pd.to_numeric(nominal.get("cons"), errors="coerce") / pd.to_numeric(nominal.get("nGDP"), errors="coerce") * 100
    nominal["imports_GDP"] = pd.to_numeric(nominal.get("imports"), errors="coerce") / pd.to_numeric(nominal.get("nGDP"), errors="coerce") * 100
    nominal["exports_GDP"] = pd.to_numeric(nominal.get("exports"), errors="coerce") / pd.to_numeric(nominal.get("nGDP"), errors="coerce") * 100
    nominal["finv_GDP"] = pd.to_numeric(nominal.get("finv"), errors="coerce") / pd.to_numeric(nominal.get("nGDP"), errors="coerce") * 100
    nominal["inv_GDP"] = pd.to_numeric(nominal.get("inv"), errors="coerce") / pd.to_numeric(nominal.get("nGDP"), errors="coerce") * 100
    nominal = nominal.rename(columns={col: f"UN_{col}" for col in nominal.columns if col not in {"ISO3", "year"}})

    # Real GDP block
    real = _read_un_real_block(raw_dir / "aggregators" / "UN" / "UN_rGDP.xlsx")
    real = real.melt(id_vars=["A", "B", "C", "D"], var_name="year", value_name="value")
    real["year"] = real["year"].astype(str).str.replace("y_", "", regex=False).astype("int32")
    real_map = {
        "Gross Domestic Product (GDP)": "rGDP",
        "Final consumption expenditure": "rcons",
    }
    real["D"] = real["D"].map(real_map)
    real = real.loc[real["D"].isin(["rGDP", "rcons"])].copy()
    real = _wide_with_all_keys(real)
    real = real.rename(columns={"A": "ISOnum", "B": "countryname"})
    real["ISOnum"] = pd.to_numeric(real["ISOnum"], errors="coerce")
    real = real.loc[real["ISOnum"].notna()].copy()
    real = real.merge(countrylist, on="ISOnum", how="inner")
    real = real[["ISO3", "year", "rGDP", "rcons"]].copy()
    real = real.rename(columns={"rGDP": "UN_rGDP", "rcons": "UN_rcons"})
    real["UN_rGDP"] = pd.to_numeric(real["UN_rGDP"], errors="coerce") / (10**6)
    real["UN_rcons"] = pd.to_numeric(real["UN_rcons"], errors="coerce") / (10**6)

    master = nominal.merge(real, on=["ISO3", "year"], how="outer")

    # Population block
    pop = pd.read_excel(raw_dir / "aggregators" / "UN" / "UN_pop.xlsx", header=None, skiprows=16, nrows=290)
    pop = pop.loc[pop.iloc[:, 5].isin(["Country/Area", "Type"])].copy()
    pop = pop.drop(columns=[0, 1, 3, 5, 6]).reset_index(drop=True)
    header_row = pop.iloc[0].copy()
    rename_map: dict[int, str] = {2: "countryname", 4: "ISOnum"}
    for col in pop.columns:
        if col in {2, 4}:
            continue
        rename_map[col] = f"UN_pop{header_row[col]}"
    pop = pop.rename(columns=rename_map).iloc[1:].copy()
    pop = pop.melt(id_vars=["countryname", "ISOnum"], var_name="year", value_name="UN_pop")
    pop["year"] = pop["year"].astype(str).str.replace("UN_pop", "", regex=False)
    pop["ISOnum"] = pd.to_numeric(pop["ISOnum"], errors="coerce")
    pop["UN_pop"] = pd.to_numeric(pop["UN_pop"], errors="coerce")
    pop["year"] = pd.to_numeric(pop["year"], errors="coerce").astype("Int64")
    pop = pop.loc[pop["ISOnum"].notna()].copy()
    pop = pop.merge(countrylist, on="ISOnum", how="inner")
    pop = pop[["ISO3", "year", "UN_pop"]].copy()
    pop["UN_pop"] = pd.to_numeric(pop["UN_pop"], errors="coerce") / (10**3)

    master = master.merge(pop, on=["ISO3", "year"], how="outer")

    level_cols = [col for col in master.columns if col not in {"ISO3", "year", "UN_pop"} and not col.endswith("_GDP")]
    for col in level_cols:
        master.loc[master["ISO3"].astype(str) == "HRV", col] = pd.to_numeric(master.loc[master["ISO3"].astype(str) == "HRV", col], errors="coerce") / 7.5345
        master.loc[master["ISO3"].astype(str) == "VEN", col] = pd.to_numeric(master.loc[master["ISO3"].astype(str) == "VEN", col], errors="coerce") / (10**6)

    late_ven = pd.to_numeric(master["year"], errors="coerce") >= 2015
    for col in ["UN_exports", "UN_imports", "UN_exports_GDP", "UN_imports_GDP"]:
        if col in master.columns:
            master.loc[late_ven, col] = pd.NA

    for col in ["UN_exports", "UN_exports_GDP"]:
        if col in master.columns:
            master.loc[master["ISO3"].astype(str) == "CSK", col] = pd.to_numeric(master.loc[master["ISO3"].astype(str) == "CSK", col], errors="coerce").abs()

    for col in [c for c in ["UN_cons_GDP", "UN_imports_GDP", "UN_exports_GDP", "UN_finv_GDP", "UN_inv_GDP"] if c in master.columns]:
        master[col] = pd.to_numeric(master[col], errors="coerce").astype("float32")

    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int32")
    master = _sort_keys(master)
    ordered_cols = [
        "ISO3",
        "year",
        "UN_pop",
        "UN_rGDP",
        "UN_rcons",
        "UN_cons",
        "UN_exports",
        "UN_finv",
        "UN_imports",
        "UN_inv",
        "UN_nGDP",
        "UN_cons_GDP",
        "UN_imports_GDP",
        "UN_exports_GDP",
        "UN_finv_GDP",
        "UN_inv_GDP",
    ]
    master = master[[col for col in ordered_cols if col in master.columns]]
    out_path = clean_dir / "aggregators" / "UN" / "UN.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_un"]
