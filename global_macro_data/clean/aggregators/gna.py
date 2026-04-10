from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_gna(
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)
    west_dir = raw_dir / "aggregators" / "Groningen" / "western_europe"
    latin_path = raw_dir / "aggregators" / "Groningen" / "latin_america" / "hna_latam_10.xls"

    def _reshape_year_row(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        start_year = pd.to_numeric(frame.iloc[3, 3], errors="coerce")
        if pd.isna(start_year):
            raise ValueError(f"Could not parse first year for {prefix}")
        work = frame.loc[frame.iloc[:, 1].astype(str).eq("GDP")].copy()
        if work.empty:
            return pd.DataFrame(columns=["year", prefix])
        work = work.iloc[[0], 3:].copy()
        year = int(start_year)
        work.columns = [f"{prefix}{year + offset}" for offset in range(work.shape[1])]
        long = work.melt(var_name="variable", value_name=prefix)
        long["year"] = pd.to_numeric(long["variable"].astype(str).str.replace(prefix, "", regex=False), errors="coerce")
        long[prefix] = pd.to_numeric(long[prefix], errors="coerce")
        return long[["year", prefix]]

    def _merge_union(master: pd.DataFrame, using: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        master_idx = master.set_index(keys)
        using_idx = using.set_index(keys)
        union_index = master_idx.index.union(using_idx.index)
        out = master_idx.reindex(union_index)
        using_idx = using_idx.reindex(union_index)
        for col in using_idx.columns:
            if col in out.columns:
                out[col] = out[col].combine_first(using_idx[col])
            else:
                out[col] = using_idx[col]
        return out.reset_index()

    real_frames: list[pd.DataFrame] = []
    nominal_frames: list[pd.DataFrame] = []
    for path in sorted(west_dir.glob("*.xls")):
        iso3 = path.stem[4:7].upper()

        if path.name not in {"hna_ger_09.xls", "hna_ita_09.xls"}:
            nominal_raw = _read_excel_compat(path, sheet_name="VA", header=None)
            nominal = _reshape_year_row(nominal_raw, "GNA_nGDP_LCU")
            nominal["ISO3"] = iso3
            nominal_frames.append(nominal[["ISO3", "year", "GNA_nGDP_LCU"]])

        sheet = "VA-Ki" if path.name == "hna_fra_09.xls" else "VA-K"
        value_col = "GNA_rGDP_LCU_index" if path.name == "hna_fra_09.xls" else "GNA_rGDP_LCU"
        real_raw = _read_excel_compat(path, sheet_name=sheet, header=None)
        real = _reshape_year_row(real_raw, value_col)
        real["ISO3"] = iso3
        real_frames.append(real[["ISO3", "year", value_col]])

    n_gdp = pd.concat(nominal_frames, ignore_index=True, sort=False) if nominal_frames else pd.DataFrame()
    r_gdp = pd.concat(real_frames, ignore_index=True, sort=False) if real_frames else pd.DataFrame()
    west = r_gdp.merge(n_gdp, on=["ISO3", "year"], how="outer")
    west["ISO3"] = west["ISO3"].replace({"GER": "DEU"})
    west = _drop_rows_with_all_missing(west, keys=("ISO3", "year"))
    fra_mask = west["ISO3"].eq("FRA")
    fin_mask = west["ISO3"].eq("FIN")
    for col in ["GNA_nGDP_LCU", "GNA_rGDP_LCU"]:
        if col in west.columns:
            west.loc[fra_mask, col] = pd.to_numeric(west.loc[fra_mask, col], errors="coerce") / 100
            west.loc[fin_mask, col] = pd.to_numeric(west.loc[fin_mask, col], errors="coerce") / 1000

    eur_fx = _load_dta(helper_dir / "EUR_irrevocable_FX.dta")[["ISO3", "EUR_irrevocable_FX"]].drop_duplicates().copy()
    west = west.merge(eur_fx, on="ISO3", how="left", indicator=True)
    merge_mask = west["_merge"].eq("both")
    for col in ["GNA_nGDP_LCU", "GNA_rGDP_LCU"]:
        if col in west.columns:
            west.loc[merge_mask, col] = (
                pd.to_numeric(west.loc[merge_mask, col], errors="coerce")
                / pd.to_numeric(west.loc[merge_mask, "EUR_irrevocable_FX"], errors="coerce")
            )
    west = west.drop(columns=["EUR_irrevocable_FX", "_merge"])

    country_lookup = _country_name_lookup(helper_dir)
    latin_frames: list[pd.DataFrame] = []
    for country in ["Argentina", "Bolivia", "Brasil", "Chile", "Colombia", "Costa Rica", "Mexico", "Peru", "Venezuela"]:
        frame = _read_excel_compat(latin_path, sheet_name=country, header=None).astype("string")
        frame = frame.iloc[3:].reset_index(drop=True)
        frame = frame.dropna(axis=1, how="all")
        frame = frame.dropna(axis=0, how="all").reset_index(drop=True)
        year_headers = [pd.to_numeric(value, errors="coerce") for value in frame.iloc[0, 1:].tolist()]
        rename_map: dict[object, str] = {frame.columns[0]: "A"}
        for idx, col in enumerate(frame.columns[1:]):
            value = year_headers[idx]
            if pd.notna(value):
                rename_map[col] = f"GNA_rGDP_LCU{int(value)}"
        frame = frame.rename(columns=rename_map)
        keep_cols = ["A"] + [col for col in frame.columns if str(col).startswith("GNA_rGDP_LCU")]
        frame = frame[keep_cols].copy()
        frame = frame.loc[frame["A"].astype(str).str.contains("GDP", na=False)].copy()
        if frame.empty:
            continue
        long = frame.melt(id_vars="A", var_name="variable", value_name="GNA_rGDP_LCU")
        long["year"] = pd.to_numeric(long["variable"].astype(str).str.replace("GNA_rGDP_LCU", "", regex=False), errors="coerce")
        long["GNA_rGDP_LCU"] = pd.to_numeric(long["GNA_rGDP_LCU"], errors="coerce")
        long["countryname"] = "Brazil" if country == "Brasil" else country
        long["ISO3"] = long["countryname"].map(country_lookup)
        latin_frames.append(long.loc[long["ISO3"].notna(), ["ISO3", "year", "GNA_rGDP_LCU"]])

    latin = pd.concat(latin_frames, ignore_index=True, sort=False) if latin_frames else pd.DataFrame(columns=["ISO3", "year", "GNA_rGDP_LCU"])
    master = _merge_union(latin, west, ["ISO3", "year"])
    master = master.rename(
        columns={
            "GNA_nGDP_LCU": "GNA_nGDP",
            "GNA_rGDP_LCU": "GNA_rGDP",
            "GNA_rGDP_LCU_index": "GNA_rGDP_index",
        }
    )
    master = master[["ISO3", "year", "GNA_rGDP", "GNA_rGDP_index", "GNA_nGDP"]].copy()
    master["ISO3"] = master["ISO3"].astype(str)
    master["year"] = pd.to_numeric(master["year"], errors="coerce")
    master = master.loc[master["year"].notna()].copy()
    master = _sort_keys(master)
    master = _coerce_numeric_dtypes(master, GNA_DTYPE_MAP)

    out_path = clean_dir / "aggregators" / "Groningen" / "GNA.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_gna"]
