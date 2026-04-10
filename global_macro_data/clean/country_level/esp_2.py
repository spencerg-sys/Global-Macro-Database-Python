from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_esp_2(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    path_a = raw_dir / "country_level" / "ESP_2a.xlsx"
    path_b = raw_dir / "country_level" / "ESP_2b.xlsx"
    path_c = raw_dir / "country_level" / "ESP_2c.xlsx"
    path_e = raw_dir / "country_level" / "ESP_2e.xlsx"

    def _esp2_value(value: object) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if text in {"", "<NA>", "nan", "None"}:
            return np.nan
        try:
            return float(format(float(text), ".16g"))
        except (TypeError, ValueError):
            return np.nan

    def _wide_header_pair(
        path: Path,
        *,
        sheet: str,
        usecols: str,
        skiprows: int,
        nrows: int,
        value_row: int,
        value_name: str,
    ) -> pd.DataFrame:
        frame = _read_excel_compat(path, sheet_name=sheet, header=None, usecols=usecols, skiprows=skiprows, nrows=nrows, dtype=str)
        years = pd.to_numeric(pd.Series(frame.iloc[0].to_list()), errors="coerce")
        values = pd.Series(frame.iloc[value_row].to_list()).map(_esp2_value)
        out = pd.DataFrame({"ISO3": "ESP", "year": years, value_name: values})
        return out.loc[out["year"].notna()].copy()

    master = _wide_header_pair(
        path_a,
        sheet="1500-2010_A",
        usecols="N:TC",
        skiprows=2,
        nrows=2,
        value_row=1,
        value_name="infl",
    )

    pop_pre = _wide_header_pair(
        path_b,
        sheet="Cuadro_3",
        usecols="M:VM",
        skiprows=2,
        nrows=2,
        value_row=1,
        value_name="pop",
    )
    master = pop_pre.merge(master, on=["ISO3", "year"], how="outer")

    pop_post = _read_excel_compat(
        path_b,
        sheet_name="Cuadro_2",
        header=None,
        usecols="M:GA",
        skiprows=2,
        nrows=3,
        dtype=str,
    )
    pop_post = pop_post.drop(index=1).reset_index(drop=True)
    pop_post = pd.DataFrame(
        {
            "ISO3": "ESP",
            "year": pd.to_numeric(pd.Series(pop_post.iloc[0].to_list()), errors="coerce"),
            "pop": pd.Series(pop_post.iloc[1].to_list()).map(_esp2_value),
        }
    )
    pop_post = pop_post.loc[pop_post["year"].notna()].copy()
    master = pop_post.merge(master, on=["ISO3", "year"], how="outer")
    if {"pop_x", "pop_y"}.issubset(master.columns):
        master["pop"] = pd.to_numeric(master["pop_x"], errors="coerce").where(
            pd.to_numeric(master["pop_x"], errors="coerce").notna(),
            pd.to_numeric(master["pop_y"], errors="coerce"),
        )
        master = master.drop(columns=["pop_x", "pop_y"], errors="ignore")

    national_accounts = _read_excel_compat(
        path_c,
        sheet_name="Prados-Escosura_A",
        header=None,
        usecols="B:GA",
        skiprows=2,
        nrows=17,
        dtype=str,
    )
    na_years = pd.to_numeric(pd.Series(national_accounts.iloc[0, 11:].to_list()), errors="coerce")
    na_map = {3: "nGDP", 4: "cons", 7: "inv", 8: "finv", 15: "exports", 16: "imports"}
    na_frames: list[pd.DataFrame] = []
    for row_idx, name in na_map.items():
        na_frames.append(
            pd.DataFrame(
                {
                    "ISO3": "ESP",
                    "year": na_years,
                    f"CS2_{name}": pd.Series(national_accounts.iloc[row_idx, 11:].to_list()).map(_esp2_value),
                }
            ).loc[na_years.notna()].copy()
        )
    na_master = na_frames[0]
    for current in na_frames[1:]:
        na_master = na_master.merge(current, on=["ISO3", "year"], how="outer")
    master = na_master.merge(master, on=["ISO3", "year"], how="outer")

    real_gdp = _read_excel_compat(
        path_c,
        sheet_name="PIB_A_1850-2020",
        header=None,
        usecols="M:FU",
        skiprows=2,
        nrows=21,
        dtype=str,
    )
    rgdp_years = pd.to_numeric(pd.Series(real_gdp.iloc[0].to_list()), errors="coerce")
    rgdp = pd.DataFrame(
        {
            "ISO3": "ESP",
            "year": rgdp_years,
            "CS2_rGDP": pd.Series(real_gdp.iloc[8].to_list()).map(_esp2_value),
            "CS2_rGDP_pc": pd.Series(real_gdp.iloc[20].to_list()).map(_esp2_value),
        }
    )
    rgdp = rgdp.loc[rgdp["year"].notna()].copy()
    master = rgdp.merge(master, on=["ISO3", "year"], how="outer")

    rgdp_index = _wide_header_pair(
        path_c,
        sheet="PIB_A_1277-1850",
        usecols="M:VN",
        skiprows=2,
        nrows=2,
        value_row=1,
        value_name="index_rGDP_pc",
    )
    master = rgdp_index.merge(master, on=["ISO3", "year"], how="outer")
    master = sh.splice(master, priority="CS2 index", generate="rGDP_pc", varname="rGDP_pc", method="chainlink", base_year=2010, save="NO")
    master = master.drop(columns=["CS2_rGDP_pc", "index_rGDP_pc"], errors="ignore")

    cbrate_raw = _read_excel_compat(
        path_e,
        sheet_name="Cuadro_1b",
        header=None,
        usecols="M:ACP",
        skiprows=3,
        nrows=3,
        dtype=str,
    )
    dates = pd.Series(cbrate_raw.iloc[0].to_list(), dtype="string")
    cbrate = pd.DataFrame(
        {
            "ISO3": "ESP",
            "date": dates,
            "CS2_cbrate_1": pd.Series(cbrate_raw.iloc[1].to_list()).map(_esp2_value),
            "CS2_cbrate_2": pd.Series(cbrate_raw.iloc[2].to_list()).map(_esp2_value),
        }
    )
    cbrate["year"] = pd.to_numeric(cbrate["date"].str.slice(0, 4), errors="coerce")
    cbrate["month"] = pd.to_numeric(cbrate["date"].str.slice(5, 7), errors="coerce")
    cbrate = cbrate.loc[cbrate["year"].notna(), ["ISO3", "year", "month", "CS2_cbrate_1", "CS2_cbrate_2"]].copy()
    cbrate = cbrate.sort_values(["year", "month"], kind="mergesort").groupby(["ISO3", "year"], sort=False).tail(1).copy()
    cbrate["CS2_cbrate"] = pd.to_numeric(cbrate["CS2_cbrate_1"], errors="coerce").where(
        pd.to_numeric(cbrate["CS2_cbrate_1"], errors="coerce").notna(),
        pd.to_numeric(cbrate["CS2_cbrate_2"], errors="coerce"),
    )
    cbrate = cbrate[["ISO3", "year", "CS2_cbrate"]].copy()
    master = cbrate.merge(master, on=["ISO3", "year"], how="outer")

    master["CS2_rGDP"] = pd.to_numeric(master["CS2_rGDP"], errors="coerce").where(
        pd.to_numeric(master["CS2_rGDP"], errors="coerce").notna(),
        pd.to_numeric(master["rGDP_pc"], errors="coerce") * pd.to_numeric(master["pop"], errors="coerce"),
    )

    work = master.rename(columns={col: col[4:] for col in master.columns if col.startswith("CS2_")}).copy()
    n_gdp = pd.to_numeric(work["nGDP"], errors="coerce")
    work["cons_GDP"] = pd.to_numeric(work["cons"], errors="coerce") / n_gdp * 100
    work["imports_GDP"] = pd.to_numeric(work["imports"], errors="coerce") / n_gdp * 100
    work["exports_GDP"] = pd.to_numeric(work["exports"], errors="coerce") / n_gdp * 100
    work["finv_GDP"] = pd.to_numeric(work["finv"], errors="coerce") / n_gdp * 100
    work["inv_GDP"] = pd.to_numeric(work["inv"], errors="coerce") / n_gdp * 100
    work = work.rename(columns={col: f"CS2_{col}" for col in work.columns if col not in {"ISO3", "year"}})
    work["CS2_pop"] = pd.to_numeric(work["CS2_pop"], errors="coerce") / 1_000_000
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("int32")
    for col in ["CS2_rGDP", "CS2_cons", "CS2_exports", "CS2_finv", "CS2_imports", "CS2_inv", "CS2_nGDP", "CS2_pop", "CS2_infl"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").astype("float64")
    for col in ["CS2_cbrate", "CS2_rGDP_pc", "CS2_cons_GDP", "CS2_imports_GDP", "CS2_exports_GDP", "CS2_finv_GDP", "CS2_inv_GDP"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").astype("float32")
    work = work[
        [
            "ISO3",
            "year",
            "CS2_cbrate",
            "CS2_rGDP",
            "CS2_cons",
            "CS2_exports",
            "CS2_finv",
            "CS2_imports",
            "CS2_inv",
            "CS2_nGDP",
            "CS2_pop",
            "CS2_infl",
            "CS2_rGDP_pc",
            "CS2_cons_GDP",
            "CS2_imports_GDP",
            "CS2_exports_GDP",
            "CS2_finv_GDP",
            "CS2_inv_GDP",
        ]
    ].copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("int32")
    for col in ["CS2_rGDP", "CS2_cons", "CS2_exports", "CS2_finv", "CS2_imports", "CS2_inv", "CS2_nGDP", "CS2_pop", "CS2_infl"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").astype("float64")
    for col in ["CS2_cbrate", "CS2_rGDP_pc", "CS2_cons_GDP", "CS2_imports_GDP", "CS2_exports_GDP", "CS2_finv_GDP", "CS2_inv_GDP"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").astype("float32")
    work = _sort_keys(work)
    out_path = clean_dir / "country_level" / "ESP_2.dta"
    _save_dta(work, out_path)
    return work
__all__ = ["clean_esp_2"]
