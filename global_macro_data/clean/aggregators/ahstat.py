from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_ahstat(
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    base_dir = raw_dir / "aggregators" / "AHSTAT"

    def _drop_one_based_rows(frame: pd.DataFrame, start: int, end: int | None = None) -> pd.DataFrame:
        stop = start if end is None else end
        idx = list(range(start - 1, stop))
        return frame.drop(frame.index.intersection(idx)).reset_index(drop=True)

    def _excel_letters(frame: pd.DataFrame) -> pd.DataFrame:
        rename_map: dict[object, str] = {}
        for idx, col in enumerate(frame.columns, start=1):
            label = ""
            current = idx
            while current:
                current, rem = divmod(current - 1, 26)
                label = chr(65 + rem) + label
            rename_map[col] = label
        return frame.rename(columns=rename_map)

    def _coerce_all_numeric(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    def _coerce_all_numeric_16g(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in out.columns:
            out[col] = out[col].map(
                lambda v: (
                    np.nan
                    if pd.isna(v) or str(v).strip() in {"", "<NA>", "nan", "None"}
                    else float(format(float(str(v).strip()), ".16g"))
                )
            )
        return out

    def _merge_master(master: pd.DataFrame, using: pd.DataFrame | None, keys: list[str]) -> pd.DataFrame:
        if using is None:
            return master.copy()
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

    chn_path = base_dir / "CHN.xlsx"
    chn = _read_excel_compat(chn_path, sheet_name="national_accounts").copy()
    for sheet in ["pop", "emp", "gov", "currency", "money"]:
        part = _read_excel_compat(chn_path, sheet_name=sheet).copy()
        chn = _merge_master(part, chn, ["year"])
    chn["ISO3"] = "CHN"
    chn["USDfx"] = pd.to_numeric(chn["USDfx"], errors="coerce") / 100

    jpn_path = base_dir / "JPN.xlsx"
    jpn = _read_excel_compat(jpn_path, sheet_name="national_accounts").copy()
    for sheet in ["pop", "finv", "money", "gov", "prices", "trade"]:
        part = _read_excel_compat(jpn_path, sheet_name=sheet).copy()
        jpn = _merge_master(part, jpn, ["year"])
    jpn["ISO3"] = "JPN"
    jpn = _sort_keys(jpn)
    prev_cpi = _lag_if_consecutive_year(jpn, "CPI")
    jpn["infl"] = ((pd.to_numeric(jpn["CPI"], errors="coerce") - prev_cpi) / prev_cpi) * 100
    jpn["pop"] = pd.to_numeric(jpn["pop"], errors="coerce") / 1_000_000
    jpn["USDfx"] = 1 / (pd.to_numeric(jpn["USDfx"], errors="coerce") / 100)

    kor_path = base_dir / "KOR.xlsx"
    kor = _read_excel_compat(kor_path, sheet_name="trade").copy()
    for sheet in ["emp", "expenditure"]:
        part = _read_excel_compat(kor_path, sheet_name=sheet).copy()
        kor = _merge_master(part, kor, ["year"])
    kor["ISO3"] = "KOR"

    rus1 = _excel_letters(_read_excel_compat(base_dir / "RUS_1.xlsx", sheet_name="9.3.3", header=None, dtype=str))
    rus1 = _drop_one_based_rows(rus1, 1, 13)
    rus1 = rus1.rename(columns={"A": "year", "B": "nGDP", "C": "cons"})[["year", "nGDP", "cons"]]
    rus1 = _drop_one_based_rows(rus1, 32, 37)
    rus1 = _coerce_all_numeric_16g(rus1)
    rus1 = rus1.loc[rus1["year"].notna()].reset_index(drop=True)
    rus1["ISO3"] = "RUS"

    rus2 = _excel_letters(_read_excel_compat(base_dir / "RUS_2.xlsx", sheet_name="9.1.1", header=None, dtype=str))
    rus2 = _drop_one_based_rows(rus2, 1, 10)
    rus2 = rus2.rename(columns={"B": "year", "M": "SUN_rGDP", "N": "RUS_rGDP"})[["year", "RUS_rGDP", "SUN_rGDP"]]
    rus2 = _drop_one_based_rows(rus2, 55, 63)
    rus2 = _coerce_all_numeric_16g(rus2)
    rus2 = rus2.loc[rus2["year"].notna()].reset_index(drop=True)
    rus = _merge_master(
        rus2[["year", "RUS_rGDP"]].rename(columns={"RUS_rGDP": "rGDP"}).assign(ISO3="RUS"),
        rus1,
        ["year", "ISO3"],
    )
    sun = rus2[["year", "SUN_rGDP"]].rename(columns={"SUN_rGDP": "rGDP"}).copy()
    sun["ISO3"] = "SUN"

    rus3 = _excel_letters(_read_excel_compat(base_dir / "RUS_3.xlsx", sheet_name="9.2.1", header=None, dtype=str))
    rus3 = _drop_one_based_rows(rus3, 1, 7)
    rus3 = rus3.rename(columns={"A": "year", "J": "rGDP_yoy"})[["year", "rGDP_yoy"]]
    rus3["year"] = rus3["year"].astype("string").str.slice(0, 4)
    yoy_values = rus3["rGDP_yoy"].astype("string").fillna("").str.strip()
    rus3 = rus3.loc[~((rus3.index > 0) & yoy_values.eq(""))].reset_index(drop=True)
    rus3 = _coerce_all_numeric_16g(rus3)
    rus3 = rus3.loc[rus3["year"].notna()].reset_index(drop=True)
    rus = _merge_master(rus3.assign(ISO3="RUS"), rus, ["year", "ISO3"])
    rus = _sort_keys(rus)
    rus_mask = rus["ISO3"].eq("RUS") & pd.to_numeric(rus["year"], errors="coerce").ge(1914)
    for idx in rus.index[rus_mask]:
        prev = rus.at[idx - 1, "rGDP"] if idx > 0 else np.nan
        yoy = rus.at[idx, "rGDP_yoy"]
        if pd.notna(prev) and pd.notna(yoy):
            rus.at[idx, "rGDP"] = prev * (1 + yoy / 100)
    rus = rus.drop(columns=["rGDP_yoy"])

    twn_nom = _excel_letters(_read_excel_compat(base_dir / "TWN.xlsx", sheet_name="Table0.1", header=None, dtype=str))
    twn_nom = twn_nom[["A", "G", "H", "I", "J"]].rename(
        columns={"A": "year", "G": "nGDP", "H": "cons", "I": "govexp", "J": "inv"}
    )
    twn_nom = _drop_one_based_rows(twn_nom, 1, 7)
    twn_nom = _drop_one_based_rows(twn_nom, 49, 55)
    twn_nom = _coerce_all_numeric_16g(twn_nom)
    twn_nom = twn_nom.loc[twn_nom["year"].notna()].reset_index(drop=True)

    twn_real = _excel_letters(_read_excel_compat(base_dir / "TWN.xlsx", sheet_name="Table0.2", header=None, dtype=str))
    twn_real = twn_real[["A", "G"]].rename(columns={"A": "year", "G": "rGDP"})
    twn_real = _drop_one_based_rows(twn_real, 1, 7)
    twn_real = _drop_one_based_rows(twn_real, 49, 55)
    twn_real = _coerce_all_numeric_16g(twn_real)
    twn_real = twn_real.loc[twn_real["year"].notna()].reset_index(drop=True)
    twn = _merge_master(twn_real, twn_nom, ["year"])
    twn["ISO3"] = "TWN"
    for col in [c for c in twn.columns if c not in {"year", "ISO3", "rGDP"}]:
        twn.loc[twn["year"].le(1948), col] = pd.to_numeric(twn.loc[twn["year"].le(1948), col], errors="coerce") / 1_000_000

    master = pd.concat([chn, jpn, kor, rus, sun, twn], ignore_index=True, sort=False)
    master["year"] = pd.to_numeric(master["year"], errors="coerce")
    master = master.loc[master["year"].notna()].copy()
    master = _sort_keys(master)
    for col in [c for c in master.columns if c not in {"ISO3", "year"}]:
        master[col] = pd.to_numeric(master[col], errors="coerce")
    for name in ["cons", "imports", "exports", "govexp", "govrev", "govtax", "finv", "inv"]:
        master[f"{name}_GDP"] = (master[name] / master["nGDP"]) * 100

    master = master.rename(columns={col: f"AHSTAT_{col}" for col in master.columns if col not in {"ISO3", "year"}})
    master = master.reindex(columns=AHSTAT_COLUMN_ORDER)
    master = _coerce_numeric_dtypes(master, AHSTAT_DTYPE_MAP)
    master["ISO3"] = master["ISO3"].astype(str)
    master = _sort_keys(master)

    out_path = clean_dir / "aggregators" / "AHSTAT" / "AHSTAT.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_ahstat"]
