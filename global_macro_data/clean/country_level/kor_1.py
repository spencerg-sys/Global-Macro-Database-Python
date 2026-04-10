from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_kor_1(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    path = raw_dir / "country_level" / "KOR_1.xlsx"

    def _kor1_value(value: object) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if text in {"", "<NA>", "nan", "None"}:
            return np.nan
        return float(format(float(text), ".16g"))

    nominal = pd.read_excel(path, sheet_name="15.3.1 ", header=None).iloc[8:].copy()
    nominal = nominal.iloc[:-1].copy()
    nominal = nominal.loc[pd.to_numeric(nominal[0], errors="coerce").notna(), [0, 3, 6, 9]].copy()
    nominal.columns = ["year", "D", "G", "J"]
    nominal["year"] = pd.to_numeric(nominal["year"], errors="coerce")
    for col in ["D", "G", "J"]:
        nominal[col] = nominal[col].map(_kor1_value)
    colonial_mask = pd.to_numeric(nominal["year"], errors="coerce") <= 1940
    for col in ["D", "G", "J"]:
        nominal.loc[colonial_mask, col] = pd.to_numeric(nominal.loc[colonial_mask, col], errors="coerce") / 1000
    nominal.iloc[0, nominal.columns.get_loc("D")] = nominal.iloc[1]["D"] * (
        (pd.to_numeric(nominal.iloc[0]["G"], errors="coerce") + pd.to_numeric(nominal.iloc[0]["J"], errors="coerce"))
        / (pd.to_numeric(nominal.iloc[1]["G"], errors="coerce") + pd.to_numeric(nominal.iloc[1]["J"], errors="coerce"))
    )
    nominal = nominal.rename(columns={"D": "CS1_nGDP"})[["year", "CS1_nGDP"]].copy()

    real = pd.read_excel(path, sheet_name="15.3.2  ", header=None).iloc[8:].copy()
    real = real.iloc[:-5].copy()
    real = real.loc[pd.to_numeric(real[0], errors="coerce").notna(), [0, 3, 6, 9]].copy()
    real.columns = ["year", "D", "G", "J"]
    real["year"] = pd.to_numeric(real["year"], errors="coerce")
    for col in ["D", "G", "J"]:
        real[col] = real[col].map(_kor1_value)
    real.iloc[0, real.columns.get_loc("D")] = real.iloc[1]["D"] * (
        (pd.to_numeric(real.iloc[0]["G"], errors="coerce") + pd.to_numeric(real.iloc[0]["J"], errors="coerce"))
        / (pd.to_numeric(real.iloc[1]["G"], errors="coerce") + pd.to_numeric(real.iloc[1]["J"], errors="coerce"))
    )
    real = real.rename(columns={"D": "CS1_rGDP"})[["year", "CS1_rGDP"]].copy()

    pop = pd.read_excel(path, sheet_name="15.3.3 ", header=None).iloc[8:].copy()
    pop = pop.loc[pd.to_numeric(pop[0], errors="coerce").notna() & pd.to_numeric(pop[3], errors="coerce").notna(), [0, 3]].copy()
    pop.columns = ["year", "CS1_pop"]
    pop["year"] = pd.to_numeric(pop["year"], errors="coerce")
    pop["CS1_pop"] = pop["CS1_pop"].map(_kor1_value)

    out = pop.merge(nominal, on="year", how="left").merge(real, on="year", how="left")
    post_1953 = pd.to_numeric(out["year"], errors="coerce") >= 1953
    out.loc[post_1953, "CS1_rGDP"] = pd.to_numeric(out.loc[post_1953, "CS1_rGDP"], errors="coerce") * 1000
    out.loc[post_1953, "CS1_nGDP"] = pd.to_numeric(out.loc[post_1953, "CS1_nGDP"], errors="coerce") * 1000
    out["CS1_pop"] = pd.to_numeric(out["CS1_pop"], errors="coerce") / 1000
    out["CS1_rGDP_pc"] = pd.to_numeric(out["CS1_rGDP"], errors="coerce") / pd.to_numeric(out["CS1_pop"], errors="coerce")
    out["ISO3"] = "KOR"
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("int16")
    for col in ["CS1_pop", "CS1_nGDP", "CS1_rGDP"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    out["CS1_rGDP_pc"] = pd.to_numeric(out["CS1_rGDP_pc"], errors="coerce").astype("float32")
    out = out[["ISO3", "year", "CS1_pop", "CS1_nGDP", "CS1_rGDP", "CS1_rGDP_pc"]].copy()
    for col in ["CS1_pop", "CS1_nGDP", "CS1_rGDP"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    out["CS1_rGDP_pc"] = pd.to_numeric(out["CS1_rGDP_pc"], errors="coerce").astype("float32")
    out = _sort_keys(out)
    out_path = clean_dir / "country_level" / "KOR_1.dta"
    _save_dta(out, out_path)
    return out
__all__ = ["clean_kor_1"]
