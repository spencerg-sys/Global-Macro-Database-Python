from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def combine_usdfx(
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
) -> pd.DataFrame:
    work = _build_splice_input(
        "USDfx",
        data_clean_dir=data_clean_dir,
        data_temp_dir=data_temp_dir,
        data_final_dir=data_final_dir,
    )
    work = work.loc[~((work["ISO3"].astype(str) == "ZWE") & (pd.to_numeric(work["year"], errors="coerce") >= 2008))].copy()
    spec = _parse_splice_spec("USDfx")
    result = sh.splice(
        work,
        priority=str(spec["priority"]),
        generate="USDfx",
        varname="USDfx",
        base_year=int(spec["base_year"]),
        method=str(spec["method"]),
        data_final_dir=_resolve(data_final_dir),
    )

    french_colonies = "SEN CIV BFA MLI NER TCD CAF CMR BEN TGO GIN GAB COG".split()
    copies: list[pd.DataFrame] = []
    for colony in french_colonies:
        fra = result.loc[(result["ISO3"].astype(str) == "FRA") & result["year"].between(1903, 1949)].copy()
        if fra.empty:
            continue
        fra = fra[[col for col in fra.columns if not col.endswith("_USDfx")]].copy()
        fra["ISO3"] = colony
        pre_1948 = pd.to_numeric(fra["year"], errors="coerce") <= 1947
        pre_1950 = pd.to_numeric(fra["year"], errors="coerce") <= 1949
        fra.loc[pre_1948, "USDfx"] = (
            pd.to_numeric(fra.loc[pre_1948, "USDfx"], errors="coerce").astype("float32") * np.float32(0.85)
        ).astype("float32")
        fra.loc[pre_1950, "USDfx"] = (
            pd.to_numeric(fra.loc[pre_1950, "USDfx"], errors="coerce").astype("float32") * np.float32(200)
        ).astype("float32")
        copies.append(fra)

    if copies:
        result = pd.concat([result] + copies, ignore_index=True, sort=False)
    result = _key_sort(result, ["ISO3", "year"])
    result["imputed"] = 0
    for colony in french_colonies:
        mask = (result["ISO3"].astype(str) == colony) & result["year"].between(1903, 1949)
        result.loc[mask, "imputed"] = 1
    result.loc[result["imputed"].eq(1), "source"] = "JST"
    result = result.drop(columns=["imputed"])

    mco = result.loc[(result["ISO3"].astype(str) == "FRA") & result["year"].between(1925, 2024)].copy()
    if not mco.empty:
        mco = mco[[col for col in mco.columns if not col.endswith("_USDfx")]].copy()
        mco["ISO3"] = "MCO"
        result = pd.concat([result, mco], ignore_index=True, sort=False)
        result = _key_sort(result, ["ISO3", "year"])
        mask = (result["ISO3"].astype(str) == "MCO") & result["year"].between(1925, 2024)
        result.loc[mask, "source"] = "JST"

    us_territories = "PRI ASM GUM UMI VIR".split()
    usa_copies: list[pd.DataFrame] = []
    for colony in us_territories:
        usa = result.loc[(result["ISO3"].astype(str) == "USA") & result["year"].between(1898, 2024)].copy()
        if usa.empty:
            continue
        usa = usa[[col for col in usa.columns if not col.endswith("_USDfx")]].copy()
        usa["ISO3"] = colony
        usa_copies.append(usa)

    if usa_copies:
        result = pd.concat([result] + usa_copies, ignore_index=True, sort=False)
    result = _key_sort(result, ["ISO3", "year"])
    result["imputed"] = 0
    for colony in us_territories:
        result.loc[result["ISO3"].astype(str) == colony, "imputed"] = 1
    result.loc[result["imputed"].eq(1), "source"] = "Tena"
    result.loc[(result["ISO3"].astype(str) == "VIR") & (pd.to_numeric(result["year"], errors="coerce") <= 1917), "USDfx"] = pd.NA
    result = result.drop(columns=["imputed"])
    result = _key_sort(result, ["ISO3", "year"])

    out_path = _resolve(data_final_dir) / "chainlinked_USDfx.dta"
    _save_dta(result, out_path)
    return result
__all__ = ["combine_usdfx"]
