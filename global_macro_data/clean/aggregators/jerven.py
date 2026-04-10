from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_jerven(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)

    def _jerven_f32(values: object) -> pd.Series | float:
        numeric = pd.to_numeric(values, errors="coerce")
        if isinstance(numeric, pd.Series):
            return numeric.astype("float32").astype("float64")
        return float(np.float32(numeric)) if pd.notna(numeric) else np.nan

    def _merge_current_master(current: pd.DataFrame, master: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        overlap = [col for col in current.columns if col not in keys and col in master.columns]
        merged = current.merge(master, on=keys, how="outer", suffixes=("_master", "_using"))
        for col in overlap:
            merged[col] = pd.to_numeric(merged[f"{col}_master"], errors="coerce").where(
                pd.to_numeric(merged[f"{col}_master"], errors="coerce").notna(),
                pd.to_numeric(merged[f"{col}_using"], errors="coerce"),
            )
            merged = merged.drop(columns=[f"{col}_master", f"{col}_using"], errors="ignore")
        return merged

    cpi = _load_dta(raw_dir / "aggregators" / "JERVEN" / "cpi_inflation.dta")
    cpi["infl"] = pd.to_numeric(cpi["inflation_frankema_waijenburg"], errors="coerce").where(
        pd.to_numeric(cpi["inflation_frankema_waijenburg"], errors="coerce").notna(),
        pd.to_numeric(cpi["inflation_reinhard_rogoff"], errors="coerce"),
    )
    master = cpi[["year", "iso", "infl"]].copy()

    for filename in ["FISCAL_PANEL_V4.dta", "FISCAL_PANEL_V4_SOMDJI.dta"]:
        current = _load_dta(raw_dir / "aggregators" / "JERVEN" / filename).copy()
        current["govtax"] = _jerven_f32(
            pd.to_numeric(current["INDIRECT_NOMINAL"], errors="coerce") + pd.to_numeric(current["DIRECT_NOMINAL"], errors="coerce")
        )
        current["govrev"] = _jerven_f32(
            pd.to_numeric(current["govtax"], errors="coerce")
            + pd.to_numeric(current["NONTAX_ORDINARY_NOMINAL"], errors="coerce")
            + pd.to_numeric(current["EXTRAORDINARY_NOMINAL"], errors="coerce")
            + pd.to_numeric(current["RESOURCES_NOMINAL"], errors="coerce")
        )
        current = current[["year", "iso", "govrev", "govtax", "POPULATION"]].copy()
        master = _merge_current_master(current, master, ["year", "iso"])

    master = master.rename(columns={"iso": "ISO3", "POPULATION": "pop"})
    master["pop"] = pd.to_numeric(master["pop"], errors="coerce") / 1_000_000
    master["govrev"] = _jerven_f32(pd.to_numeric(master["govrev"], errors="coerce") / 1_000_000)
    master["govtax"] = _jerven_f32(pd.to_numeric(master["govtax"], errors="coerce") / 1_000_000)

    for col in ["govrev", "govtax"]:
        def _replace_jerven(mask: pd.Series, expr: pd.Series) -> None:
            master.loc[mask, col] = _jerven_f32(expr)

        _replace_jerven(master["year"].lt(1960) & master["ISO3"].eq("DZA"), pd.to_numeric(master.loc[master["year"].lt(1960) & master["ISO3"].eq("DZA"), col], errors="coerce") / 100)
        _replace_jerven(master["ISO3"].eq("ZMB"), pd.to_numeric(master.loc[master["ISO3"].eq("ZMB"), col], errors="coerce") * 1000)
        _replace_jerven(master["ISO3"].eq("UGA") & master["year"].le(1965), pd.to_numeric(master.loc[master["ISO3"].eq("UGA") & master["year"].le(1965), col], errors="coerce") * 20)
        _replace_jerven(master["ISO3"].eq("TUN") & master["year"].le(1953), pd.to_numeric(master.loc[master["ISO3"].eq("TUN") & master["year"].le(1953), col], errors="coerce") / 1000)
        _replace_jerven(master["ISO3"].eq("TZA") & master["year"].le(1962), pd.to_numeric(master.loc[master["ISO3"].eq("TZA") & master["year"].le(1962), col], errors="coerce") * 20)
        _replace_jerven(master["ISO3"].eq("TZA") & master["year"].le(1914), pd.to_numeric(master.loc[master["ISO3"].eq("TZA") & master["year"].le(1914), col], errors="coerce") / 20)
        _replace_jerven(master["ISO3"].eq("KEN") & master["year"].le(1963), pd.to_numeric(master.loc[master["ISO3"].eq("KEN") & master["year"].le(1963), col], errors="coerce") * 20)
        _replace_jerven(master["ISO3"].eq("SDN") & master["year"].le(1995), pd.to_numeric(master.loc[master["ISO3"].eq("SDN") & master["year"].le(1995), col], errors="coerce") / 10)
        _replace_jerven(master["ISO3"].eq("SDN") & master["year"].le(2005), pd.to_numeric(master.loc[master["ISO3"].eq("SDN") & master["year"].le(2005), col], errors="coerce") / 100)
        _replace_jerven(master["ISO3"].eq("MOZ") & master["year"].le(1941), pd.to_numeric(master.loc[master["ISO3"].eq("MOZ") & master["year"].le(1941), col], errors="coerce") / 1000)
        _replace_jerven(master["ISO3"].eq("MOZ") & master["year"].le(1973), pd.to_numeric(master.loc[master["ISO3"].eq("MOZ") & master["year"].le(1973), col], errors="coerce") * 1000)
        _replace_jerven(master["ISO3"].eq("MOZ") & master["year"].between(2002, 2004), pd.to_numeric(master.loc[master["ISO3"].eq("MOZ") & master["year"].between(2002, 2004), col], errors="coerce") / 1000)
        _replace_jerven(master["ISO3"].eq("MOZ") & master["year"].le(2001), pd.to_numeric(master.loc[master["ISO3"].eq("MOZ") & master["year"].le(2001), col], errors="coerce") / 1000)
        _replace_jerven(master["ISO3"].eq("MAR") & master["year"].le(1962), pd.to_numeric(master.loc[master["ISO3"].eq("MAR") & master["year"].le(1962), col], errors="coerce") / 100)
        _replace_jerven(master["ISO3"].eq("MRT"), pd.to_numeric(master.loc[master["ISO3"].eq("MRT"), col], errors="coerce") / 10)
        _replace_jerven(master["ISO3"].eq("GHA") & master["year"].le(2004), pd.to_numeric(master.loc[master["ISO3"].eq("GHA") & master["year"].le(2004), col], errors="coerce") / 10000)
        _replace_jerven(master["ISO3"].eq("AGO") & master["year"].between(1974, 1994), pd.to_numeric(master.loc[master["ISO3"].eq("AGO") & master["year"].between(1974, 1994), col], errors="coerce") * (10**-3))
        _replace_jerven(master["ISO3"].eq("GNB") & master["year"].between(1942, 1972), pd.to_numeric(master.loc[master["ISO3"].eq("GNB") & master["year"].between(1942, 1972), col], errors="coerce") * (10**3))
        _replace_jerven(master["ISO3"].eq("GNB") & master["year"].between(1930, 1934), pd.to_numeric(master.loc[master["ISO3"].eq("GNB") & master["year"].between(1930, 1934), col], errors="coerce") * (10**3))
        _replace_jerven(master["ISO3"].eq("NAM") & master["year"].le(1911), pd.to_numeric(master.loc[master["ISO3"].eq("NAM") & master["year"].le(1911), col], errors="coerce") / 20)
        _replace_jerven(master["ISO3"].eq("ZMB"), pd.to_numeric(master.loc[master["ISO3"].eq("ZMB"), col], errors="coerce") / 1000)
        _replace_jerven(master["ISO3"].eq("LBY"), pd.to_numeric(master.loc[master["ISO3"].eq("LBY"), col], errors="coerce") / 480)
        _replace_jerven(master["ISO3"].eq("COD") & master["year"].le(1993), pd.to_numeric(master.loc[master["ISO3"].eq("COD") & master["year"].le(1993), col], errors="coerce") / 3000)
        _replace_jerven(master["ISO3"].eq("COD") & master["year"].le(1995), pd.to_numeric(master.loc[master["ISO3"].eq("COD") & master["year"].le(1995), col], errors="coerce") / 1000)
        _replace_jerven(master["ISO3"].eq("COD") & master["year"].le(1962), pd.to_numeric(master.loc[master["ISO3"].eq("COD") & master["year"].le(1962), col], errors="coerce") / 1000)

    master = master.rename(columns={"pop": "JERVEN_pop", "govtax": "JERVEN_govtax", "govrev": "JERVEN_govrev", "infl": "JERVEN_infl"})
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    master["JERVEN_pop"] = pd.to_numeric(master["JERVEN_pop"], errors="coerce").astype("float64")
    master["JERVEN_govtax"] = pd.to_numeric(master["JERVEN_govtax"], errors="coerce").astype("float32")
    master["JERVEN_govrev"] = pd.to_numeric(master["JERVEN_govrev"], errors="coerce").astype("float32")
    master["JERVEN_infl"] = pd.to_numeric(master["JERVEN_infl"], errors="coerce").astype("float32")
    master = master[["ISO3", "year", "JERVEN_pop", "JERVEN_govtax", "JERVEN_govrev", "JERVEN_infl"]].copy()
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("int16")
    master["JERVEN_pop"] = pd.to_numeric(master["JERVEN_pop"], errors="coerce").astype("float64")
    master["JERVEN_govtax"] = pd.to_numeric(master["JERVEN_govtax"], errors="coerce").astype("float32")
    master["JERVEN_govrev"] = pd.to_numeric(master["JERVEN_govrev"], errors="coerce").astype("float32")
    master["JERVEN_infl"] = pd.to_numeric(master["JERVEN_infl"], errors="coerce").astype("float32")
    master = _sort_keys(master)
    out_path = clean_dir / "aggregators" / "JERVEN" / "JERVEN.dta"
    _save_dta(master, out_path)
    return master
__all__ = ["clean_jerven"]
