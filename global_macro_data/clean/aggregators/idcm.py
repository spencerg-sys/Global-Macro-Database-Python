from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def clean_idcm(
    *,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_helper_dir: Path | str = REPO_ROOT / "data" / "helpers",
) -> pd.DataFrame:
    raw_dir = _resolve(data_raw_dir)
    clean_dir = _resolve(data_clean_dir)
    helper_dir = _resolve(data_helper_dir)

    df = _load_dta(raw_dir / "aggregators" / "IDCM" / "IDCM.dta")
    df = df.loc[~df["ref_area"].astype(str).str.contains(r"[0-9]", regex=True, na=False)].copy()
    df = df.rename(columns={"ref_area": "ISO2", "period": "year"})

    countrylist = _load_dta(helper_dir / "countrylist.dta")[["ISO2", "ISO3"]].copy()
    df = df.merge(countrylist, on="ISO2", how="left")
    df = df.loc[df["ISO3"].notna()].copy()
    df = df.drop(columns=["ISO2"])

    series_code = df["series_code"].astype(str)
    df["type"] = ""
    type_map = {
        "W2.S1.S1.B.B1GQ._Z._Z._Z.XDC.V.N": "IDCM_nGDP_LCU",
        "W2.S1.S1.B.B1GQ._Z._Z._Z.XDC.Q.N": "IDCM_rGDP_LCU",
        "W0.S1.S1.B.B9._Z._Z._Z.XDC.V.N": "IDCM_CA",
        "W0.S1.S1.D.P3._Z._Z._T.XDC.V.N": "IDCM_cons",
        "W0.S1.S1.D.P5.N1G._T._Z.XDC.V.N": "IDCM_inv",
        "W0.S1.S1.D.P51G.N11G._T._Z.XDC.V.N": "IDCM_finv",
        "W1.S1.S1.D.P6._Z._Z._Z.XDC.V.N": "IDCM_exports",
        "W1.S1.S1.C.P7._Z._Z._Z.XDC.V.N": "IDCM_imports",
        "W0.S1.S1.B.B8N._Z._Z._Z.XDC.V.N": "IDCM_savings_1",
        "W2.S1.S1.D.P51C.N1G._T._Z.XDC.V.N": "IDCM_savings_2",
        "W2.S1.S1._Z.EMP._Z._T._Z.PS._Z.N": "IDCM_emp",
    }
    for marker, target in type_map.items():
        df.loc[series_code.str.contains(marker, regex=False, na=False), "type"] = target
    df = df.loc[df["type"].ne(""), ["ISO3", "year", "type", "value"]].copy()

    wide = df.pivot_table(index=["ISO3", "year"], columns="type", values="value", aggfunc="first").reset_index()
    wide.columns.name = None

    if {"IDCM_savings_1", "IDCM_savings_2"}.issubset(wide.columns):
        wide["IDCM_sav"] = pd.to_numeric(wide["IDCM_savings_1"], errors="coerce") + pd.to_numeric(wide["IDCM_savings_2"], errors="coerce")
    wide = wide.drop(columns=["IDCM_savings_1", "IDCM_savings_2"], errors="ignore")

    for col in [c for c in wide.columns if c not in {"ISO3", "year"}]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce")
        wide.loc[wide[col].eq(0), col] = pd.NA

    wide = wide.rename(columns={col: col.replace("_LCU", "") for col in wide.columns})
    if "IDCM_sav" in wide.columns:
        wide["IDCM_sav"] = pd.to_numeric(wide["IDCM_sav"], errors="coerce").astype("float32")

    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("int16")
    wide = _sort_keys(wide)
    if wide.duplicated(["ISO3", "year"]).any():
        raise sh.PipelineRuntimeError("ISO3 year do not uniquely identify observations", code=459)
    wide = wide[["ISO3", "year"] + [col for col in wide.columns if col not in {"ISO3", "year"}]]

    out_path = clean_dir / "aggregators" / "IDCM" / "IDCM.dta"
    _save_dta(wide, out_path)
    return wide
__all__ = ["clean_idcm"]
