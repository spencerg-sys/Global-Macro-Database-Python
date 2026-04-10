from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def combine_variable(
    varname: str,
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
) -> pd.DataFrame:
    if varname == "USDfx":
        from .combine_usdfx import combine_usdfx

        return combine_usdfx(
            data_clean_dir=data_clean_dir,
            data_temp_dir=data_temp_dir,
            data_final_dir=data_final_dir,
        )
    if varname in {"BankingCrisis", "CurrencyCrisis", "SovDebtCrisis"}:
        from .build_crisis_indicator import build_crisis_indicator

        return build_crisis_indicator(varname, data_clean_dir=data_clean_dir, data_temp_dir=data_temp_dir, data_final_dir=data_final_dir)
    if varname == "CA_GDP":
        from .combine_ca_gdp import combine_ca_gdp

        return combine_ca_gdp(
            data_clean_dir=data_clean_dir,
            data_temp_dir=data_temp_dir,
            data_helper_dir=sh.DATA_HELPER_DIR,
            data_final_dir=data_final_dir,
        )
    if varname == "rGDP":
        from .combine_rgdp import combine_rgdp

        return combine_rgdp(
            data_clean_dir=data_clean_dir,
            data_temp_dir=data_temp_dir,
            data_final_dir=data_final_dir,
        )
    if varname == "rGDP_USD":
        from .combine_rgdp_usd import combine_rgdp_usd

        return combine_rgdp_usd(
            data_clean_dir=data_clean_dir,
            data_temp_dir=data_temp_dir,
            data_final_dir=data_final_dir,
        )
    from .combine_splice_variable import combine_splice_variable

    return combine_splice_variable(varname, data_clean_dir=data_clean_dir, data_final_dir=data_final_dir, data_temp_dir=data_temp_dir)
__all__ = ["combine_variable"]
