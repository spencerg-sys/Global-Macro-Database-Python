from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def make_blank_panel(
    *,
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    maxdate: int | None = None,
    mindate: int = 0,
) -> pd.DataFrame:
    helper_dir = _resolve(data_helper_dir)
    temp_dir = _resolve(data_temp_dir)
    max_year = (date.today().year + 10) if maxdate is None else int(maxdate)
    diff = max_year - int(mindate)

    countries = pd.read_dta(helper_dir / "countrylist.dta", convert_categoricals=False)[["ISO3"]].copy()
    panel = countries.loc[countries.index.repeat(diff)].reset_index(drop=True)
    panel["year"] = panel.groupby("ISO3").cumcount() + int(mindate) + 1

    temp_dir.mkdir(parents=True, exist_ok=True)
    _save_dta(panel, temp_dir / "blank_panel.dta")
    return panel
__all__ = ["make_blank_panel"]
