from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def combine_all(
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
) -> list[tuple[str, tuple[int, int]]]:
    from ..initialize.make_sources_dataset import make_sources_dataset
    from ..initialize.validate_outputs import validate_outputs
    from ..merge.merge_clean_data import merge_clean_data
    from .combine_variable import combine_variable

    merge_clean_data(data_clean_dir=data_clean_dir, data_temp_dir=data_temp_dir, data_final_dir=data_final_dir)
    validate_outputs(data_final_dir=data_final_dir)
    make_sources_dataset(data_temp_dir=data_temp_dir)

    completed: list[tuple[str, tuple[int, int]]] = []
    for path in _combine_relative_paths():
        target = _extract_generate_var(path)
        if target is None:
            stem = path.stem.replace(" ", "")
            if stem in {"BankingCrisis", "CurrencyCrisis", "SovDebtCrisis", "rGDP_USD"}:
                target = stem
            else:
                continue
        result = combine_variable(
            target,
            data_clean_dir=data_clean_dir,
            data_temp_dir=data_temp_dir,
            data_final_dir=data_final_dir,
        )
        completed.append((target, tuple(result.shape)))
    return completed
__all__ = ["combine_all"]
