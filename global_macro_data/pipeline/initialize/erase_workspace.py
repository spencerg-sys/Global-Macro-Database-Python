from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})


def erase_workspace(
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    data_distribute_dir: Path | str = REPO_ROOT / "data" / "distribute",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
) -> dict[str, int]:
    roots = {
        "data_clean": _resolve(data_clean_dir),
        "data_final": _resolve(data_final_dir),
        "data_distribute": _resolve(data_distribute_dir),
        "data_temp": _resolve(data_temp_dir),
    }
    deleted: dict[str, int] = {}
    for label, root in roots.items():
        count = 0
        if root.exists():
            for path in root.rglob("*.dta"):
                if path.is_file():
                    path.unlink()
                    count += 1
        deleted[label] = count
    return deleted


__all__ = ["erase_workspace"]
