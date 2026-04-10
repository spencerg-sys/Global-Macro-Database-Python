from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def sync_packaged_final_artifacts(
    *,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    overwrite: bool = False,
) -> list[Path]:
    final_dir = _resolve(data_final_dir)
    return _sync_tree(REPO_ROOT / "data" / "final", final_dir, overwrite=overwrite)
__all__ = ["sync_packaged_final_artifacts"]
