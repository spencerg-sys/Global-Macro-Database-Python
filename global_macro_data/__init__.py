from __future__ import annotations

from .gmd import (
    GMDCommandError,
    VALID_VARIABLES,
    get_available_versions,
    get_current_version,
    gmd,
    list_countries,
    list_variables,
)
from . import clean_api as _clean_api
from . import download_api as _download_api
from . import helpers as _helpers
from . import pipeline_api as _pipeline_api


def _export_public(module: object) -> list[str]:
    names = list(getattr(module, "__all__", []))
    for name in names:
        globals()[name] = getattr(module, name)
    return names


_helper_exports = _export_public(_helpers)
_pipeline_exports = _export_public(_pipeline_api)
_clean_exports = _export_public(_clean_api)
_download_exports = _export_public(_download_api)


__all__ = list(
    dict.fromkeys(
        [
            "gmd",
            "GMDCommandError",
            "get_available_versions",
            "get_current_version",
            "list_variables",
            "list_countries",
            "VALID_VARIABLES",
            *_helper_exports,
            *_pipeline_exports,
            *_clean_exports,
            *_download_exports,
        ]
    )
)
