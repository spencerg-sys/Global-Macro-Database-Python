from __future__ import annotations

import importlib.util
import re
import shutil

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})


_IMPORT_NAME_MAP = {
    "pyshp": "shapefile",
}
_EXTERNAL_TOOLS = ("latexmk", "pdflatex", "bibtex", "kpsewhich")


def _parse_requirement_name(requirement: str) -> str:
    token = requirement.split(";", 1)[0].strip()
    token = token.split("[", 1)[0].strip()
    match = re.match(r"([A-Za-z0-9_.-]+)", token)
    if match is None:
        raise ValueError(f"Unable to parse requirement name from {requirement!r}")
    return match.group(1)


def check_runtime_packages(
    *,
    repo_root: Path | str = REPO_ROOT,
    requirements_path: Path | str | None = None,
) -> dict[str, object]:
    root = _resolve(repo_root)
    req_path = _resolve(requirements_path) if requirements_path is not None else root / "requirements.txt"
    if not req_path.exists():
        raise FileNotFoundError(f"Missing requirements file: {req_path}")

    requirements = [
        line.strip()
        for line in req_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    present: list[str] = []
    missing: list[str] = []
    for requirement in requirements:
        package_name = _parse_requirement_name(requirement)
        import_name = _IMPORT_NAME_MAP.get(package_name, package_name)
        if importlib.util.find_spec(import_name) is None:
            missing.append(package_name)
        else:
            present.append(package_name)

    if missing:
        missing_text = ", ".join(missing)
        raise ModuleNotFoundError(
            f"Missing required Python packages: {missing_text}. Install them before running the pipeline."
        )

    external_tools = {tool: shutil.which(tool) is not None for tool in _EXTERNAL_TOOLS}
    return {
        "python_modules": present,
        "external_tools": external_tools,
    }


__all__ = ["check_runtime_packages"]
