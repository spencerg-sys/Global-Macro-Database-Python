from __future__ import annotations

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})


def _cleanup_swap_files(*, repo_root: Path | str = REPO_ROOT) -> int:
    root = _resolve(repo_root)
    deleted = 0
    for path in root.rglob("*.stswp"):
        if path.is_file():
            path.unlink()
            deleted += 1
    return deleted


def run_master_pipeline(
    *,
    validate: bool = True,
    erase: bool = False,
    download: bool = False,
    clean: bool = False,
    combine: bool = False,
    output_data: bool = False,
    document: bool = False,
    paper: bool = False,
    packages: bool = False,
    repo_root: Path | str = REPO_ROOT,
    data_raw_dir: Path | str = REPO_ROOT / "data" / "raw",
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    data_distribute_dir: Path | str = REPO_ROOT / "data" / "distribute",
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
    doc_dir: Path | str = sh.OUTPUT_DOC_DIR,
) -> dict[str, object]:
    from ...clean_api import rebuild_clean_sources
    from ...download_api import download_all_sources
    from ..combine.combine_all import combine_all
    from ..documentation.build_documentation_all import build_documentation_all
    from ..merge.merge_final_data import merge_final_data
    from ..paper.build_paper_all import build_paper_all
    from .check_runtime_packages import check_runtime_packages
    from .erase_workspace import erase_workspace
    from .make_blank_panel import make_blank_panel
    from .make_download_dates import make_download_dates
    from .make_notes_dataset import make_notes_dataset
    from .validate_inputs import validate_inputs

    summary: dict[str, object] = {}
    root = _resolve(repo_root)
    if packages:
        summary["packages"] = check_runtime_packages(repo_root=root)

    if erase:
        summary["erase"] = erase_workspace(
            data_clean_dir=data_clean_dir,
            data_final_dir=data_final_dir,
            data_distribute_dir=data_distribute_dir,
            data_temp_dir=data_temp_dir,
        )
        summary["download_dates"] = tuple(make_download_dates(data_temp_dir=data_temp_dir).shape)
        summary["blank_panel_after_erase"] = tuple(
            make_blank_panel(data_helper_dir=data_helper_dir, data_temp_dir=data_temp_dir).shape
        )
        summary["notes"] = tuple(make_notes_dataset(data_temp_dir=data_temp_dir).shape)

    summary["blank_panel"] = tuple(
        make_blank_panel(data_helper_dir=data_helper_dir, data_temp_dir=data_temp_dir).shape
    )

    if validate:
        summary["validate_inputs"] = validate_inputs(
            data_raw_dir=data_raw_dir,
            data_clean_dir=data_clean_dir,
            data_helper_dir=data_helper_dir,
        )
        summary["swap_cleanup"] = _cleanup_swap_files(repo_root=root)

    if download:
        summary["download"] = download_all_sources(
            data_raw_dir=data_raw_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
        )

    if clean:
        summary["clean"] = rebuild_clean_sources(
            data_raw_dir=data_raw_dir,
            data_clean_dir=data_clean_dir,
            data_helper_dir=data_helper_dir,
            data_temp_dir=data_temp_dir,
        )

    if combine:
        summary["combine"] = combine_all(
            data_clean_dir=data_clean_dir,
            data_temp_dir=data_temp_dir,
            data_final_dir=data_final_dir,
        )

    if document:
        summary["documentation"] = tuple(
            build_documentation_all(
                data_clean_dir=data_clean_dir,
                data_final_dir=data_final_dir,
                data_temp_dir=data_temp_dir,
                data_helper_dir=data_helper_dir,
                doc_dir=doc_dir,
                compile_pdf=False,
                compile_country_docs=False,
            ).shape
        )

    if output_data:
        summary["data_final"] = tuple(
            merge_final_data(
                data_temp_dir=data_temp_dir,
                data_final_dir=data_final_dir,
                data_helper_dir=data_helper_dir,
            ).shape
        )

    if paper:
        summary["paper"] = build_paper_all(
            repo_root=root,
            data_clean_dir=data_clean_dir,
            data_temp_dir=data_temp_dir,
            data_final_dir=data_final_dir,
            data_helper_dir=data_helper_dir,
        )

    return summary
__all__ = ["run_master_pipeline"]
