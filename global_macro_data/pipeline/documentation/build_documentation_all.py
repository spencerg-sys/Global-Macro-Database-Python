from __future__ import annotations

from .. import _core as _core
from ._core import (
    build_country_heatmaps,
    compile_documentation_pdfs,
    ensure_documentation_assets,
)

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

def build_documentation_all(
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
    doc_dir: Path | str = sh.OUTPUT_DOC_DIR,
    compile_pdf: bool = False,
    compile_country_docs: bool = False,
) -> pd.DataFrame:
    final_dir = _resolve(data_final_dir)
    temp_dir = _resolve(data_temp_dir)
    doc_path = _resolve(doc_dir)
    ensure_documentation_assets(doc_dir=doc_path)
    blocks, local_vars = _parse_documentation_spec()
    notes_sources_path = temp_dir / "notes_sources.dta"
    if not notes_sources_path.exists():
        raise FileNotFoundError(
            f"Missing source notes dataset at {notes_sources_path}. Run combine_all() or make_sources_dataset() first."
        )

    for block in blocks:
        input_var = str(block["input"])
        docvar = str(block.get("docvar", input_var))
        savevar = str(block.get("savevar", input_var))
        definition = str(block.get("variable_definition", savevar))
        doc_dataset_path = final_dir / f"documentation_{savevar}.dta"
        source_path = final_dir / f"chainlinked_{input_var}.dta"
        if not source_path.exists():
            if bool(block.get("cap_use")):
                continue
            raise FileNotFoundError(f"Missing documentation input dataset: {source_path}")

        df = _load_dta(source_path)
        effective_docvar = docvar
        if effective_docvar not in df.columns and input_var in df.columns:
            effective_docvar = input_var

        opts = _parse_gmdmakedoc_options(str(block.get("docopts", "")))
        sh.gmdmakedoc(
            df,
            effective_docvar,
            log=bool(opts["log"]),
            ylabel=opts["ylabel"] if isinstance(opts["ylabel"], str) or opts["ylabel"] is None else None,
            transformation=opts["transformation"] if isinstance(opts["transformation"], str) or opts["transformation"] is None else None,
            graphformat=opts["graphformat"] if isinstance(opts["graphformat"], str) or opts["graphformat"] is None else None,
            data_helper_dir=data_helper_dir,
            doc_dir=doc_dir,
        )
        doc_df = sh._build_doc_spells(df, effective_docvar, data_helper_dir=data_helper_dir)
        doc_df["variable"] = savevar
        doc_df["variable_definition"] = definition
        _save_dta(doc_df, doc_dataset_path)

    documentation = pd.DataFrame()
    for var in local_vars:
        path = final_dir / f"documentation_{var}.dta"
        if not path.exists():
            continue
        current = _load_dta(path)
        documentation = current if documentation.empty else pd.concat([current, documentation], ignore_index=True, sort=False)

    if documentation.empty:
        _save_dta(documentation, final_dir / "documentation.dta")
        return documentation

    notes_sources = _load_dta(notes_sources_path)
    documentation = documentation.merge(notes_sources, on=["source", "variable"], how="outer")
    documentation["notes"] = documentation["notes"].fillna("").astype(str) + ". " + documentation["note"].fillna("").astype(str)
    documentation = documentation.drop(columns=["note"], errors="ignore")
    doc_order = ["ISO3", "source", "countryname", "range", "notes", "tiny", "variable", "variable_definition"]
    documentation = documentation[[col for col in doc_order if col in documentation.columns] + [col for col in documentation.columns if col not in doc_order]]
    _save_dta(documentation, final_dir / "documentation.dta")

    build_country_heatmaps(
        data_clean_dir=data_clean_dir,
        data_temp_dir=data_temp_dir,
        data_final_dir=data_final_dir,
        doc_dir=doc_path,
        graphformat="pdf",
    )

    if local_vars:
        sh.gmdcombinedocs(local_vars, doc_dir=doc_dir)
    sh.gmdmakedoc_cs(documentation, doc_dir=doc_dir)
    if compile_pdf:
        compile_documentation_pdfs(
            doc_dir=doc_path,
            master=True,
            country_specific=compile_country_docs,
        )
    return documentation
__all__ = ["build_documentation_all"]
