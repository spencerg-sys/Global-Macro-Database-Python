from __future__ import annotations

from .pipeline import _core as _core
from .pipeline.combine import (
    build_crisis_indicator,
    combine_all,
    combine_ca_gdp,
    combine_rgdp,
    combine_rgdp_usd,
    combine_splice_variable,
    combine_usdfx,
    combine_variable,
)
from .pipeline.documentation import build_documentation_all
from .pipeline.documentation import (
    build_country_heatmap,
    build_country_heatmaps,
    compile_documentation_pdfs,
    compile_latex_pdf,
    ensure_documentation_assets,
)
from .pipeline.initialize import (
    check_runtime_packages,
    erase_workspace,
    make_blank_panel,
    make_download_dates,
    make_notes_dataset,
    make_sources_dataset,
    run_master_pipeline,
    validate_inputs,
    validate_outputs,
)
from .pipeline.merge import merge_clean_data, merge_final_data
from .pipeline.paper import (
    build_paper_all,
    build_paper_fig_boxplots_var,
    build_paper_fig_chile,
    build_paper_fig_fra,
    build_paper_fig_gbr,
    build_paper_fig_gdp_share_per_var,
    build_paper_fig_source_comparison,
    build_paper_fig_sources_per_var,
    build_paper_fig_stylized_fact_rates,
    build_paper_fig_stylized_fact_trade,
    build_paper_fig_stylized_fact_usd,
    build_paper_fig_sweden,
    build_paper_fig_worldmap_firstdata,
    build_paper_numbers,
    build_paper_tab_comparison,
    build_paper_tab_no_sources,
    build_paper_tab_obs_count,
    build_paper_tab_variable_descriptions,
)
from .pipeline.sync.sync_packaged_final_artifacts import sync_packaged_final_artifacts

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})


__all__ = [
    "build_crisis_indicator",
    "build_country_heatmap",
    "build_country_heatmaps",
    "build_documentation_all",
    "build_paper_all",
    "build_paper_fig_boxplots_var",
    "build_paper_fig_chile",
    "build_paper_fig_fra",
    "build_paper_fig_gbr",
    "build_paper_fig_gdp_share_per_var",
    "build_paper_fig_source_comparison",
    "build_paper_fig_sources_per_var",
    "build_paper_fig_stylized_fact_rates",
    "build_paper_fig_stylized_fact_trade",
    "build_paper_fig_stylized_fact_usd",
    "build_paper_fig_sweden",
    "build_paper_fig_worldmap_firstdata",
    "build_paper_numbers",
    "build_paper_tab_comparison",
    "build_paper_tab_no_sources",
    "build_paper_tab_obs_count",
    "build_paper_tab_variable_descriptions",
    "combine_all",
    "combine_ca_gdp",
    "combine_rgdp",
    "combine_rgdp_usd",
    "combine_splice_variable",
    "combine_usdfx",
    "combine_variable",
    "compile_documentation_pdfs",
    "compile_latex_pdf",
    "check_runtime_packages",
    "erase_workspace",
    "ensure_documentation_assets",
    "make_blank_panel",
    "make_download_dates",
    "make_notes_dataset",
    "make_sources_dataset",
    "merge_clean_data",
    "merge_final_data",
    "run_master_pipeline",
    "sync_packaged_final_artifacts",
    "validate_inputs",
    "validate_outputs",
]
