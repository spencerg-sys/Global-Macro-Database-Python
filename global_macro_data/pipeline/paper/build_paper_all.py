from __future__ import annotations

from pathlib import Path

from . import _core as _core
from .figures import (
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
)
from .numbers import build_paper_numbers
from .tables import (
    build_paper_tab_comparison,
    build_paper_tab_no_sources,
    build_paper_tab_obs_count,
    build_paper_tab_variable_descriptions,
)

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})


def build_paper_all(
    *,
    repo_root: Path | str = REPO_ROOT,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
    numbers_dir: Path | str = sh.OUTPUT_NUMBERS_DIR,
    tables_dir: Path | str = sh.OUTPUT_TABLES_DIR,
    graphs_dir: Path | str = PAPER_OUTPUT_GRAPH_DIR,
) -> dict[str, object]:
    summary: dict[str, object] = {}
    summary["numbers"] = sorted(build_paper_numbers(repo_root=repo_root, data_final_dir=data_final_dir, numbers_dir=numbers_dir).keys())
    summary["tables"] = [
        build_paper_tab_obs_count(
            data_clean_dir=data_clean_dir,
            data_temp_dir=data_temp_dir,
            data_final_dir=data_final_dir,
            data_helper_dir=data_helper_dir,
            tables_dir=tables_dir,
        ).name,
        build_paper_tab_no_sources(
            data_clean_dir=data_clean_dir,
            data_temp_dir=data_temp_dir,
            data_final_dir=data_final_dir,
            data_helper_dir=data_helper_dir,
            tables_dir=tables_dir,
        ).name,
        build_paper_tab_comparison(
            data_clean_dir=data_clean_dir,
            data_temp_dir=data_temp_dir,
            data_final_dir=data_final_dir,
            data_helper_dir=data_helper_dir,
            tables_dir=tables_dir,
        ).name,
        *[path.name for path in build_paper_tab_variable_descriptions(data_final_dir=data_final_dir, data_helper_dir=data_helper_dir, tables_dir=tables_dir)],
    ]

    figure_outputs = [
        build_paper_fig_source_comparison(
            data_clean_dir=data_clean_dir,
            data_temp_dir=data_temp_dir,
            data_final_dir=data_final_dir,
            data_helper_dir=data_helper_dir,
            graphs_dir=graphs_dir,
        ),
        build_paper_fig_sources_per_var(
            data_clean_dir=data_clean_dir,
            data_temp_dir=data_temp_dir,
            data_final_dir=data_final_dir,
            graphs_dir=graphs_dir,
        ),
        *build_paper_fig_boxplots_var(
            data_clean_dir=data_clean_dir,
            data_temp_dir=data_temp_dir,
            data_final_dir=data_final_dir,
            graphs_dir=graphs_dir,
        ),
        *build_paper_fig_gdp_share_per_var(data_final_dir=data_final_dir, graphs_dir=graphs_dir),
        build_paper_fig_worldmap_firstdata(
            data_clean_dir=data_clean_dir,
            data_temp_dir=data_temp_dir,
            data_final_dir=data_final_dir,
            data_helper_dir=data_helper_dir,
            graphs_dir=graphs_dir,
        ),
        build_paper_fig_stylized_fact_rates(data_final_dir=data_final_dir, data_helper_dir=data_helper_dir, graphs_dir=graphs_dir),
        build_paper_fig_stylized_fact_trade(data_final_dir=data_final_dir, graphs_dir=graphs_dir),
        *build_paper_fig_stylized_fact_usd(data_final_dir=data_final_dir, data_helper_dir=data_helper_dir, graphs_dir=graphs_dir),
    ]
    chile = build_paper_fig_chile(
        data_clean_dir=data_clean_dir,
        data_temp_dir=data_temp_dir,
        data_final_dir=data_final_dir,
        graphs_dir=graphs_dir,
    )
    sweden = build_paper_fig_sweden(
        data_clean_dir=data_clean_dir,
        data_temp_dir=data_temp_dir,
        data_final_dir=data_final_dir,
        graphs_dir=graphs_dir,
    )
    if chile is not None:
        figure_outputs.append(chile)
    if sweden is not None:
        figure_outputs.append(sweden)
    figure_outputs.extend(build_paper_fig_fra(data_final_dir=data_final_dir, data_helper_dir=data_helper_dir, graphs_dir=graphs_dir))
    figure_outputs.extend(build_paper_fig_gbr(data_final_dir=data_final_dir, data_helper_dir=data_helper_dir, graphs_dir=graphs_dir))
    summary["figures"] = [path.name for path in figure_outputs if path is not None]
    return summary


__all__ = ["build_paper_all"]
