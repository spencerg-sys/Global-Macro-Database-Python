from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from .. import _core as _core

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})


HEATMAP_VARIABLE_SPECS: list[tuple[str, str]] = [
    ("cbrate", "Central bank policy rate"),
    ("strate", "Short-term interest rate"),
    ("ltrate", "Long-term interest rate"),
    ("M0", "Money supply (M0)"),
    ("M1", "Money supply (M1)"),
    ("M2", "Money supply (M2)"),
    ("M3", "Money supply (M3)"),
    ("rGDP", "Real GDP"),
    ("nGDP", "Nominal GDP"),
    ("cons", "Consumption"),
    ("inv", "Gross capital formation"),
    ("finv", "Gross fixed capital formation"),
    ("CA_GDP", "Current account"),
    ("exports", "Exports"),
    ("imports", "Imports"),
    ("REER", "Real effective exchange rate"),
    ("USDfx", "USD exchange rate"),
    ("govrev", "Government revenue"),
    ("govtax", "Government tax revenue"),
    ("govexp", "Government expenditure"),
    ("govdebt_GDP", "Government debt"),
    ("govdef_GDP", "Government deficit"),
    ("unemp", "Unemployment"),
    ("infl", "Inflation"),
    ("CPI", "Consumer price index"),
    ("HPI", "House prices index"),
    ("pop", "Population"),
]

HEATMAP_SUFFIXES = [var for var, _ in HEATMAP_VARIABLE_SPECS]
HEATMAP_SUFFIXES_SORTED = sorted(HEATMAP_SUFFIXES, key=len, reverse=True)
HEATMAP_LABELS = {var: label for var, label in HEATMAP_VARIABLE_SPECS}


def _heatmap_year_increment(years: list[int]) -> int:
    n_years = len(years)
    if n_years <= 100:
        return 10
    if n_years <= 200:
        return 20
    return 30


def _normalize_clean_wide_for_heatmaps(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    rename_map = {
        col: col.replace("WDI_ARC_", "WDIARC_")
        for col in work.columns
        if col.startswith("WDI_ARC_")
    }
    if "WB_CC_infl" in work.columns and "WBCC_infl" not in work.columns:
        rename_map["WB_CC_infl"] = "WBCC_infl"
    if "OECD_HPI" in work.columns and "OECD_EO_HPI" not in work.columns:
        rename_map["OECD_HPI"] = "OECD_EO_HPI"
    if rename_map:
        work = work.rename(columns=rename_map)

    drop_patterns = ("BVX", "LV", "RR_crisis")
    drop_cols = [
        col
        for col in work.columns
        if col.endswith("_rcons")
        or col.endswith("_rHPI")
        or any(col.startswith(pattern) for pattern in drop_patterns)
    ]
    if drop_cols:
        work = work.drop(columns=drop_cols, errors="ignore")
    return work


def _match_source_variable(column: str) -> tuple[str, str] | None:
    for variable in HEATMAP_SUFFIXES_SORTED:
        suffix = f"_{variable}"
        if column.endswith(suffix):
            source = column[: -len(suffix)]
            if source:
                return source, variable
    return None


def _heatmap_long_frame(df: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for column in df.columns:
        if column in {"ISO3", "year", "countryname"}:
            continue
        match = _match_source_variable(str(column))
        if match is None:
            continue
        source, variable = match
        piece = df.loc[df[column].notna(), ["year", column]].copy()
        if piece.empty:
            continue
        piece["variable"] = variable
        piece["source"] = source
        pieces.append(piece[["year", "variable", "source"]])
    if not pieces:
        return pd.DataFrame(columns=["year", "variable", "source"])
    return pd.concat(pieces, ignore_index=True)


def _heatmap_count_matrix(
    clean_wide: pd.DataFrame,
    *,
    iso3: str,
    min_year: int | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    country = clean_wide.loc[clean_wide["ISO3"].astype(str) == str(iso3)].copy()
    if country.empty:
        raise KeyError(f"No clean data found for {iso3}")

    country = _normalize_clean_wide_for_heatmaps(country)
    if min_year is not None:
        country = country.loc[pd.to_numeric(country["year"], errors="coerce").ge(min_year)].copy()
    country = country.dropna(axis=1, how="all")
    if country.empty:
        return pd.DataFrame(index=HEATMAP_SUFFIXES, columns=[]), []

    long_df = _heatmap_long_frame(country)
    if long_df.empty:
        return pd.DataFrame(index=HEATMAP_SUFFIXES, columns=[]), []

    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    long_df = long_df.dropna(subset=["year"]).copy()
    long_df["year"] = long_df["year"].astype(int)
    long_df = long_df.drop_duplicates(subset=["year", "variable", "source"]).copy()

    counts = (
        long_df.groupby(["variable", "year"], dropna=False)["source"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(HEATMAP_SUFFIXES, fill_value=0)
    )
    years = [int(year) for year in counts.columns.tolist()]
    return counts, years


def build_country_heatmap(
    clean_wide: pd.DataFrame,
    *,
    iso3: str,
    output_path: Path | str,
    min_year: int | None = None,
) -> Path | None:
    counts, years = _heatmap_count_matrix(clean_wide, iso3=iso3, min_year=min_year)
    output = _resolve(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if counts.empty or not years:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    ordered_counts = counts.loc[counts.sum(axis=1).gt(0)].copy()
    if ordered_counts.empty:
        return None

    labels = [HEATMAP_LABELS.get(var, var) for var in ordered_counts.index.tolist()]
    matrix = ordered_counts.to_numpy(dtype=float)
    vmax = int(np.nanmax(matrix)) if matrix.size else 0
    vmax = max(vmax, 1)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "gmd_heatmap",
        ["#ffffff", "#e7d982", "#d8aa3a", "#d86b2d", "#c33426", "#6e0f0f"],
        N=max(vmax + 1, 2),
    )
    norm = mcolors.BoundaryNorm(np.arange(-0.5, vmax + 1.5, 1), cmap.N)

    fig_height = max(6.0, len(labels) * 0.34)
    fig, ax = plt.subplots(figsize=(12.0, fig_height))
    image = ax.imshow(matrix, aspect="auto", interpolation="none", cmap=cmap, norm=norm)
    ax.set_facecolor("white")

    increment = _heatmap_year_increment(years)
    tick_years = list(range(int(years[0]), int(years[-1]) + 1, increment))
    if years[-1] not in tick_years:
        tick_years.append(int(years[-1]))
    year_to_pos = {year: idx for idx, year in enumerate(years)}
    tick_positions = [year_to_pos[year] for year in tick_years if year in year_to_pos]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(year) for year in tick_years if year in year_to_pos], rotation=90)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticks(np.arange(-0.5, len(years), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="#d0d0d0", linewidth=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.ax.set_title("")
    colorbar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def build_country_heatmaps(
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    doc_dir: Path | str = sh.OUTPUT_DOC_DIR,
    graphformat: str = "pdf",
) -> list[Path]:
    final_dir = _resolve(data_final_dir)
    doc_path = _resolve(doc_dir)
    graphs_dir = doc_path / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    clean_wide = _load_clean_data_wide(
        data_clean_dir=data_clean_dir,
        data_temp_dir=data_temp_dir,
        data_final_dir=final_dir,
    )
    outputs: list[Path] = []
    for iso3 in sorted({str(value) for value in clean_wide["ISO3"].dropna().tolist()}):
        built = build_country_heatmap(
            clean_wide,
            iso3=iso3,
            output_path=graphs_dir / f"{iso3}_heatmap.{graphformat}",
        )
        if built is not None:
            outputs.append(built)
    return outputs


def ensure_documentation_assets(
    *,
    doc_dir: Path | str = sh.OUTPUT_DOC_DIR,
) -> dict[str, Path]:
    doc_path = _resolve(doc_dir)
    doc_path.mkdir(parents=True, exist_ok=True)

    bib_path = doc_path / "bib.bib"
    if not bib_path.exists():
        bib_path.write_text("% Auto-generated placeholder bibliography.\n", encoding="utf-8")

    qje_path = doc_path / "qje.bst"
    if not qje_path.exists():
        qje_source = shutil.which("kpsewhich")
        found_qje: str | None = None
        found_plainnat: str | None = None
        if qje_source:
            result = subprocess.run(
                [qje_source, "qje.bst"],
                capture_output=True,
                text=True,
                check=False,
            )
            found_qje = result.stdout.strip() or None
            if not found_qje:
                result = subprocess.run(
                    [qje_source, "plainnat.bst"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                found_plainnat = result.stdout.strip() or None
        if found_qje:
            shutil.copy2(found_qje, qje_path)
        elif found_plainnat:
            shutil.copy2(found_plainnat, qje_path)

    return {"bib": bib_path, "qje": qje_path}


def compile_latex_pdf(
    tex_path: Path | str,
    *,
    doc_dir: Path | str = sh.OUTPUT_DOC_DIR,
) -> Path | None:
    tex_file = _resolve(tex_path)
    if not tex_file.exists():
        return None

    ensure_documentation_assets(doc_dir=doc_dir)
    doc_path = _resolve(doc_dir)

    latexmk = shutil.which("latexmk")
    if latexmk:
        result = subprocess.run(
            [
                latexmk,
                "-pdf",
                "-interaction=nonstopmode",
                "-f",
                tex_file.name,
            ],
            cwd=doc_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            pdf_path = tex_file.with_suffix(".pdf")
            return pdf_path if pdf_path.exists() else None

    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if not pdflatex:
        return None

    stem = tex_file.stem
    subprocess.run(
        [pdflatex, "-interaction=nonstopmode", tex_file.name],
        cwd=doc_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if bibtex:
        subprocess.run(
            [bibtex, stem],
            cwd=doc_path,
            capture_output=True,
            text=True,
            check=False,
        )
    subprocess.run(
        [pdflatex, "-interaction=nonstopmode", tex_file.name],
        cwd=doc_path,
        capture_output=True,
        text=True,
        check=False,
    )
    subprocess.run(
        [pdflatex, "-interaction=nonstopmode", tex_file.name],
        cwd=doc_path,
        capture_output=True,
        text=True,
        check=False,
    )

    pdf_path = tex_file.with_suffix(".pdf")
    return pdf_path if pdf_path.exists() else None


def compile_documentation_pdfs(
    *,
    doc_dir: Path | str = sh.OUTPUT_DOC_DIR,
    master: bool = True,
    country_specific: bool = False,
) -> list[Path]:
    doc_path = _resolve(doc_dir)
    built: list[Path] = []
    if master:
        compiled = compile_latex_pdf(doc_path / "master.tex", doc_dir=doc_path)
        if compiled is not None:
            built.append(compiled)

    if country_specific:
        for tex_path in sorted(doc_path.glob("*.tex")):
            if tex_path.name == "master.tex":
                continue
            compiled = compile_latex_pdf(tex_path, doc_dir=doc_path)
            if compiled is not None:
                built.append(compiled)
    return built


__all__ = [
    "build_country_heatmap",
    "build_country_heatmaps",
    "compile_documentation_pdfs",
    "compile_latex_pdf",
    "ensure_documentation_assets",
]
