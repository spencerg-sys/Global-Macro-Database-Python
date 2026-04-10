from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import _core as _core
from .tables import _gfd_variable_counts, _gmd_variable_counts, _source_variable_counts

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig, path: Path | str) -> Path:
    output = _resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    return output


def _source_long(
    clean_wide: pd.DataFrame,
    *,
    variables: set[str] | None = None,
    years: set[int] | None = None,
    iso3: str | None = None,
) -> pd.DataFrame:
    work = clean_wide.copy()
    if years is not None:
        work = work.loc[pd.to_numeric(work["year"], errors="coerce").isin(sorted(years))].copy()
    if iso3 is not None:
        work = work.loc[work["ISO3"].astype(str) == str(iso3)].copy()

    pieces: list[pd.DataFrame] = []
    for column in work.columns:
        if column in {"ISO3", "year", "countryname"}:
            continue
        match = _match_source_variable(str(column))
        if match is None:
            continue
        source, variable = match
        if variables is not None and variable not in variables:
            continue
        piece = work.loc[work[column].notna(), ["ISO3", "year", column]].copy()
        if piece.empty:
            continue
        piece["variable"] = variable
        piece["source"] = source
        pieces.append(piece[["ISO3", "year", "variable", "source"]])
    if not pieces:
        return pd.DataFrame(columns=["ISO3", "year", "variable", "source"])
    out = pd.concat(pieces, ignore_index=True)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["year"]).copy()
    out["year"] = out["year"].astype(int)
    return out.drop_duplicates(["ISO3", "year", "variable", "source"]).reset_index(drop=True)


def build_paper_fig_source_comparison(
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
    graphs_dir: Path | str = PAPER_OUTPUT_GRAPH_DIR,
) -> Path:
    clean_wide = _load_clean_wide(
        data_clean_dir=data_clean_dir,
        data_temp_dir=data_temp_dir,
        data_final_dir=data_final_dir,
    )
    data_final = _load_data_final(data_final_dir=data_final_dir)
    gmd_counts = _gmd_variable_counts(data_final)
    source_counts = {
        source: _source_variable_counts(clean_wide, source)
        for source in PAPER_COMPARISON_SOURCES
        if source != "GFD"
    }
    source_counts["GFD"] = _gfd_variable_counts(data_helper_dir=data_helper_dir)

    rows: list[dict[str, object]] = []
    for var in PAPER_VARIABLE_ORDER:
        gmd = pd.to_numeric(pd.Series([gmd_counts.get(var)]), errors="coerce").iloc[0]
        if pd.isna(gmd) or gmd <= 0:
            continue
        competitor_counts = [
            float(pd.to_numeric(pd.Series([source_counts[source].get(var)]), errors="coerce").iloc[0])
            for source in PAPER_COMPARISON_SOURCES
        ]
        competitor_counts = [value for value in competitor_counts if not np.isnan(value)]
        next_best = max(competitor_counts) if competitor_counts else np.nan
        rows.append(
            {
                "variable": PAPER_VARIABLE_LABELS.get(var, var),
                "count_GMD": float(gmd),
                "count_next": next_best,
            }
        )

    out = pd.DataFrame(rows).sort_values("count_GMD", ascending=False).reset_index(drop=True)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    y = np.arange(len(out))
    ax.barh(y - 0.2, out["count_GMD"], height=0.38, color="#337ab7", label="Global Macro Database")
    ax.barh(y + 0.2, out["count_next"], height=0.38, color="#d9534f", label="Next best source")
    ax.set_yticks(y)
    ax.set_yticklabels(out["variable"])
    ax.invert_yaxis()
    ax.set_xlabel("Number of country-year observations")
    ax.legend(frameon=False)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    path = _save_figure(fig, _figure_path("sources_comparison_total.eps", graphs_dir=graphs_dir))
    plt.close(fig)
    return path


def build_paper_fig_sources_per_var(
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    graphs_dir: Path | str = PAPER_OUTPUT_GRAPH_DIR,
) -> Path:
    clean_wide = _load_clean_wide(
        data_clean_dir=data_clean_dir,
        data_temp_dir=data_temp_dir,
        data_final_dir=data_final_dir,
    )
    variables = set(PAPER_VARIABLE_ORDER)
    long_df = _source_long(clean_wide, variables=variables, years={1900, 1980, 2008})
    coverage = (
        long_df.drop_duplicates(["ISO3", "year", "variable"])
        .groupby(["variable", "year"])["ISO3"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(PAPER_VARIABLE_ORDER)
        .dropna(how="all")
    )
    coverage = coverage.loc[coverage.sum(axis=1).gt(0)].copy()
    coverage["label"] = [PAPER_VARIABLE_LABELS.get(var, var) for var in coverage.index]
    coverage = coverage.reset_index(drop=True)

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(13, max(7, len(coverage) * 0.33)))
    y = np.arange(len(coverage))
    x1900 = coverage.get(1900, pd.Series(0, index=coverage.index)).astype(float).to_numpy()
    x1980 = coverage.get(1980, pd.Series(0, index=coverage.index)).astype(float).to_numpy()
    x2008 = coverage.get(2008, pd.Series(0, index=coverage.index)).astype(float).to_numpy()

    for idx in range(len(coverage)):
        ax.plot([x1900[idx], x1980[idx]], [y[idx], y[idx]], color="#999999", linestyle="--", linewidth=1)
        ax.plot([x1980[idx], x2008[idx]], [y[idx], y[idx]], color="#999999", linestyle="--", linewidth=1)
    ax.scatter(x1900, y, color="navy", label="1900", zorder=3)
    ax.scatter(x1980, y, color="maroon", label="1980", zorder=3)
    ax.scatter(x2008, y, color="magenta", label="2008", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(coverage["label"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Number of countries covered")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    path = _save_figure(fig, _figure_path("source_per_var.eps", graphs_dir=graphs_dir))
    plt.close(fig)
    return path


def build_paper_fig_boxplots_var(
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    graphs_dir: Path | str = PAPER_OUTPUT_GRAPH_DIR,
) -> list[Path]:
    clean_wide = _load_clean_wide(
        data_clean_dir=data_clean_dir,
        data_temp_dir=data_temp_dir,
        data_final_dir=data_final_dir,
    )
    long_df = _source_long(clean_wide, variables=set(PAPER_VARIABLE_ORDER), years={1900, 1950, 1980, 2008})
    source_counts = (
        long_df.groupby(["variable", "year", "ISO3"])["source"]
        .nunique()
        .reset_index(name="num_sources")
    )
    outputs: list[Path] = []
    plt = _mpl()
    for variable in PAPER_VARIABLE_ORDER:
        subset = source_counts.loc[source_counts["variable"].eq(variable)].copy()
        if subset.empty:
            continue
        ordered_years = [year for year in [1900, 1950, 1980, 2008] if year in subset["year"].tolist()]
        data = [subset.loc[subset["year"].eq(year), "num_sources"].to_numpy() for year in ordered_years]
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.boxplot(data, patch_artist=True, labels=[str(year) for year in ordered_years])
        ax.set_title(PAPER_VARIABLE_LABELS.get(variable, variable))
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="y", color="#efefef")
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        outputs.append(_save_figure(fig, _figure_path(f"Boxplot_{variable}.eps", graphs_dir=graphs_dir)))
        plt.close(fig)
    return outputs


def build_paper_fig_gdp_share_per_var(
    *,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    graphs_dir: Path | str = PAPER_OUTPUT_GRAPH_DIR,
) -> list[Path]:
    data_final = _load_data_final(data_final_dir=data_final_dir).drop(columns=["countryname"], errors="ignore")
    if "USDfx" in data_final.columns:
        work = data_final.copy()
    else:
        usdfx = _load_chainlinked("USDfx", data_final_dir=data_final_dir)[["ISO3", "year", "USDfx"]].copy()
        work = data_final.merge(usdfx, on=["ISO3", "year"], how="inner")

    work = work.loc[~work["ISO3"].astype(str).isin(GDP_SHARE_EXCLUSIONS)].copy()
    work = work.loc[~((work["ISO3"].astype(str) == "DEU") & pd.to_numeric(work["year"], errors="coerce").le(1945))].copy()
    work["nGDP_USDfx"] = pd.to_numeric(work["nGDP"], errors="coerce") / pd.to_numeric(work["USDfx"], errors="coerce")
    work["total_gdp"] = work.groupby("year")["nGDP_USDfx"].transform("sum")
    work["gdp_share"] = (work["nGDP_USDfx"] / work["total_gdp"]) * 100
    work = work.loc[pd.to_numeric(work["year"], errors="coerce").ge(1900) & work["gdp_share"].gt(0.1)].copy()

    for column in [col for col in work.columns if col not in {"ISO3", "year", "USDfx", "nGDP_USDfx", "total_gdp", "gdp_share"}]:
        work[column] = pd.to_numeric(work[column], errors="coerce").notna().astype(float) * work["gdp_share"]

    keep_vars = ["govexp", "govrev", "govtax", "govdebt_GDP", "govdef_GDP", "M0", "M1", "M2", "M3", "strate", "ltrate", "cbrate", "cons", "inv", "finv", "exports", "imports", "CA_GDP", "infl", "USDfx", "REER", "HPI", "unemp"]
    collapsed = work.groupby("year")[keep_vars].sum(min_count=1).reset_index()
    collapsed = collapsed.loc[pd.to_numeric(collapsed["year"], errors="coerce").le(2020)].copy()

    groups = [
        ("government_finances.eps", ["govexp", "govrev", "govtax", "govdebt_GDP", "govdef_GDP"], ["Expenditure", "Revenue", "Tax revenue", "Debt-to-GDP", "Deficit"]),
        ("Money_Rates.eps", ["M0", "M1", "M2", "M3", "strate", "ltrate", "cbrate"], ["Money supply (M0)", "Money supply (M1)", "Money supply (M2)", "Money supply (M3)", "Short-term interest rate", "Long-term interest rate", "Central bank policy rate"]),
        ("National_accounts_gdp.eps", ["cons", "inv", "finv", "exports", "imports", "CA_GDP"], ["Consumption", "Gross capital formation", "Gross fixed capital formation", "Exports", "Imports", "Current account"]),
        ("Prices_labor.eps", ["infl", "USDfx", "REER", "HPI", "unemp"], ["Inflation", "USD exchange rate", "Real effective exchange rate", "House price index", "Unemployment rate"]),
    ]

    plt = _mpl()
    outputs: list[Path] = []
    for filename, columns, labels in groups:
        fig, ax = plt.subplots(figsize=(11.5, 6.2))
        for column, label in zip(columns, labels):
            ax.plot(collapsed["year"], pd.to_numeric(collapsed[column], errors="coerce"), linewidth=2, label=label)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(color="#efefef")
        ax.legend(frameon=False, ncol=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        outputs.append(_save_figure(fig, _figure_path(filename, graphs_dir=graphs_dir)))
        plt.close(fig)
    return outputs


def build_paper_fig_worldmap_firstdata(
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
    graphs_dir: Path | str = PAPER_OUTPUT_GRAPH_DIR,
) -> Path:
    import shapefile
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon
    import matplotlib.colors as mcolors

    clean_wide = _load_clean_wide(
        data_clean_dir=data_clean_dir,
        data_temp_dir=data_temp_dir,
        data_final_dir=data_final_dir,
    )
    long_df = _source_long(clean_wide)
    first_year = (
        long_df.drop_duplicates(["ISO3", "year"])
        .groupby("ISO3")["year"]
        .min()
        .rename("year")
        .reset_index()
    )

    helper_dir = _resolve(data_helper_dir)
    shp_path = helper_dir / "WB_countries_Admin0_10m" / "WB_countries_Admin0_10m.shp"
    reader = shapefile.Reader(str(shp_path))
    fields = [field[0] for field in reader.fields[1:]]
    records = [dict(zip(fields, record)) for record in reader.records()]

    year_map = dict(zip(first_year["ISO3"].astype(str), first_year["year"].astype(float)))
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(12, 6.8))
    patches: list[Polygon] = []
    colors: list[float] = []

    for shape_rec, meta in zip(reader.shapes(), records):
        iso3 = str(meta.get("ISO_A3", "")).strip()
        iso3_eh = str(meta.get("ISO_A3_EH", "")).strip()
        if iso3 == "-99" and iso3_eh == "FRA":
            iso3 = "FRA"
        if not iso3 or iso3 == "-99":
            continue
        value = year_map.get(iso3)
        points = shape_rec.points
        parts = list(shape_rec.parts) + [len(points)]
        for start, end in zip(parts[:-1], parts[1:]):
            polygon = Polygon(points[start:end], closed=True)
            patches.append(polygon)
            colors.append(np.nan if value is None else float(value))

    valid = [value for value in colors if not np.isnan(value)]
    cmap = plt.get_cmap("RdYlBu_r")
    bounds = sorted(
        {
            int(min(valid or [1800])),
            1800,
            1850,
            1900,
            1925,
            1950,
            1975,
            int(max(valid or [2020])) + 1,
        }
    )
    if len(bounds) < 2:
        bounds = [0, 1]
    norm = mcolors.BoundaryNorm(
        boundaries=bounds,
        ncolors=cmap.N,
        clip=False,
    )
    collection = PatchCollection(patches, cmap=cmap, edgecolor="#888888", linewidth=0.2)
    collection.set_array(np.array([0 if np.isnan(value) else value for value in colors], dtype=float))
    collection.set_norm(norm)
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.axis("off")
    cbar = fig.colorbar(collection, ax=ax, orientation="horizontal", fraction=0.035, pad=0.02)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    path = _save_figure(fig, _figure_path("world_map.eps", graphs_dir=graphs_dir))
    plt.close(fig)
    return path


def build_paper_fig_stylized_fact_rates(
    *,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
    graphs_dir: Path | str = PAPER_OUTPUT_GRAPH_DIR,
) -> Path:
    work = _load_chainlinked("ltrate", data_final_dir=data_final_dir)[["ISO3", "year", "ltrate", "Schmelzing_ltrate"]].copy()
    keep = {"ITA", "GBR", "NLD", "DEU", "FRA", "USA", "ESP", "JPN", "BEL", "CHE", "SWE", "NOR", "DNK", "CAN"}
    work = work.loc[work["ISO3"].astype(str).isin(keep) & pd.to_numeric(work["year"], errors="coerce").ge(1875)].copy()
    work.loc[work["ISO3"].astype(str).isin({"ITA", "GBR", "NLD", "DEU", "FRA", "USA", "ESP", "JPN"}), "ltrate"] = np.nan
    schmelzing = work.groupby("year")["Schmelzing_ltrate"].mean().rename("ltrate").reset_index()
    schmelzing["ISO3"] = "Schmelzing"
    work = pd.concat([work[["ISO3", "year", "ltrate"]], schmelzing], ignore_index=True)
    work = work.drop_duplicates(["ISO3", "year"]).copy()
    work = work.loc[pd.to_numeric(work["year"], errors="coerce").le(2024)].copy()

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    order = ["Schmelzing", "BEL", "CHE", "SWE", "NOR", "DNK", "CAN"]
    labels = {
        "Schmelzing": "Schmelzing (2019)",
        "BEL": "Belgium",
        "CHE": "Switzerland",
        "SWE": "Sweden",
        "NOR": "Norway",
        "DNK": "Denmark",
        "CAN": "Canada",
    }
    for iso in order:
        subset = work.loc[work["ISO3"].eq(iso)].sort_values("year")
        if subset.empty:
            continue
        ax.plot(subset["year"], subset["ltrate"], linewidth=2.3 if iso == "Schmelzing" else 1.2, label=labels[iso], color="black" if iso == "Schmelzing" else None)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(color="#efefef")
    ax.legend(frameon=False, ncol=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    path = _save_figure(fig, _figure_path("stylized_fact_rates.eps", graphs_dir=graphs_dir))
    plt.close(fig)
    return path


def build_paper_fig_stylized_fact_trade(
    *,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    graphs_dir: Path | str = PAPER_OUTPUT_GRAPH_DIR,
) -> Path:
    exports = _load_chainlinked("exports", data_final_dir=data_final_dir)[["ISO3", "year", "exports"]].copy()
    usdfx = _load_chainlinked("USDfx", data_final_dir=data_final_dir)[["ISO3", "year", "USDfx"]].copy()
    work = exports.merge(usdfx, on=["ISO3", "year"], how="inner")
    work["exports_USD"] = pd.to_numeric(work["exports"], errors="coerce") / pd.to_numeric(work["USDfx"], errors="coerce")
    work = work.loc[~work["ISO3"].astype(str).isin({"MMR", "SLE", "ROU", "ZWE", "POL", "YUG"})].copy()
    work = work.loc[pd.to_numeric(work["year"], errors="coerce").between(1850, 2024)].copy()
    work = work.dropna(subset=["exports_USD"])
    work["total_exports"] = work.groupby("year")["exports_USD"].transform("sum")
    work["export_share"] = (work["exports_USD"] / work["total_exports"]) * 100
    selected = ["USA", "FRA", "GBR", "JPN", "CHN", "DEU"]
    work["country"] = np.where(work["ISO3"].astype(str).isin(selected), work["ISO3"], "ROW")
    collapsed = work.groupby(["year", "country"])["export_share"].sum().reset_index()
    pivot = collapsed.pivot(index="year", columns="country", values="export_share").fillna(0.0)
    order = ["USA", "FRA", "GBR", "JPN", "CHN", "DEU", "ROW"]
    labels = ["United States", "France", "United Kingdom", "Japan", "China", "Germany", "Rest of World"]

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.stackplot(pivot.index, [pivot.get(country, pd.Series(0.0, index=pivot.index)) for country in order], labels=labels, alpha=0.9)
    ax.set_xlabel("")
    ax.set_ylabel("Share of Global Exports (%)")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    path = _save_figure(fig, _figure_path("stylized_fact_trade.eps", graphs_dir=graphs_dir))
    plt.close(fig)
    return path


def build_paper_fig_stylized_fact_usd(
    *,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
    graphs_dir: Path | str = PAPER_OUTPUT_GRAPH_DIR,
) -> list[Path]:
    work = _load_chainlinked("USDfx", data_final_dir=data_final_dir)[["ISO3", "year", "USDfx"]].copy()
    work = work.dropna(subset=["USDfx"]).loc[pd.to_numeric(work["year"], errors="coerce").le(2023)].copy()
    work["logfx"] = np.log(pd.to_numeric(work["USDfx"], errors="coerce"))
    work = work.sort_values(["ISO3", "year"])
    work["dlogfx"] = work.groupby("ISO3")["logfx"].diff()
    summary = work.groupby("ISO3")["dlogfx"].std().rename("dlogfx").reset_index()
    summary = summary.loc[summary["dlogfx"].ne(0)].copy()
    summary["rank"] = summary["dlogfx"].rank(method="first")
    n_countries = len(summary)
    summary["performance"] = "Middle"
    summary.loc[summary["rank"].le(20), "performance"] = "Top 20"
    summary.loc[summary["rank"].gt(n_countries - 20), "performance"] = "Bottom 20"
    summary = summary.loc[summary["performance"].ne("Middle")].copy()

    country_names = _country_names(data_helper_dir=data_helper_dir)
    ranges = work.groupby("ISO3")["year"].agg(["min", "max"]).reset_index()
    summary = summary.merge(country_names, on="ISO3", how="left").merge(ranges, on="ISO3", how="left")
    summary["countryname"] = summary["countryname"].fillna(summary["ISO3"])
    summary["countryname"] = summary["countryname"].replace(
        {
            "Democratic Republic of the Congo": "Congo DRC",
            "Saint Vincent and the Grenadines": "St-Vincent",
            "Saint Kitts and Nevis": "St-Kitts and Nevis",
        }
    )
    summary["countryname"] = summary["countryname"] + " (" + summary["min"].astype(int).astype(str) + "-" + summary["max"].astype(int).astype(str) + ")"

    plt = _mpl()
    outputs: list[Path] = []
    for filename, mask, color in [
        ("stylized_fact_rates_USD_1.eps", summary["performance"].eq("Bottom 20"), "#dc1414"),
        ("stylized_fact_rates_USD_2.eps", summary["performance"].eq("Top 20"), "#1f6b2a"),
    ]:
        subset = summary.loc[mask].sort_values("dlogfx")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(subset["countryname"], subset["dlogfx"], color=color)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="x", color="#efefef")
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        outputs.append(_save_figure(fig, _figure_path(filename, graphs_dir=graphs_dir)))
        plt.close(fig)
    return outputs


def build_paper_fig_chile(
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    graphs_dir: Path | str = PAPER_OUTPUT_GRAPH_DIR,
) -> Path | None:
    clean_wide = _load_clean_wide(
        data_clean_dir=data_clean_dir,
        data_temp_dir=data_temp_dir,
        data_final_dir=data_final_dir,
    )
    return build_country_heatmap(clean_wide, iso3="CHL", output_path=_figure_path("CHL_heatmap.eps", graphs_dir=graphs_dir))


def build_paper_fig_sweden(
    *,
    data_clean_dir: Path | str = REPO_ROOT / "data" / "clean",
    data_temp_dir: Path | str = sh.DATA_TEMP_DIR,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    graphs_dir: Path | str = PAPER_OUTPUT_GRAPH_DIR,
) -> Path | None:
    clean_wide = _load_clean_wide(
        data_clean_dir=data_clean_dir,
        data_temp_dir=data_temp_dir,
        data_final_dir=data_final_dir,
    )
    return build_country_heatmap(
        clean_wide,
        iso3="SWE",
        output_path=_figure_path("SWE_heatmap.eps", graphs_dir=graphs_dir),
        min_year=1300,
    )


def build_paper_fig_fra(
    *,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
    graphs_dir: Path | str = PAPER_OUTPUT_GRAPH_DIR,
) -> list[Path]:
    df = _load_chainlinked("inv", data_final_dir=data_final_dir)
    df = df.loc[df["ISO3"].astype(str).eq("FRA")].copy()
    return sh.gmdmakeplot_cs(
        df,
        "inv",
        log=True,
        ylabel="Investment, millions of LCU (Log scale)",
        y_axislabel='0 "1" 2 "10000" 4 "100000" 6 "500000" 8 "1000000"',
        graphformat="eps",
        data_helper_dir=data_helper_dir,
        graphs_dir=graphs_dir,
    )


def build_paper_fig_gbr(
    *,
    data_final_dir: Path | str = sh.DATA_FINAL_DIR,
    data_helper_dir: Path | str = sh.DATA_HELPER_DIR,
    graphs_dir: Path | str = PAPER_OUTPUT_GRAPH_DIR,
) -> list[Path]:
    outputs: list[Path] = []
    specs = [
        ("nGDP", True, "Nominal GDP, millions of LCU (Log scale)", '0 "1" 2 "10000" 4 "100000" 6 "2000000" 8 "4000000"'),
        ("exports", True, "Exports, millions of LCU (Log scale)", '0 "1" 2 "1000" 4 "100000" 6 "500000" 8 "1000000"'),
        ("govdebt_GDP", False, "Government debt, % of GDP", '0 "0" 50 "50" 100 "100" 150 "150" 200 "200" 250 "250" 300 "300"'),
    ]
    for varname, log, ylabel, ticks in specs:
        df = _load_chainlinked(varname, data_final_dir=data_final_dir)
        df = df.loc[df["ISO3"].astype(str).eq("GBR")].copy()
        outputs.extend(
            sh.gmdmakeplot_cs(
                df,
                varname,
                log=log,
                ylabel=ylabel,
                y_axislabel=ticks,
                graphformat="eps",
                data_helper_dir=data_helper_dir,
                graphs_dir=graphs_dir,
            )
        )
    return outputs


__all__ = [
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
]
