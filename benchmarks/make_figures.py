#!/usr/bin/env python3
"""Draw the benchmark figures from the result JSON. No numbers are typed in here."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from libbench import FIGURES, RESULTS, ensure_dirs, read_json, write_json

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

INK = "#1c2b33"
TEAL = "#0f6e6b"
CLAY = "#c4492c"
SAND = "#f6f3ec"
GOLD = "#c4a15a"


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _box(ax, xy, w, h, text, fc):
    patch = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.012,rounding_size=0.08",
        facecolor=fc, edgecolor=INK, linewidth=0.8,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=8, color=INK, wrap=True)


def figure_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    _box(ax, (0.3, 4.7), 2.6, 1.1, "NIfTI or NRRD\nimage and mask", SAND)
    _box(ax, (3.6, 4.7), 2.8, 1.1, "Python\nload, geometry, YAML", "#d7ebe9")
    _box(ax, (7.1, 4.7), 2.5, 1.1, "mode\nibsi | pyradiomics", "#f3e6c8")
    _box(ax, (1.5, 2.5), 7.0, 1.5, "Rust engine\nresample  ·  resegment  ·  discretise  ·  filters  ·  features", "#d9e4ea")
    _box(ax, (2.2, 0.4), 5.6, 1.2, "Feature table  +  SHA-256 configuration hash", SAND)
    for x0, x1, y in ((2.9, 3.6, 5.25), (6.4, 7.1, 5.25)):
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
    ax.annotate("", xy=(5.0, 4.0), xytext=(5.0, 4.7), arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
    ax.annotate("", xy=(5.0, 1.6), xytext=(5.0, 2.5), arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
    ax.set_title("pyradix data flow", loc="left", color=INK, fontsize=11)
    save(fig, "fig1_architecture")


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def figure_ibsi(ibsi: dict) -> None:
    groups = []
    for row in ibsi.get("digital_phantom_rows", []):
        groups.append(("Digital phantom", row))
    for row in ibsi.get("pat1_rows", []):
        groups.append((f"PAT1 {row.get('config', '')}", row))
    if not groups:
        return
    # Maximum |error| / published tolerance, by family, for each case.
    cases = []
    for label, _row in groups:
        if label not in cases:
            cases.append(label)
    families = []
    for _label, row in groups:
        fam = row.get("family") or "other"
        if fam not in families:
            families.append(fam)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    width = 0.18
    x0 = list(range(len(families)))
    for i, case in enumerate(cases):
        heights = []
        for fam in families:
            ratios = []
            for label, row in groups:
                if label != case or (row.get("family") or "other") != fam:
                    continue
                tol = float(row["tol"]) if row.get("tol") not in (None, "") else float("nan")
                err = float(row["abs_err"]) if row.get("abs_err") not in (None, "") else float("nan")
                if tol and tol == tol and tol > 0 and err == err:
                    ratios.append(err / tol)
            heights.append(max(ratios) if ratios else float("nan"))
        offset = (i - (len(cases) - 1) / 2) * width
        ax.bar([x + offset for x in x0], heights, width=width, label=case, color=[TEAL, "#3d8b88", GOLD, CLAY, "#6b7c85"][i % 5])
    ax.axhline(1.0, color=CLAY, lw=0.8, ls="--", label="published tolerance")
    ax.set_xticks(x0)
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_ylabel("max |error| / published tolerance")
    ax.set_title("IBSI-1 reference rows", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_ylim(bottom=0)
    save(fig, "fig2_ibsi_agreement")


def _percent(a: float, b: float) -> float | None:
    mean = 0.5 * (a + b)
    if abs(mean) < 1e-8:
        return None
    return 100.0 * (a - b) / mean


def figure_bland(feature_rows: list[dict[str, str]]) -> dict:
    """Bland–Altman on percent difference, plus Lin CCC and ICC(A,1)."""
    wanted = [("d4_brain1", "bw25"), ("d2_pat1", "bw25")]
    stats = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=False)
    for ax, (dataset, config) in zip(axes, wanted):
        xs, ys = [], []
        paired = []
        for row in feature_rows:
            if row["dataset"] != dataset or row["config"] != config:
                continue
            if row["radix"] in ("", None) or row["reference"] in ("", None):
                continue
            a, b = float(row["radix"]), float(row["reference"])
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            pct = _percent(a, b)
            paired.append((a, b))
            if pct is None or not math.isfinite(pct):
                continue
            xs.append(0.5 * (a + b))
            ys.append(pct)
        if not ys:
            ax.set_visible(False)
            stats[f"{dataset}__{config}"] = {"n": 0}
            continue
        bias = sum(ys) / len(ys)
        var = sum((y - bias) ** 2 for y in ys) / max(len(ys) - 1, 1)
        sd = math.sqrt(var)
        ax.scatter([abs(x) for x in xs], ys, s=16, c=TEAL, alpha=0.8, linewidths=0)
        ax.axhline(bias, color=INK, lw=0.8)
        ax.axhline(bias + 1.96 * sd, color=CLAY, lw=0.7, ls="--")
        ax.axhline(bias - 1.96 * sd, color=CLAY, lw=0.7, ls="--")
        ax.set_xscale("log")
        span = max(abs(y) for y in ys) * 1.6
        ax.set_ylim(-span, span)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_title(f"{dataset} {config}\n(y zoomed; tolerance is ±0.01%)", loc="left", fontsize=8)
        ax.set_xlabel("|mean| of the two values")
        ccc = lin_ccc([p[0] for p in paired], [p[1] for p in paired])
        icc = icc_a1([p[0] for p in paired], [p[1] for p in paired])
        stats[f"{dataset}__{config}"] = {
            "n_plotted": len(ys),
            "n_paired": len(paired),
            "bias_percent": bias,
            "sd_percent": sd,
            "loa_low_percent": bias - 1.96 * sd,
            "loa_high_percent": bias + 1.96 * sd,
            "max_abs_percent": max(abs(y) for y in ys),
            "lin_ccc": ccc,
            "icc_a1": icc,
            "icc_unit": "one feature on one image is one subject; two softwares are the raters",
        }
    axes[0].set_ylabel("percent difference\n(pyradix − PyRadiomics) / mean")
    fig.suptitle("Bland–Altman, original features", x=0.02, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    save(fig, "fig3_bland_altman")
    return stats


def lin_ccc(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x) / n
    syy = sum((b - my) ** 2 for b in y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    den = sxx + syy + (mx - my) ** 2
    if den == 0:
        return None
    return 2 * sxy / den


def icc_a1(x: list[float], y: list[float]) -> float | None:
    """Two-way agreement ICC, single measures, from McGraw and Wong."""
    n = len(x)
    k = 2
    if n < 3:
        return None
    rows = [(a, b) for a, b in zip(x, y)]
    grand = sum(a + b for a, b in rows) / (n * k)
    mean_i = [(a + b) / k for a, b in rows]
    mean_j = [sum(a for a, _b in rows) / n, sum(b for _a, b in rows) / n]
    ms_rows = k * sum((m - grand) ** 2 for m in mean_i) / (n - 1)
    ms_cols = n * sum((m - grand) ** 2 for m in mean_j) / (k - 1)
    ss_err = 0.0
    for (a, b), mi in zip(rows, mean_i):
        ss_err += (a - mi - mean_j[0] + grand) ** 2
        ss_err += (b - mi - mean_j[1] + grand) ** 2
    ms_err = ss_err / ((n - 1) * (k - 1))
    den = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    if den == 0:
        return None
    return (ms_rows - ms_err) / den


def figure_runtime(runtime: dict) -> None:
    rows = runtime.get("rows", [])
    size = [r for r in rows if r["series"] == "roi_size" and not r.get("error") and r.get("median_s") is not None]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for engine, color in (("radix", TEAL), ("pyradiomics", CLAY)):
        pts = [r for r in size if r["engine"] == engine and r.get("roi_voxels_reported")]
        pts.sort(key=lambda r: r["roi_voxels_reported"])
        if not pts:
            continue
        ax.plot(
            [r["roi_voxels_reported"] for r in pts],
            [r["median_s"] for r in pts],
            marker="o",
            color=color,
            label=engine,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ROI voxels (PAT1 centroid subset, or the full mask)")
    ax.set_ylabel("median extract time (s)")
    ax.set_title("Original features, 4 threads, bin width 25", loc="left", color=INK)
    ax.legend(frameon=False)
    save(fig, "fig4_runtime_size")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=False)
    panels = (
        ("threads_original", "brain1 original"),
        ("threads_log", "brain1 LoG, first order"),
    )
    for ax, (series, title) in zip(axes, panels):
        for engine, color in (("radix", TEAL), ("pyradiomics", CLAY)):
            pts = [
                r
                for r in rows
                if r["series"] == series and r["engine"] == engine and not r.get("error") and r.get("median_s") is not None
            ]
            pts.sort(key=lambda r: r["threads"])
            if not pts:
                continue
            ax.plot([r["threads"] for r in pts], [r["median_s"] for r in pts], marker="o", color=color, label=engine)
        ax.set_xticks([1, 2, 4])
        ax.set_xlabel("threads")
        ax.set_title(title, loc="left", fontsize=9)
    axes[0].set_ylabel("median time (s)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Thread pin on public brain1", x=0.02, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    save(fig, "fig5_runtime_threads")


def figure_coverage(ibsi: dict, feature_rows: list[dict[str, str]]) -> list[list[str]]:
    from pyradix._radix import feature_catalog

    catalog = feature_catalog()
    ibsi_rows = ibsi.get("digital_phantom_rows", []) + ibsi.get("pat1_rows", [])
    brain = [r for r in feature_rows if r["dataset"] == "d4_brain1" and r["config"] == "bw25"]
    header = ["family", "implemented", "IBSI rows", "IBSI pass", "brain1 shared", "brain1 pass"]
    table = [header]
    order = list(catalog.keys())
    for fam in order:
        names = catalog[fam]["features"]
        ibsi_fam = [r for r in ibsi_rows if r.get("family") == fam]
        brain_fam = [r for r in brain if r.get("family") == fam]
        table.append(
            [
                fam,
                str(len(names)),
                str(len(ibsi_fam)),
                str(sum(1 for r in ibsi_fam if str(r.get("pass")).lower() == "true" or r.get("pass") is True)),
                str(len(brain_fam)),
                str(sum(1 for r in brain_fam if str(r.get("pass")).lower() == "true")),
            ]
        )
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axis("off")
    cell = ax.table(cellText=table[1:], colLabels=table[0], loc="center", cellLoc="center")
    cell.auto_set_font_size(False)
    cell.set_fontsize(8)
    cell.scale(1, 1.35)
    for (r, c), cell_obj in cell.get_celld().items():
        cell_obj.set_edgecolor("#d0d5d8")
        cell_obj.set_linewidth(0.4)
        if r == 0:
            cell_obj.set_facecolor(TEAL)
            cell_obj.set_text_props(color="white")
        elif r % 2 == 0:
            cell_obj.set_facecolor(SAND)
        else:
            cell_obj.set_facecolor("white")
    ax.set_title("Feature coverage against the checks that were run", loc="left", color=INK, fontsize=11, pad=12)
    save(fig, "fig6_feature_coverage")
    return table


def _tex(value) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def _fmt(value, digits=4) -> str:
    if value is None:
        return "[TO RUN]"
    if isinstance(value, float):
        if value != value:
            return "[TO RUN]"
        return f"{value:.{digits}g}"
    return str(value)


def write_generated(ibsi: dict, agree: dict, runtime: dict, bland: dict, table: list[list[str]]) -> None:
    lines = [
        "% Generated by benchmarks/make_figures.py from the result JSON. Do not edit.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{IBSI-1 reference rows inside the published tolerance.}",
        "\\label{tab:ibsi}",
        "\\begin{tabular}{lrrr}",
        "\\hline",
        "Check & rows & pass & fail \\\\",
        "\\hline",
    ]
    gates = ibsi.get("digital_phantom", {}).get("gates", {})
    for gate, block in gates.items():
        lines.append(
            f"Digital phantom {gate} & {block['n_rows']} & {block['n_pass']} & {block['n_fail']} \\\\"
        )
    for config, block in ibsi.get("pat1", {}).get("configs", {}).items():
        lines.append(
            f"PAT1 {config} & {block['n_rows']} & {block['n_pass']} & {block['n_fail']} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    lines += [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Live Original-feature agreement with PyRadiomics 3.1.0. Tolerance $\\max(10^{-4},10^{-4}|b|)$.}",
        "\\label{tab:agree}",
        "\\begin{tabular}{llrrrr}",
        "\\hline",
        "Data & setting & shared & pass & fail & speedup \\\\",
        "\\hline",
    ]
    for job in agree.get("jobs", []):
        agr = job.get("agreement") or {}
        speed = job.get("speedup")
        lines.append(
            f"{_tex(job.get('dataset',''))} & {_tex(job.get('config',''))} & {agr.get('n_shared','')} & "
            f"{agr.get('n_pass','')} & {agr.get('n_fail','')} & {_fmt(speed, 3)} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    if table:
        lines += [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\caption{Names implemented, IBSI-1 rows exercised, and brain1 bin-width 25 keys shared with PyRadiomics 3.1.0.}",
            "\\label{tab:coverage}",
            "\\begin{tabular}{lrrrrr}",
            "\\hline",
            " & ".join(table[0]) + " \\\\",
            "\\hline",
        ]
        for row in table[1:]:
            lines.append(" & ".join(row) + " \\\\")
        lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    out = Path(os.environ.get("RESULTS_TEX", RESULTS / "generated_results.tex"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log = ibsi.get("ibsi2_log_impulse", {})
    summary = {
        "ibsi_digital": gates,
        "ibsi_pat1": {
            k: {kk: vv for kk, vv in block.items() if kk != "rows"}
            for k, block in ibsi.get("pat1", {}).get("configs", {}).items()
        },
        "ibsi2_log_max_abs_err": log.get("max_abs_err"),
        "ibsi2_pytest_returncode": (ibsi.get("ibsi2_pytest") or {}).get("returncode"),
        "ibsi2_pytest_summary": (ibsi.get("ibsi2_pytest") or {}).get("summary"),
        "agreement": [
            {
                "dataset": job.get("dataset"),
                "config": job.get("config"),
                "n_shared": (job.get("agreement") or {}).get("n_shared"),
                "n_pass": (job.get("agreement") or {}).get("n_pass"),
                "n_fail": (job.get("agreement") or {}).get("n_fail"),
                "worst_feature": (job.get("agreement") or {}).get("worst_feature"),
                "worst_rel_err": (job.get("agreement") or {}).get("worst_rel_err"),
                "speedup": job.get("speedup"),
                "radix_median_s": job.get("radix_median_s"),
                "pyradiomics_median_s": job.get("pyradiomics_median_s"),
                "radix_max_rss_kb": job.get("radix_max_rss_kb"),
                "pyradiomics_max_rss_kb": job.get("pyradiomics_max_rss_kb"),
                "radix_error": job.get("radix_error"),
                "pyradiomics_error": job.get("pyradiomics_error"),
            }
            for job in agree.get("jobs", [])
        ],
        "bland_altman": bland,
        "runtime_rows": [
            {
                "series": r.get("series"),
                "engine": r.get("engine"),
                "threads": r.get("threads"),
                "requested_roi_voxels": r.get("requested_roi_voxels"),
                "roi_voxels_reported": r.get("roi_voxels_reported"),
                "median_s": r.get("median_s"),
                "max_rss_kb": r.get("max_rss_kb"),
                "error": r.get("error"),
            }
            for r in runtime.get("rows", [])
        ],
    }
    write_json(RESULTS / "summary.json", summary)


def main() -> int:
    ensure_dirs()
    figure_architecture()
    ibsi = read_json(RESULTS / "ibsi.json") if (RESULTS / "ibsi.json").exists() else {}
    agree = read_json(RESULTS / "pyradiomics_agreement.json") if (RESULTS / "pyradiomics_agreement.json").exists() else {"jobs": []}
    runtime = read_json(RESULTS / "runtime.json") if (RESULTS / "runtime.json").exists() else {"rows": []}
    if ibsi:
        figure_ibsi(ibsi)
    rows = _rows(RESULTS / "pyradiomics_features.csv")
    bland = figure_bland(rows) if rows else {}
    if runtime.get("rows"):
        figure_runtime(runtime)
    table = figure_coverage(ibsi, rows) if ibsi else []
    write_generated(ibsi, agree, runtime, bland, table)
    print(f"figures in {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
