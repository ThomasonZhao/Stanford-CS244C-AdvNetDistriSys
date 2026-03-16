from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.visualize import CaseView, parse_summary

TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw zoomed and overview throughput-vs-bandwidth plots")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--zoom-output", type=str, required=True)
    parser.add_argument("--overview-output", type=str, required=True)
    return parser.parse_args()


def _successful_cases_by_stage(cases: List[CaseView]) -> Dict[int, List[CaseView]]:
    out: Dict[int, List[CaseView]] = {}
    for case in cases:
        if case.return_code != 0 or case.mean_tflops_per_s is None:
            continue
        out.setdefault(case.stage, []).append(case)
    return out


def _plot_zoom(cases_by_stage: Dict[int, List[CaseView]], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    for stage in sorted(cases_by_stage):
        stage_cases = sorted(
            (case for case in cases_by_stage[stage] if 0.0 < case.bandwidth_gbps <= 10.0),
            key=lambda case: case.bandwidth_gbps,
        )
        xs = [0.0]
        ys = [0.0]
        xs.extend(case.bandwidth_gbps for case in stage_cases)
        ys.extend(case.mean_tflops_per_s for case in stage_cases if case.mean_tflops_per_s is not None)
        ax.plot(xs, ys, marker="o", linewidth=2, label=f"stage {stage}")

    ax.set_xlim(0.0, 10.0)
    ax.set_xticks([0.5, 1.0, 2.5, 5.0, 10.0])
    ax.set_xlabel("Bandwidth limit (Gbps)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Approx. training TFLOPs / s", fontsize=LABEL_FONTSIZE)
    ax.set_title("TFLOPs vs Bandwidth (0-10 Gbps)", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)


def _plot_overview(cases_by_stage: Dict[int, List[CaseView]], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (ax, ax_inf) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(10.8, 5.8),
        gridspec_kw={"width_ratios": [8, 1], "wspace": 0.05},
    )

    for stage in sorted(cases_by_stage):
        finite_cases = sorted(
            (case for case in cases_by_stage[stage] if case.bandwidth_gbps > 0.0),
            key=lambda case: case.bandwidth_gbps,
        )
        unlimited_case = next((case for case in cases_by_stage[stage] if case.bandwidth_gbps <= 0.0), None)

        xs = [0.0]
        ys = [0.0]
        xs.extend(case.bandwidth_gbps for case in finite_cases)
        ys.extend(case.mean_tflops_per_s for case in finite_cases if case.mean_tflops_per_s is not None)
        line = ax.plot(xs, ys, marker="o", linewidth=2, label=f"stage {stage}")[0]
        if unlimited_case is not None and unlimited_case.mean_tflops_per_s is not None:
            ax_inf.plot([0.0], [unlimited_case.mean_tflops_per_s], marker="o", linestyle="None", color=line.get_color())

    ax.set_xlim(0.0, 100.0)
    ax.set_xticks([1.0, 10.0, 100.0])
    ax.set_xticklabels(["1", "10", "100"])
    ax.set_xlabel("Bandwidth limit (Gbps)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Approx. training TFLOPs / s", fontsize=LABEL_FONTSIZE)
    ax.set_title("TFLOPs vs Bandwidth Overview", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE)

    ax_inf.set_xlim(-0.5, 0.5)
    ax_inf.set_xticks([0.0])
    ax_inf.set_xticklabels(["inf"])
    ax_inf.grid(True, axis="y", alpha=0.3)
    ax_inf.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax_inf.tick_params(axis="y", left=False, labelleft=False)

    ax.spines["right"].set_visible(False)
    ax_inf.spines["left"].set_visible(False)

    d = 0.015
    kwargs = dict(transform=ax.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs = dict(transform=ax_inf.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax_inf.plot((-d, +d), (-d, +d), **kwargs)
    ax_inf.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.90, wspace=0.05)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    cases = parse_summary(run_dir / "summary.json")
    cases_by_stage = _successful_cases_by_stage(cases)
    if not cases_by_stage:
        raise RuntimeError("no successful throughput cases found")

    _plot_zoom(cases_by_stage, Path(args.zoom_output).resolve())
    _plot_overview(cases_by_stage, Path(args.overview_output).resolve())


if __name__ == "__main__":
    main()
