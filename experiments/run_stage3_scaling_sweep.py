from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import harness
from experiments.run_fit_memory_bandwidth import (
    TuningTrial,
    _detect_gpu_total_memory_mb,
    _select_max_batch_size,
    _training_extra_args,
    _trial_from_result,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep ZeRO Stage 3 across GPU counts with OOM-boundary microbatch tuning to recreate a Figure-3-style scaling curve"
    )
    parser.add_argument("--config", type=str, default="")

    parser.add_argument("--name", type=str, default="stage3_gpu_scaling")
    parser.add_argument("--results-dir", type=str, default="experiments/results")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--model-size", type=str, default="medium", choices=["tiny", "small", "medium"])
    parser.add_argument("--gpu-counts", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port-base", type=int, default=29500)
    parser.add_argument("--case-timeout-s", type=float, default=3600.0)

    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--collective-impl", type=str, default="torch", choices=["ring", "torch"])
    parser.add_argument("--data-mode", type=str, default="synthetic", choices=["synthetic", "fineweb"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--stage2-grad-bucket-mb", type=float, default=64.0)
    parser.add_argument("--profile-memory-interval", type=int, default=0)
    parser.add_argument("--metrics-warmup-steps", type=int, default=2)
    parser.add_argument("--tflops-mode", type=str, default="profile", choices=["estimate", "profile", "off"])
    parser.add_argument("--profile-collectives", action="store_true")

    parser.add_argument("--theory-vocab-size", type=int, default=0)
    parser.add_argument("--extra-args", type=str, default="")

    parser.add_argument("--fit-mode", type=str, default="oom_boundary", choices=["oom_boundary"])
    parser.add_argument(
        "--memory-metric",
        type=str,
        default="peak_cuda_max_allocated_mb",
        choices=[
            "peak_cuda_allocated_mb",
            "peak_cuda_reserved_mb",
            "peak_cuda_max_allocated_mb",
            "peak_cuda_max_reserved_mb",
        ],
    )
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--initial-batch-size", type=int, default=1)
    parser.add_argument("--batch-size-multiple", type=int, default=1)
    parser.add_argument("--growth-factor", type=float, default=2.0)
    parser.add_argument("--tuning-memory-warmup-steps", type=int, default=2)
    parser.add_argument(
        "--fixed-batch-size",
        type=int,
        default=0,
        help="If >0, skip OOM-boundary tuning entirely and benchmark this exact per-GPU microbatch for every GPU count.",
    )
    return parser


def _apply_config_to_parser(parser: argparse.ArgumentParser, config_path: str) -> None:
    payload = json.loads(Path(config_path).read_text())
    defaults = payload.get("defaults", {})
    matrix = payload.get("matrix", {})

    parser_defaults: Dict[str, object] = {}
    if isinstance(defaults, dict):
        parser_defaults.update(defaults)
    if isinstance(matrix, dict) and "gpu_counts" in matrix:
        parser_defaults["gpu_counts"] = matrix["gpu_counts"]
    if parser_defaults:
        parser.set_defaults(**parser_defaults)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    preliminary_args, _ = parser.parse_known_args(argv)
    if preliminary_args.config:
        _apply_config_to_parser(parser, preliminary_args.config)
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _normalized_gpu_counts(values: List[int]) -> List[int]:
    cleaned = sorted({int(value) for value in values if int(value) > 0})
    if not cleaned:
        raise ValueError("gpu_counts must contain at least one positive integer")
    return cleaned


def _benchmark_extra_args(args: argparse.Namespace) -> str:
    parts: List[str] = []
    if args.extra_args.strip():
        parts.append(args.extra_args.strip())
    parts.append(f"--tflops-mode {args.tflops_mode}")
    if args.profile_collectives:
        parts.append("--profile-collectives")
    return " ".join(parts)


def _make_case(
    args: argparse.Namespace,
    *,
    gpu_count: int,
    batch_size: int,
    steps: int,
    metrics_warmup_steps: int,
    extra_args: str,
) -> harness.CaseConfig:
    return harness.CaseConfig(
        stage=3,
        model_size=str(args.model_size),
        bandwidth_gbps=0.0,
        nproc_per_node=int(gpu_count),
        steps=int(steps),
        seq_len=int(args.seq_len),
        batch_size=int(batch_size),
        grad_accum_steps=int(args.grad_accum_steps),
        collective_impl=str(args.collective_impl),
        data_mode=str(args.data_mode),
        seed=int(args.seed),
        dtype=str(args.dtype),
        max_grad_norm=float(args.max_grad_norm),
        stage2_grad_bucket_mb=float(args.stage2_grad_bucket_mb),
        profile_memory_interval=int(args.profile_memory_interval),
        metrics_warmup_steps=int(metrics_warmup_steps),
        bandwidth_mode="none",
        sim_latency_ms=0.0,
        tc_interface="eth0",
        socket_interface="lo",
        socket_shaper_burst_bytes=262_144,
        theory_vocab_size=int(args.theory_vocab_size),
        extra_args=str(extra_args),
    )


def _summary_row(
    *,
    gpu_count: int,
    selected_batch_size: int,
    selected_peak_memory_mb: float | None,
    benchmark_result: harness.CaseResult,
) -> Dict[str, object]:
    total_tflops = benchmark_result.mean_tflops_per_s
    if total_tflops is None:
        raise RuntimeError(
            f"missing TFLOPs for gpu_count={gpu_count}; rerun with a TFLOPs-enabled mode such as --tflops-mode profile"
        )

    global_tokens_per_step = int(
        benchmark_result.config["batch_size"]
        * benchmark_result.config["grad_accum_steps"]
        * benchmark_result.config["seq_len"]
        * benchmark_result.config["nproc_per_node"]
    )
    return {
        "gpu_count": int(gpu_count),
        "selected_batch_size": int(selected_batch_size),
        "selected_peak_memory_mb": None if selected_peak_memory_mb is None else float(selected_peak_memory_mb),
        "global_tokens_per_step": global_tokens_per_step,
        "mean_tokens_per_s": None
        if benchmark_result.mean_tokens_per_s is None
        else float(benchmark_result.mean_tokens_per_s),
        "mean_tflops_per_s": float(total_tflops),
        "per_gpu_tflops_per_s": float(total_tflops) / float(gpu_count),
        "peak_cuda_max_allocated_mb": None
        if benchmark_result.peak_cuda_max_allocated_mb is None
        else float(benchmark_result.peak_cuda_max_allocated_mb),
        "mean_comm_ms": None if benchmark_result.mean_comm_ms is None else float(benchmark_result.mean_comm_ms),
        "mean_fb_ms": None if benchmark_result.mean_fb_ms is None else float(benchmark_result.mean_fb_ms),
        "mean_opt_ms": None if benchmark_result.mean_opt_ms is None else float(benchmark_result.mean_opt_ms),
        "return_code": int(benchmark_result.return_code),
        "case_id": str(benchmark_result.case_id),
        "log_path": str(benchmark_result.log_path),
        "profile_path": str(benchmark_result.profile_path),
    }


def main() -> None:
    args = parse_args()
    gpu_counts = _normalized_gpu_counts(list(args.gpu_counts))

    launch = harness._launch_config_from_args(args)
    run_dir = Path(args.results_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    tuning_dir = run_dir / "tuning"

    per_gpu_total_memory_mb = _detect_gpu_total_memory_mb(max(gpu_counts))
    tuning_steps = int(args.tuning_memory_warmup_steps) + 1
    case_counter = itertools.count()
    benchmark_extra_args = _benchmark_extra_args(args)
    scaling_mode = "fixed_batch" if int(args.fixed_batch_size) > 0 else "fit_to_oom"

    per_gpu_count: Dict[str, object] = {}
    benchmark_results: List[harness.CaseResult] = []
    scaling_points: List[Dict[str, object]] = []
    previous_selected_batch_size: int | None = None

    for gpu_count in gpu_counts:
        if int(args.fixed_batch_size) > 0:
            fixed_batch_size = int(args.fixed_batch_size)
            benchmark_case = _make_case(
                args,
                gpu_count=gpu_count,
                batch_size=fixed_batch_size,
                steps=int(args.steps),
                metrics_warmup_steps=int(args.metrics_warmup_steps),
                extra_args=benchmark_extra_args,
            )
            print(
                f"[stage3-scaling] benchmark gpu_count={gpu_count} fixed_batch_size={fixed_batch_size}",
                flush=True,
            )
            benchmark_result = harness._run_case(
                case=benchmark_case,
                run_dir=run_dir,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
                launch=launch,
                case_index=next(case_counter),
            )
            benchmark_results.append(benchmark_result)
            per_gpu_count[str(gpu_count)] = {
                "gpu_count": int(gpu_count),
                "fit_status": "fit" if benchmark_result.return_code == 0 else "benchmark_failed",
                "reason": "fixed_batch",
                "selected_batch_size": fixed_batch_size,
                "selected_peak_memory_mb": None,
                "selected_case_id": None,
                "selected_log_path": None,
                "selected_profile_path": None,
                "tuning_trials": [],
                "benchmark_result": asdict(benchmark_result),
            }
            if benchmark_result.return_code != 0:
                print(
                    f"[stage3-scaling] fixed-batch benchmark failed gpu_count={gpu_count} (see {benchmark_result.log_path})",
                    flush=True,
                )
                continue
            if not args.dry_run:
                scaling_points.append(
                    _summary_row(
                        gpu_count=gpu_count,
                        selected_batch_size=fixed_batch_size,
                        selected_peak_memory_mb=None,
                        benchmark_result=benchmark_result,
                    )
                )
            continue

        initial_batch_size = int(args.initial_batch_size)
        if previous_selected_batch_size is not None:
            initial_batch_size = max(initial_batch_size, previous_selected_batch_size)

        def evaluate(batch_size: int) -> TuningTrial:
            tuning_case = _make_case(
                args,
                gpu_count=gpu_count,
                batch_size=batch_size,
                steps=tuning_steps,
                metrics_warmup_steps=0,
                extra_args=_training_extra_args(benchmark_extra_args, int(args.tuning_memory_warmup_steps)),
            )
            result = harness._run_case(
                case=tuning_case,
                run_dir=tuning_dir,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
                launch=launch,
                case_index=next(case_counter),
            )
            return _trial_from_result(
                result,
                fit_mode=str(args.fit_mode),
                memory_metric=str(args.memory_metric),
                memory_budget_mb=None,
            )

        print(f"[stage3-scaling] tuning gpu_count={gpu_count}", flush=True)
        try:
            best_trial, trials = _select_max_batch_size(
                min_batch_size=int(args.min_batch_size),
                max_batch_size=int(args.max_batch_size),
                batch_size_multiple=int(args.batch_size_multiple),
                growth_factor=float(args.growth_factor),
                initial_batch_size=int(initial_batch_size),
                evaluator=evaluate,
            )
        except RuntimeError as exc:
            per_gpu_count[str(gpu_count)] = {
                "gpu_count": int(gpu_count),
                "fit_status": "no_fit",
                "reason": str(exc),
                "selected_batch_size": None,
                "benchmark_result": None,
                "tuning_trials": [],
            }
            print(f"[stage3-scaling] gpu_count={gpu_count} no fit: {exc}", flush=True)
            continue

        previous_selected_batch_size = int(best_trial.batch_size)
        tuning_payload = {
            "gpu_count": int(gpu_count),
            "fit_status": "fit",
            "reason": str(best_trial.reason),
            "selected_batch_size": int(best_trial.batch_size),
            "selected_peak_memory_mb": None
            if best_trial.peak_memory_mb is None
            else float(best_trial.peak_memory_mb),
            "selected_case_id": str(best_trial.case_id),
            "selected_log_path": str(best_trial.log_path),
            "selected_profile_path": str(best_trial.profile_path),
            "tuning_trials": [asdict(trial) for trial in trials],
        }

        benchmark_case = _make_case(
            args,
            gpu_count=gpu_count,
            batch_size=int(best_trial.batch_size),
            steps=int(args.steps),
            metrics_warmup_steps=int(args.metrics_warmup_steps),
            extra_args=benchmark_extra_args,
        )
        print(
            f"[stage3-scaling] benchmark gpu_count={gpu_count} batch_size={best_trial.batch_size}",
            flush=True,
        )
        benchmark_result = harness._run_case(
            case=benchmark_case,
            run_dir=run_dir,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
            launch=launch,
            case_index=next(case_counter),
        )
        tuning_payload["benchmark_result"] = asdict(benchmark_result)
        per_gpu_count[str(gpu_count)] = tuning_payload
        benchmark_results.append(benchmark_result)

        if benchmark_result.return_code != 0:
            print(
                f"[stage3-scaling] benchmark failed gpu_count={gpu_count} (see {benchmark_result.log_path})",
                flush=True,
            )
            continue

        if not args.dry_run:
            scaling_points.append(
                _summary_row(
                    gpu_count=gpu_count,
                    selected_batch_size=int(best_trial.batch_size),
                    selected_peak_memory_mb=best_trial.peak_memory_mb,
                    benchmark_result=benchmark_result,
                )
            )

    scaling_points.sort(key=lambda item: int(item["gpu_count"]))
    base_point = scaling_points[0] if scaling_points else None
    if base_point is not None:
        base_gpu_count = int(base_point["gpu_count"])
        base_total_tflops = float(base_point["mean_tflops_per_s"])
        for point in scaling_points:
            gpu_count = int(point["gpu_count"])
            total_tflops = float(point["mean_tflops_per_s"])
            perfect_linear_tflops = base_total_tflops * (float(gpu_count) / float(base_gpu_count))
            speedup = total_tflops / base_total_tflops
            ideal_speedup = float(gpu_count) / float(base_gpu_count)
            point["perfect_linear_tflops_per_s"] = perfect_linear_tflops
            point["speedup_vs_base"] = speedup
            point["scaling_efficiency_vs_base"] = speedup / ideal_speedup if ideal_speedup > 0 else None
            point["superlinear_gain_vs_perfect_linear"] = total_tflops / perfect_linear_tflops

    summary = harness._build_summary(results=benchmark_results, args=args)
    summary["experiment_type"] = "stage3_gpu_scaling"
    if scaling_mode == "fixed_batch":
        summary["methodology_note"] = (
            "This run adapts ZeRO Figure 3 to a single-host GPU-count sweep using one fixed per-GPU microbatch "
            "for every GPU count, without OOM-boundary tuning."
        )
    else:
        summary["methodology_note"] = (
            "This run adapts ZeRO Figure 3 to a single-host GPU-count sweep. For each GPU count, "
            "ZeRO Stage 3 is tuned to the largest per-GPU microbatch that fits at the OOM boundary, "
            "then benchmarked without artificial bandwidth shaping."
        )
    summary["figure3_reference_note"] = (
        "The original Figure 3 in the ZeRO paper reports a fixed 60B model from 64 to 400 GPUs. "
        "This repo-level adaptation uses the same scaling idea on one 16-GPU host with the largest supported preset model."
    )
    summary["scaling_mode"] = scaling_mode
    summary["fixed_batch_size"] = None if int(args.fixed_batch_size) <= 0 else int(args.fixed_batch_size)
    summary["gpu_counts"] = gpu_counts
    summary["per_gpu_total_memory_mb"] = float(per_gpu_total_memory_mb)
    summary["per_gpu_count"] = per_gpu_count
    summary["scaling_points"] = scaling_points
    summary["num_successful_scaling_points"] = len(scaling_points)
    summary["base_gpu_count"] = None if base_point is None else int(base_point["gpu_count"])
    summary["base_total_tflops_per_s"] = None if base_point is None else float(base_point["mean_tflops_per_s"])

    summary_path = run_dir / "summary.json"
    _write_json(summary_path, summary)
    print(f"[stage3-scaling] wrote {summary_path}", flush=True)

    if not scaling_points and not args.dry_run:
        raise RuntimeError("stage3 scaling sweep produced no successful benchmark points")


if __name__ == "__main__":
    main()
