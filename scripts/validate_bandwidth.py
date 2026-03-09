from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THROTTLE_SCRIPT = PROJECT_ROOT / "infra" / "throttle.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate tc shaping with iperf3")
    subparsers = parser.add_subparsers(dest="command", required=True)

    server = subparsers.add_parser("server", help="Run an iperf3 server")
    server.add_argument("--bind", type=str, default="0.0.0.0")
    server.add_argument("--port", type=int, default=5201)
    server.add_argument("--one-off", action="store_true", help="Exit after one client run")

    validate = subparsers.add_parser("validate", help="Measure baseline and shaped throughput")
    validate.add_argument("--target-host", type=str, required=True)
    validate.add_argument("--port", type=int, default=5201)
    validate.add_argument("--device", type=str, required=True)
    validate.add_argument("--rate", type=str, required=True, help="tc rate, e.g. 10gbit or 500mbit")
    validate.add_argument("--burst", type=str, default="1mb")
    validate.add_argument("--latency", type=str, default="10ms")
    validate.add_argument("--duration-s", type=int, default=5)
    validate.add_argument("--baseline-runs", type=int, default=1)
    validate.add_argument("--shaped-runs", type=int, default=1)
    validate.add_argument(
        "--max-shaped-ratio",
        type=float,
        default=0.8,
        help="Fail validation if shaped throughput exceeds this fraction of baseline throughput.",
    )
    validate.add_argument(
        "--allow-route-mismatch",
        action="store_true",
        help="Proceed even if the route to the target does not use the selected device.",
    )
    validate.add_argument("--json-output", type=str, default="")

    return parser.parse_args()


def _require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(f"required command not found: {name}")


def _run(cmd: list[str], *, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def _route_device(target_host: str) -> str:
    proc = _run(["ip", "route", "get", target_host])
    tokens = proc.stdout.strip().split()
    if "dev" not in tokens:
        raise RuntimeError(f"could not determine route device for {target_host!r}: {proc.stdout!r}")
    return tokens[tokens.index("dev") + 1]


def _iperf_bits_per_second(target_host: str, port: int, duration_s: int) -> tuple[float, dict[str, object]]:
    proc = _run(
        [
            "iperf3",
            "-c",
            target_host,
            "-p",
            str(port),
            "-t",
            str(duration_s),
            "-J",
        ]
    )
    payload = json.loads(proc.stdout)
    end = payload.get("end", {})
    summary = end.get("sum_received") or end.get("sum_sent")
    if not isinstance(summary, dict) or "bits_per_second" not in summary:
        raise RuntimeError(f"iperf3 JSON did not contain a summary throughput: {payload}")
    return float(summary["bits_per_second"]), payload


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _run_iperf_series(target_host: str, port: int, duration_s: int, count: int) -> list[float]:
    results: list[float] = []
    for _ in range(count):
        bps, _ = _iperf_bits_per_second(target_host=target_host, port=port, duration_s=duration_s)
        results.append(bps)
    return results


def _apply_tc(device: str, rate: str, burst: str, latency: str) -> None:
    try:
        _run([str(THROTTLE_SCRIPT), "apply", device, rate, burst, latency])
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or str(exc)
        raise RuntimeError(
            "failed to apply tc shaping. This usually means the process lacks CAP_NET_ADMIN, "
            f"the device is invalid, or the environment blocks qdisc changes: {detail}"
        ) from exc


def _clear_tc(device: str) -> None:
    subprocess.run(
        [str(THROTTLE_SCRIPT), "delete", device],
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def run_server(args: argparse.Namespace) -> int:
    _require_binary("iperf3")
    cmd = ["iperf3", "-s", "-B", args.bind, "-p", str(args.port)]
    if args.one_off:
        cmd.append("-1")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return 0


def run_validate(args: argparse.Namespace) -> int:
    _require_binary("iperf3")
    _require_binary("ip")
    if not THROTTLE_SCRIPT.exists():
        raise FileNotFoundError(f"missing throttle helper: {THROTTLE_SCRIPT}")

    routed_device = _route_device(args.target_host)
    if routed_device != args.device and not args.allow_route_mismatch:
        raise RuntimeError(
            "selected device does not match route to target host: "
            f"target_host={args.target_host} routed_device={routed_device} requested_device={args.device}. "
            "Use the actual egress device, or pass --allow-route-mismatch if this is intentional."
        )

    baseline_runs = _run_iperf_series(
        target_host=args.target_host,
        port=args.port,
        duration_s=args.duration_s,
        count=args.baseline_runs,
    )

    try:
        _apply_tc(device=args.device, rate=args.rate, burst=args.burst, latency=args.latency)
        shaped_runs = _run_iperf_series(
            target_host=args.target_host,
            port=args.port,
            duration_s=args.duration_s,
            count=args.shaped_runs,
        )
    finally:
        _clear_tc(args.device)

    baseline_bps = _mean(baseline_runs)
    shaped_bps = _mean(shaped_runs)
    shaped_ratio = shaped_bps / baseline_bps if baseline_bps > 0 else float("inf")
    passed = shaped_ratio <= args.max_shaped_ratio

    summary = {
        "target_host": args.target_host,
        "port": args.port,
        "requested_device": args.device,
        "routed_device": routed_device,
        "rate": args.rate,
        "burst": args.burst,
        "latency": args.latency,
        "duration_s": args.duration_s,
        "baseline_runs_bps": baseline_runs,
        "shaped_runs_bps": shaped_runs,
        "baseline_mean_gbps": baseline_bps / 1e9,
        "shaped_mean_gbps": shaped_bps / 1e9,
        "shaped_ratio": shaped_ratio,
        "max_shaped_ratio": args.max_shaped_ratio,
        "passed": passed,
    }

    text = json.dumps(summary, indent=2)
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")
    print(text)
    return 0 if passed else 2


def main() -> int:
    args = parse_args()
    if args.command == "server":
        return run_server(args)
    if args.command == "validate":
        return run_validate(args)
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)