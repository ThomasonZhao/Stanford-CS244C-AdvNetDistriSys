from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = PROJECT_ROOT / "infra" / "build_socket_shaper.py"
SHARED_OBJECT = PROJECT_ROOT / "infra" / "socket_shaper.so"


pytestmark = pytest.mark.skipif(platform.system() != "Linux", reason="socket shaper integration test is Linux-only")


def _transfer_script() -> str:
    return textwrap.dedent(
        """
        import json
        import socket
        import threading
        import time

        TOTAL = 8 * 1024 * 1024
        payload = b"x" * (1 << 20)

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        ready = threading.Event()

        def server():
            ready.set()
            conn, _ = listener.accept()
            received = 0
            while received < TOTAL:
                chunk = conn.recv(1 << 20)
                if not chunk:
                    break
                received += len(chunk)
            conn.close()
            listener.close()

        thread = threading.Thread(target=server, daemon=True)
        thread.start()
        ready.wait()

        client = socket.socket()
        client.connect(("127.0.0.1", port))
        sent = 0
        t0 = time.perf_counter()
        while sent < TOTAL:
            chunk = payload[: min(len(payload), TOTAL - sent)]
            client.sendall(chunk)
            sent += len(chunk)
        client.shutdown(socket.SHUT_WR)
        thread.join()
        dt = time.perf_counter() - t0
        print(json.dumps({"seconds": dt, "mbps": (TOTAL * 8) / (dt * 1e6)}))
        """
    )


def test_socket_shaper_builds_and_limits_local_tcp_throughput(tmp_path: Path) -> None:
    subprocess.run([sys.executable, str(BUILD_SCRIPT)], cwd=PROJECT_ROOT, check=True)
    assert SHARED_OBJECT.exists()

    unshaped = subprocess.run(
        [sys.executable, "-c", _transfer_script()],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    unshaped_metrics = json.loads(unshaped.stdout)

    env = os.environ.copy()
    env["LD_PRELOAD"] = str(SHARED_OBJECT)
    env["ZERO_SOCKET_SHAPER_BW_GBPS"] = "0.1"
    env["ZERO_SOCKET_SHAPER_LATENCY_MS"] = "0"
    env["ZERO_SOCKET_SHAPER_BURST_BYTES"] = "65536"
    shaped = subprocess.run(
        [sys.executable, "-c", _transfer_script()],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    shaped_metrics = json.loads(shaped.stdout)

    assert shaped_metrics["seconds"] > unshaped_metrics["seconds"]
    assert shaped_metrics["mbps"] < unshaped_metrics["mbps"]
    assert shaped_metrics["mbps"] < 200.0
