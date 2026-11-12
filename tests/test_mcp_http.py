"""Transport MCP `streamable-http` : mode de déploiement réseau.

Le reste de la suite utilise `stdio`. Ce test démarre un vrai serveur HTTP sur
un port libre, s'y connecte avec le **client de production**
(`LpoDataClient(url=...)`) et vérifie qu'on retrouve exactement les mêmes outils
et les mêmes sorties structurées. C'est ce qui rend le basculement
stdio → HTTP sûr côté exploitation.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from orchestrator.mcp_client import McpDataClient

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int, process: subprocess.Popen, timeout: float = 25.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "le serveur MCP HTTP s'est arrêté : "
                + (process.stderr.read() if process.stderr else "")
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"port {port} jamais ouvert")


@pytest.fixture()
def http_server(monkeypatch, demo_db_path):
    from src.database import engine

    port = _free_port()
    env = dict(os.environ)
    env["DATABASE_URL"] = f"sqlite:///{demo_db_path}"
    env["LPO_PROFILE"] = "observations"
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lpo_data_mcp",
            "--transport",
            "streamable-http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "WARNING",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_port(port, process)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            process.kill()
        engine.reset_engine()


async def test_http_transport_exposes_the_same_contract(http_server):
    async with McpDataClient(url=http_server) as client:
        tools = set(client.tool_names)
        overview = await client.call("get_dataset_overview")
        terms = await client.call("list_species", {"limit": 3})
        aggregate = await client.call("aggregate_observations", {"group_by": "year", "limit": 5})

    assert tools == {
        "get_dataset_overview",
        "list_species",
        "get_observations",
        "aggregate_observations",
        "get_species_info",
    }
    assert overview["profile"] == "observations"
    assert overview["total_rows"] == 900
    assert terms["species"]
    assert aggregate["points"] and aggregate["total"] > 0
