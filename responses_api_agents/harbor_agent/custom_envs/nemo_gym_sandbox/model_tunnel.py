# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-sandbox reverse tunnel from the sandbox to a model server next to the Harbor process.

For clusters where sandboxes cannot open connections to the training nodes, but the node can reach
the sandbox through the provider's service endpoint (e.g. OpenSandbox's server proxy, which carries
WebSockets). The node opens the tunnel, the sandbox uses it:

    in sandbox:  127.0.0.1:<pod_port>  ->  chisel server --reverse  (on <server_port>)
                     <= WebSocket opened by the node via sandbox.endpoint(<server_port>) =>
    on node:     chisel client  ->  <target> (the model server host:port)

`chisel <https://github.com/jpillora/chisel>`_ binaries are not shipped: ``chisel_dir`` must
hold ``chisel_<version>_linux_<arch>`` for the node and for the sandbox architecture (amd64,
arm64), as released (gunzipped). The sandbox copy is uploaded at every start, so the task image
needs no internet and no preinstalled tool, only ``uname``, ``setsid``/``nohup`` and, for the
health check, ``bash``.
"""

import asyncio
import os
import platform
import secrets
import shlex
import time
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional


_ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}
_CONNECTED_MARKER = "Connected (Latency"


@dataclass(frozen=True)
class ModelTunnelConfig:
    """``harbor_environment_kwargs.model_tunnel``.

    target: ``host:port`` the node-side client dials; harbor_agent fills it with the model server
        when unset.
    chisel_dir: directory with ``chisel_<chisel_version>_linux_{amd64,arm64}``.
    pod_port: loopback port in the sandbox that forwards to ``target``.
    server_port: sandbox port of the chisel server (reached through ``sandbox.endpoint``).
    connect_timeout_s: bound on the client's first ``Connected``.
    health_path: when set, one HTTP GET from inside the sandbox through the tunnel must answer
        2xx before the trial starts (e.g. ``/health`` of a Gym server).
    tls_skip_verify: skip certificate checks on the endpoint's TLS.
    max_retry_count: client reconnect attempts after a drop.
    """

    chisel_dir: str
    target: Optional[str] = None
    chisel_version: str = "1.12.0"
    pod_port: int = 8000
    server_port: int = 8765
    connect_timeout_s: float = 60.0
    health_path: Optional[str] = "/health"
    tls_skip_verify: bool = False
    max_retry_count: int = 10
    pod_binary: str = "/tmp/.nemo-gym-chisel"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ModelTunnelConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Unknown model_tunnel keys {unknown}; known: {sorted(known)}")
        if not raw.get("chisel_dir"):
            raise ValueError("model_tunnel.chisel_dir is required")
        return cls(**dict(raw))

    def binary(self, arch: str) -> Path:
        return Path(self.chisel_dir) / f"chisel_{self.chisel_version}_linux_{arch}"


def server_start_script(cfg: ModelTunnelConfig, wait_s: float = 15.0) -> str:
    """sh: start the chisel server detached and return only once it listens (or fail with its log).

    The command must not return before the server has left the exec's process group: the sandbox's
    exec service may signal that group when the command ends, and a fire-and-forget ``(setsid ... &)``
    races with that. Waiting for chisel's "Listening on" line also turns a server that dies at start
    (bad binary, port taken) into an error here instead of a client timeout later.
    """
    b = shlex.quote(cfg.pod_binary)
    log = shlex.quote(cfg.pod_binary + ".log")
    tries = max(1, int(wait_s / 0.2))
    return (
        f"chmod +x {b} || exit 1; (setsid nohup {b} server --host 0.0.0.0 --port {cfg.server_port} --reverse "
        f"> {log} 2>&1 &); "
        f"i=0; while [ $i -lt {tries} ]; do "
        f"grep -q 'Listening on' {log} 2>/dev/null && exit 0; i=$((i+1)); sleep 0.2; done; "
        f"echo 'chisel server not listening after {wait_s:g}s:'; cat {log}; exit 1"
    )


def pod_diagnostics_script(cfg: ModelTunnelConfig) -> str:
    """sh: the server log tail and how many chisel server processes run (no ps in slim images)."""
    log = shlex.quote(cfg.pod_binary + ".log")
    return (
        f"tail -n 20 {log}; n=0; for c in /proc/[0-9]*/cmdline; do "
        "tr '\\0' ' ' < \"$c\" 2>/dev/null | grep -q '[c]hisel server' && n=$((n+1)); done; "
        'echo "chisel server processes: $n"'
    )


def health_check_script(pod_port: int, path: str, attempts: int = 40) -> str:
    """bash: GET ``path`` on 127.0.0.1:``pod_port`` (retried while the reverse listener comes up),
    print the status line, exit 0 on any answer."""
    request = f"GET {path} HTTP/1.0\\r\\nHost: 127.0.0.1\\r\\n\\r\\n"
    return (
        f"for i in $(seq 1 {attempts}); do "
        f'r=$( (exec 3<>/dev/tcp/127.0.0.1/{pod_port} && printf "{request}" >&3 && head -n1 <&3) 2>/dev/null ); '
        'if [ -n "$r" ]; then echo "$r"; exit 0; fi; sleep 0.5; done; '
        f"echo 'no answer on 127.0.0.1:{pod_port}'; exit 1"
    )


class ModelTunnel:
    """One tunnel for one sandbox: :meth:`open` after the sandbox started, :meth:`close` before it stops."""

    def __init__(self, sandbox: Any, config: ModelTunnelConfig, logger: Any, label: str = "") -> None:
        self._sandbox = sandbox
        self._config = config
        self._logger = logger
        self._label = label
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader: Optional[asyncio.Task] = None
        self._connected = asyncio.Event()
        self._tail: deque[str] = deque(maxlen=40)

    @property
    def client_log_tail(self) -> list[str]:
        return list(self._tail)

    async def _exec(self, command: str, *, env: Optional[dict[str, str]] = None, timeout_s: float = 60) -> Any:
        return await self._sandbox.exec(command, env=env, timeout_s=timeout_s)

    async def open(self) -> None:
        """Start both ends and wait until the sandbox reaches the target; on any failure nothing is left running
        on the node (the sandbox side goes with the sandbox)."""
        t0 = time.monotonic()
        try:
            status = await self._open()
        except BaseException:
            await self.close()
            raise
        cfg = self._config
        self._logger.info(
            "model_tunnel%s: sandbox 127.0.0.1:%s -> %s up in %.1fs (%s)",
            self._label,
            cfg.pod_port,
            cfg.target,
            time.monotonic() - t0,
            status,
        )

    async def _open(self) -> str:
        """Returns the health check's status line, or "no health check"."""
        cfg = self._config
        if not cfg.target:
            raise ValueError("model_tunnel.target is not set (harbor_agent sets it to the model server)")
        node_arch = _ARCH.get(platform.machine())
        res = await self._exec("uname -m", timeout_s=30)
        pod_arch = _ARCH.get((res.stdout or "").strip())
        if node_arch is None or pod_arch is None:
            raise RuntimeError(
                f"model_tunnel: unsupported architecture (node {platform.machine()!r}, sandbox {res.stdout!r})"
            )
        node_bin, pod_bin = cfg.binary(node_arch), cfg.binary(pod_arch)
        for path in {node_bin, pod_bin}:
            if not path.is_file():
                raise FileNotFoundError(f"model_tunnel: chisel binary {path} not found")

        await self._sandbox.upload(pod_bin, cfg.pod_binary)
        # One credential per sandbox: the endpoint is reachable by every holder of the provider API key.
        # Passed as AUTH (chisel reads it from the environment), so it stays out of argv on both sides.
        auth = f"nemo-gym:{secrets.token_hex(16)}"
        res = await self._exec(server_start_script(cfg), env={"AUTH": auth}, timeout_s=60)
        if res.return_code != 0:
            raise RuntimeError(
                f"model_tunnel{self._label}: chisel server in the sandbox did not come up: "
                f"rc={res.return_code} {res.stdout!r} {res.stderr!r}"
            )

        endpoint = await self._sandbox.endpoint(cfg.server_port)
        args = [str(node_bin), "client", "--max-retry-count", str(cfg.max_retry_count), "--max-retry-interval", "10s"]
        if cfg.tls_skip_verify:
            args.append("--tls-skip-verify")
        # chisel takes extra handshake headers only as flags; they carry the provider's auth headers.
        for key, value in endpoint.headers.items():
            args += ["--header", f"{key}: {value}"]
        args += [endpoint.endpoint.rstrip("/"), f"R:127.0.0.1:{cfg.pod_port}:{cfg.target}"]
        self._proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "AUTH": auth},
            start_new_session=True,
        )
        self._reader = asyncio.create_task(self._read_client_output())

        exited = asyncio.ensure_future(self._proc.wait())
        connected = asyncio.ensure_future(self._connected.wait())
        try:
            await asyncio.wait({exited, connected}, timeout=cfg.connect_timeout_s, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for fut in (exited, connected):
                if not fut.done():
                    fut.cancel()
        if not self._connected.is_set():
            raise RuntimeError(
                f"model_tunnel{self._label}: chisel client did not connect within {cfg.connect_timeout_s}s; "
                f"client output: {self.client_log_tail[-10:]}; sandbox side: {await self._pod_log()!r}"
            )
        self._logger.debug(
            "model_tunnel%s: connected, sandbox 127.0.0.1:%s -> %s", self._label, cfg.pod_port, cfg.target
        )

        if cfg.health_path:
            script = health_check_script(cfg.pod_port, cfg.health_path)
            res = await self._exec(f"bash -c {shlex.quote(script)}", timeout_s=60)
            status_line = (res.stdout or "").strip()
            parts = status_line.split()
            if res.return_code != 0 or len(parts) < 2 or not parts[1].startswith("2"):
                raise RuntimeError(
                    f"model_tunnel{self._label}: health check GET {cfg.health_path} through the tunnel failed: "
                    f"rc={res.return_code} {status_line!r} {res.stderr!r}"
                )
            return f"GET {cfg.health_path}: {status_line}"
        return "no health check"

    async def _pod_log(self) -> str:
        """Best effort: the sandbox-side chisel log and whether it still runs, for error messages."""
        try:
            res = await self._exec(pod_diagnostics_script(self._config), timeout_s=20)
            return (res.stdout or "").strip()
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            return f"<unavailable: {exc!r}>"

    async def _read_client_output(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        async for raw in self._proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            self._tail.append(line)
            if _CONNECTED_MARKER in line:
                self._connected.set()

    async def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        reader, self._reader = self._reader, None
        if reader is not None:
            try:
                await asyncio.wait_for(reader, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                reader.cancel()
