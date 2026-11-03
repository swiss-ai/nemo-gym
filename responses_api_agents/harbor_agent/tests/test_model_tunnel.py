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
import http.server
import json
import logging
import os
import platform
import re
import shutil
import signal
import subprocess
import time
import threading
from pathlib import Path
from typing import Optional

import pytest

from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.sandbox.providers.base import SandboxEndpoint
from responses_api_agents.harbor_agent.custom_envs.nemo_gym_sandbox.model_tunnel import (
    _ARCH,
    ModelTunnel,
    ModelTunnelConfig,
    health_check_script,
    pod_diagnostics_script,
    server_start_script,
)


NODE_ARCH = _ARCH.get(platform.machine())
pytestmark = pytest.mark.skipif(NODE_ARCH is None or shutil.which("bash") is None, reason="needs bash on amd64/arm64")

ENDPOINT = SandboxEndpoint(
    endpoint="https://osb.example/v1/sandboxes/sbx-1/proxy/8765/",
    headers={"OPEN-SANDBOX-API-KEY": "secret-key", "OpenSandbox-Secure-Access": "tok"},
)


def _fake_chisel(path: Path, record: Path, *, connect: bool = True) -> None:
    """Stand-in for the node-side chisel client: records argv and AUTH, prints chisel's log lines."""
    body = (
        "#!/bin/bash\n"
        f'python3 -c \'import json,os,sys; json.dump({{"argv": sys.argv[1:], "auth": os.environ.get("AUTH")}}, '
        f'open("{record}", "w"))\' "$@"\n'
        'echo "client: Connecting"\n'
    )
    if connect:
        body += 'echo "client: Connected (Latency 1.2ms)"\nexec sleep 60\n'
    else:
        body += 'echo "client: Give up"\nexit 1\n'
    path.write_text(body)
    path.chmod(0o755)


class FakeSandbox:
    def __init__(self, *, pod_arch: str = "x86_64", health: str = "HTTP/1.1 200 OK", health_rc: int = 0):
        self.pod_arch = pod_arch
        self.health = health
        self.health_rc = health_rc
        self.exec_calls: list[dict] = []
        self.uploads: dict[str, bytes] = {}

    async def exec(self, command: str, *, env: Optional[dict] = None, timeout_s=None) -> SandboxExecResult:
        self.exec_calls.append({"command": command, "env": env})
        if command == "uname -m":
            return SandboxExecResult(stdout=f"{self.pod_arch}\n", stderr=None, return_code=0)
        if command.startswith("bash -c"):
            return SandboxExecResult(stdout=self.health + "\n", stderr=None, return_code=self.health_rc)
        return SandboxExecResult(stdout="", stderr=None, return_code=0)

    async def upload(self, local_path, remote_path: str) -> None:
        self.uploads[remote_path] = Path(local_path).read_bytes()

    async def endpoint(self, port: int) -> SandboxEndpoint:
        assert port == 8765
        return ENDPOINT


@pytest.fixture
def chisel_dir(tmp_path: Path) -> Path:
    d = tmp_path / "chisel"
    d.mkdir()
    for arch in ("amd64", "arm64"):
        (d / f"chisel_1.12.0_linux_{arch}").write_bytes(f"pod-binary-{arch}".encode())
    return d


def _config(chisel_dir: Path, **kw) -> ModelTunnelConfig:
    return ModelTunnelConfig.from_mapping(
        {"chisel_dir": str(chisel_dir), "target": "policy-host:9000", "connect_timeout_s": 10, **kw}
    )


def _pid_alive(proc) -> bool:
    return proc is not None and proc.returncode is None


class TestConfig:
    def test_rejects_unknown_keys(self, chisel_dir):
        with pytest.raises(ValueError, match="Unknown model_tunnel keys"):
            ModelTunnelConfig.from_mapping({"chisel_dir": str(chisel_dir), "pod_prot": 8000})

    def test_requires_chisel_dir(self):
        with pytest.raises(ValueError, match="chisel_dir"):
            ModelTunnelConfig.from_mapping({"target": "h:1"})

    def test_binary_names_follow_release_assets(self, chisel_dir):
        assert _config(chisel_dir).binary("arm64") == chisel_dir / "chisel_1.12.0_linux_arm64"


class TestOpenClose:
    @pytest.mark.asyncio
    async def test_open_starts_both_ends_and_close_stops_client(self, chisel_dir, tmp_path, caplog):
        record = tmp_path / "client.json"
        _fake_chisel(chisel_dir / f"chisel_1.12.0_linux_{NODE_ARCH}", record)
        sandbox = FakeSandbox(pod_arch="x86_64")
        tunnel = ModelTunnel(sandbox, _config(chisel_dir), logging.getLogger("t"), label=" [trial-1]")
        with caplog.at_level(logging.INFO, logger="t"):
            await tunnel.open()
        assert "model_tunnel [trial-1]: sandbox 127.0.0.1:8000 -> policy-host:9000 up in" in caplog.text
        assert "GET /health: HTTP/1.1 200 OK" in caplog.text
        try:
            # the sandbox gets the binary for ITS architecture
            assert sandbox.uploads["/tmp/.nemo-gym-chisel"] == (chisel_dir / "chisel_1.12.0_linux_amd64").read_bytes()
            start = next(c for c in sandbox.exec_calls if "server" in c["command"])
            assert "--reverse" in start["command"] and "--port 8765" in start["command"]
            server_auth = start["env"]["AUTH"]
            assert server_auth.startswith("nemo-gym:") and server_auth not in start["command"]

            client = json.loads(record.read_text())
            argv = client["argv"]
            assert argv[0] == "client"
            assert client["auth"] == server_auth  # same per-sandbox credential, via the environment
            assert not any(server_auth in a for a in argv)
            assert argv[-2:] == [
                "https://osb.example/v1/sandboxes/sbx-1/proxy/8765",
                "R:127.0.0.1:8000:policy-host:9000",
            ]
            assert "OPEN-SANDBOX-API-KEY: secret-key" in argv and "OpenSandbox-Secure-Access: tok" in argv
            assert "--tls-skip-verify" not in argv

            health = sandbox.exec_calls[-1]["command"]
            assert health.startswith("bash -c") and "/dev/tcp/127.0.0.1/8000" in health and "GET /health" in health
            proc = tunnel._proc
            assert _pid_alive(proc)
        finally:
            await tunnel.close()
        assert proc.returncode is not None

    @pytest.mark.asyncio
    async def test_client_that_never_connects_fails_with_its_output(self, chisel_dir, tmp_path):
        _fake_chisel(chisel_dir / f"chisel_1.12.0_linux_{NODE_ARCH}", tmp_path / "c.json", connect=False)
        tunnel = ModelTunnel(FakeSandbox(), _config(chisel_dir), logging.getLogger("t"))
        with pytest.raises(RuntimeError, match="did not connect") as exc:
            await tunnel.open()
        assert "Give up" in str(exc.value)
        assert tunnel._proc is None

    @pytest.mark.asyncio
    async def test_failed_health_check_fails_open_and_stops_client(self, chisel_dir, tmp_path):
        _fake_chisel(chisel_dir / f"chisel_1.12.0_linux_{NODE_ARCH}", tmp_path / "c.json")
        tunnel = ModelTunnel(
            FakeSandbox(health="HTTP/1.1 502 Bad Gateway"),
            _config(chisel_dir, tls_skip_verify=True),
            logging.getLogger("t"),
        )
        with pytest.raises(RuntimeError, match="health check"):
            await tunnel.open()
        assert tunnel._proc is None
        assert "--tls-skip-verify" in json.loads((tmp_path / "c.json").read_text())["argv"]

    @pytest.mark.asyncio
    async def test_no_health_check_when_disabled(self, chisel_dir, tmp_path):
        _fake_chisel(chisel_dir / f"chisel_1.12.0_linux_{NODE_ARCH}", tmp_path / "c.json")
        sandbox = FakeSandbox(health="garbage", health_rc=1)
        tunnel = ModelTunnel(sandbox, _config(chisel_dir, health_path=None), logging.getLogger("t"))
        await tunnel.open()
        await tunnel.close()
        assert not any(c["command"].startswith("bash -c") for c in sandbox.exec_calls)

    @pytest.mark.asyncio
    async def test_missing_target_or_binary(self, chisel_dir):
        with pytest.raises(ValueError, match="target"):
            await ModelTunnel(FakeSandbox(), _config(chisel_dir, target=None), logging.getLogger("t")).open()
        (chisel_dir / "chisel_1.12.0_linux_amd64").unlink()
        with pytest.raises(FileNotFoundError, match="chisel_1.12.0_linux_amd64"):
            await ModelTunnel(FakeSandbox(), _config(chisel_dir), logging.getLogger("t")).open()

    @pytest.mark.asyncio
    async def test_unsupported_sandbox_arch(self, chisel_dir):
        with pytest.raises(RuntimeError, match="unsupported architecture"):
            await ModelTunnel(FakeSandbox(pod_arch="s390x"), _config(chisel_dir), logging.getLogger("t")).open()


class TestServerStartFailure:
    @pytest.mark.asyncio
    async def test_failed_server_start_fails_open_before_the_client(self, chisel_dir, tmp_path):
        record = tmp_path / "c.json"
        _fake_chisel(chisel_dir / f"chisel_1.12.0_linux_{NODE_ARCH}", record)

        class DeadServer(FakeSandbox):
            async def exec(self, command, *, env=None, timeout_s=None):
                if "server" in command and "--reverse" in command:
                    self.exec_calls.append({"command": command, "env": env})
                    out = "chisel server not listening after 15s:\nboom"
                    return SandboxExecResult(stdout=out, stderr=None, return_code=1)
                return await super().exec(command, env=env, timeout_s=timeout_s)

        tunnel = ModelTunnel(DeadServer(), _config(chisel_dir), logging.getLogger("t"))
        with pytest.raises(RuntimeError, match="did not come up") as exc:
            await tunnel.open()
        assert "boom" in str(exc.value)
        assert not record.exists()  # the node-side client was never started

    @pytest.mark.asyncio
    async def test_client_timeout_reports_the_sandbox_side(self, chisel_dir, tmp_path):
        _fake_chisel(chisel_dir / f"chisel_1.12.0_linux_{NODE_ARCH}", tmp_path / "c.json", connect=False)

        class WithLog(FakeSandbox):
            async def exec(self, command, *, env=None, timeout_s=None):
                if command.startswith("tail -n 20"):
                    out = "server: Listening\nchisel server processes: 0"
                    return SandboxExecResult(stdout=out, stderr=None, return_code=0)
                return await super().exec(command, env=env, timeout_s=timeout_s)

        with pytest.raises(RuntimeError, match="chisel server processes: 0"):
            await ModelTunnel(WithLog(), _config(chisel_dir), logging.getLogger("t")).open()


class _Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def log_message(self, *args):
        pass


class TestHealthCheckScript:
    """The script runs for real under bash against a local HTTP server."""

    def test_reports_status_line(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Health)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            port = server.server_address[1]
            ok = subprocess.run(["bash", "-c", health_check_script(port, "/health")], capture_output=True, text=True)
            assert ok.returncode == 0 and ok.stdout.split()[1] == "200"
            missing = subprocess.run(
                ["bash", "-c", health_check_script(port, "/nope")], capture_output=True, text=True
            )
            assert missing.returncode == 0 and missing.stdout.split()[1] == "404"
        finally:
            server.shutdown()

    def test_no_listener_fails(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Health)
        port = server.server_address[1]
        server.server_close()  # nothing listens on the port any more
        res = subprocess.run(
            ["bash", "-c", health_check_script(port, "/health", attempts=2)],
            capture_output=True,
            text=True,
            env={**os.environ},
            timeout=30,
        )
        assert res.returncode == 1 and "no answer" in res.stdout


def _fake_server(path: Path, body: str) -> ModelTunnelConfig:
    path.write_text("#!/bin/bash\n" + body)
    return ModelTunnelConfig(chisel_dir=str(path.parent), pod_binary=str(path))


def _running(pod_binary: Path) -> list[int]:
    pids = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            if f"{pod_binary} server".encode() in cmdline.read_bytes().replace(b"\0", b" "):
                pids.append(int(cmdline.parent.name))
        except OSError:
            pass
    return pids


class TestPodScripts:
    """The sandbox-side shell scripts run for real (sh, as the sandbox exec service runs them)."""

    def test_server_start_returns_once_listening_and_survives_a_group_kill(self, tmp_path):
        # chisel prints "Listening on" after a moment; the exec service may SIGKILL the command's
        # process group as soon as the command returns (job 5245999: no server behind the proxy).
        binary = tmp_path / ".nemo-gym-chisel"  # its cmdline must keep "<binary> server": no exec
        cfg = _fake_server(binary, 'sleep 0.5; echo "server: Listening on http://0.0.0.0:$5"; sleep 30\n')
        proc = subprocess.Popen(
            ["sh", "-c", server_start_script(cfg)], start_new_session=True, stdout=subprocess.PIPE, text=True
        )
        try:
            out, _ = proc.communicate(timeout=20)
            assert proc.returncode == 0, out
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            time.sleep(0.3)
            assert _running(binary), "server died with the exec's process group"
            diag = subprocess.run(["sh", "-c", pod_diagnostics_script(cfg)], capture_output=True, text=True)
            assert "Listening on http://0.0.0.0:8765" in diag.stdout
            # >= 1: on a dev machine unrelated shells may mention "chisel server" in their command line
            assert re.search(r"chisel server processes: [1-9]", diag.stdout), diag.stdout
        finally:
            for pid in _running(binary):
                os.kill(pid, signal.SIGKILL)

    def test_server_that_dies_fails_the_start_with_its_log(self, tmp_path):
        binary = tmp_path / "chisel-srv"
        cfg = _fake_server(binary, 'echo "server: listen tcp 0.0.0.0:$5: bind: address already in use"; exit 1\n')
        res = subprocess.run(
            ["sh", "-c", server_start_script(cfg, wait_s=1)], capture_output=True, text=True, timeout=20
        )
        assert res.returncode == 1
        assert "not listening after 1s" in res.stdout and "address already in use" in res.stdout

    def test_server_start_fails_without_binary(self, tmp_path):
        cfg = ModelTunnelConfig(chisel_dir=str(tmp_path), pod_binary=str(tmp_path / "missing"))
        res = subprocess.run(["sh", "-c", server_start_script(cfg, wait_s=1)], capture_output=True, timeout=20)
        assert res.returncode == 1
