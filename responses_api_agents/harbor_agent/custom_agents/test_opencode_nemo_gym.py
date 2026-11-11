# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest


pytest.importorskip("harbor")

from jinja2 import Environment  # noqa: E402

from responses_api_agents.harbor_agent.custom_agents import opencode_nemo_gym as mod  # noqa: E402
from responses_api_agents.harbor_agent.custom_agents.opencode_nemo_gym import OpenCodeNemoGym, deep_merge  # noqa: E402


API_BASE = "http://127.0.0.1:8000/ng-rollout/3-1/training-token-capture/v1"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
# Top-level keys of OpenCode's config schema at v1.17.11 (core/src/v1/config/config.ts:32-187);
# config/parse.ts:40-53 rejects any other top-level key.
OPENCODE_1_17_11_TOP_LEVEL_KEYS = {
    "$schema", "shell", "logLevel", "server", "command", "skills", "references", "reference", "watcher",
    "snapshot", "plugin", "share", "autoshare", "autoupdate", "disabled_providers", "enabled_providers",
    "model", "small_model", "default_agent", "username", "mode", "agent", "provider", "mcp", "formatter",
    "lsp", "instructions", "layout", "permission", "tools", "attachment", "enterprise", "tool_output",
    "compaction", "experimental",
}  # fmt: skip
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _agent(tmp_path: Path, **kwargs) -> OpenCodeNemoGym:
    return OpenCodeNemoGym(logs_dir=tmp_path, model_name=MODEL, api_base=API_BASE, **kwargs)


def _config(agent: OpenCodeNemoGym) -> dict:
    (cmd,) = agent.create_run_agent_commands("x")
    return json.loads(cmd.env["OPENCODE_CONFIG_CONTENT"])


def test_requires_api_base(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="api_base"):
        OpenCodeNemoGym(logs_dir=tmp_path, model_name=MODEL)


# --- run command ---------------------------------------------------------------------------------


def test_run_command(tmp_path: Path) -> None:
    (cmd,) = _agent(tmp_path).create_run_agent_commands("print 'hi' > /app/out.txt")
    assert cmd.command == (
        f"{mod.PIPEFAIL_IF_SUPPORTED}opencode --model nemogym/{MODEL} run --format=json --title 'NeMo Gym' "
        "-- 'print '\"'\"'hi'\"'\"' > /app/out.txt' 2>&1 </dev/null | tee /logs/agent/opencode.txt"
    )
    assert "timeout" not in cmd.command
    assert "--print-logs" not in cmd.command


def test_run_command_timeout_wrapper_and_debug_logs(tmp_path: Path) -> None:
    (cmd,) = _agent(tmp_path, opencode_timeout_sec=870, opencode_debug=True).create_run_agent_commands("-x")
    assert cmd.command.startswith(
        f"{mod.PIPEFAIL_IF_SUPPORTED}timeout -k 10 870 opencode --model nemogym/{MODEL} run "
    )
    assert "--title 'NeMo Gym' --print-logs --log-level DEBUG -- -x 2>&1 </dev/null |" in cmd.command


@pytest.mark.parametrize("bad", [0, -5])
def test_rejects_non_positive_timeout(tmp_path: Path, bad: int) -> None:
    with pytest.raises(ValueError, match="opencode_timeout_sec"):
        _agent(tmp_path, opencode_timeout_sec=bad)


@needs_bash
@pytest.mark.parametrize("timeout_sec", [None, 30])
def test_run_command_in_bash_closes_stdin_and_keeps_the_exit_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeout_sec: int | None
) -> None:
    if timeout_sec and shutil.which("timeout") is None:
        pytest.skip("needs coreutils timeout")
    log = tmp_path / "opencode.txt"
    monkeypatch.setattr(mod, "LOG_FILE", str(log))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Stands in for `opencode run`: reads stdin to EOF as OpenCode does for a non-TTY stdin
    # (cli/cmd/run.ts:399), prints its argv and exits 3.
    fake = bin_dir / "opencode"
    fake.write_text('#!/bin/bash\nstdin=$(cat)\nprintf "%s\\n" "$@"\necho "stdin=[$stdin]"\nexit 3\n')
    fake.chmod(0o755)
    (cmd,) = _agent(tmp_path, opencode_timeout_sec=timeout_sec).create_run_agent_commands("-fix it")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    # stdin is a pipe whose write end stays open, like the sandbox exec's: without </dev/null the
    # fake (like OpenCode) would block on it until the timeout below
    read_end, write_end = os.pipe()
    try:
        proc = subprocess.run(
            ["bash", "-c", cmd.command], stdin=read_end, capture_output=True, text=True, env=env, timeout=20
        )
    finally:
        os.close(read_end)
        os.close(write_end)
    assert proc.returncode == 3  # opencode's status, not tee's 0
    lines = log.read_text().splitlines()
    assert lines == proc.stdout.splitlines()
    assert lines[-3:] == ["--", "-fix it", "stdin=[]"]
    assert lines[:5] == ["--model", f"nemogym/{MODEL}", "run", "--format=json", "--title"]


@pytest.mark.parametrize("shell", ["sh", "dash"])
def test_run_command_starts_opencode_in_a_posix_sh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shell: str
) -> None:
    """The sandbox exec may use /bin/sh; a shell without pipefail must still run OpenCode."""
    if shutil.which(shell) is None:
        pytest.skip(f"needs {shell}")
    log = tmp_path / "opencode.txt"
    monkeypatch.setattr(mod, "LOG_FILE", str(log))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "opencode"
    fake.write_text("#!/bin/sh\necho started\nexit 3\n")
    fake.chmod(0o755)
    (cmd,) = _agent(tmp_path).create_run_agent_commands("x")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    proc = subprocess.run([shell, "-c", cmd.command], capture_output=True, text=True, env=env, timeout=20)
    supports_pipefail = subprocess.run([shell, "-c", "set -o pipefail"], capture_output=True).returncode == 0
    assert log.read_text().splitlines() == ["started"]
    assert proc.returncode == (3 if supports_pipefail else 0)


def test_run_command_workdir(tmp_path: Path) -> None:
    (plain,) = _agent(tmp_path).create_run_agent_commands("x")
    assert not plain.command.startswith("mkdir")
    (cmd,) = _agent(tmp_path, opencode_workdir="/app").create_run_agent_commands("x")
    assert cmd.command == f"mkdir -p /app && cd /app; {plain.command}"
    with pytest.raises(ValueError, match="absolute"):
        _agent(tmp_path, opencode_workdir="app")


@needs_bash
def test_run_command_starts_opencode_in_the_created_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "opencode.txt"
    monkeypatch.setattr(mod, "LOG_FILE", str(log))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "opencode"
    fake.write_text("#!/bin/sh\npwd\n")
    fake.chmod(0o755)
    workdir = tmp_path / "does" / "not" / "exist yet"
    (cmd,) = _agent(tmp_path, opencode_workdir=str(workdir)).create_run_agent_commands("x")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    proc = subprocess.run(["bash", "-c", cmd.command], cwd="/", capture_output=True, text=True, env=env, timeout=20)
    assert proc.returncode == 0
    assert log.read_text().splitlines() == [str(workdir)]


# --- environment ---------------------------------------------------------------------------------


def test_env(tmp_path: Path) -> None:
    (cmd,) = _agent(
        tmp_path, model_info={"max_input_tokens": 16384, "max_output_tokens": 4096}
    ).create_run_agent_commands("x")
    env = dict(cmd.env)
    assert json.loads(env.pop("OPENCODE_CONFIG_CONTENT"))["model"] == f"nemogym/{MODEL}"
    assert env == {
        "XDG_DATA_HOME": "/logs/agent/opencode-data",
        "OPENCODE_DISABLE_MODELS_FETCH": "true",
        "OPENCODE_DISABLE_AUTOUPDATE": "true",
        "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
        "OPENCODE_FAKE_VCS": "git",
        "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "4096",
    }


def test_no_output_token_env_without_model_info(tmp_path: Path) -> None:
    (cmd,) = _agent(tmp_path).create_run_agent_commands("x")
    assert "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX" not in cmd.env


# --- config --------------------------------------------------------------------------------------


def test_config(tmp_path: Path) -> None:
    config = _config(
        _agent(tmp_path, max_turns=20, model_info={"max_input_tokens": 16384, "max_output_tokens": 16384})
    )
    assert set(config) <= OPENCODE_1_17_11_TOP_LEVEL_KEYS
    assert config["enabled_providers"] == ["nemogym"]
    provider = config["provider"]["nemogym"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"] == {"baseURL": API_BASE, "apiKey": "EMPTY", "timeout": False, "headerTimeout": False}
    assert "chunkTimeout" not in provider["options"]  # positive int only; unset = off
    assert provider["models"] == {MODEL: {"name": MODEL, "limit": {"context": 16384, "input": 16384, "output": 16384}}}
    assert config["model"] == config["small_model"] == f"nemogym/{MODEL}"
    assert config["agent"] == {"title": {"disable": True}, "build": {"steps": 20}}
    assert config["compaction"] == {"auto": False, "prune": False}
    assert config["share"] == "disabled"
    assert config["autoupdate"] is False
    assert config["snapshot"] is False
    assert config["permission"] == {
        "*": "allow",
        "task": "deny",
        "webfetch": "deny",
        "websearch": "deny",
        "todowrite": "deny",
    }
    assert next(iter(config["permission"])) == "*"  # last match wins in OpenCode


def test_no_limit_or_steps_when_unknown(tmp_path: Path) -> None:
    config = _config(_agent(tmp_path))
    assert config["provider"]["nemogym"]["models"] == {MODEL: {"name": MODEL}}
    assert config["agent"] == {"title": {"disable": True}}


def test_output_limit_capped_by_request_max_output_tokens(tmp_path: Path) -> None:
    agent = _agent(
        tmp_path,
        model_info={"max_input_tokens": 16384, "max_output_tokens": 16384},
        responses_create_params={"max_output_tokens": 2048, "temperature": 1.0},
    )
    assert agent.model_limits() == {"context": 16384, "input": 16384, "output": 2048}
    assert _agent(tmp_path, model_info={"max_input_tokens": 8192}).model_limits() == {
        "context": 8192,
        "input": 8192,
        "output": 8192,
    }


def test_opencode_config_overlay_is_deep_merged_last(tmp_path: Path) -> None:
    overlay = {
        "provider": {"nemogym": {"options": {"chunkTimeout": 900000}}},
        "permission": {"bash": "allow", "todowrite": None},
        "agent": {"build": {"prompt": "Solve the task."}},
        "snapshot": None,
        "instructions": ["/app/AGENTS.md"],
    }
    config = _config(_agent(tmp_path, max_turns=7, opencode_config=overlay))
    options = config["provider"]["nemogym"]["options"]
    assert options["chunkTimeout"] == 900000 and options["timeout"] is False and options["baseURL"] == API_BASE
    assert list(config["permission"]) == ["*", "task", "webfetch", "websearch", "bash"]
    assert config["agent"]["build"] == {"steps": 7, "prompt": "Solve the task."}
    assert config["agent"]["title"] == {"disable": True}
    assert "snapshot" not in config
    assert config["instructions"] == ["/app/AGENTS.md"]


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"b": 1}, "c": [1]}
    overlay = {"a": {"d": 2}, "c": [2]}
    assert deep_merge(base, overlay) == {"a": {"b": 1, "d": 2}, "c": [2]}
    assert base == {"a": {"b": 1}, "c": [1]} and overlay == {"a": {"d": 2}, "c": [2]}


def test_consumes_harbor_agent_kwargs(tmp_path: Path) -> None:
    # harbor_agent passes these (plus Terminus-2 kwargs merged from the overlay); all must be accepted
    agent = _agent(
        tmp_path,
        responses_create_params={"temperature": 1.0, "top_p": 1.0, "input": []},
        temperature=1.0,
        max_turns=3,
        model_info={"max_input_tokens": 4096, "max_output_tokens": 1024},
        interleaved_thinking=False,
    )
    assert agent._responses_create_params == {"temperature": 1.0, "top_p": 1.0, "input": []}
    assert agent._max_turns == 3
    assert agent.model_limits() == {"context": 4096, "input": 4096, "output": 1024}
    assert agent.create_run_agent_commands("x")


# --- install -------------------------------------------------------------------------------------


def test_default_install_is_the_pinned_binary(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    assert agent.version() == "1.17.11"
    assert agent.install_variables() == {
        "version": "1.17.11",
        "url": "https://github.com/anomalyco/opencode/releases/download/v1.17.11/opencode-linux-x64-baseline.tar.gz",
        "sha256": "36064327d5a3fba662264f5726f517eafa26d1aefb0d59bc95e66bddccf3497b",
        "install_dir": "/usr/local/bin",
    }


def test_other_version_has_no_known_digest_unless_given(tmp_path: Path) -> None:
    assert _agent(tmp_path, version="1.18.1").install_variables()["sha256"] == "''"
    digest = "a" * 64
    assert _agent(tmp_path, version="1.18.1", opencode_sha256=digest).install_variables()["sha256"] == digest


class _Env:
    """Records exec/upload calls; every command exits with return_code."""

    def __init__(self, return_code: int = 0) -> None:
        self.calls: list[dict] = []
        self.uploads: list[tuple[str, str]] = []
        self.return_code = return_code
        self.default_user = None

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        from harbor.environments.base import ExecResult

        self.calls.append({"command": command, "cwd": cwd, "env": env, "timeout_sec": timeout_sec, "user": user})
        return ExecResult(stdout="out", stderr=None, return_code=self.return_code)

    async def upload_file(self, source_path, target_path):
        self.uploads.append((Path(source_path).read_text(), target_path))


@pytest.mark.asyncio
async def test_binary_install_renders_and_runs_the_script_as_root(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    env = _Env()
    await agent.install(env)
    ((script, target),) = env.uploads
    assert target == "/installed-agent/install.sh"
    assert script == Environment().from_string(mod.BINARY_INSTALL_TEMPLATE.read_text()).render(
        **agent.install_variables()
    )
    assert env.calls == [
        {"command": "bash /installed-agent/install.sh", "cwd": None, "env": None, "timeout_sec": None, "user": "root"}
    ]
    assert (tmp_path / "setup" / "return-code.txt").read_text() == "0"


@pytest.mark.asyncio
async def test_failed_binary_install_fails_setup(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="exit code 1"):
        await _agent(tmp_path).install(_Env(return_code=1))


@pytest.mark.asyncio
async def test_npm_install_is_harbors(tmp_path: Path) -> None:
    agent = _agent(tmp_path, opencode_install="npm", version="1.18.1")
    env = _Env()
    await agent.install(env)
    assert env.uploads == []
    assert "npm i -g opencode-ai@1.18.1" in env.calls[-1]["command"]
    (cmd,) = agent.create_run_agent_commands("x")
    assert cmd.command.startswith(mod.NVM_PREFIX)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"opencode_install": "docker"}, "opencode_install"),
        ({"version": "1.17.11; rm -rf /"}, "version"),
        ({"opencode_sha256": "XYZ"}, "opencode_sha256"),
    ],
)
def test_rejects_bad_install_kwargs(tmp_path: Path, kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _agent(tmp_path, **kwargs)


def _release(tmp_path: Path, printed_version: str) -> tuple[str, str]:
    """A fake release archive (file:// URL, sha256) whose `opencode` prints `printed_version`."""
    script = f"#!/bin/sh\necho {printed_version}\n".encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("opencode")
        info.size, info.mode = len(script), 0o755
        tar.addfile(info, io.BytesIO(script))
    archive = tmp_path / "release" / "opencode.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(buf.getvalue())
    return archive.as_uri(), hashlib.sha256(buf.getvalue()).hexdigest()


def _run_install(tmp_path: Path, agent: OpenCodeNemoGym, url: str, sha256: str | None = None):
    variables = {
        **agent.install_variables(),
        "url": shlex.quote(url),
        "install_dir": shlex.quote(str(tmp_path / "bin")),
    }
    if sha256 is not None:
        variables["sha256"] = shlex.quote(sha256)
    script = tmp_path / "install.sh"
    script.write_text(Environment().from_string(mod.BINARY_INSTALL_TEMPLATE.read_text()).render(**variables))
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=60)


needs_install_tools = pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("bash", "python3", "tar", "install")),
    reason="needs bash, python3, tar and install",
)


@needs_install_tools
def test_install_script_downloads_verifies_and_installs(tmp_path: Path) -> None:
    url, digest = _release(tmp_path, "1.17.11")
    result = _run_install(tmp_path, _agent(tmp_path), url, digest)
    assert result.returncode == 0, result.stderr
    installed = tmp_path / "bin" / "opencode"
    assert subprocess.run([str(installed), "--version"], capture_output=True, text=True).stdout == "1.17.11\n"
    assert f"sha256 {digest}" in result.stdout
    # a second run finds the right version and downloads nothing
    again = _run_install(tmp_path, _agent(tmp_path), "file:///nonexistent.tar.gz", digest)
    assert again.returncode == 0 and "already installed" in again.stdout


@needs_install_tools
def test_install_script_without_digest_skips_the_check(tmp_path: Path) -> None:
    url, _ = _release(tmp_path, "1.18.1")
    result = _run_install(tmp_path, _agent(tmp_path, version="1.18.1"), url)
    assert result.returncode == 0, result.stderr


@needs_install_tools
def test_install_script_rejects_a_wrong_digest(tmp_path: Path) -> None:
    url, _ = _release(tmp_path, "1.17.11")
    result = _run_install(tmp_path, _agent(tmp_path), url, "0" * 64)
    assert result.returncode != 0
    assert "sha256 mismatch" in result.stderr
    assert not (tmp_path / "bin" / "opencode").exists()


@needs_install_tools
def test_install_script_rejects_a_wrong_version(tmp_path: Path) -> None:
    url, digest = _release(tmp_path, "1.18.1")
    result = _run_install(tmp_path, _agent(tmp_path), url, digest)
    assert result.returncode != 0
    assert "version mismatch" in result.stderr


def test_unknown_kwargs_still_fail(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown option 'bogus'"):
        _agent(tmp_path, bogus=1)


@pytest.mark.asyncio
async def test_run_records_the_command_and_a_failed_exit_without_raising(tmp_path: Path) -> None:
    from harbor.models.agent.context import AgentContext

    agent = _agent(tmp_path, opencode_timeout_sec=870)
    env = _Env(return_code=124)
    await agent.run("do it", env, AgentContext())
    (call,) = env.calls
    (expected,) = agent.create_run_agent_commands("do it")
    assert call["command"] == expected.command and call["env"] == expected.env
    assert call["user"] is None  # the environment's default user, as Harbor 0.1.x ran it
    assert (tmp_path / "command-0" / "return-code.txt").read_text() == "124"


def test_writes_no_atif_trajectory(tmp_path: Path) -> None:
    from harbor.models.agent.context import AgentContext

    (tmp_path / "opencode.txt").write_text('{"type": "text", "part": {"type": "text", "text": "hi"}}\n')
    context = AgentContext()
    _agent(tmp_path).populate_context_post_run(context)
    assert not (tmp_path / "trajectory.json").exists()
    assert context.is_empty()


# Harbor 0.25's Job.create runs AgentFactory.run_preflight -> cls.preflight -> cls.parse_options on the
# config's kwargs before any trial exists (harbor job.py _preflight_agents). Job 5329860: every kwarg was
# "Unknown option" there, so no trial ran and every rollout came back empty.
def test_preflight_accepts_the_kwargs_harbor_agent_passes() -> None:
    kwargs = {
        "api_base": API_BASE,
        "responses_create_params": {"temperature": 1.0},
        "temperature": 1.0,
        "max_turns": 50,
        "interleaved_thinking": True,
        "model_info": {"max_input_tokens": 16384},
        "opencode_config": {"share": "disabled"},
        "opencode_install": "binary",
        "opencode_timeout_sec": 870,
        "opencode_debug": False,
        "opencode_workdir": "/app",
        "version": "1.17.11",
    }
    OpenCodeNemoGym.preflight(kwargs=kwargs, env={})


def test_preflight_still_rejects_an_unknown_option() -> None:
    with pytest.raises(ValueError, match="Unknown option 'not_an_option'"):
        OpenCodeNemoGym.preflight(kwargs={"api_base": API_BASE, "not_an_option": 1}, env={})
