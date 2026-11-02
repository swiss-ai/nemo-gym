# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest


pytest.importorskip("harbor")

from responses_api_agents.harbor_agent.custom_agents.opencode_nemo_gym import NVM_PREFIX, OpenCodeNemoGym  # noqa: E402


API_BASE = "http://policy-host:9000/ng-rollout/3-1/training-token-capture/v1"


def _agent(tmp_path: Path, **kwargs) -> OpenCodeNemoGym:
    return OpenCodeNemoGym(logs_dir=tmp_path, model_name="Qwen/Qwen3-4B-Instruct-2507", api_base=API_BASE, **kwargs)


def test_requires_api_base(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="api_base"):
        OpenCodeNemoGym(logs_dir=tmp_path, model_name="Qwen/Qwen3-4B-Instruct-2507")


def test_run_command_points_opencode_at_the_model_server(tmp_path: Path) -> None:
    (cmd,) = _agent(tmp_path).create_run_agent_commands("print 'hi' > /app/out.txt")
    assert cmd.command.startswith(
        NVM_PREFIX + "opencode --model nemogym/Qwen/Qwen3-4B-Instruct-2507 run --format=json "
    )
    assert "'print '\"'\"'hi'\"'\"' > /app/out.txt'" in cmd.command
    config = json.loads(cmd.env["OPENCODE_CONFIG_CONTENT"])
    provider = config["provider"]["nemogym"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == API_BASE
    assert list(provider["models"]) == ["Qwen/Qwen3-4B-Instruct-2507"]
    assert config["model"] == config["small_model"] == "nemogym/Qwen/Qwen3-4B-Instruct-2507"


def test_ignores_harbor_agent_extra_kwargs(tmp_path: Path) -> None:
    # harbor_agent always passes these; the agent must accept and ignore them
    # (Harbor 0.25 rejects agent kwargs its options model lacks, so they must not reach it), as must the
    # Terminus-2 options of the shared harbor_agent_kwargs
    agent = _agent(
        tmp_path,
        responses_create_params={"temperature": 1.0},
        temperature=1.0,
        interleaved_thinking=False,
        max_turns=30,
        model_info={"max_input_tokens": 1000},
    )
    assert agent.create_run_agent_commands("x")


def test_unknown_kwargs_still_fail(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown option 'bogus'"):
        _agent(tmp_path, bogus=1)


class _Env:
    def __init__(self, return_code: int = 0) -> None:
        self.calls: list[dict] = []
        self.return_code = return_code

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        from harbor.environments.base import ExecResult

        self.calls.append({"command": command, "cwd": cwd, "env": env, "timeout_sec": timeout_sec, "user": user})
        return ExecResult(stdout="out", stderr=None, return_code=self.return_code)


@pytest.mark.asyncio
async def test_run_records_the_command_and_a_failed_exit_without_raising(tmp_path: Path) -> None:
    from harbor.models.agent.context import AgentContext

    agent = _agent(tmp_path)
    env = _Env(return_code=124)
    await agent.run("do it", env, AgentContext())
    (call,) = env.calls
    assert call["command"] == agent.create_run_agent_commands("do it")[0].command
    assert call["user"] is None  # the environment's default user, as Harbor 0.1.x ran it
    assert json.loads(call["env"]["OPENCODE_CONFIG_CONTENT"])["model"] == "nemogym/Qwen/Qwen3-4B-Instruct-2507"
    assert (tmp_path / "command-0" / "return-code.txt").read_text() == "124"
    assert (tmp_path / "command-0" / "stdout.txt").read_text() == "out"


def test_writes_no_atif_trajectory(tmp_path: Path) -> None:
    from harbor.models.agent.context import AgentContext

    (tmp_path / "opencode.txt").write_text('{"type": "text", "part": {"type": "text", "text": "hi"}}\n')
    context = AgentContext()
    _agent(tmp_path).populate_context_post_run(context)
    assert not (tmp_path / "trajectory.json").exists()
    assert context.is_empty()
