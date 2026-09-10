# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import shlex
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.workspace_swe.app import WorkspaceSWEConfig, WorkspaceSWEServer, _pass_fraction, _write_file


def _response(text):
    return NeMoGymResponse.model_validate(
        {
            "id": "r",
            "created_at": 0,
            "model": "m",
            "object": "response",
            "output": [
                {
                    "id": "msg",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    )


@pytest.fixture
def env(tmp_path):
    config = WorkspaceSWEConfig(host="", port=0, entrypoint="", name="", workspace_dir=str(tmp_path))
    return WorkspaceSWEServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.mark.asyncio
async def test_edit_submit_real_tests_and_cleanup(env):
    fixture = Path(__file__).parents[1] / "data/example.jsonl"
    task = json.loads(fixture.read_text())
    task["test_cmd"] = f"{shlex.quote(sys.executable)} -m pytest -q"
    observation, info = await env.reset(task, "a")
    root = env.session_state["a"]["root"]
    assert "mathutils.py" in observation
    assert info == {}
    assert not (root / "test_mathutils.py").exists()
    await env.step(
        _response('<write path="mathutils.py">def add(a,b): return a+b\ndef divide(a,b): return a/b</write>'), {}, "a"
    )
    _, reward, terminated, truncated, info = await env.step(_response("<submit/>"), {}, "a")
    assert (reward, terminated, truncated) == (1.0, True, False)
    assert "2 passed" in info["test_output"]
    assert not root.exists()
    assert "a" not in env.session_state


@pytest.mark.asyncio
async def test_max_steps_grades_and_sessions_are_independent(env):
    task = {"files": {"value": "a"}, "test_cmd": "printf '1 passed, 1 failed'; exit 1", "max_steps": 1}
    await env.reset(task, "a")
    await env.reset({"files": {"value": "b"}}, "b")
    _, reward, terminated, truncated, info = await env.step(_response("<cmd>cat value</cmd>"), {}, "a")
    assert (reward, terminated, truncated) == (0.5, False, True)
    assert info["truncated_at_max_steps"]
    assert (env.session_state["b"]["root"] / "value").read_text() == "b"
    await env.close_session("b")


@pytest.mark.asyncio
async def test_reset_replaces_workspace_and_failed_setup_cleans_up(env):
    await env.reset({}, "a")
    first = env.session_state["a"]["workspace"]
    await env.reset({}, "a")
    assert not first.exists()
    observation, info = await env.reset({"setup_cmd": "exit 2"}, "a")
    assert "Setup failed" in observation
    assert info["error"] == "setup_failed"
    assert not env.session_state
    assert not list(Path(env.config.workspace_dir).iterdir())


@pytest.mark.parametrize("relative", ["../repo-other/file", "/tmp/escape"])
def test_reject_paths_outside_workspace(tmp_path, relative):
    root = tmp_path / "repo"
    root.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        _write_file(root, relative, "bad")


@pytest.mark.asyncio
async def test_commands_timeout_without_blocking_event_loop(env):
    task = asyncio.create_task(env._run("sleep 30", Path(env.config.workspace_dir), timeout=0.05))
    await asyncio.sleep(0.01)
    assert not task.done()
    output, code = await task
    assert "timed out" in output
    assert code == -1


def test_final_summary_before_output_truncation():
    assert _pass_fraction("1 passed\n" + "x" * 5000 + "\n2 passed, 1 failed, 1 error in 0.2s") == 0.5
    assert _pass_fraction("no tests ran") == 0.0


@pytest.mark.asyncio
async def test_failed_test_runner_does_not_reward_output(env):
    await env.reset({"test_cmd": "printf '10 passed'; exit 2"}, "a")
    _, reward, terminated, _, _ = await env.step(_response("<submit/>"), {}, "a")
    assert reward == 0.0
    assert terminated


@pytest.mark.asyncio
async def test_invalid_hidden_test_path_returns_error_and_cleans_up(env):
    await env.reset({"hidden_tests": {"../repo-other/test.py": "bad"}}, "a")
    root = env.session_state["a"]["root"]
    _, reward, terminated, truncated, info = await env.step(_response("<submit/>"), {}, "a")
    assert (reward, terminated, truncated) == (0.0, True, False)
    assert info["error"] == "grading_failed"
    assert not root.exists()
    assert not env.session_state
