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
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from responses_api_agents.harbor_agent.custom_agents import mini_swe_agent_nemo_gym as mod
from responses_api_agents.harbor_agent.custom_agents.mini_swe_agent_nemo_gym import MiniSweAgentNemoGym


API_BASE = "http://127.0.0.1:8000/ng-rollout/3-1/training-token-capture/v1"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def _agent(tmp_path: Path, **kwargs) -> MiniSweAgentNemoGym:
    return MiniSweAgentNemoGym(logs_dir=tmp_path, model_name=MODEL, api_base=API_BASE, **kwargs)


def test_requires_api_base(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="api_base"):
        MiniSweAgentNemoGym(logs_dir=tmp_path, model_name=MODEL)


def test_settings_carry_rollout_url_and_on_policy_sampling(tmp_path: Path) -> None:
    agent = _agent(
        tmp_path,
        responses_create_params={"temperature": 1.0, "top_p": 1.0, "max_output_tokens": 2048, "input": []},
        max_turns=20,
    )
    assert agent.mini_settings() == {
        "model_kwargs": {
            "api_base": API_BASE,
            "api_key": "EMPTY",
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 2048,
        },
        "max_calls": 20,
    }


def test_unset_sampling_is_not_sent_and_no_limit_by_default(tmp_path: Path) -> None:
    assert _agent(tmp_path).mini_settings() == {
        "model_kwargs": {"api_base": API_BASE, "api_key": "EMPTY"},
        "max_calls": 0,
    }


def test_run_command(tmp_path: Path) -> None:
    (cmd,) = _agent(tmp_path, max_turns=5).create_run_agent_commands("print 'hi' > /app/out.txt")
    assert f"PYTHONPATH={mod.POD_DIR} mini -m openai/{MODEL} --model-class nemo_gym_mini.NemoGymModel " in cmd.command
    assert "-t 'print '\"'\"'hi'\"'\"' > /app/out.txt' -y " in cmd.command
    assert "-o /logs/agent/mini-swe-agent.trajectory.json -l 0 --exit-immediately" in cmd.command
    assert cmd.command.endswith("| tee /logs/agent/mini-swe-agent.txt")
    assert cmd.env["MSWEA_CONFIGURED"] == "true"
    assert json.loads(cmd.env["NEMO_GYM_MINI"])["max_calls"] == 5
    # Harbor's MiniSweAgent reads OPENAI_API_BASE/API keys from the node's environment; this one must not
    assert set(cmd.env) == {"MSWEA_CONFIGURED", "NEMO_GYM_MINI"}


def test_ignores_harbor_agent_extra_kwargs(tmp_path: Path) -> None:
    # harbor_agent passes Terminus-2's kwargs too (temperature, model_info); they must be accepted, and on
    # Harbor 0.25 kept away from its options validation
    agent = _agent(
        tmp_path, temperature=1.0, model_info={"max_input_tokens": 1}, interleaved_thinking=False, version="abc123"
    )
    assert agent.create_run_agent_commands("x")
    assert agent._version == "abc123"


def test_unknown_kwargs_still_fail(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown option 'bogus'"):
        _agent(tmp_path, bogus=1)


def test_install_command_installs_the_pinned_fork_commit(tmp_path: Path) -> None:
    command = _agent(tmp_path, version="8e8a515fdcecf3a8e45c3909f7f196bfe18ca89a").install_command()
    assert command.endswith(
        f"uv tool install --python {mod.MINI_PYTHON} "
        "git+https://github.com/li-boxuan/mini-swe-agent.git@8e8a515fdcecf3a8e45c3909f7f196bfe18ca89a"
    )
    assert "mini-swe-agent==" not in command  # not Harbor 0.25's PyPI install
    if shutil.which("bash"):
        assert subprocess.run(["bash", "-n", "-c", command]).returncode == 0


class _Env:
    def __init__(self, return_code: int = 0) -> None:
        self.calls: list[dict] = []
        self.return_code = return_code
        self.default_user = None

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        from harbor.environments.base import ExecResult

        self.calls.append({"command": command, "env": env, "user": user})
        return ExecResult(stdout="", stderr=None, return_code=self.return_code)


@pytest.mark.asyncio
async def test_install_runs_the_fork_install_as_the_agent(tmp_path: Path) -> None:
    agent = _agent(tmp_path, version="abc123")
    env = _Env()
    await agent.install(env)
    last = env.calls[-1]
    assert last["user"] is None
    assert last["command"].endswith(agent.install_command())


@pytest.mark.asyncio
async def test_run_records_the_command_and_a_failed_exit_without_raising(tmp_path: Path) -> None:
    from harbor.models.agent.context import AgentContext

    agent = _agent(tmp_path, max_turns=3)
    env = _Env(return_code=1)
    await agent.run("do it", env, AgentContext())
    (call,) = env.calls
    assert call["command"] == agent.create_run_agent_commands("do it")[0].command
    assert json.loads(call["env"]["NEMO_GYM_MINI"])["max_calls"] == 3
    assert (tmp_path / "command-0" / "return-code.txt").read_text() == "1"


# Stubs of the two packages nemo_gym_mini.py imports, shaped like mini-swe-agent 1.x / litellm.
_STUBS = {
    "litellm/__init__.py": "from . import cost_calculator\n",
    "litellm/cost_calculator.py": (
        "def completion_cost(response):\n    raise RuntimeError('model not in price list')\n"
    ),
    "minisweagent/__init__.py": "",
    "minisweagent/agents/__init__.py": "",
    "minisweagent/agents/default.py": "class TerminatingException(Exception): pass\n"
    "class LimitsExceeded(TerminatingException): pass\n",
    "minisweagent/models/__init__.py": "",
    "minisweagent/models/litellm_model.py": textwrap.dedent(
        """
        import litellm

        class LitellmModel:
            def __init__(self, *, model_name, model_kwargs=None):
                self.model_name, self.model_kwargs = model_name, model_kwargs or {}
                self.n_calls, self.cost = 0, 0.0

            def query(self, messages, **kwargs):
                cost = litellm.cost_calculator.completion_cost(None)  # mini re-raises errors here
                self.n_calls += 1
                self.cost += cost
                return {
                    "content": f"call {self.n_calls}",
                    "kwargs": self.model_kwargs,
                    "sent": messages,
                }
        """
    ),
}

# Stands in for `mini`: loads the model class the way mini does and records what it sees.
_FAKE_MINI = textwrap.dedent(
    """\
    #!{python}
    import importlib, json, sys
    args = sys.argv[1:]
    module, cls = args[args.index("--model-class") + 1].rsplit(".", 1)
    Model = getattr(importlib.import_module(module), cls)
    m = Model(model_name=args[args.index("-m") + 1], model_kwargs={{"temperature": 0.0, "drop_params": True}})
    out = []
    from minisweagent.agents.default import LimitsExceeded, TerminatingException
    # mini's history: assistant replies carry `extra` (the raw response), as in DefaultAgent.add_message
    history = [{{"role": "system", "content": "sys"}}, {{"role": "user", "content": "task"}}]
    for _ in range(3):
        try:
            reply = m.query(list(history))
            out.append(reply)
            history += [{{"role": "assistant", "content": reply["content"], "extra": {{"response": {{}}}}}},
                        {{"role": "user", "content": "observation"}}]
        except LimitsExceeded:
            out.append({{"limit": "LimitsExceeded"}})  # InteractiveAgent would prompt on stdin here
        except TerminatingException as e:
            out.append({{"limit": type(e).__name__ + ": " + str(e)}})
    print(json.dumps({{"task": args[args.index("-t") + 1], "out": out, "cost": m.cost}}))
    """
)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_run_command_in_bash_loads_the_model_class(tmp_path: Path) -> None:
    stubs = tmp_path / "stubs"
    for rel, src in _STUBS.items():
        (stubs / rel).parent.mkdir(parents=True, exist_ok=True)
        (stubs / rel).write_text(src)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "mini").write_text(_FAKE_MINI.format(python=sys.executable))
    (bin_dir / "mini").chmod(0o755)
    pod_dir = tmp_path / "pod"
    logs = tmp_path / "logs"
    (logs / "agent").mkdir(parents=True)

    agent = _agent(tmp_path, responses_create_params={"temperature": 1.0, "top_p": 0.9}, max_turns=2)
    (cmd,) = agent.create_run_agent_commands("fix it; don't 'break' $HOME")
    command = cmd.command.replace(mod.POD_DIR, str(pod_dir)).replace("/logs/agent", str(logs / "agent"))
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        **cmd.env,
    }
    # the pod module's dir comes first via the command's own PYTHONPATH; the stubs stand in for mini's venv
    command = command.replace(f"PYTHONPATH={pod_dir} ", f"PYTHONPATH={pod_dir}{os.pathsep}{stubs} ")
    res = subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr

    assert (pod_dir / "nemo_gym_mini.py").read_text() == mod.POD_MODULE.read_text()
    result = json.loads(res.stdout.strip().splitlines()[-1])
    assert result["task"] == "fix it; don't 'break' $HOME"
    first, second, third = result["out"]
    assert first["kwargs"] == {
        "temperature": 1.0,  # the rollout's, not mini's 0.0
        "top_p": 0.9,
        "drop_params": True,
        "api_base": API_BASE,
        "api_key": "EMPTY",
    }
    assert second["content"] == "call 2"
    # the second call carries mini's assistant reply; only role/content may reach the model server,
    # which rejects unknown message fields with 422 (job 5253837: every second call failed)
    assert second["sent"][2] == {"role": "assistant", "content": "call 1"}
    assert all(set(msg) == {"role", "content"} for msg in second["sent"])
    assert third == {"limit": "CallLimitReached: Model call limit reached (2)."}
    assert result["cost"] == 0.0
    assert (logs / "agent" / "mini-swe-agent.txt").read_text() == res.stdout


# Harbor 0.25's Job.create validates the config's kwargs with cls.preflight before any trial exists
# (harbor job.py _preflight_agents); see the OpenCode test of the same name (job 5329860).
def test_preflight_accepts_the_kwargs_harbor_agent_passes() -> None:
    kwargs = {
        "api_base": "http://127.0.0.1:8000/v1",
        "responses_create_params": {"temperature": 1.0},
        "temperature": 1.0,
        "max_turns": 50,
        "interleaved_thinking": True,
        "model_info": {"max_input_tokens": 16384},
        "version": "8e8a515fdcecf3a8e45c3909f7f196bfe18ca89a",
    }
    MiniSweAgentNemoGym.preflight(kwargs=kwargs, env={})


def test_preflight_still_rejects_an_unknown_option() -> None:
    with pytest.raises(ValueError, match="Unknown option 'not_an_option'"):
        MiniSweAgentNemoGym.preflight(kwargs={"api_base": "http://127.0.0.1:8000/v1", "not_an_option": 1}, env={})
