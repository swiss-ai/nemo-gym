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
"""Command-list helpers for our installed agents (OpenCode, mini-swe-agent) on Harbor 0.25.

Harbor up to 0.1.x had installed agents return their run commands as a list of ``ExecInput`` and ran
them in ``BaseInstalledAgent.run``, recording each command, its return code and output under
``<agent logs>/command-<i>/`` without failing on a non-zero exit; setup rendered a Jinja install
script and ran it. Harbor 0.25 dropped both: agents implement ``install()`` and ``run()`` themselves,
and its ``exec_as_agent`` prefixes every command with ``set -o pipefail;`` and raises on a non-zero
exit (harbor agents/installed/base.py:999-1083). Our adapters keep the old contract with these
helpers, so their commands, logs and exit handling stay as they were validated (e.g. OpenCode's own
``timeout`` exit status is recorded, the verifier still runs).

Harbor 0.25 also validates every agent kwarg it does not consume against the agent's
``options_model``, which forbids unknown keys (agents/base.py:191, 283-305; agents/options.py:53);
Harbor 0.1.x ignored them. harbor_agent passes the same kwargs to every agent (app.py
``_build_job_config``: ``responses_create_params``, ``temperature``), and the Terminus-2 configs'
``harbor_agent_kwargs`` carry options other agents do not have; see ``drop_shared_kwargs``.
"""

import inspect
from pathlib import Path
from typing import Any, Iterable, Mapping

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from jinja2 import Environment
from pydantic import BaseModel


# kwargs that harbor_agent or the shared Terminus-2 harbor_agent_kwargs give to whichever agent runs. The last four
# are Terminus-2 options from Gym's configs/harbor_agent_opensandbox_training.yaml, which every run loads; an agent
# overlay merged on top cannot delete them (OmegaConf), so OpenCode and mini get them too (job 5330618).
SHARED_AGENT_KWARGS = (
    "responses_create_params",
    "temperature",
    "interleaved_thinking",
    "max_turns",
    "model_info",
    "enable_summarize",
    "collect_rollout_details",
    "trajectory_config",
    "nemo_model_server_timeout_sec",
)


def drop_shared_kwargs(kwargs: dict[str, Any], keep: Iterable[str] = ()) -> dict[str, Any]:
    """``kwargs`` without the SHARED_AGENT_KWARGS this agent does not use (``keep`` lists those it does)."""
    unused = set(SHARED_AGENT_KWARGS) - set(keep)
    return {key: value for key, value in kwargs.items() if key not in unused}


def kwargs_for_harbor_options(agent_cls: type, kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """``kwargs`` without the ones ``agent_cls.__init__`` takes itself and the SHARED_AGENT_KWARGS.

    Harbor 0.25's Job.create validates each agent config's kwargs against the agent's ``options_model``
    before any trial exists (job.py ``_preflight_agents`` -> agents/base.py ``preflight`` ->
    ``parse_options``). Our adapters consume their own kwargs in ``__init__`` and pass the rest on, so their
    ``preflight`` must hand Harbor only that rest; otherwise every one of them is an "Unknown option".
    """
    own = {
        name
        for name, param in inspect.signature(agent_cls.__init__).parameters.items()
        if param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD) and name != "self"
    }
    return {key: value for key, value in drop_shared_kwargs(dict(kwargs or {})).items() if key not in own}


class ExecInput(BaseModel):
    """One command of an installed agent's run (Harbor 0.1.x ``harbor.agents.installed.base.ExecInput``)."""

    command: str
    cwd: str | None = None
    env: dict[str, str] | None = None
    timeout_sec: int | None = None


async def run_commands(agent: BaseAgent, environment: BaseEnvironment, commands: Iterable[ExecInput]) -> None:
    """Run ``commands`` in order as the environment's default user, as Harbor 0.1.x's installed-agent ``run``.

    Each command, its return code and its output go to ``<agent logs>/command-<i>/``. A non-zero exit is
    recorded, not raised. Harbor 0.25's Trial calls ``populate_context_post_run`` itself after it has
    downloaded the agent logs (harbor trial/trial.py ``_sync_agent_output``), so this does not.
    """
    for i, exec_input in enumerate(commands):
        command_dir = agent.logs_dir / f"command-{i}"
        command_dir.mkdir(parents=True, exist_ok=True)
        (command_dir / "command.txt").write_text(exec_input.command)
        result = await environment.exec(
            command=exec_input.command,
            cwd=exec_input.cwd,
            env=exec_input.env,
            timeout_sec=exec_input.timeout_sec,
        )
        (command_dir / "return-code.txt").write_text(str(result.return_code))
        if result.stdout:
            (command_dir / "stdout.txt").write_text(result.stdout)
        if result.stderr:
            (command_dir / "stderr.txt").write_text(result.stderr)


async def install_from_template(
    agent: BaseAgent,
    environment: BaseEnvironment,
    template_path: Path,
    variables: Mapping[str, Any],
    user: str | int | None = "root",
) -> None:
    """Render a Jinja install script, upload it and run it, as Harbor 0.1.x's installed-agent ``setup``.

    Harbor 0.1.x ran it without a user (the sandbox's default, root in the task images); Harbor 0.25's
    agents install system packages with ``exec_as_root``, so root is the default here as well. The
    script's return code and output go to ``<agent logs>/setup/``; a non-zero exit raises.
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Install agent template file not found: {template_path}")
    script_path = agent.logs_dir / "install.sh"
    script_path.write_text(Environment().from_string(template_path.read_text()).render(**variables))
    await environment.upload_file(source_path=script_path, target_path="/installed-agent/install.sh")
    result = await environment.exec(command="bash /installed-agent/install.sh", user=user)

    setup_dir = agent.logs_dir / "setup"
    setup_dir.mkdir(parents=True, exist_ok=True)
    (setup_dir / "return-code.txt").write_text(str(result.return_code))
    if result.stdout:
        (setup_dir / "stdout.txt").write_text(result.stdout)
    if result.stderr:
        (setup_dir / "stderr.txt").write_text(result.stderr)
    if result.return_code != 0:
        raise RuntimeError(f"Agent setup failed with exit code {result.return_code}. See logs in {setup_dir}")
