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
"""Harbor's installed mini-swe-agent, pointed at a NeMo Gym model server.

Harbor's own agent passes the model URL only through the node's global ``OPENAI_API_BASE`` and keeps
mini's default config (``temperature: 0.0``, no step limit, a litellm cost lookup that fails for a
model litellm does not know). This subclass gives every run its own ``api_base`` (harbor_agent's
rollout URL, with the capture prefix under token capture), the rollout's sampling parameters and a
call limit, through a small model class (``mini_swe_agent_pod/nemo_gym_mini.py``) that it writes into
the sandbox and loads with ``mini --model-class``.

mini-swe-agent runs inside the task sandbox and calls the model with litellm, without streaming,
so it works with worker-owned token capture (which rejects ``stream: true``). ``api_base`` must be
reachable from the sandbox (harbor_agent's ``harbor_model_base_url``, e.g. the per-sandbox model
tunnel at http://127.0.0.1:8000). Harbor's ATIF conversion is inherited.

On Harbor 0.25 the run goes through ``installed_compat`` (one recorded command, no failure on a non-zero
exit, as on Harbor 0.1.x). Harbor 0.25 installs mini-swe-agent 2.x from PyPI (``mini-swe-agent==<version>``,
harbor agents/installed/mini_swe_agent.py:756-791); ``version`` here is a commit of the li-boxuan fork
(mini-swe-agent 1.x with the ``mini`` CLI and ``--model-class``), which Harbor 0.1.x installed from git, so
``install`` keeps doing that. Harbor 0.25's ATIF conversion reads the 1.x trajectory (messages, observations,
token usage) but no longer turns its ``bash`` blocks into tool calls.
"""

import json
import shlex
from pathlib import Path
from typing import Any, Mapping

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from responses_api_agents.harbor_agent.custom_agents.installed_compat import (
    ExecInput,
    drop_shared_kwargs,
    kwargs_for_harbor_options,
    run_commands,
)


POD_MODULE = Path(__file__).parent / "mini_swe_agent_pod" / "nemo_gym_mini.py"
POD_DIR = "/tmp/nemo-gym-mini"
# Harbor 0.1.x's install-mini-swe-agent.sh.j2: `uv tool install git+<this>@<version>`
MINI_GIT_URL = "git+https://github.com/li-boxuan/mini-swe-agent.git"
# where `uv tool install` puts `mini`; Harbor 0.1.x relied on the interactive shell's .bashrc for it
LOCAL_BIN_PATH = 'export PATH="$HOME/.local/bin:$PATH"; '
# mini-swe-agent's own Python, as Harbor 0.25 (mini_swe_agent.py:50); the fork needs >= 3.10
MINI_PYTHON = "3.12"

# Responses API names -> chat-completions names (litellm). Unset ones are not sent.
SAMPLING_KEYS = {"temperature": "temperature", "top_p": "top_p", "max_output_tokens": "max_tokens"}


class MiniSweAgentNemoGym(MiniSweAgent):
    @classmethod
    def preflight(cls, kwargs: dict[str, Any] | None = None, env: Mapping[str, str] | None = None) -> None:
        # Harbor 0.25 checks the config's kwargs before any trial (Job.create); ours are taken by __init__
        super().preflight(kwargs=kwargs_for_harbor_options(cls, kwargs), env=env)

    def __init__(
        self,
        *args: Any,
        api_base: str | None = None,
        api_key: str = "EMPTY",
        responses_create_params: dict[str, Any] | None = None,
        max_turns: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **drop_shared_kwargs(kwargs))
        if not api_base:
            raise ValueError("MiniSweAgentNemoGym needs api_base (harbor_agent passes the model server URL)")
        self._api_base = api_base
        self._api_key = api_key
        self._responses_create_params = responses_create_params or {}
        self._max_turns = max_turns

    @staticmethod
    def name() -> str:
        return "mini-swe-agent-nemo-gym"

    def install_command(self) -> str:
        spec = f"{MINI_GIT_URL}@{self._version}" if self._version else MINI_GIT_URL
        python = shlex.quote(MINI_PYTHON)
        return (
            "set -euo pipefail; "
            "if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi && "
            'if ! grep -q \'export PATH="$HOME/.local/bin:$PATH"\' "$HOME/.bashrc" 2>/dev/null; then '
            'echo \'export PATH="$HOME/.local/bin:$PATH"\' >> "$HOME/.bashrc"; fi && '
            f"{LOCAL_BIN_PATH}"
            f"uv python install {python} && uv tool install --python {python} {shlex.quote(spec)}"
        )

    async def install(self, environment: BaseEnvironment) -> None:
        # system packages as Harbor 0.25's MiniSweAgent.install (as root, per package manager), then the fork
        await self.ensure_system_dependencies(environment, ("curl", "bash", "build_tools", "git"))
        await self.exec_as_agent(environment, command=self.install_command())

    def _litellm_model(self) -> str:
        if not self.model_name:
            raise ValueError("MiniSweAgentNemoGym needs a model name")
        # litellm's OpenAI-compatible provider; the rest is sent as the request's model name
        return f"openai/{self.model_name}"

    def mini_settings(self) -> dict[str, Any]:
        model_kwargs: dict[str, Any] = {"api_base": self._api_base, "api_key": self._api_key}
        for src, dst in SAMPLING_KEYS.items():
            if self._responses_create_params.get(src) is not None:
                model_kwargs[dst] = self._responses_create_params[src]
        return {"model_kwargs": model_kwargs, "max_calls": int(self._max_turns or 0)}

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        module = shlex.quote(POD_MODULE.read_text())
        command = (
            f"mkdir -p {POD_DIR} && printf '%s' {module} > {POD_DIR}/nemo_gym_mini.py && "
            f"{LOCAL_BIN_PATH}PYTHONPATH={POD_DIR} mini -m {shlex.quote(self._litellm_model())} "
            f"--model-class nemo_gym_mini.NemoGymModel -t {shlex.quote(instruction)} -y "
            f"-o {self._mini_swe_agent_trajectory_path} -l 0 --exit-immediately "
            f"2>&1 </dev/null | tee /logs/agent/mini-swe-agent.txt"
        )
        env = {
            "MSWEA_CONFIGURED": "true",  # no interactive first-run setup
            "NEMO_GYM_MINI": json.dumps(self.mini_settings()),
        }
        return [ExecInput(command=command, env=env)]

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await run_commands(self, environment, self.create_run_agent_commands(instruction))
