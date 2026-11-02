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
"""Harbor's installed OpenCode agent, pointed at a NeMo Gym model server.

Harbor's own OpenCode agent (at the Gym pin) only knows hosted providers selected by the
`provider/model` prefix of the model name and their API-key environment variables. This subclass
declares one OpenAI-compatible provider whose base URL is the `api_base` that harbor_agent passes
(the model server, with the rollout's capture prefix when token capture is on), so every call of
the in-sandbox OpenCode reaches the Gym model server and can be attributed to its rollout.

OpenCode runs inside the task sandbox, so `api_base` must be reachable from there (see
harbor_agent's `harbor_model_base_url`). The agent writes no ATIF trajectory; for training the
tokens come from model-server capture, not from this agent.

On Harbor 0.25 the run goes through `installed_compat` (one recorded command, no failure on a
non-zero exit, as on Harbor 0.1.x) and Harbor's ATIF conversion of `opencode.txt` stays off.
"""

import json
import shlex
from typing import Any

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from responses_api_agents.harbor_agent.custom_agents.installed_compat import (
    ExecInput,
    drop_shared_kwargs,
    run_commands,
)


PROVIDER_ID = "nemogym"
# Harbor 0.25 installs OpenCode with npm under nvm and sources it before running (opencode.py:913)
NVM_PREFIX = "[ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; "


class OpenCodeNemoGym(OpenCode):
    def __init__(self, *args: Any, api_base: str | None = None, api_key: str = "EMPTY", **kwargs: Any) -> None:
        super().__init__(*args, **drop_shared_kwargs(kwargs))
        if not api_base:
            raise ValueError("OpenCodeNemoGym needs api_base (harbor_agent passes the model server URL)")
        self._api_base = api_base
        self._api_key = api_key

    @staticmethod
    def name() -> str:
        return "opencode-nemo-gym"

    def _opencode_model(self) -> str:
        if not self.model_name:
            raise ValueError("OpenCodeNemoGym needs a model name")
        return f"{PROVIDER_ID}/{self.model_name}"

    def opencode_config(self) -> dict[str, Any]:
        model = self._opencode_model()
        return {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                PROVIDER_ID: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "NeMo Gym model server",
                    "options": {"baseURL": self._api_base, "apiKey": self._api_key},
                    "models": {self.model_name: {"name": self.model_name}},
                }
            },
            "model": model,
            # Title generation and similar helper calls use small_model; keep them on the same
            # endpoint so no call leaves for a hosted provider.
            "small_model": model,
            "share": "disabled",
            "autoupdate": False,
        }

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        env = {
            "OPENCODE_CONFIG_CONTENT": json.dumps(self.opencode_config()),
            # as in Harbor's OpenCode agent
            "OPENCODE_FAKE_VCS": "git",
        }
        return [
            ExecInput(
                command=(
                    f"{NVM_PREFIX}opencode --model {shlex.quote(self._opencode_model())} run --format=json "
                    f"{shlex.quote(instruction)} 2>&1 | tee /logs/agent/opencode.txt"
                ),
                env=env,
            )
        ]

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await run_commands(self, environment, self.create_run_agent_commands(instruction))

    def populate_context_post_run(self, context: AgentContext) -> None:
        # Harbor 0.25's OpenCode converts opencode.txt into an ATIF trajectory.json, which harbor_agent
        # would turn into output items; Harbor 0.1.x's wrote none and this adapter keeps it that way.
        pass
