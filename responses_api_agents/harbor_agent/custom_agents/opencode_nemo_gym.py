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

Under token capture a rollout without a terminal witness (OpenCode returns no output items) is
masked as soon as its calls form more than one root or chain. The configuration therefore switches
off every model call that is not the main agent loop: session title, sub-agents (`task`),
compaction and pruning; provider timeouts are off, because a timed-out request is retried and the
retry is one more captured call. Sampling parameters are not set here (OpenCode does not send the
rollout's temperature/top_p); set them with the Gym model server's `sampling_overrides`.

OpenCode keys and flags below were checked against the OpenCode source at tag v1.17.11
(github.com/anomalyco/opencode, commit 67aec2212); file:line references are relative to that tree.
Another version may name them differently, so `version` defaults to that release.

OpenCode runs inside the task sandbox, so `api_base` must be reachable from there (see
harbor_agent's `harbor_model_base_url`, e.g. the per-sandbox model tunnel). The agent writes no
ATIF trajectory; for training the tokens come from model-server capture, not from this agent.

On Harbor 0.25 the run goes through `installed_compat` (one recorded command, no failure on a
non-zero exit, as on Harbor 0.1.x) and Harbor's ATIF conversion of `opencode.txt` stays off.
"""

import copy
import json
import re
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from responses_api_agents.harbor_agent.custom_agents.installed_compat import (
    ExecInput,
    drop_shared_kwargs,
    install_from_template,
    run_commands,
)


PROVIDER_ID = "nemogym"

# Release whose config schema, flags and CLI this adapter was checked against.
DEFAULT_VERSION = "1.17.11"
# Standalone Bun-compiled binary for glibc x86_64 without AVX2 requirements (no Node/npm needed).
DEFAULT_ASSET = "opencode-linux-x64-baseline.tar.gz"
RELEASE_URL = "https://github.com/anomalyco/opencode/releases/download/v{version}/{asset}"
# sha256 of release archives as listed by the GitHub release API ("digest" of the asset).
KNOWN_ARCHIVE_SHA256 = {
    ("1.17.11", "opencode-linux-x64-baseline.tar.gz"): (
        "36064327d5a3fba662264f5726f517eafa26d1aefb0d59bc95e66bddccf3497b"
    ),
}
INSTALL_METHODS = ("binary", "npm")
BINARY_INSTALL_TEMPLATE = Path(__file__).parent / "opencode_pod" / "install-opencode-binary.sh.j2"
BINARY_INSTALL_DIR = "/usr/local/bin"
# Harbor 0.25 installs OpenCode with npm under nvm and sources it before running (opencode.py:913)
NVM_PREFIX = "[ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; "

# A fixed, non-default session title: OpenCode skips its title model call for such a session
# (session/prompt.ts:204, `--title` -> cli/cmd/run.ts:186-189, 433-436, 501-504).
SESSION_TITLE = "NeMo Gym"
LOG_FILE = "/logs/agent/opencode.txt"
# OpenCode's data dir (sqlite DB with sessions/messages, its log files) under the trial's agent
# logs, so Harbor downloads it with /logs/agent (core/src/global.ts:11,23 via xdg-basedir).
DATA_HOME = "/logs/agent/opencode-data"
TIMEOUT_KILL_AFTER_SEC = 10

_VERSION_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.+-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


# The sandbox exec may run the command with /bin/sh (OpenSandbox runs `su -s /bin/sh -c` for a
# named user). A plain `set -o pipefail` is a special-builtin error in a shell without it (older
# dash), which ends a non-interactive shell before OpenCode starts. Probe in a subshell first.
PIPEFAIL_IF_SUPPORTED = "(set -o pipefail) 2>/dev/null && set -o pipefail; "


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """`base` updated by `overlay`, recursing into dicts; a None value in `overlay` deletes the key."""
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class OpenCodeNemoGym(OpenCode):
    def __init__(
        self,
        *args: Any,
        api_base: str | None = None,
        api_key: str = "EMPTY",
        responses_create_params: dict[str, Any] | None = None,
        max_turns: int | None = None,
        model_info: dict[str, Any] | None = None,
        opencode_config: dict[str, Any] | None = None,
        opencode_timeout_sec: int | None = None,
        opencode_debug: bool = False,
        opencode_install: str = "binary",
        opencode_sha256: str | None = None,
        opencode_asset: str = DEFAULT_ASSET,
        opencode_workdir: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Harbor's base keeps `version` (the OpenCode release, or the npm version for "npm").
        super().__init__(*args, **drop_shared_kwargs(kwargs))
        if not api_base:
            raise ValueError("OpenCodeNemoGym needs api_base (harbor_agent passes the model server URL)")
        if opencode_install not in INSTALL_METHODS:
            raise ValueError(f"opencode_install must be one of {INSTALL_METHODS}, got {opencode_install!r}")
        if opencode_timeout_sec is not None and int(opencode_timeout_sec) <= 0:
            raise ValueError(f"opencode_timeout_sec must be positive, got {opencode_timeout_sec!r}")
        if self._version is None:
            self._version = DEFAULT_VERSION
        if not _VERSION_RE.match(self._version):
            raise ValueError(f"invalid OpenCode version {self._version!r}")
        if opencode_sha256 is not None and not _SHA256_RE.match(opencode_sha256):
            raise ValueError("opencode_sha256 must be 64 lowercase hex digits")
        if opencode_workdir is not None and not str(opencode_workdir).startswith("/"):
            raise ValueError(f"opencode_workdir must be an absolute path, got {opencode_workdir!r}")
        self._api_base = api_base
        self._api_key = api_key
        # Kept for reference; sampling values reach vLLM through vllm_model.sampling_overrides.
        self._responses_create_params = responses_create_params or {}
        self._max_turns = max_turns
        self._model_info = model_info or {}
        self._opencode_config_overlay = opencode_config or {}
        self._timeout_sec = int(opencode_timeout_sec) if opencode_timeout_sec is not None else None
        self._debug = bool(opencode_debug)
        self._install_method = opencode_install
        self._asset = opencode_asset
        self._workdir = opencode_workdir
        self._sha256 = opencode_sha256 or KNOWN_ARCHIVE_SHA256.get((self._version, opencode_asset))

    @staticmethod
    def name() -> str:
        return "opencode-nemo-gym"

    # --- install -------------------------------------------------------------------------------

    def install_variables(self) -> dict[str, str]:
        """Variables of the binary install script (BINARY_INSTALL_TEMPLATE)."""
        return {
            "version": self._version,
            "url": shlex.quote(RELEASE_URL.format(version=self._version, asset=self._asset)),
            "sha256": shlex.quote(self._sha256 or ""),
            "install_dir": shlex.quote(BINARY_INSTALL_DIR),
        }

    async def install(self, environment: BaseEnvironment) -> None:
        if self._install_method == "npm":
            # Harbor's: system packages, Node (nvm, or the packaged one on musl), `npm i -g opencode-ai@<v>`
            await super().install(environment)
            return
        # as root: it writes to BINARY_INSTALL_DIR
        await install_from_template(self, environment, BINARY_INSTALL_TEMPLATE, self.install_variables())

    # --- configuration -------------------------------------------------------------------------

    def _opencode_model(self) -> str:
        if not self.model_name:
            raise ValueError("OpenCodeNemoGym needs a model name")
        # parseModel splits on the first "/" only (provider/provider.ts:1974-1980)
        return f"{PROVIDER_ID}/{self.model_name}"

    def model_limits(self) -> dict[str, int] | None:
        """OpenCode's model `limit` from Terminus-2-style `model_info` (None when unknown).

        `max_input_tokens` is the whole context window in our overlays (max_total_sequence_length),
        as for Terminus-2; the output limit is additionally capped by the request's max_output_tokens.
        """
        context = self._model_info.get("max_input_tokens") or self._model_info.get("max_tokens")
        if not context:
            return None
        output = self._model_info.get("max_output_tokens") or context
        request_cap = self._responses_create_params.get("max_output_tokens")
        if request_cap:
            output = min(int(output), int(request_cap))
        # limit.context and limit.output are required, limit.input optional
        # (core/src/v1/config/provider.ts:41-47)
        return {"context": int(context), "input": int(context), "output": int(output)}

    def opencode_config(self) -> dict[str, Any]:
        model = self._opencode_model()
        model_entry: dict[str, Any] = {"name": self.model_name}
        limits = self.model_limits()
        if limits:
            model_entry["limit"] = limits
        build_agent: dict[str, Any] = {}
        if self._max_turns:
            # steps: max agentic iterations, the last one without tools
            # (core/src/v1/config/agent.ts:34-36; session/prompt.ts:1281-1282)
            build_agent["steps"] = int(self._max_turns)
        config: dict[str, Any] = {
            "$schema": "https://opencode.ai/config.json",
            # Top-level keys must all exist in the schema, unknown ones fail the config load
            # (config/parse.ts:40-53; schema core/src/v1/config/config.ts:32-187).
            "enabled_providers": [PROVIDER_ID],  # config.ts:71-73, provider/provider.ts:1358-1364
            "provider": {
                PROVIDER_ID: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "NeMo Gym model server",
                    "options": {
                        "baseURL": self._api_base,
                        "apiKey": self._api_key,
                        # No request or header timeout: under stream replay the model server sends
                        # no byte before generation ends, and a timed-out request is retried as a
                        # new captured call. Both accept false (core/src/v1/config/provider.ts:95-110;
                        # applied in provider/provider.ts:1704-1730, which also passes Bun's fetch
                        # `timeout: false`). chunkTimeout takes only a positive int and is off when
                        # unset (provider.ts:111-114, provider/provider.ts:1712).
                        "timeout": False,
                        "headerTimeout": False,
                    },
                    "models": {self.model_name: model_entry},
                }
            },
            "model": model,
            # Helper calls (title, ...) use small_model; keep it on the same endpoint so no call
            # leaves for a hosted provider (config.ts:77-79).
            "small_model": model,
            "share": "disabled",  # config.ts:57-60
            "autoupdate": False,  # config.ts:64-67; cli/upgrade.ts:10
            # No git snapshots of the work tree per step (config.ts:52-55; session/summary.ts:115).
            "snapshot": False,
            # Compaction makes a summary call and starts a new chain; pruning rewrites old tool
            # results so the next prompt no longer extends its parent (config.ts:146-165;
            # session/overflow.ts:28, session/compaction.ts:248).
            "compaction": {"auto": False, "prune": False},
            "agent": {
                # `disable` deletes the built-in title agent; ensureTitle then returns before any
                # call (agent/agent.ts:234-249, 267-270; session/prompt.ts:220-221).
                "title": {"disable": True},
                **({"build": build_agent} if build_agent else {}),
            },
            # Last matching rule wins and user rules come after OpenCode's defaults, so "*" must be
            # first (permission/index.ts:29-38; agent/agent.ts:119-151). A tool denied for pattern
            # "*" is removed from the request's tool list (session/llm/request.ts:199-203,
            # permission/index.ts:205-216): no sub-agents (separate roots), no web access, no todo
            # list (prompt tokens only). Without "*": allow, `opencode run` auto-rejects every "ask"
            # (e.g. doom_loop, external_directory) (cli/cmd/run.ts:779-797).
            "permission": {
                "*": "allow",
                "task": "deny",
                "webfetch": "deny",
                "websearch": "deny",
                "todowrite": "deny",
            },
        }
        return deep_merge(config, self._opencode_config_overlay)

    def opencode_env(self) -> dict[str, str]:
        env = {
            "OPENCODE_CONFIG_CONTENT": json.dumps(self.opencode_config()),
            "XDG_DATA_HOME": DATA_HOME,
            # Boolean flags are true for "true" or "1" (core/src/flag/flag.ts:3-6).
            "OPENCODE_DISABLE_MODELS_FETCH": "true",  # flag.ts:29; core/src/models-dev.ts:200, 233
            "OPENCODE_DISABLE_AUTOUPDATE": "true",  # flag.ts:23; cli/upgrade.ts:10
            # No opencode.json/.opencode from the task's work tree (flag.ts:54-56; config/config.ts:405).
            "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
            # as in Harbor's OpenCode agent (flag.ts:31)
            "OPENCODE_FAKE_VCS": "git",
        }
        limits = self.model_limits()
        if limits:
            # Caps max output tokens per request: min(limit.output, this), default 32000
            # (effect/runtime-flags.ts:51; provider/transform.ts:18, 1307-1309).
            env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(limits["output"])
        return env

    # --- run -----------------------------------------------------------------------------------

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        argv: list[str] = []
        if self._timeout_sec is not None:
            # Ends OpenCode before Harbor's agent timeout so the command's return code is recorded.
            argv += ["timeout", "-k", str(TIMEOUT_KILL_AFTER_SEC), str(self._timeout_sec)]
        argv += ["opencode", "--model", self._opencode_model(), "run", "--format=json", "--title", SESSION_TITLE]
        if self._debug:
            argv += ["--print-logs", "--log-level", "DEBUG"]  # global options (index.ts:53-61)
        # "--": an instruction starting with "-" stays the message (cli/cmd/run.ts:256, 271).
        argv += ["--", instruction]
        # stdin from /dev/null: with a non-TTY stdin `opencode run` reads it to EOF before doing
        # anything (cli/cmd/run.ts:399), and the sandbox exec leaves it open. pipefail keeps
        # OpenCode's exit status (e.g. 124 from timeout) instead of tee's.
        command = f"{PIPEFAIL_IF_SUPPORTED}{shlex.join(argv)} 2>&1 </dev/null | tee {LOG_FILE}"
        if self._workdir:
            # OpenCode's project is its cwd; the sandbox exec starts in / (or $HOME), where OpenCode
            # warns "failed to initialize fff ... file system root or home directory" and the whole
            # file system is the project (job 5314432). Tasks may expect the agent to create the
            # directory, so create it; if that fails, OpenCode still runs in the exec's cwd.
            workdir = shlex.quote(self._workdir)
            command = f"mkdir -p {workdir} && cd {workdir}; {command}"
        if self._install_method == "npm":
            command = f"{NVM_PREFIX}{command}"
        return [ExecInput(command=command, env=self.opencode_env())]

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await run_commands(self, environment, self.create_run_agent_commands(instruction))

    def populate_context_post_run(self, context: AgentContext) -> None:
        # Harbor 0.25's OpenCode converts opencode.txt into an ATIF trajectory.json, which harbor_agent
        # would turn into output items; Harbor 0.1.x's wrote none and this adapter keeps it that way.
        pass
