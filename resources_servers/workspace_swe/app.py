# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 swiss-ai contributors.
# SPDX-License-Identifier: Apache-2.0
"""Interactive coding workspace. Run only inside an isolated execution service.

Commands intentionally execute arbitrary agent code. Workspace directories and
path validation are conveniences, not a security boundary between sessions.
"""

import asyncio
import os
import re
import shlex
import shutil
import signal
import tempfile
import time
from pathlib import Path
from typing import Optional

from pydantic import Field, PrivateAttr

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.gymnasium import GymnasiumServer, extract_text


CMD_RE = re.compile(r"<cmd>(.*?)</cmd>", re.DOTALL)
WRITE_RE = re.compile(r'<write path="([^"]+)">(.*?)</write>', re.DOTALL)
SUBMIT_RE = re.compile(r"<submit\s*/?>")
OUTPUT_LIMIT = 4000
PROTOCOL = (
    "One action per turn: <cmd>shell command</cmd> (60s timeout), "
    '<write path="relative/path">full file contents</write>, or <submit/> to run hidden tests.'
)


class WorkspaceSWEConfig(BaseResourcesServerConfig):
    workspace_dir: str = "/workspaces"
    max_concurrent_commands: int = Field(default=8, ge=1)
    session_ttl_s: int = Field(default=1800, ge=1)


def _truncate(output: str) -> str:
    return output if len(output) <= OUTPUT_LIMIT else output[:OUTPUT_LIMIT] + "\n... [truncated]"


def _write_file(root: Path, relative: str, content: str) -> None:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes the workspace: {relative}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _pass_fraction(output: str) -> float:
    # Use the last summary; earlier captured output can also contain strings
    # such as '1 passed'. Parse before truncating the observation.
    for line in reversed(output.splitlines()):
        passed = re.search(r"\b(\d+) passed\b", line)
        failed = re.findall(r"\b(\d+) (?:failed|errors?)\b", line)
        if passed or failed:
            count = int(passed[1]) if passed else 0
            total = count + sum(map(int, failed))
            return count / total if total else 0.0
    return 0.0


class WorkspaceSWEServer(GymnasiumServer):
    config: WorkspaceSWEConfig
    _commands: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        self._commands = asyncio.Semaphore(self.config.max_concurrent_commands)

    async def _run(self, command: str, cwd: Path, timeout: int = 60) -> tuple[str, int]:
        async with self._commands:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                output, _ = await asyncio.wait_for(process.communicate(), timeout)
            except (TimeoutError, asyncio.CancelledError) as exc:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.communicate()
                if isinstance(exc, asyncio.CancelledError):
                    raise
                return f"[timed out after {timeout}s]", -1
            return output.decode(errors="replace"), process.returncode

    async def reset(self, metadata: dict, session_id: Optional[str] = None):
        now = time.monotonic()
        for sid in list(self.session_state):
            if now - self.session_state[sid]["created"] > self.config.session_ttl_s:
                await self.close_session(sid)
        await self.close_session(session_id)
        workspaces = Path(self.config.workspace_dir)
        workspaces.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix="ws_", dir=workspaces))
        root = workspace / "repo"
        self.session_state[session_id] = {
            "root": root,
            "workspace": workspace,
            "metadata": metadata,
            "steps": 0,
            "created": now,
        }
        try:
            if metadata.get("repo_url"):
                command = f"git clone --depth 50 -- {shlex.quote(metadata['repo_url'])} repo"
                output, code = await self._run(command, workspace, timeout=180)
                if code:
                    raise ValueError(f"Clone failed: {_truncate(output)}")
                if metadata.get("commit"):
                    output, code = await self._run(
                        f"git checkout --detach {shlex.quote(metadata['commit'])}", root, timeout=180
                    )
                    if code:
                        raise ValueError(f"Checkout failed: {_truncate(output)}")
            else:
                root.mkdir()
                for relative, content in (metadata.get("files") or {}).items():
                    _write_file(root, relative, content)
            if metadata.get("setup_cmd"):
                output, code = await self._run(metadata["setup_cmd"], root, timeout=180)
                if code:
                    raise ValueError(f"Setup failed: {_truncate(output)}")
            listing, _ = await self._run("find . -type f -not -path './.git/*' | head -60", root)
        except (OSError, ValueError) as exc:
            await self.close_session(session_id)
            return f"Workspace setup failed: {exc}", {"error": "setup_failed"}
        task = metadata.get("task", "Fix the failing tests in this repository.")
        return f"TASK:\n{task}\n\nWORKSPACE FILES:\n{_truncate(listing)}\n\n{PROTOCOL}", {}

    async def step(self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None):
        state = self.session_state.get(session_id)
        if state is None:
            return "No active workspace; call /reset first.", 0.0, True, False, {"error": "no_session"}
        root, meta = state["root"], state["metadata"]
        state["steps"] += 1
        text = extract_text(action)
        submitted = bool(SUBMIT_RE.search(text))
        if not submitted:
            write = WRITE_RE.search(text)
            command = CMD_RE.search(text)
            if write:
                try:
                    _write_file(root, write[1], write[2])
                    observation = f"Wrote {len(write[2])} characters to {write[1]}"
                except (OSError, ValueError) as exc:
                    observation = f"Write failed: {exc}"
            elif command:
                output, code = await self._run(command[1].strip(), root)
                observation = f"{_truncate(output)}\n[exit {code}]"
            else:
                observation = "No action found. " + PROTOCOL
            if state["steps"] < int(meta.get("max_steps", 12)):
                return observation, 0.0, False, False, {}
        try:
            for relative, content in (meta.get("hidden_tests") or {}).items():
                _write_file(root, relative, content)
            output, code = await self._run(meta.get("test_cmd", "python -m pytest hidden_tests -q"), root, timeout=120)
            reward = _pass_fraction(output) if code in (0, 1) else 0.0
            info = {"test_output": _truncate(output), "reward": reward}
            if not submitted:
                info["truncated_at_max_steps"] = True
            return None, reward, submitted, not submitted, info
        except (OSError, ValueError) as exc:
            return None, 0.0, True, False, {"error": "grading_failed", "test_output": str(exc)}
        finally:
            await self.close_session(session_id)

    async def close_session(self, session_id: Optional[str]) -> None:
        state = self.session_state.pop(session_id, None)
        if state:
            await asyncio.to_thread(shutil.rmtree, state["workspace"], ignore_errors=True)


if __name__ == "__main__":
    WorkspaceSWEServer.run_webserver()
