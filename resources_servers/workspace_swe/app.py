# SPDX-FileCopyrightText: Copyright (c) 2026 swiss-ai contributors.
# SPDX-License-Identifier: Apache-2.0
"""Interactive software-engineering environment (Gymnasium reset/step).

State is a per-session workspace (a path -> text dict) held IN THIS POD. The
pod never executes anything itself: every shell command runs in the separate
code-gym-sandbox service via POST /run_code, with the workspace shipped in
`files` and mutated files fetched back. Execution isolation therefore lives in
the sandbox (one audited surface), not here.

Action protocol (plain text tags, driven by the stock gymnasium_agent):
    <cmd>pytest -q</cmd>                   run a shell command in the sandbox
    <write path="pkg/mod.py">...</write>   edit a file (in-pod, no execution)
    <submit/>                              run hidden tests -> reward, done

Task spec (dataset row extras, forwarded via reset metadata):
    task         str   description shown to the agent
    files        dict  path -> contents seeded into the workspace
    hidden_tests dict  path -> contents injected ONLY at grade time
    test_cmd     str   command whose pass fraction is the reward
    setup_cmd    str   optional one-off run at reset (e.g. git clone, pip install)
    max_steps    int   soft cap; exceeding truncates and grades
"""

import base64
import os
import re
import time
from typing import Optional

import aiohttp

from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.gymnasium import GymnasiumServer, extract_text

SANDBOX_URL = os.environ.get(
    "SANDBOX_URL", "http://sandbox-dev-internal.rob-poc.svc.cluster.local"
)
CMD_RE = re.compile(r"<cmd>(.*?)</cmd>", re.DOTALL)
WRITE_RE = re.compile(r'<write path="([^"]+)">(.*?)</write>', re.DOTALL)
SUBMIT_RE = re.compile(r"<submit\s*/?>")
OUTPUT_LIMIT = 4000
CMD_TIMEOUT_S = 60
SESSION_TTL_S = 1800
PROTOCOL = (
    "Work in the workspace. One action per turn:\n"
    "<cmd>shell command</cmd> to run something (60s timeout),\n"
    '<write path="relative/path">full new file contents</write> to edit a file,\n'
    "<submit/> when solved (hidden tests decide your reward)."
)


def _b64e(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def _b64d(s: str) -> str:
    return base64.b64decode(s).decode("utf-8", "replace")


def _truncate(s: str) -> str:
    return s if len(s) <= OUTPUT_LIMIT else s[:OUTPUT_LIMIT] + f"\n... [truncated, {len(s)} bytes]"


def _pass_fraction(output: str) -> float:
    passed = sum(int(m) for m in re.findall(r"(\d+) passed", output))
    failed = sum(int(m) for m in re.findall(r"(\d+) (?:failed|error)", output))
    total = passed + failed
    return passed / total if total else 0.0


class WorkspaceSWEServer(GymnasiumServer):
    async def _sandbox_run(self, command: str, files: dict, timeout: int = CMD_TIMEOUT_S):
        """Run `command` (bash) in the sandbox with `files` staged; return
        (stdout_text, updated_files). Fetches back all known paths so in-place
        edits made by the command are reflected in session state."""
        payload = {
            "code": command,
            "language": "bash",
            "files": {p: _b64e(c) for p, c in files.items()},
            "fetch_files": list(files.keys()),
            "run_timeout": timeout,
        }
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(f"{SANDBOX_URL}/run_code", json=payload,
                                  timeout=aiohttp.ClientTimeout(total=timeout + 30)) as r:
                    data = await r.json()
        except Exception as e:  # sandbox unreachable/timeout — surface, don't crash the episode
            return f"[sandbox error: {e}]", files
        rr = data.get("run_result") or {}
        out = (rr.get("stdout") or "") + (("\n" + rr["stderr"]) if rr.get("stderr") else "")
        out += f"\n[exit {rr.get('return_code')}]"
        updated = dict(files)
        for p, c in (data.get("files") or {}).items():
            try:
                updated[p] = _b64d(c)
            except Exception:
                pass
        return _truncate(out), updated

    async def reset(self, metadata: dict, session_id: Optional[str] = None):
        self._sweep_expired()
        files = dict(metadata.get("files") or {})
        task = metadata.get("task", "Fix the failing tests in this repository.")
        listing = "\n".join(sorted(files))
        if metadata.get("setup_cmd"):  # e.g. git clone / pip install, run in the sandbox
            out, files = await self._sandbox_run(metadata["setup_cmd"], files, timeout=180)
            listing = out
        self.session_state[session_id] = {
            "files": files, "meta": metadata, "steps": 0, "created": time.time(),
        }
        return f"TASK:\n{task}\n\nWORKSPACE:\n{listing}\n\n{PROTOCOL}", {}

    async def step(self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None):
        state = self.session_state.get(session_id)
        if state is None:
            return "No active workspace for this session — was /reset called?", 0.0, True, False, {"error": "no_session"}
        state["steps"] += 1
        files, meta = state["files"], state["meta"]
        max_steps = int(meta.get("max_steps", 12))
        text = extract_text(action)

        if SUBMIT_RE.search(text):
            reward, info = await self._grade(files, meta)
            self.session_state.pop(session_id, None)
            return None, reward, True, False, info

        m = WRITE_RE.search(text)
        if m:
            rel, content = m.group(1), m.group(2)
            if rel.startswith("/") or ".." in rel.split("/"):
                obs = f"Refused: illegal path {rel}"
            else:
                files[rel] = content
                obs = f"Wrote {len(content)} bytes to {rel}"
        else:
            m = CMD_RE.search(text)
            if m:
                obs, files = await self._sandbox_run(m.group(1).strip(), files)
                state["files"] = files
            else:
                obs = "No action found. " + PROTOCOL

        if state["steps"] >= max_steps:
            reward, info = await self._grade(files, meta)
            info["truncated_at_max_steps"] = True
            self.session_state.pop(session_id, None)
            return None, reward, False, True, info
        return obs, 0.0, False, False, {}

    async def _grade(self, files: dict, meta: dict):
        graded = dict(files)
        graded.update(meta.get("hidden_tests") or {})  # injected only now — model never saw them
        test_cmd = meta.get("test_cmd", "python -m pytest hidden_tests -q")
        out, _ = await self._sandbox_run(test_cmd, graded, timeout=120)
        reward = _pass_fraction(out)
        return reward, {"test_output": _truncate(out), "reward": reward}

    def _sweep_expired(self):
        now = time.time()
        for sid in [s for s, st in self.session_state.items() if now - st["created"] > SESSION_TTL_S]:
            self.session_state.pop(sid, None)


if __name__ == "__main__":
    WorkspaceSWEServer.run_webserver()
