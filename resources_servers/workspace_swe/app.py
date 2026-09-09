# SPDX-FileCopyrightText: Copyright (c) 2026 swiss-ai contributors.
# SPDX-License-Identifier: Apache-2.0
"""Interactive software-engineering environment (Gymnasium reset/step).

Vanilla NeMo Gym: execution happens in-process (like NVIDIA's cvdp/swerl_gen),
in a real per-session workspace directory on disk. Clone a repo once at reset,
then run shell commands and edit files across turns; the reward is the hidden
test suite's pass fraction at <submit/>.

Isolation is the pod boundary (crun today; set runtimeClassName to a sandboxed
runtime — e.g. gvisor — when CSCS provides one). Pair with a default-deny
egress NetworkPolicy unless the task needs to clone/pull over the network.

Action protocol (plain text tags, driven by the stock gymnasium_agent):
    <cmd>pytest -q</cmd>                   run a shell command in the workspace
    <write path="pkg/mod.py">...</write>   replace a file's contents
    <submit/>                              run hidden tests -> reward, done

Task spec (dataset row extras, forwarded via reset metadata):
    task         str   description shown to the agent
    files        dict  path -> contents materialized into the workspace
    hidden_tests dict  path -> contents injected ONLY at grade time
    test_cmd     str   command whose pass fraction is the reward
    setup_cmd    str   optional one-off run at reset (e.g. git clone / pip install)
    repo_url     str   optional shallow-clone instead of `files`
    commit       str   optional checkout after clone
    max_steps    int   soft cap; exceeding truncates and grades
"""

import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.gymnasium import GymnasiumServer, extract_text

CMD_RE = re.compile(r"<cmd>(.*?)</cmd>", re.DOTALL)
WRITE_RE = re.compile(r'<write path="([^"]+)">(.*?)</write>', re.DOTALL)
SUBMIT_RE = re.compile(r"<submit\s*/?>")
CMD_TIMEOUT_S = 60
OUTPUT_LIMIT = 4000
SESSION_TTL_S = 1800
WORKSPACES = Path("/workspaces")
PROTOCOL = (
    "Work inside the workspace. One action per turn:\n"
    "<cmd>shell command</cmd> to run something (60s timeout),\n"
    '<write path="relative/path">full new file contents</write> to edit a file,\n'
    "<submit/> when solved (hidden tests decide your reward)."
)


def _truncate(s: str) -> str:
    return s if len(s) <= OUTPUT_LIMIT else s[:OUTPUT_LIMIT] + f"\n... [truncated, {len(s)} bytes]"


def _run(cmd: str, cwd: Path, timeout: int = CMD_TIMEOUT_S) -> str:
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
        return _truncate(f"$ {cmd}\n{out}\n[exit {p.returncode}]")
    except subprocess.TimeoutExpired:
        return f"$ {cmd}\n[timed out after {timeout}s]"


def _pass_fraction(output: str) -> float:
    passed = sum(int(m) for m in re.findall(r"(\d+) passed", output))
    failed = sum(int(m) for m in re.findall(r"(\d+) (?:failed|error)", output))
    total = passed + failed
    return passed / total if total else 0.0


class WorkspaceSWEServer(GymnasiumServer):
    async def reset(self, metadata: dict, session_id: Optional[str] = None):
        self._sweep_expired()
        WORKSPACES.mkdir(exist_ok=True)
        ws = Path(tempfile.mkdtemp(prefix="ws_", dir=str(WORKSPACES)))
        root = ws / "repo"
        task = metadata.get("task", "Fix the failing tests in this repository.")

        if metadata.get("repo_url"):
            clone = _run(
                f"git clone --depth 50 {metadata['repo_url']} repo"
                + (f" && cd repo && git checkout {metadata['commit']}" if metadata.get("commit") else ""),
                ws, timeout=180,
            )
            if not root.exists():
                shutil.rmtree(ws, ignore_errors=True)
                return f"Workspace setup failed:\n{clone}", {}
        else:
            root.mkdir(parents=True)
            for rel, content in (metadata.get("files") or {}).items():
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)

        setup_out = ""
        if metadata.get("setup_cmd"):
            setup_out = "\n" + _run(metadata["setup_cmd"], root, timeout=180)

        self.session_state[session_id] = {
            "root": root, "ws": ws, "meta": metadata, "steps": 0, "created": time.time(),
        }
        listing = _run("find . -type f -not -path './.git/*' | head -60", root)
        return f"TASK:\n{task}\n\nWORKSPACE FILES:\n{listing}{setup_out}\n\n{PROTOCOL}", {}

    async def step(self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None):
        state = self.session_state.get(session_id)
        if state is None:
            return "No active workspace for this session — was /reset called?", 0.0, True, False, {"error": "no_session"}
        root, meta = state["root"], state["meta"]
        state["steps"] += 1
        max_steps = int(meta.get("max_steps", 12))
        text = extract_text(action)

        if SUBMIT_RE.search(text):
            reward, info = self._grade(root, meta)
            self._cleanup(session_id)
            return None, reward, True, False, info

        m = WRITE_RE.search(text)
        if m:
            rel, content = m.group(1), m.group(2)
            p = (root / rel).resolve()
            if not str(p).startswith(str(root.resolve())):
                obs = f"Refused: path escapes the workspace: {rel}"
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
                obs = f"Wrote {len(content)} bytes to {rel}"
        else:
            m = CMD_RE.search(text)
            obs = _run(m.group(1).strip(), root) if m else "No action found. " + PROTOCOL

        if state["steps"] >= max_steps:
            reward, info = self._grade(root, meta)
            info["truncated_at_max_steps"] = True
            self._cleanup(session_id)
            return None, reward, False, True, info
        return obs, 0.0, False, False, {}

    def _grade(self, root: Path, meta: dict):
        for rel, content in (meta.get("hidden_tests") or {}).items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        out = _run(meta.get("test_cmd", "python -m pytest hidden_tests -q"), root, timeout=120)
        reward = _pass_fraction(out)
        return reward, {"test_output": _truncate(out), "reward": reward}

    def _cleanup(self, session_id: Optional[str]) -> None:
        st = self.session_state.pop(session_id, None)
        if st:
            shutil.rmtree(st["ws"], ignore_errors=True)

    def _sweep_expired(self) -> None:
        now = time.time()
        for sid in [s for s, st in self.session_state.items() if now - st["created"] > SESSION_TTL_S]:
            self._cleanup(sid)


if __name__ == "__main__":
    WORKSPACES.mkdir(exist_ok=True)
    WorkspaceSWEServer.run_webserver()
