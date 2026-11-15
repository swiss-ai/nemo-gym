# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reproduce ``unresolved_parent`` after a content-less assistant response.

Observed in a Terminus-2 GRPO run: the policy often answers with an empty string,
the harness echoes ``{"role": "assistant", "content": ""}`` and asks again
("Previous response had parsing errors ..."). The next call is then poisoned.

Mechanism (before the fix): ``assistant_fingerprint`` hashed only the *content* of assistant turns
and ``_content_of`` drops empty text, so a content-less, tool-less assistant turn
adds nothing to the hash. A call whose response is empty therefore gets the same
continuation fingerprint as its parent. The next request matches two ledger rows
(parent and empty child), both of whose recorded contexts are a prefix of the
request, so the resolver refuses to choose:

* ``RolloutLineage.resolve_node`` (in-memory index): two candidates with different
  cumulative digests -> UNRESOLVED ``ambiguous``.
* ``FileLineageStore._resolve_row`` (ledger rows, the production path): two rows
  share the fingerprint -> ``len(records) != 1`` -> None, and the reported reason
  is the incremental index's ``no_match``.

An empty response on the *first* call does not trigger it: its fingerprint is that
of "one content-less assistant turn", which no other row shares. That is why
rollouts whose only empty response was call 0 survived.

The fix hashes an explicit marker for a model-authored turn with no content and no tool call.
The reproductions now pass; the controls guard the cases that already resolved.
"""

from __future__ import annotations

import pytest

from nemo_gym.token_id_capture.fingerprint import FINGERPRINT_VERSION, assistant_fingerprint
from nemo_gym.token_id_capture.lineage import FileLineageStore, InMemoryLineageStore
from nemo_gym.token_id_capture.records import ParentResolutionStatus
from nemo_gym.token_id_capture.sink import (
    UNRESOLVED_PARENT_REASON,
    CaptureContext,
    reset_token_sink,
    resolve_parent,
    set_token_sink,
)
from nemo_gym.token_id_capture.staging.digest import (
    EMPTY_EXTRAS_DIGEST,
    compute_chain_hash,
    hash_token_ids,
)
from nemo_gym.token_id_capture.staging.records import CallRecord, CaptureLedgerCommit, RolloutManifest


# Shapes taken from Terminus-2 trial micro-json-062 (job 5256202).
SYSTEM_TASK = {
    "role": "user",
    "content": "You are an AI assistant tasked with solving command-line tasks ... Format your response as JSON.",
}
ASSISTANT_JSON = {
    "role": "assistant",
    "content": '{"analysis": "look around", "plan": "list files", "commands": [{"keystrokes": "ls\\n"}]}',
}
TERMINAL_OUTPUT = {"role": "user", "content": "New Terminal Output:\nroot@box:/# ls\napp  server\n"}
PARSE_ERROR = {
    "role": "user",
    "content": (
        "Previous response had parsing errors:\nERROR: No valid JSON found in response\n"
        "WARNINGS: - No valid JSON object found\n\nPlease fix these issues and provide a proper JSON response."
    ),
}
ASSISTANT_EMPTY = {"role": "assistant", "content": ""}
ASSISTANT_NONE = {"role": "assistant", "content": None}
ASSISTANT_WHITESPACE = {"role": "assistant", "content": "\n\n"}
ASSISTANT_TOOL_ONLY = {
    "role": "assistant",
    "content": None,
    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}],
}
TOOL_RESULT = {"role": "tool", "tool_call_id": "call_1", "content": "app  server"}


def _served(message: dict) -> dict:
    """The response item the model server records (``response_to_output_items`` on a chat payload)."""
    return {"type": "message", **message}


class _Ledger:
    """Commit calls into a ledger exactly as the external-staging commit hook does."""

    def __init__(self, store, rollout_id: str = "r1") -> None:
        self.store = store
        self.rollout_id = rollout_id
        self.tokens: list[int] = []
        self.chain_hash: str | None = None
        self.staging_chain: list[str] = []
        self.calls: list[str] = []

    async def commit(self, request_items: list[dict], response: dict) -> str:
        call_id = f"c{len(self.calls) + 1}"
        prev_len = len(self.tokens)
        # Every call generates at least one token (EOS even for an empty answer),
        # so cumulative digests of different calls always differ.
        delta = list(range(prev_len + 1000, prev_len + 1010))
        self.tokens += delta
        self.chain_hash = compute_chain_hash(self.chain_hash, delta)
        parent = self.calls[-1] if self.calls else None
        staging_key = f"{self.rollout_id}/{call_id}"
        self.staging_chain = self.staging_chain + [staging_key]
        record = CallRecord(
            model_call_id=call_id,
            parent_call_id=parent,
            staging_key=staging_key,
            weight_version=1,
            prev_len=prev_len,
            delta_len=len(delta),
            cum_len=len(self.tokens),
            digest="a" * 64,
            extras_digest=EMPTY_EXTRAS_DIGEST,
            mode="text" if parent is None else "token_in",
            response_id=f"chatcmpl-{call_id}",
            admitted_at=1.0,
            chain_hash=self.chain_hash,
            cumulative_hash=hash_token_ids(self.tokens),
            fingerprint_version=FINGERPRINT_VERSION,
        )
        await self.store.record(
            CaptureLedgerCommit(
                rollout_id=self.rollout_id,
                record=record,
                staging_chain=tuple(self.staging_chain),
                request_items=request_items,
                response_items=[_served(response)],
            )
        )
        self.calls.append(call_id)
        return call_id


async def _replay(store, turns: list[tuple[dict, dict]]) -> tuple[_Ledger, list[dict]]:
    """Commit ``(user, assistant)`` turns in order; return the ledger and the full history."""
    ledger = _Ledger(store)
    history: list[dict] = []
    for user, assistant in turns:
        history.append(user)
        await ledger.commit(list(history), assistant)
        history.append(assistant)
    return ledger, history


async def _admit(store, request_items: list[dict], model_call_id: str) -> CaptureContext:
    context = CaptureContext(
        rollout_id="r1",
        model_call_id=model_call_id,
        token_sink=None,
        lineage_store=store,
        external_staging=True,
    )
    token = set_token_sink(context)
    try:
        await resolve_parent(request_items)
    finally:
        reset_token_sink(token)
    return context


@pytest.fixture(params=["file", "memory"])
def store(request, tmp_path):
    if request.param == "file":
        return FileLineageStore(tmp_path)
    return InMemoryLineageStore()


# --- fingerprint level -------------------------------------------------------


def test_empty_content_spellings_fingerprint_identically():
    """Served ``""`` and echoed ``None``/missing/``[]`` must keep matching (passes today, must survive a fix)."""
    spellings = [
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": None},
        {"role": "assistant"},
        {"role": "assistant", "content": []},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": ""}]},
    ]
    prints = {assistant_fingerprint([SYSTEM_TASK, ASSISTANT_JSON, PARSE_ERROR, m]) for m in spellings}
    assert len(prints) == 1


def test_empty_assistant_turn_changes_fingerprint():
    before = assistant_fingerprint([SYSTEM_TASK, ASSISTANT_JSON])
    after = assistant_fingerprint([SYSTEM_TASK, ASSISTANT_JSON, PARSE_ERROR, ASSISTANT_EMPTY])
    assert before != after


def test_repeated_empty_assistant_turns_change_fingerprint():
    one = assistant_fingerprint([SYSTEM_TASK, ASSISTANT_EMPTY])
    two = assistant_fingerprint([SYSTEM_TASK, ASSISTANT_EMPTY, PARSE_ERROR, ASSISTANT_EMPTY])
    assert one != two


# --- resolver level: failing reproductions -----------------------------------


async def test_empty_response_after_nonempty_call_resolves(store):
    """c1 answers JSON, c2 answers "", c3 must continue c2 (56/62 Run B rejections)."""
    ledger, history = await _replay(store, [(SYSTEM_TASK, ASSISTANT_JSON), (TERMINAL_OUTPUT, ASSISTANT_EMPTY)])
    resolution = await store.resolve("r1", history + [PARSE_ERROR])
    assert resolution.status == ParentResolutionStatus.RESOLVED, resolution.reason
    assert resolution.match.model_call_id == ledger.calls[-1]


async def test_consecutive_empty_responses_from_first_call_resolve(store):
    """c1 and c2 both answer "": c3 must continue c2 (5/62 Run B rejections)."""
    ledger, history = await _replay(store, [(SYSTEM_TASK, ASSISTANT_EMPTY), (PARSE_ERROR, ASSISTANT_EMPTY)])
    resolution = await store.resolve("r1", history + [PARSE_ERROR])
    assert resolution.status == ParentResolutionStatus.RESOLVED, resolution.reason
    assert resolution.match.model_call_id == ledger.calls[-1]


async def test_empty_response_echoed_as_none_resolves(store):
    """Served ``""``, echoed by the harness as ``content: None`` (litellm may do this)."""
    ledger, history = await _replay(store, [(SYSTEM_TASK, ASSISTANT_JSON), (TERMINAL_OUTPUT, ASSISTANT_EMPTY)])
    history[-1] = ASSISTANT_NONE
    resolution = await store.resolve("r1", history + [PARSE_ERROR])
    assert resolution.status == ParentResolutionStatus.RESOLVED, resolution.reason
    assert resolution.match.model_call_id == ledger.calls[-1]


# --- resolver level: controls that pass today --------------------------------


async def test_empty_response_on_first_call_resolves(store):
    """Survivor pattern: the only empty answer is call 0 (18/18 Run B survivors with an empty row)."""
    ledger, history = await _replay(
        store,
        [(SYSTEM_TASK, ASSISTANT_EMPTY), (PARSE_ERROR, ASSISTANT_JSON), (TERMINAL_OUTPUT, ASSISTANT_JSON)],
    )
    resolution = await store.resolve("r1", history + [TERMINAL_OUTPUT])
    assert resolution.status == ParentResolutionStatus.RESOLVED, resolution.reason
    assert resolution.match.model_call_id == ledger.calls[-1]


async def test_empty_response_on_first_call_then_next_call_resolves(store):
    """The call right after a first-call empty answer has a unique parent."""
    ledger, history = await _replay(store, [(SYSTEM_TASK, ASSISTANT_EMPTY)])
    resolution = await store.resolve("r1", history + [PARSE_ERROR])
    assert resolution.status == ParentResolutionStatus.RESOLVED, resolution.reason
    assert resolution.match.model_call_id == "c1"


async def test_whitespace_only_response_resolves(store):
    """``"\\n\\n"`` is non-empty content and is hashed, so it does not collapse."""
    ledger, history = await _replay(store, [(SYSTEM_TASK, ASSISTANT_JSON), (TERMINAL_OUTPUT, ASSISTANT_WHITESPACE)])
    resolution = await store.resolve("r1", history + [PARSE_ERROR])
    assert resolution.status == ParentResolutionStatus.RESOLVED, resolution.reason
    assert resolution.match.model_call_id == ledger.calls[-1]


async def test_tool_call_only_response_resolves(store):
    """``content: None`` with tool calls is fingerprinted by its tool calls."""
    ledger, history = await _replay(store, [(SYSTEM_TASK, ASSISTANT_JSON), (TERMINAL_OUTPUT, ASSISTANT_TOOL_ONLY)])
    resolution = await store.resolve("r1", history + [TOOL_RESULT])
    assert resolution.status == ParentResolutionStatus.RESOLVED, resolution.reason
    assert resolution.match.model_call_id == ledger.calls[-1]


async def test_identical_nonempty_responses_resolve(store):
    """Repeating the same non-empty answer still extends the fingerprint."""
    ledger, history = await _replay(store, [(SYSTEM_TASK, ASSISTANT_JSON), (TERMINAL_OUTPUT, ASSISTANT_JSON)])
    resolution = await store.resolve("r1", history + [TERMINAL_OUTPUT])
    assert resolution.status == ParentResolutionStatus.RESOLVED, resolution.reason
    assert resolution.match.model_call_id == ledger.calls[-1]


# --- sink level: the rollout is poisoned -------------------------------------


async def test_sink_admits_call_after_empty_response(store):
    """Run B: ``Unresolved parent for model call ... poisoning the call``."""
    ledger, history = await _replay(
        store,
        [(SYSTEM_TASK, ASSISTANT_JSON), (TERMINAL_OUTPUT, ASSISTANT_JSON), (TERMINAL_OUTPUT, ASSISTANT_EMPTY)],
    )
    context = await _admit(store, history + [PARSE_ERROR], model_call_id="c4")
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    assert [failure.reason for failure in manifest.failures] == []
    assert context.capture_admission is not None
    assert context.capture_admission.mode == "token_in"
    assert context.capture_admission.parent_call_id == ledger.calls[-1]


async def test_sink_admits_call_after_first_call_empty(store):
    """Control at the sink level: a first-call empty answer does not poison."""
    ledger, history = await _replay(store, [(SYSTEM_TASK, ASSISTANT_EMPTY)])
    context = await _admit(store, history + [PARSE_ERROR], model_call_id="c2")
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    assert manifest.failures == []
    assert context.capture_admission is not None and context.capture_admission.parent_call_id == "c1"
    assert UNRESOLVED_PARENT_REASON not in [failure.reason for failure in manifest.failures]
