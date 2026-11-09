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

"""Resolve the parent of a call whose reply carried reasoning that the harness does not echo.

Observed with Terminus-2 (``interleaved_thinking: false``) on a model that thinks on its first turn (job 5302914):
173/192 rollouts were rejected with ``unresolved_parent``.

With ``uses_reasoning_parser: true`` the vLLM model server folds the parsed reasoning into the reply content
as ``<think>REASONING</think>`` + content before it records the call.
LiteLLM splits the block back out, and the harness echoes only the content.
Before the fix, the recorded fingerprint covered the think block and the echo matched no ledger row.

The fix strips the leading think blocks from model-authored text on both sides,
so a call resolves whether the harness echoes its reasoning or not.
"""

from __future__ import annotations

import pytest

from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint, conversation_digest
from nemo_gym.token_id_capture.lineage import FileLineageStore, InMemoryLineageStore
from nemo_gym.token_id_capture.records import ParentResolutionStatus
from nemo_gym.token_id_capture.staging.records import RolloutManifest
from tests.unit_tests.test_token_capture_empty_assistant import (
    ASSISTANT_EMPTY,
    ASSISTANT_JSON,
    PARSE_ERROR,
    SYSTEM_TASK,
    TERMINAL_OUTPUT,
    TOOL_RESULT,
    _admit,
    _Ledger,
)


REASONING = "The user wants the files listed.\nI should run ls first."
TOOL_CALLS = [{"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}]


def _think(reasoning: str, content: str | None) -> dict:
    """The reply as the model server records it (``_wrap_reasoning_in_think_tags`` + content)."""
    return {"role": "assistant", "content": f"<think>{reasoning}</think>" + (content or "")}


ASSISTANT_JSON_THINKING = _think(REASONING, ASSISTANT_JSON["content"])
ASSISTANT_THINK_ONLY = _think(REASONING, "")
ASSISTANT_TOOL_THINKING = {**_think(REASONING, None), "tool_calls": TOOL_CALLS}
ASSISTANT_TOOL_ONLY = {"role": "assistant", "content": None, "tool_calls": TOOL_CALLS}


async def _replay(store, turns: list[tuple[dict, dict, dict]]) -> tuple[_Ledger, list[dict]]:
    """Commit ``(user, served, echoed)`` turns; the history carries what the harness echoes."""
    ledger = _Ledger(store)
    history: list[dict] = []
    for user, served, echoed in turns:
        history.append(user)
        await ledger.commit(list(history), served)
        history.append(echoed)
    return ledger, history


@pytest.fixture(params=["file", "memory"])
def store(request, tmp_path):
    if request.param == "file":
        return FileLineageStore(tmp_path)
    return InMemoryLineageStore()


# --- fingerprint level -------------------------------------------------------


def test_think_wrapped_reply_and_its_content_hash_identically():
    served = assistant_fingerprint([SYSTEM_TASK, ASSISTANT_JSON_THINKING])
    assert served == assistant_fingerprint([SYSTEM_TASK, ASSISTANT_JSON])
    assert served != ""


def test_think_block_is_stripped_in_every_dialect():
    text = ASSISTANT_JSON["content"]
    wrapped = f"<think>{REASONING}</think>{text}"
    shapes = [
        {"role": "assistant", "content": wrapped},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": wrapped}]},
        {"role": "assistant", "content": [{"type": "text", "text": wrapped}]},
        # Responses harness: the reasoning comes back as a standalone item next to the visible text.
        [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": REASONING}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]},
        ],
    ]
    fingerprints = {assistant_fingerprint(shape if isinstance(shape, list) else [shape]) for shape in shapes}
    assert fingerprints == {assistant_fingerprint([ASSISTANT_JSON])}


def test_reasoning_only_reply_hashes_as_an_empty_turn():
    spellings = [ASSISTANT_THINK_ONLY, ASSISTANT_EMPTY, {"role": "assistant", "content": None}]
    prints = {assistant_fingerprint([SYSTEM_TASK, ASSISTANT_JSON, PARSE_ERROR, m]) for m in spellings}
    assert len(prints) == 1
    assert prints != {assistant_fingerprint([SYSTEM_TASK, ASSISTANT_JSON])}


def test_tool_call_with_reasoning_hashes_as_the_bare_tool_call():
    assert assistant_fingerprint([ASSISTANT_TOOL_THINKING]) == assistant_fingerprint([ASSISTANT_TOOL_ONLY])
    # Not an empty turn: the tool call is hashed.
    assert assistant_fingerprint([ASSISTANT_TOOL_THINKING]) != assistant_fingerprint([ASSISTANT_EMPTY])


def test_visible_text_that_opens_with_a_think_block_matches_a_litellm_echo():
    # LiteLLM splits off one leading block and keeps the rest verbatim.
    served = {"role": "assistant", "content": "<think>a</think><think>b</think>answer"}
    echoed = {"role": "assistant", "content": "<think>b</think>answer"}
    assert assistant_fingerprint([served]) == assistant_fingerprint([echoed])
    assert assistant_fingerprint([served]) == assistant_fingerprint([{"role": "assistant", "content": "answer"}])


def test_think_blocks_after_visible_text_stay_hashed():
    plain = assistant_fingerprint([{"role": "assistant", "content": "answer"}])
    assert assistant_fingerprint([{"role": "assistant", "content": "answer<think>x</think>"}]) != plain
    assert assistant_fingerprint([{"role": "assistant", "content": " <think>x</think>answer"}]) != plain
    # An unclosed block is not reasoning folded in by the server.
    assert assistant_fingerprint([{"role": "assistant", "content": "<think>answer"}]) != plain


def test_user_content_with_think_tags_is_not_altered():
    question = {"role": "user", "content": "<think>keep this</think>What is in the directory?"}
    stripped = {"role": "user", "content": "What is in the directory?"}
    assert conversation_digest([question]) != conversation_digest([stripped])
    assert conversation_digest([question, ASSISTANT_JSON]) != conversation_digest([stripped, ASSISTANT_JSON])
    tool_result = {"role": "tool", "tool_call_id": "call_1", "content": "<think>x</think>app"}
    assert conversation_digest([tool_result]) != conversation_digest([{**tool_result, "content": "app"}])


# --- resolver level ----------------------------------------------------------


async def test_reasoning_reply_echoed_without_reasoning_resolves(store):
    """Job 5302914: turn 0 thinks, Terminus-2 echoes only the content."""
    ledger, history = await _replay(store, [(SYSTEM_TASK, ASSISTANT_JSON_THINKING, ASSISTANT_JSON)])
    resolution = await store.resolve("r1", history + [TERMINAL_OUTPUT])
    assert resolution.status == ParentResolutionStatus.RESOLVED, resolution.reason
    assert resolution.match.model_call_id == ledger.calls[-1]


async def test_reasoning_reply_echoed_with_think_block_resolves(store):
    """Terminus-2 with ``interleaved_thinking: true`` echoes the reply verbatim."""
    ledger, history = await _replay(store, [(SYSTEM_TASK, ASSISTANT_JSON_THINKING, ASSISTANT_JSON_THINKING)])
    resolution = await store.resolve("r1", history + [TERMINAL_OUTPUT])
    assert resolution.status == ParentResolutionStatus.RESOLVED, resolution.reason
    assert resolution.match.model_call_id == ledger.calls[-1]


@pytest.mark.parametrize("echo_reasoning", [False, True])
async def test_multi_turn_mix_resolves_every_call(store, echo_reasoning):
    """Reasoning on some turns only; every request must continue the call right before it."""
    other_json = {"role": "assistant", "content": '{"analysis": "done", "plan": "", "commands": []}'}
    turns = [
        (SYSTEM_TASK, ASSISTANT_JSON_THINKING, ASSISTANT_JSON),
        (TERMINAL_OUTPUT, ASSISTANT_JSON, ASSISTANT_JSON),
        (TERMINAL_OUTPUT, _think("check again", other_json["content"]), other_json),
        (TERMINAL_OUTPUT, ASSISTANT_THINK_ONLY, ASSISTANT_EMPTY),
        (PARSE_ERROR, other_json, other_json),
    ]
    if echo_reasoning:
        turns = [(user, served, served) for user, served, _ in turns]
    ledger = _Ledger(store)
    history: list[dict] = []
    for user, served, echoed in turns:
        history.append(user)
        if ledger.calls:
            resolution = await store.resolve("r1", list(history))
            assert resolution.status == ParentResolutionStatus.RESOLVED, (len(ledger.calls), resolution.reason)
            assert resolution.match.model_call_id == ledger.calls[-1]
        await ledger.commit(list(history), served)
        history.append(echoed)


@pytest.mark.parametrize("echo", [ASSISTANT_EMPTY, {"role": "assistant", "content": None}, ASSISTANT_THINK_ONLY])
async def test_reasoning_only_reply_resolves_to_itself_not_its_parent(store, echo):
    """A reply with reasoning and no content, echoed as ``""``, must not collapse onto its parent."""
    ledger, history = await _replay(
        store, [(SYSTEM_TASK, ASSISTANT_JSON, ASSISTANT_JSON), (TERMINAL_OUTPUT, ASSISTANT_THINK_ONLY, echo)]
    )
    resolution = await store.resolve("r1", history + [PARSE_ERROR])
    assert resolution.status == ParentResolutionStatus.RESOLVED, resolution.reason
    assert resolution.match.model_call_id == "c2"


async def test_reasoning_only_first_reply_resolves(store):
    ledger, history = await _replay(store, [(SYSTEM_TASK, ASSISTANT_THINK_ONLY, ASSISTANT_EMPTY)])
    resolution = await store.resolve("r1", history + [PARSE_ERROR])
    assert resolution.status == ParentResolutionStatus.RESOLVED, resolution.reason
    assert resolution.match.model_call_id == "c1"


async def test_tool_call_with_reasoning_and_empty_content_resolves(store):
    ledger, history = await _replay(
        store,
        [
            (SYSTEM_TASK, ASSISTANT_JSON_THINKING, ASSISTANT_JSON),
            (TERMINAL_OUTPUT, ASSISTANT_TOOL_THINKING, ASSISTANT_TOOL_ONLY),
        ],
    )
    resolution = await store.resolve("r1", history + [TOOL_RESULT])
    assert resolution.status == ParentResolutionStatus.RESOLVED, resolution.reason
    assert resolution.match.model_call_id == "c2"


async def test_responses_harness_echoing_a_reasoning_item_resolves(store):
    """A Responses harness gets the reasoning as a standalone item and echoes it beside the text."""
    echoed = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": REASONING}]},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": ASSISTANT_JSON["content"]}],
        },
    ]
    ledger = _Ledger(store)
    await ledger.commit([SYSTEM_TASK], ASSISTANT_JSON_THINKING)
    resolution = await store.resolve("r1", [SYSTEM_TASK, *echoed, TERMINAL_OUTPUT])
    assert resolution.status == ParentResolutionStatus.RESOLVED, resolution.reason
    assert resolution.match.model_call_id == "c1"


async def test_empty_reply_after_reasoning_turn_still_resolves(store):
    """The previous fix's case (an empty reply after a non-empty call) keeps resolving."""
    ledger, history = await _replay(
        store,
        [(SYSTEM_TASK, ASSISTANT_JSON_THINKING, ASSISTANT_JSON), (TERMINAL_OUTPUT, ASSISTANT_EMPTY, ASSISTANT_EMPTY)],
    )
    resolution = await store.resolve("r1", history + [PARSE_ERROR])
    assert resolution.status == ParentResolutionStatus.RESOLVED, resolution.reason
    assert resolution.match.model_call_id == "c2"


async def test_user_content_with_think_tags_resolves(store):
    question = {"role": "user", "content": "<think>not reasoning</think>List the files."}
    ledger, history = await _replay(store, [(question, ASSISTANT_JSON_THINKING, ASSISTANT_JSON)])
    resolution = await store.resolve("r1", history + [TERMINAL_OUTPUT])
    assert resolution.status == ParentResolutionStatus.RESOLVED, resolution.reason
    assert resolution.match.model_call_id == "c1"
    # The context digest still sees the user's tags: dropping them is a different context.
    rewritten = [{"role": "user", "content": "List the files."}, ASSISTANT_JSON, TERMINAL_OUTPUT]
    resolution = await store.resolve("r1", rewritten)
    assert resolution.status == ParentResolutionStatus.UNRESOLVED


# --- sink level ----------------------------------------------------------------


async def test_sink_admits_call_after_reasoning_reply(store):
    """Job 5302914: ``Unresolved parent for model call ... poisoning the call`` after a thinking turn 0."""
    ledger, history = await _replay(store, [(SYSTEM_TASK, ASSISTANT_JSON_THINKING, ASSISTANT_JSON)])
    context = await _admit(store, history + [TERMINAL_OUTPUT], model_call_id="c2")
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    assert [failure.reason for failure in manifest.failures] == []
    assert context.capture_admission is not None
    assert context.capture_admission.mode == "token_in"
    assert context.capture_admission.parent_call_id == "c1"
