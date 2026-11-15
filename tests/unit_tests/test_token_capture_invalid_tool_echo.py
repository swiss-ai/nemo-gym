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

"""A tool call with unparseable arguments, echoed by OpenCode as an ``invalid`` call, keeps its lineage.

Observed in an OpenCode GRPO rollout: the policy called ``bash`` with arguments that are not
JSON. OpenCode (AI SDK tool-call repair) does not echo that call as served. It keeps the call
id and rewrites the call as ``invalid`` with ``{"tool": "bash", "error": "Invalid input for
tool bash: JSON parsing failed: ..."}``, and answers it with a tool result that reports the
error. The served call hashed ``(id, "bash", <raw arguments>)``, the echo ``(id, "invalid",
<error object>)``, so the next call matched no parent and every later call of the rollout
was poisoned.

The served tokens are what the policy generated, and the echo names the same call by id.
Unparseable arguments of a call with an id therefore hash as one marker, and the echo hashes
as the call it repairs: the named tool with that marker.
"""

from __future__ import annotations

import json

import pytest

from nemo_gym.token_id_capture.fingerprint import FINGERPRINT_VERSION, assistant_fingerprint
from nemo_gym.token_id_capture.lineage import FileLineageStore, InMemoryLineageStore
from nemo_gym.token_id_capture.records import ParentResolutionStatus
from nemo_gym.token_id_capture.sink import (
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


# Shapes from OpenCode 1.17.11 on Apertus 2 (job 5316800, rollout g6, calls 4-6).
SYSTEM = {"role": "system", "content": "You are opencode, an interactive CLI tool ..."}
USER = {"role": "user", "content": '"Create /app/text.txt ... write the count to /app/count.txt."'}

BAD_ID = "chatcmpl-tool-80eb114f4a73ce73"
# Not JSON: the quote before the comma closes the string early.
BAD_ARGUMENTS = '{"command": "cat /app/count.txt | xargs -I{} echo \\"Count: {}"", "description": "Verify count.txt"}'
JSON_ERROR = (
    "Invalid input for tool bash: JSON parsing failed: Text: " + BAD_ARGUMENTS + ".\nError message: "
    "JSON Parse error: Expected '}'"
)


def _assistant(content: str, call_id: str, name: str, arguments: str) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def _tool(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


READ_CALL = _assistant("Let me verify count.txt:", "chatcmpl-tool-b974", "read", '{"filePath": "/app/count.txt"}')
READ_RESULT = _tool("chatcmpl-tool-b974", "<content>\n1: 3\n</content>")

# What the model server served for call 4.
SERVED_BAD_CALL = _assistant("I see there's an issue. Let me verify the output:", BAD_ID, "bash", BAD_ARGUMENTS)
# What OpenCode echoes for it in the request of call 5.
ECHOED_BAD_CALL = _assistant(
    "I see there's an issue. Let me verify the output:",
    BAD_ID,
    "invalid",
    json.dumps({"tool": "bash", "error": JSON_ERROR}, separators=(",", ":")),
)
BAD_CALL_RESULT = _tool(BAD_ID, "The arguments provided to the tool are invalid: " + JSON_ERROR)

# Call 5's served reply and its result.
GOOD_CALL = _assistant(
    "Let me use the bash tool properly:",
    "chatcmpl-tool-a007",
    "bash",
    '{"command": "cat /app/count.txt", "description": "Display contents of count.txt"}',
)
GOOD_CALL_RESULT = _tool("chatcmpl-tool-a007", "3\n")


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


async def _serve_bad_call(store) -> _Ledger:
    """Commit call 1 (a read) and call 2 (the call with unparseable arguments), as served."""
    ledger = _Ledger(store)
    await ledger.commit([SYSTEM, USER], READ_CALL)
    await ledger.commit([SYSTEM, USER, READ_CALL, READ_RESULT], SERVED_BAD_CALL)
    return ledger


HISTORY_AFTER_BAD_CALL = [SYSTEM, USER, READ_CALL, READ_RESULT, ECHOED_BAD_CALL, BAD_CALL_RESULT]


@pytest.fixture(params=["file", "memory"])
def store(request, tmp_path):
    if request.param == "file":
        return FileLineageStore(tmp_path)
    return InMemoryLineageStore()


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


# --- fingerprint level -------------------------------------------------------


def test_invalid_echo_fingerprints_as_the_served_call():
    served = assistant_fingerprint([SYSTEM, USER, SERVED_BAD_CALL])
    echoed = assistant_fingerprint([SYSTEM, USER, ECHOED_BAD_CALL])
    assert served == echoed


def test_invalid_echo_with_arguments_as_object_fingerprints_as_the_served_call():
    echo = _assistant(ECHOED_BAD_CALL["content"], BAD_ID, "invalid", "")
    echo["tool_calls"][0]["function"]["arguments"] = {"tool": "bash", "error": JSON_ERROR}
    assert assistant_fingerprint([SERVED_BAD_CALL]) == assistant_fingerprint([echo])


def test_invalid_echo_with_other_id_does_not_match():
    echo = _assistant(
        ECHOED_BAD_CALL["content"],
        "chatcmpl-tool-ffff",
        "invalid",
        ECHOED_BAD_CALL["tool_calls"][0]["function"]["arguments"],
    )
    assert assistant_fingerprint([SERVED_BAD_CALL]) != assistant_fingerprint([echo])


def test_invalid_echo_naming_other_tool_does_not_match():
    echo = _assistant(
        ECHOED_BAD_CALL["content"],
        BAD_ID,
        "invalid",
        json.dumps({"tool": "write", "error": JSON_ERROR.replace("tool bash", "tool write")}),
    )
    assert assistant_fingerprint([SERVED_BAD_CALL]) != assistant_fingerprint([echo])


def test_invalid_echo_of_parseable_arguments_does_not_match():
    """A schema error (arguments were JSON) is not a parse failure: the served arguments stay hashed."""
    served = _assistant("Run it:", BAD_ID, "bash", '{"cmd": "ls"}')
    echo = _assistant(
        "Run it:",
        BAD_ID,
        "invalid",
        json.dumps({"tool": "bash", "error": "Invalid input for tool bash: Type validation failed: ..."}),
    )
    assert assistant_fingerprint([served]) != assistant_fingerprint([echo])


def test_tool_named_invalid_with_other_arguments_hashes_as_is():
    """A real tool that happens to be named ``invalid`` keeps its own identity."""
    call = _assistant("", "call_1", "invalid", '{"path": "/app"}')
    other = _assistant("", "call_1", "invalid", '{"path": "/tmp"}')
    assert assistant_fingerprint([call]) != assistant_fingerprint([other])


def test_unparseable_arguments_of_different_calls_do_not_collide():
    one = _assistant("x", "chatcmpl-tool-1", "bash", '{"command": "a""}')
    two = _assistant("x", "chatcmpl-tool-2", "bash", '{"command": "b""}')
    assert assistant_fingerprint([one]) != assistant_fingerprint([two])


def test_reserialized_json_arguments_still_match():
    """Control: OpenCode echoes valid arguments without spaces; that already resolved before the fix."""
    served = _assistant("x", "chatcmpl-tool-a007", "bash", '{"command": "cat /app/count.txt"}')
    echo = _assistant("x", "chatcmpl-tool-a007", "bash", '{"command":"cat /app/count.txt"}')
    assert assistant_fingerprint([served]) == assistant_fingerprint([echo])


# --- resolver level ------------------------------------------------------------


async def test_call_after_invalid_echo_resolves_to_the_served_call(store):
    ledger = await _serve_bad_call(store)
    resolution = await store.resolve("r1", HISTORY_AFTER_BAD_CALL)
    assert resolution.status == ParentResolutionStatus.RESOLVED, resolution.reason
    assert resolution.match.model_call_id == ledger.calls[-1]


async def test_lineage_continues_after_the_call_that_followed_the_echo(store):
    """Call 3 was served on the echoed history; call 4 continues it with the echo still in place."""
    ledger = await _serve_bad_call(store)
    await ledger.commit(list(HISTORY_AFTER_BAD_CALL), GOOD_CALL)
    resolution = await store.resolve("r1", HISTORY_AFTER_BAD_CALL + [GOOD_CALL, GOOD_CALL_RESULT])
    assert resolution.status == ParentResolutionStatus.RESOLVED, resolution.reason
    assert resolution.match.model_call_id == ledger.calls[-1]


async def test_sink_admits_call_after_invalid_echo_token_in(store):
    """The production symptom: ``Unresolved parent for model call ... poisoning the call``."""
    ledger = await _serve_bad_call(store)
    context = await _admit(store, HISTORY_AFTER_BAD_CALL, model_call_id="c3")
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    assert [failure.reason for failure in manifest.failures] == []
    assert context.capture_admission is not None
    assert context.capture_admission.mode == "token_in"
    assert context.capture_admission.parent_call_id == ledger.calls[-1]
