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

"""Fingerprint conversation content across serving dialects.

These are pure functions over message lists.
They hold no state and read no store.
Any server may import them.
Lineage state stays confined to the capture locus and the finalizer.

``assistant_fingerprint`` hashes only model-authored turns.
It identifies the call that produced the last model-authored turn.
``conversation_digest`` hashes every conversation turn, including tool results.
A parent resolver uses it to verify context before reusing tokens.
Harness instruction items (``system`` / ``developer``) count by position only:
harnesses regenerate them per request with volatile fields (OpenCode's system
prompt carries today's date), and hashing their content poisoned every rollout
that crossed a UTC day boundary. Version 3 of the digest skips their content.

Chat, Responses, and Anthropic shapes normalize to the same hash input.
The hash layout is tagged and length-delimited.
No concatenation of fields can collide with another field boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import orjson


# Increment when fingerprint canonicalization or hash layout changes.
# Resolvers ignore entries stamped with a different version.
FINGERPRINT_VERSION = 5

_FINGERPRINT_DOMAIN = b"nemo-gym-lineage"
_CONTEXT_DOMAIN = b"nemo-gym-lineage-context"
_INSTRUCTION_ROLES = frozenset({"system", "developer"})

# The reasoning that a model server folds into assistant text: one or more ``<think>...</think>`` blocks at the start.
_LEADING_THINK_BLOCKS = re.compile(r"(?:<think>.*?</think>)+", re.DOTALL)


def assistant_fingerprint(messages: list[dict]) -> str:
    """Fingerprint the model-authored turns of a request, in order.

    The fingerprint identifies the call that produced the last model-authored turn.
    User and tool content is excluded from the lookup key.
    Dialect-specific tool-call shapes normalize to the same hash input.

    A turn is a run of consecutive model-authored items.
    A turn with no text and no tool call (an empty reply) hashes an explicit marker.
    Without it, the call that produced an empty reply would share its parent's fingerprint.
    The next request would then match both calls and resolve as ambiguous.
    Non-empty turns hash exactly as before, so their fingerprints do not change.

    Reasoning folded into assistant text is excluded like standalone reasoning items.
    A server with a reasoning parser returns ``<think>REASONING</think>`` + content.
    A harness may echo that text verbatim or only the content after the block.
    Both echoes must find the call, so leading think blocks are stripped before hashing.
    A turn left with no text and no tool call hashes the empty-turn marker.
    """
    hasher = hashlib.sha256(_FINGERPRINT_DOMAIN)
    count = 0
    # Whether the current run of consecutive model-authored items hashed any field.
    # ``None`` means no run is open.
    run_hashed: bool | None = None
    for message in messages or []:
        if not isinstance(message, dict):
            raise ValueError(f"request item is not an object: {type(message).__name__}")
        if not _is_assistant_authored(message):
            if message.get("type") == "reasoning":
                # Excluded like reasoning content; it neither opens nor closes a turn.
                continue
            if run_hashed is False:
                _update_field(hasher, b"\x05", "")
            run_hashed = None
            continue
        count += 1
        run_hashed = bool(run_hashed)
        for content_type, payload in _strip_leading_reasoning(_content_of(message.get("content"))):
            _update_field(hasher, b"\x00", content_type)
            _update_field(hasher, b"\x01", payload)
            run_hashed = True
        for call_id, name, arguments in _tools_of(message):
            _update_field(hasher, b"\x02", call_id)
            _update_field(hasher, b"\x03", name)
            _update_field(hasher, b"\x04", arguments)
            run_hashed = True
    if run_hashed is False:
        _update_field(hasher, b"\x05", "")
    if count == 0:
        return ""
    return hasher.hexdigest()


def conversation_digest(messages: list[dict]) -> str:
    """Hash every turn of a conversation, model-authored or not.

    ``assistant_fingerprint`` ignores user and tool content.
    This digest covers that omitted context.
    A mismatch rejects the parent before its tokens are reused.

    Instruction items (``system`` / ``developer`` roles) contribute their role
    and position but not their content: the harness owns them and may rewrite
    volatile fields between calls (a date, a clock), which is not a rewritten
    conversation. Reusing the parent's tokens then serves the instruction text
    the model actually saw, which is what training needs.
    """
    hasher = hashlib.sha256(_CONTEXT_DOMAIN)
    for message in messages or []:
        if not isinstance(message, dict):
            raise ValueError(f"request item is not an object: {type(message).__name__}")
        role = str(message.get("role") or message.get("type") or "")
        _update_field(hasher, b"\x00", role)
        if _is_instruction_item(message):
            continue
        for content_type, payload in _content_of(message.get("content")):
            _update_field(hasher, b"\x01", content_type)
            _update_field(hasher, b"\x02", payload)
        for call_id, name, arguments in _tools_of(message):
            _update_field(hasher, b"\x03", call_id)
            _update_field(hasher, b"\x04", name)
            _update_field(hasher, b"\x05", arguments)
        # Include tool results.
        # Summarizing, redacting, or truncating a result changes the request context.
        for call_id, output in _tool_results_of(message):
            _update_field(hasher, b"\x06", call_id)
            _update_field(hasher, b"\x07", output)
    return hasher.hexdigest()


def canonicalize_tool_arguments(value: Any) -> str:
    """Normalize a tool call's arguments for comparison only.

    Harnesses can reserialize tool-call arguments between turns.
    Comparison uses sorted-key JSON with normalized separators.
    The record retains the model's original string.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return value.strip()
    else:
        parsed = value
    return _canonical_json(parsed)


def _is_instruction_item(message: dict) -> bool:
    """Return whether the harness, not the conversation, authored this item."""
    return message.get("role") in _INSTRUCTION_ROLES


def _is_assistant_authored(message: dict) -> bool:
    """Return whether the model produced this item.

    Chat and Anthropic use the ``assistant`` role.
    Responses tool calls are roleless ``function_call`` items.
    """
    if message.get("role") == "assistant":
        return True
    # Reasoning is deliberately excluded.
    # A harness need not echo standalone reasoning items.
    # Including reasoning would make fingerprints depend on the dialect and echo behavior.
    # Reasoning-only collisions resolve as ambiguous and fall back.
    return message.get("type") == "function_call"


def _content_of(content: Any) -> list[tuple[str, str]]:
    """Return typed content parts without discarding prompt-shaping blocks.

    Tool calls are normalized separately by ``_tools_of``.
    Tool results are normalized separately by ``_tool_results_of``.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [("text", content)] if content else []
    if not isinstance(content, list):
        raise ValueError(f"unsupported message content: {type(content).__name__}")
    parts: list[tuple[str, str]] = []
    for block in content:
        if isinstance(block, str):
            if block:
                parts.append(("text", block))
            continue
        if not isinstance(block, dict):
            raise ValueError(f"unsupported content block: {type(block).__name__}")
        block_type = str(block.get("type") or "")
        if block_type in {"tool_use", "tool_result"}:
            continue
        if isinstance(block.get("text"), str) and block_type in {
            "",
            "text",
            "input_text",
            "output_text",
        }:
            if block["text"]:
                parts.append(("text", block["text"]))
            continue
        if not block_type:
            raise ValueError("content block has no supported type")
        parts.append((block_type, _canonical_json(block)))
    return parts


def _strip_leading_reasoning(parts: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop the think blocks that open a model-authored item's text.

    ``_wrap_reasoning_in_think_tags`` puts reasoning only at the start of the content.
    Clients that split it back out (LiteLLM) take one leading block and keep the rest verbatim.
    Stripping the whole leading run makes both echoes agree, even when the visible text starts with a block.
    Think blocks later in the text are visible model output and stay hashed.
    """
    if not parts or parts[0][0] != "text":
        return parts
    match = _LEADING_THINK_BLOCKS.match(parts[0][1])
    if match is None:
        return parts
    rest = parts[0][1][match.end() :]
    return ([("text", rest)] if rest else []) + parts[1:]


def _tools_of(message: dict) -> list[tuple[str, str, str]]:
    """Return tool calls as ``(id, name, canonical arguments)`` tuples.

    Chat stores calls in the message's ``tool_calls`` field.
    Anthropic stores calls in ``tool_use`` content blocks.
    Responses stores each call as a standalone ``function_call`` item.
    """
    tools: list[tuple[str, str, str]] = []
    # A Responses item is the tool call.
    if message.get("type") == "function_call":
        name = str(message.get("name", ""))
        if message.get("namespace"):
            # SSE separates these fields; the backend and echoed request sanitizer join them.
            # Preserve the namespace so different tools with the same short name cannot match.
            name = f"{message['namespace']}__{name}"
        tools.append(
            (
                str(message.get("call_id") or message.get("id") or ""),
                name,
                canonicalize_tool_arguments(message.get("arguments")),
            )
        )
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tools.append(
                    (
                        str(block.get("id") or ""),
                        str(block.get("name", "")),
                        canonicalize_tool_arguments(block.get("input")),
                    )
                )
    for call in message.get("tool_calls") or []:
        function = (call or {}).get("function") or {}
        tools.append(
            (
                str((call or {}).get("id") or ""),
                str(function.get("name", "")),
                canonicalize_tool_arguments(function.get("arguments")),
            )
        )
    return tools


def _tool_results_of(message: dict) -> list[tuple[str, str]]:
    """Return tool result identities and payloads across dialects.

    Responses stores results in standalone ``function_call_output`` items.
    Anthropic stores results in ``tool_result`` content blocks.
    Chat stores results as plain message content.
    """
    parts: list[tuple[str, str]] = []
    if message.get("type") == "function_call_output":
        output = message.get("output")
        parts.append(
            (
                str(message.get("call_id") or message.get("id") or ""),
                output if isinstance(output, str) else _canonical_json(output),
            )
        )
    elif message.get("role") == "tool":
        content = message.get("content")
        parts.append(
            (
                str(message.get("tool_call_id") or ""),
                content if isinstance(content, str) else _canonical_json(content),
            )
        )
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                inner = block.get("content")
                payload = inner if isinstance(inner, str) else _canonical_json(inner)
                parts.append((str(block.get("tool_use_id") or block.get("id") or ""), payload))
    return parts


def _canonical_json(value: Any) -> str:
    """Serialize JSON-compatible prompt content without losing structure."""
    try:
        return orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    except (TypeError, orjson.JSONEncodeError) as error:
        raise ValueError(f"unsupported prompt content: {type(value).__name__}") from error


def _update_field(hasher: Any, tag: bytes, value: str) -> None:
    """Hash one tagged, length-delimited UTF-8 field."""
    encoded = value.encode("utf-8")
    hasher.update(tag)
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)
