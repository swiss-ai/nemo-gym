# SPDX-FileCopyrightText: Copyright (c) 2026 Swiss AI Initiative. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reply cleaning for the factuality environment (stdlib only, so scripts can use it without nemo_gym)."""

import re


_UNCLOSED_THINK_RE = re.compile(r"<think>(?:(?!</think>).)*\Z", re.S | re.I)
_LEADING_CLOSE_RE = re.compile(r"\A(?:(?!<think>).)*?</think>", re.S | re.I)


def strip_thinking(text: str) -> str:
    """Remove thinking the reasoning parser did not separate, beyond what the grader's clean_reply does
    (complete <think>...</think> blocks): text before an unpaired leading </think> (the template opened
    the block in the prompt), and an unclosed trailing <think>... (a loop cut off mid-thought)."""
    if _LEADING_CLOSE_RE.search(text) and "<think>" not in text.split("</think>", 1)[0].lower():
        text = _LEADING_CLOSE_RE.sub("", text, count=1)
    return _UNCLOSED_THINK_RE.sub("", text)


# How an abstention is phrased (heuristic, English). RL on the Qwen stand-in (2026-10-22) learned to "abstain" mostly
# by deflecting: claiming a clear question is unclear and asking for clarification, which the rubric labels
# NOT_ATTEMPTED like an honest "I don't know". These flags make that visible per step; they do not change the reward.
_DEFLECTION_RE = re.compile(
    r"clarif|more (context|details?|specifics?|information)|"
    r"(could|can) you (please )?(provide|specify|clarify|tell me|give)|"
    r"(isn't|is not|wasn't|was not) (clear|specific)|\bunclear\b|\bincomplete\b|\bambiguous\b",
    re.I,
)
_NOT_IN_DOCUMENT_RE = re.compile(
    r"\b(document|passage|text|context)s? (provided |given |above )?(does not|doesn't|do not|don't)"
    r"( explicitly)? (say|state|mention|contain|specify|provide|include)",
    re.I,
)
_IDK_RE = re.compile(
    r"\bI (don't|do not|can't|cannot|couldn't) (know|recall|remember|say|tell)\b|"
    r"\bI('m| am) (not (sure|certain|aware)|unsure|uncertain)\b|"
    r"\bI (don't|do not) have (specific |enough |reliable |accurate )?(information|knowledge|data)\b|"
    r"\bI('m| am) (unable|not able) to (find|determine|confirm|verify|answer)\b",
    re.I,
)
ABSTAIN_STYLES = ("deflection", "not_in_document", "idk", "empty", "other")


def abstain_style(reply: str) -> str:
    """Classify an abstaining reply: deflection (asks for clarification / calls the question unclear; wins over the
    others, as it shifts the problem to the user), not_in_document (grounded), idk (states not knowing), empty, other."""
    text = reply.strip()
    if not text:
        return "empty"
    if _DEFLECTION_RE.search(text):
        return "deflection"
    if _NOT_IN_DOCUMENT_RE.search(text):
        return "not_in_document"
    if _IDK_RE.search(text):
        return "idk"
    return "other"
