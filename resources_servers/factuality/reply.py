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
