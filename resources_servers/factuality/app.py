# SPDX-FileCopyrightText: Copyright (c) 2026 Swiss AI Initiative. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Factuality resources server: the factuality eval grader as the RL reward (swiss-ai issue #1011).

One grading core with the eval harness (``factuality_grading``, a verbatim vendored copy of
swiss-ai/factuality-sft-data ``src/evals``; provenance in ``factuality_grading/VENDORED.json``):

    rubric prompt -> judge call -> strict parser -> label (CORRECT / INCORRECT / NOT_ATTEMPTED) + hedged

``verify()`` only adds what RL needs around it:

* **Final reply only.** The judge sees the last assistant message of the trajectory, never reasoning
  items or earlier output (the grader also strips inline ``<think>`` blocks).
* **Reward as configuration.** ``reward`` maps the five outcomes (correct, hedged-correct, abstain,
  hedged-wrong, confident-wrong) to values, plus truncation (default: the minimum), an optional
  length term and optional outside-answer adjustments. Nothing about the values is in code.
* **Truncation** (the policy hit ``max_output_tokens``) gets the truncation value without a judge call.
* **Judge failures are masked, never scored as wrong.** If every judge attempt raises or is
  unparseable, the sample is flagged with ``instance_config.mask_sample`` (read by NeMo RL's GRPO,
  ``env.should_mask_flagged_samples``, default true): it is dropped from the loss. It still enters the
  group baseline, so it gets a neutral reward (default: the abstain value).
* **Per-window rates and an abstain guard.** Every numeric field of the verify response is averaged
  per step by NeMo RL (``train/<agent>/<field>/mean`` in W&B). In addition the server writes one line
  per window of ``stats.window_rollouts`` rollouts (= one generation batch) to ``stats.path`` and,
  if ``guard.enabled``, stops the run when the abstain rate jumps (see ``GuardConfig``). Rows with
  ``split: validation`` are left out of the windows and the guard.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from resources_servers.factuality.factuality_grading import VENDORED, grader
from resources_servers.factuality.factuality_grading import data as fdata
from resources_servers.factuality.reply import strip_thinking


GRADER_VERSION = (
    f"harness {VENDORED['harness_version']} / {VENDORED['judge_prompt_version']} / "
    f"{VENDORED['source_commit'][:8]}{'+dirty' if VENDORED['source_dirty'] else ''}"
)

OUTCOMES = ("correct", "hedged_correct", "abstain", "hedged_wrong", "confident_wrong", "truncated", "judge_failed")
TASK_INDEX_KEY = "_ng_task_index"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class RewardConfig(BaseModel):
    """Reward values per outcome. Undecided (Valentina meeting 2026-10-22): keep them configurable.

    Abstaining is optimal on a question exactly when P(correct) < (abstain - wrong) / (correct - wrong)
    (confident values, no hedging); with the defaults that threshold is 0.5."""

    correct: float = 1.0
    hedged_correct: float = 1.0
    abstain: float = 0.0
    hedged_wrong: float = -1.0
    confident_wrong: float = -1.0
    # Ran out of max_output_tokens. None = the minimum of the five values above (minus length_coef),
    # so a loop is never cheaper than a confident wrong answer.
    truncated: Optional[float] = None
    # Judge failed on every attempt. The sample is masked from the loss but still counts in the group
    # baseline, so it should be neutral. None = the abstain value.
    judge_failure: Optional[float] = None
    # Optional length term: reward -= length_coef * min(1, reply_tokens / length_norm_tokens), applied
    # to every judged outcome (not to truncation, which is already the minimum). 0 = off.
    length_coef: float = 0.0
    length_norm_tokens: int = 1024
    # Grounded-unanswerable items: added when the reply correctly says the documents lack the answer and
    # then adds an answer from memory (judge OUTSIDE_ANSWER). 0 = no effect (default).
    outside_answer_correct: float = 0.0
    outside_answer_incorrect: float = 0.0
    outside_answer_ungraded: float = 0.0  # OUTSIDE_ANSWER: GIVEN (no reference to check it against)

    def resolved_truncated(self) -> float:
        if self.truncated is not None:
            return self.truncated
        return min(self.correct, self.hedged_correct, self.abstain, self.hedged_wrong, self.confident_wrong) - max(
            self.length_coef, 0.0
        )

    def resolved_judge_failure(self) -> float:
        return self.abstain if self.judge_failure is None else self.judge_failure


class StatsConfig(BaseModel):
    # Rollouts per window; set to num_prompts_per_step * num_generations_per_prompt so a window is one
    # generation batch. 0 disables windows (and the guard).
    window_rollouts: int = 0
    # JSONL file, one line per window. None = windows are only printed to the server log.
    path: Optional[str] = None


class GuardConfig(BaseModel):
    """Stop training when the policy starts collapsing into "always abstain" (R1 collapsed in 6 steps).

    After each window: trip if abstain_rate >= max_abstain_rate, or abstain_rate minus the lowest
    abstain rate of the earlier windows >= max_abstain_jump. Once tripped, every verify call raises,
    so the NeMo RL job fails within a step (GymTransportError, HTTP 500) with the reason in the stats
    file and the server log. Rates are over judged + truncated rollouts (judge failures excluded)."""

    enabled: bool = False
    max_abstain_rate: float = 0.8
    max_abstain_jump: float = 0.25


class FactualityConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_max_tokens: int = 256
    judge_retries: int = 2  # extra attempts after an error or an unparseable answer
    judge_retry_temperature: float = 0.7
    judge_error_backoff_s: float = 2.0
    judge_max_concurrency: int = 256
    # Until the judge has answered once, judge calls that fail (e.g. the judge vLLM is still loading) are retried every
    # judge_startup_poll_s for up to judge_startup_wait_s instead of counting as judge failures. Lets training set up
    # while the judge loads. 0 = no wait (a judge that is down at the first call fails as usual).
    judge_startup_wait_s: float = 0.0
    judge_startup_poll_s: float = 15.0

    reward: RewardConfig = Field(default_factory=RewardConfig)
    # "any": every rollout that hit max_output_tokens gets the truncation value (default; a cut-off
    # reply is a failure). "no_answer": the eval harness's rule, only if the cut-off reply is empty
    # after removing thinking; otherwise the partial reply is judged.
    truncation_rule: Literal["any", "no_answer"] = "any"
    stats: StatsConfig = Field(default_factory=StatsConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)


# ---------------------------------------------------------------------------
# Requests / responses
# ---------------------------------------------------------------------------


class FactualityRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    id: Optional[Union[int, str]] = None
    source: Optional[str] = None  # pool name, e.g. nq_rl / squad2_rl (eval set names are refused)
    question: Optional[str] = None
    answers: List[str] = Field(default_factory=list)
    answerable: Optional[bool] = None
    documents: Optional[Union[List[Dict[str, Any]], str]] = None  # grounded items
    kind: Optional[str] = None  # rubric override (crepe_false_premise, crepe_sound, ambiguous)
    meta: Optional[Union[Dict[str, Any], str]] = None
    gold_readings: Optional[List[Dict[str, Any]]] = None
    # "validation" rows are graded and rewarded as usual but kept out of the stats windows and the guard,
    # so a window stays one training generation batch.
    split: Optional[str] = None


class FactualityVerifyRequest(FactualityRunRequest, BaseVerifyRequest):
    pass


class FactualityVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    outcome: str
    label: Optional[str] = None
    hedged: Optional[bool] = None
    truncated: bool = False
    rubric: Optional[str] = None
    reply: str = ""
    reply_tokens: Optional[int] = None
    judge_output: Optional[str] = None
    judge_rationale: Optional[str] = None
    judge_attempts: int = 0
    judge_error: Optional[str] = None
    outside_answer: Optional[str] = None
    ambiguity: Optional[str] = None
    grader_version: str = GRADER_VERSION
    # Per-rollout indicators; NeMo RL averages every numeric field per step -> rates in W&B.
    is_correct: float = 0.0  # correct, hedged or not
    is_hedged_correct: float = 0.0
    is_abstain: float = 0.0
    is_hedged_wrong: float = 0.0
    is_confident_wrong: float = 0.0
    is_wrong: float = 0.0
    is_truncated: float = 0.0
    judge_failed: float = 0.0
    # NeMo RL reads instance_config.mask_sample to drop a sample from the loss.
    instance_config: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class JudgeCallError(RuntimeError):
    pass


class GuardTripped(RuntimeError):
    pass


def final_reply(response: NeMoGymResponse) -> tuple[str, Optional[int]]:
    """Text and generated-token count of the last assistant message. Reasoning items, tool calls and
    earlier messages are ignored; an output without an assistant message gives ""."""
    for item in reversed(response.output or []):
        if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            content = getattr(item, "content", None)
            if isinstance(content, str):
                text = content
            else:
                text = "".join(getattr(c, "text", "") or "" for c in (content or []))
            ids = getattr(item, "generation_token_ids", None)
            return text, (len(ids) if ids is not None else None)
    return "", None


def hit_max_tokens(response: NeMoGymResponse) -> bool:
    details = getattr(response, "incomplete_details", None)
    reason = getattr(details, "reason", None) if details is not None else None
    if reason is None and isinstance(details, dict):
        reason = details.get("reason")
    return reason == "max_output_tokens"


def item_from_request(body: FactualityRunRequest) -> fdata.Item:
    raw = {
        "id": body.id,
        "set": body.source or "rl",
        "question": body.question or "",
        "answers": body.answers,
        "answerable": body.answerable,
        "documents": body.documents,
        "meta": body.meta,
        "kind": body.kind,
        "gold_readings": body.gold_readings,
    }
    return fdata.normalise_pool_row(raw)


def outcome_of(label: Optional[str], hedged: Optional[bool]) -> str:
    if label == "CORRECT":
        return "hedged_correct" if hedged else "correct"
    if label == "INCORRECT":
        return "hedged_wrong" if hedged else "confident_wrong"
    if label == "NOT_ATTEMPTED":
        return "abstain"
    return "judge_failed"


def reward_for(cfg: RewardConfig, outcome: str, reply_tokens: Optional[int], outside_answer: Optional[str]) -> float:
    if outcome == "truncated":
        return cfg.resolved_truncated()
    if outcome == "judge_failed":
        return cfg.resolved_judge_failure()
    value = getattr(cfg, outcome)
    if outside_answer == "CORRECT":
        value += cfg.outside_answer_correct
    elif outside_answer == "INCORRECT":
        value += cfg.outside_answer_incorrect
    elif outside_answer == "GIVEN":
        value += cfg.outside_answer_ungraded
    if cfg.length_coef and reply_tokens is not None and cfg.length_norm_tokens > 0:
        value -= cfg.length_coef * min(1.0, reply_tokens / cfg.length_norm_tokens)
    return value


# ---------------------------------------------------------------------------
# Windows and guard
# ---------------------------------------------------------------------------


class RateWindows:
    """Counts outcomes per window of N verify calls; writes one JSONL line per window; runs the guard."""

    def __init__(self, stats: StatsConfig, guard: GuardConfig) -> None:
        self.stats, self.guard = stats, guard
        self.index = 0
        self.history: List[float] = []  # abstain rate per closed window
        self.tripped: Optional[str] = None
        self._reset()

    def _reset(self) -> None:
        self.counts: Counter = Counter()
        self.rewards: List[float] = []
        self.tokens: List[int] = []
        self.groups: Dict[Any, List[float]] = defaultdict(list)
        self.t0 = time.time()

    def add(self, outcome: str, reward: float, reply_tokens: Optional[int], task_index: Any) -> Optional[dict]:
        if self.stats.window_rollouts <= 0:
            return None
        self.counts[outcome] += 1
        self.rewards.append(reward)
        if reply_tokens is not None:
            self.tokens.append(reply_tokens)
        if task_index is not None:
            self.groups[task_index].append(reward)
        if sum(self.counts.values()) < self.stats.window_rollouts:
            return None
        return self._close()

    def _close(self) -> dict:
        n = sum(self.counts.values())
        graded = n - self.counts["judge_failed"]

        def rate(*names: str) -> Optional[float]:
            return round(sum(self.counts[x] for x in names) / graded, 4) if graded else None

        multi = [g for g in self.groups.values() if len(g) > 1]
        rec = {
            "window": self.index,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds": round(time.time() - self.t0, 1),
            "n": n,
            "counts": {k: self.counts[k] for k in OUTCOMES},
            "correct_rate": rate("correct", "hedged_correct"),
            "abstain_rate": rate("abstain"),
            "wrong_rate": rate("hedged_wrong", "confident_wrong"),
            "confident_wrong_rate": rate("confident_wrong"),
            "truncated_rate": rate("truncated"),
            "judge_failed": self.counts["judge_failed"],
            "mean_reward": round(sum(self.rewards) / len(self.rewards), 4) if self.rewards else None,
            "mean_reply_tokens": round(sum(self.tokens) / len(self.tokens), 1) if self.tokens else None,
            "groups": len(multi),
            "zero_variance_group_share": (
                round(sum(1 for g in multi if max(g) - min(g) < 1e-9) / len(multi), 4) if multi else None
            ),
            "mean_group_reward_std": (round(sum(_std(g) for g in multi) / len(multi), 4) if multi else None),
            "grader_version": GRADER_VERSION,
        }
        abst = rec["abstain_rate"]
        if self.guard.enabled and abst is not None:
            if abst >= self.guard.max_abstain_rate:
                self.tripped = f"abstain rate {abst:.3f} >= max_abstain_rate {self.guard.max_abstain_rate}"
            elif self.history and abst - min(self.history) >= self.guard.max_abstain_jump:
                self.tripped = (
                    f"abstain rate {abst:.3f} rose by {abst - min(self.history):.3f} over the lowest earlier "
                    f"window ({min(self.history):.3f}) >= max_abstain_jump {self.guard.max_abstain_jump}"
                )
            if self.tripped:
                rec["guard_tripped"] = self.tripped
        if abst is not None:
            self.history.append(abst)
        self._write(rec)
        self.index += 1
        self._reset()
        return rec

    def _write(self, rec: dict) -> None:
        line = json.dumps(rec)
        print(f"[factuality-stats] {line}", file=sys.stderr, flush=True)
        if self.stats.path:
            path = Path(self.stats.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(line + "\n")


def _std(xs: List[float]) -> float:
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class FactualityServer(SimpleResourcesServer):
    config: FactualityConfig
    _judge_sem: asyncio.Semaphore = PrivateAttr()
    _windows: RateWindows = PrivateAttr()
    _judge_up: bool = PrivateAttr(default=False)
    _judge_wait_logged: bool = PrivateAttr(default=False)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._judge_sem = asyncio.Semaphore(max(1, self.config.judge_max_concurrency))
        self._windows = RateWindows(self.config.stats, self.config.guard)
        r = self.config.reward
        print(
            f"[factuality] grader {GRADER_VERSION}; reward {r.model_dump()}; truncated -> {r.resolved_truncated()}, "
            f"judge failure -> {r.resolved_judge_failure()} (masked); truncation_rule {self.config.truncation_rule}",
            file=sys.stderr,
            flush=True,
        )

    async def _call_judge(self, prompt: str, temperature: float, max_tokens: int) -> str:
        deadline = time.monotonic() + self.config.judge_startup_wait_s
        while True:
            try:
                text = await self._call_judge_once(prompt, temperature, max_tokens)
            except Exception as e:  # noqa: BLE001 - only retried before the judge has ever answered
                if self._judge_up or time.monotonic() >= deadline:
                    raise
                if not self._judge_wait_logged:
                    self._judge_wait_logged = True
                    print(
                        f"[factuality] judge not answering yet ({type(e).__name__}: {str(e)[:200]}); retrying every "
                        f"{self.config.judge_startup_poll_s:.0f} s for up to {self.config.judge_startup_wait_s:.0f} s",
                        file=sys.stderr,
                        flush=True,
                    )
                await asyncio.sleep(self.config.judge_startup_poll_s)
                continue
            if not self._judge_up:
                self._judge_up = True
                print("[factuality] judge answering", file=sys.stderr, flush=True)
            return text

    async def _call_judge_once(self, prompt: str, temperature: float, max_tokens: int) -> str:
        params = self.config.judge_responses_create_params.model_copy(deep=True)
        params.input = [NeMoGymEasyInputMessage(role="user", content=prompt)]
        params.temperature = temperature
        params.max_output_tokens = max_tokens
        async with self._judge_sem:
            resp = await self.server_client.post(
                server_name=self.config.judge_model_server.name,
                url_path="/v1/responses",
                json=params,
            )
            status = getattr(resp, "status", 200)
            if isinstance(status, int) and status >= 400:
                body = await resp.text() if hasattr(resp, "text") else ""
                raise JudgeCallError(f"judge HTTP {status}: {str(body)[:300]}")
            judge_response = NeMoGymResponse.model_validate(await resp.json())
        text, _ = final_reply(judge_response)
        return text

    async def verify(self, body: FactualityVerifyRequest) -> FactualityVerifyResponse:
        if self._windows.tripped:
            raise GuardTripped(f"factuality guard tripped, stopping the run: {self._windows.tripped}")

        item = item_from_request(body)
        raw_reply, reply_tokens = final_reply(body.response)
        reply = strip_thinking(raw_reply)
        if reply_tokens is None and body.response.usage is not None:
            reply_tokens = body.response.usage.output_tokens
        truncated = hit_max_tokens(body.response)

        result: Dict[str, Any] = {
            "label": None,
            "hedged": None,
            "rationale": None,
            "judge_raw": None,
            "attempts": 0,
            "judge_error": None,
        }
        if truncated and (self.config.truncation_rule == "any" or not grader.clean_reply(reply)):
            outcome = "truncated"
        else:
            result = await grader.grade_with(
                item,
                reply,
                self._call_judge,
                max_tokens=self.config.judge_max_tokens,
                retries=self.config.judge_retries,
                retry_temperature=self.config.judge_retry_temperature,
                error_backoff_s=self.config.judge_error_backoff_s,
            )
            outcome = outcome_of(result["label"], result["hedged"])

        outside = result.get("outside_answer")
        reward = reward_for(self.config.reward, outcome, reply_tokens, outside)
        failed = outcome == "judge_failed"
        task_index = (body.model_extra or {}).get(TASK_INDEX_KEY)

        response = FactualityVerifyResponse(
            **body.model_dump(),
            reward=reward,
            outcome=outcome,
            label=result["label"],
            hedged=result["hedged"],
            truncated=outcome == "truncated",
            rubric=item.kind,
            reply=reply,
            reply_tokens=reply_tokens,
            judge_output=result["judge_raw"],
            judge_rationale=result["rationale"],
            judge_attempts=result["attempts"],
            judge_error=result["judge_error"],
            outside_answer=outside,
            ambiguity=(
                None
                if "ambiguity_addressed" not in result
                else ("ADDRESSED" if result["ambiguity_addressed"] else "IGNORED")
            ),
            is_correct=float(outcome in ("correct", "hedged_correct")),
            is_hedged_correct=float(outcome == "hedged_correct"),
            is_abstain=float(outcome == "abstain"),
            is_hedged_wrong=float(outcome == "hedged_wrong"),
            is_confident_wrong=float(outcome == "confident_wrong"),
            is_wrong=float(outcome in ("hedged_wrong", "confident_wrong")),
            is_truncated=float(outcome == "truncated"),
            judge_failed=float(failed),
            instance_config={"mask_sample": True} if failed else None,
        )
        if body.split != "validation":
            self._windows.add(outcome, reward, reply_tokens, task_index)
        return response


if __name__ == "__main__":
    FactualityServer.run_webserver()
