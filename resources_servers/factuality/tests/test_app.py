# SPDX-FileCopyrightText: Copyright (c) 2026 Swiss AI Initiative. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reward tests for the factuality environment.

``reward_cases.jsonl`` holds 52 hand-checked replies (closed-book, grounded answerable and
unanswerable, CREPE, AmbigNQ; hedging, commitment rule, truncation, judge failures, thinking).
Each case carries the judge's canned output(s); the tests check outcome, label, hedged flag, the
judge prompt, masking and the reward under two reward configurations. The same file feeds
``scripts/check_reward_cases_live.py``, which sends the replies to the real judge on Clariden.
"""

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import approx

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.factuality.app import (
    FactualityConfig,
    FactualityServer,
    FactualityVerifyRequest,
    GuardConfig,
    GuardTripped,
    RateWindows,
    RewardConfig,
    StatsConfig,
    final_reply,
    hit_max_tokens,
    reward_for,
)
from resources_servers.factuality.factuality_grading import VENDORED, grader


HERE = Path(__file__).parent
CASES = [json.loads(line) for line in (HERE / "reward_cases.jsonl").read_text().splitlines() if line.strip()]

DEFAULT = RewardConfig()  # correct +1 / abstain 0 / wrong -1, truncation = min
FIVE_LEVEL = RewardConfig(
    correct=1.0, hedged_correct=0.6, abstain=0.0, hedged_wrong=-0.4, confident_wrong=-1.0, judge_failure=0.1
)
EXPECTED = {
    "default": {
        "correct": 1,
        "hedged_correct": 1,
        "abstain": 0,
        "hedged_wrong": -1,
        "confident_wrong": -1,
        "truncated": -1,
        "judge_failed": 0,
    },
    "five_level": {
        "correct": 1,
        "hedged_correct": 0.6,
        "abstain": 0,
        "hedged_wrong": -0.4,
        "confident_wrong": -1,
        "truncated": -1,
        "judge_failed": 0.1,
    },
}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _msg(text: str, i: int = 0) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id=f"msg{i}",
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _response(output, finish: str = "stop", output_tokens: int | None = None) -> NeMoGymResponse:
    kw = {}
    if finish == "length":
        kw["incomplete_details"] = {"reason": "max_output_tokens"}
    if output_tokens is not None:
        kw["usage"] = {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 10 + output_tokens,
        }
    return NeMoGymResponse(
        id="r",
        created_at=0.0,
        model="policy",
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        **kw,
    )


def _policy_response(case: dict) -> NeMoGymResponse:
    if case.get("output"):
        items = []
        for i, o in enumerate(case["output"]):
            if o["type"] == "reasoning":
                items.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rs{i}", summary=[NeMoGymSummary(text=o["text"], type="summary_text")]
                    )
                )
            else:
                items.append(_msg(o["text"], i))
        return _response(items, case.get("finish", "stop"))
    return _response([_msg(case["reply"])], case.get("finish", "stop"))


def _judge_http(outputs: list) -> tuple[AsyncMock, list]:
    """A mocked ServerClient.post that serves judge outputs in order; records the prompts sent."""
    prompts: list[str] = []
    seq = iter(outputs)

    async def post(server_name, url_path, json):
        assert server_name == "judge_model" and url_path == "/v1/responses"
        prompts.append(json.input[0].content)
        out = next(seq)
        if isinstance(out, dict):
            if out["error"] == "raise":
                raise ConnectionError("judge unreachable")
            resp = MagicMock()
            resp.status = out["error"]
            resp.text = AsyncMock(return_value="Internal Server Error")
            return resp
        resp = MagicMock()
        resp.status = 200
        resp.json = AsyncMock(return_value=_response([_msg(out)]).model_dump())
        return resp

    return AsyncMock(side_effect=post), prompts


def _config(reward: RewardConfig = DEFAULT, **kw) -> FactualityConfig:
    return FactualityConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[], temperature=0.0),
        judge_error_backoff_s=0.0,
        reward=reward,
        **kw,
    )


def _server(cfg: FactualityConfig, judge_outputs: list) -> tuple[FactualityServer, list, MagicMock]:
    client = MagicMock(spec=ServerClient)
    client.post, prompts = _judge_http(judge_outputs)
    return FactualityServer(config=cfg, server_client=client), prompts, client


def _request(case: dict, response: NeMoGymResponse | None = None, **extra) -> FactualityVerifyRequest:
    fields = {
        k: case[k]
        for k in ("id", "source", "question", "answers", "answerable", "documents", "kind", "meta", "gold_readings")
        if k in case
    }
    fields.setdefault("id", case["case"])
    return FactualityVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=response or _policy_response(case),
        **fields,
        **extra,
    )


# ---------------------------------------------------------------------------
# The 52 hand-checked cases
# ---------------------------------------------------------------------------


def test_case_file_covers_every_outcome_and_rubric():
    assert len(CASES) >= 50
    assert {c["expect"]["outcome"] for c in CASES} == set(EXPECTED["default"])
    kinds = {c.get("kind") or ("grounded" if c.get("documents") else "closed_book") for c in CASES}
    assert kinds == {"closed_book", "grounded", "crepe_false_premise", "crepe_sound", "ambiguous"}
    assert len({c["case"] for c in CASES}) == len(CASES)


@pytest.mark.parametrize("which", ["default", "five_level"])
@pytest.mark.parametrize("case", CASES, ids=[c["case"] for c in CASES])
async def test_reward_case(case: dict, which: str) -> None:
    reward_cfg = DEFAULT if which == "default" else FIVE_LEVEL
    server, prompts, _ = _server(_config(reward_cfg), case["judge"])
    res = await server.verify(_request(case))
    exp = case["expect"]

    assert res.outcome == exp["outcome"]
    assert res.label == exp["label"]
    assert res.hedged == exp["hedged"]
    assert res.reward == approx(EXPECTED[which][exp["outcome"]])
    assert len(prompts) == case.get("judge_calls", len(case["judge"]))
    # judge failures are masked from the loss, everything else is not
    assert (res.instance_config or {}).get("mask_sample", False) == (exp["outcome"] == "judge_failed")
    assert res.judge_failed == float(exp["outcome"] == "judge_failed")
    assert res.truncated == (exp["outcome"] == "truncated")
    for s in case.get("judge_prompt_excludes", []):
        assert all(s not in p for p in prompts), s
    if "outside" in case:
        assert res.outside_answer == case["outside"]
    if "ambiguity" in case:
        assert res.ambiguity == case["ambiguity"]
    # rates: exactly one indicator fires (none for judge failures)
    indicators = [res.is_correct, res.is_abstain, res.is_wrong, res.is_truncated, res.judge_failed]
    assert sum(indicators) == 1.0
    assert res.grader_version.startswith(f"harness {VENDORED['harness_version']}")


async def test_judge_sees_final_reply_in_the_harness_prompt() -> None:
    case = next(c for c in CASES if c["case"] == "last_message_only")
    server, prompts, _ = _server(_config(), case["judge"])
    await server.verify(_request(case))
    assert "<reply>\nCorrection: Canberra.\n</reply>" in prompts[0]
    assert "Gold answer: Canberra" in prompts[0]


async def test_grounded_prompt_uses_grounded_rubric() -> None:
    ans = next(c for c in CASES if c["case"] == "grounded_correct")
    unans = next(c for c in CASES if c["case"] == "unans_correct_abstention")
    s1, p1, _ = _server(_config(), ans["judge"])
    s2, p2, _ = _server(_config(), unans["judge"])
    await s1.verify(_request(ans))
    await s2.verify(_request(unans))
    assert "The model was given documents" in p1[0]
    assert "The documents do NOT contain the answer" in p2[0] and "OUTSIDE_ANSWER: <NONE|GIVEN>" in p2[0]


async def test_judge_retries_use_harness_sampling() -> None:
    case = next(c for c in CASES if c["case"] == "judge_unparseable_then_ok")
    client = MagicMock(spec=ServerClient)
    client.post, _ = _judge_http(case["judge"])
    server = FactualityServer(config=_config(), server_client=client)
    await server.verify(_request(case))
    temps = [call.kwargs["json"].temperature for call in client.post.call_args_list]
    max_toks = {call.kwargs["json"].max_output_tokens for call in client.post.call_args_list}
    assert temps == [0.0, 0.7] and max_toks == {256}


# ---------------------------------------------------------------------------
# Reward configuration
# ---------------------------------------------------------------------------


def test_truncation_defaults_to_the_minimum_minus_length_term() -> None:
    assert RewardConfig().resolved_truncated() == -1.0
    assert RewardConfig(confident_wrong=-2.0).resolved_truncated() == -2.0
    assert RewardConfig(length_coef=0.1).resolved_truncated() == approx(-1.1)
    assert RewardConfig(truncated=-5.0).resolved_truncated() == -5.0
    assert RewardConfig(abstain=0.3).resolved_judge_failure() == 0.3


def test_length_term_and_outside_answer_adjustments() -> None:
    cfg = RewardConfig(length_coef=0.2, length_norm_tokens=100, outside_answer_ungraded=-0.5)
    assert reward_for(cfg, "correct", 50, None) == approx(0.9)
    assert reward_for(cfg, "correct", 500, None) == approx(0.8)  # capped at length_coef
    assert reward_for(cfg, "correct", None, None) == approx(1.0)  # no token count: no length term
    assert reward_for(cfg, "correct", 0, "GIVEN") == approx(0.5)
    assert reward_for(cfg, "truncated", 999, None) == approx(-1.2)


async def test_length_term_uses_policy_token_count() -> None:
    case = next(c for c in CASES if c["case"] == "plain_correct")
    server, _, _ = _server(_config(RewardConfig(length_coef=0.5, length_norm_tokens=100)), case["judge"])
    resp = _response([_msg(case["reply"])], output_tokens=40)
    res = await server.verify(_request(case, response=resp))
    assert res.reply_tokens == 40 and res.reward == approx(0.8)


async def test_no_answer_truncation_rule_judges_partial_replies() -> None:
    case = next(c for c in CASES if c["case"] == "truncated_partial")
    server, prompts, _ = _server(
        _config(truncation_rule="no_answer"), ["RATIONALE: cut.\nLABEL: INCORRECT\nCONFIDENCE: CONFIDENT"]
    )
    res = await server.verify(_request(case))
    assert res.outcome == "confident_wrong" and len(prompts) == 1 and not res.truncated
    loop = next(c for c in CASES if c["case"] == "truncated_loop")  # empty after stripping thinking
    server, prompts, _ = _server(_config(truncation_rule="no_answer"), [])
    res = await server.verify(_request(loop))
    assert res.outcome == "truncated" and prompts == []


async def test_eval_set_rows_are_refused() -> None:
    case = dict(next(c for c in CASES if c["case"] == "plain_correct"), source="nq_open")
    server, _, _ = _server(_config(), case["judge"])
    with pytest.raises(ValueError, match="eval set name"):
        await server.verify(_request(case))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_final_reply_and_truncation_detection() -> None:
    assert final_reply(_response([])) == ("", None)
    assert final_reply(_response([_msg("a", 0), _msg("b", 1)]))[0] == "b"
    assert hit_max_tokens(_response([_msg("x")], finish="length"))
    assert not hit_max_tokens(_response([_msg("x")]))


def test_vendored_grader_is_verbatim() -> None:
    root = Path(grader.__file__).parent
    for name, meta in VENDORED["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == meta["sha256"], (
            f"{name} differs from the vendored copy; change the harness and re-run scripts/sync_grader.py"
        )
    assert grader.JUDGE_PROMPT_VERSION == VENDORED["judge_prompt_version"]


# ---------------------------------------------------------------------------
# Windows and guard
# ---------------------------------------------------------------------------


def test_windows_write_rates_and_group_variance(tmp_path) -> None:
    w = RateWindows(StatsConfig(window_rollouts=4, path=str(tmp_path / "s.jsonl")), GuardConfig())
    for outcome, r, g in (("correct", 1, 0), ("confident_wrong", -1, 0), ("abstain", 0, 1), ("abstain", 0, 1)):
        rec = w.add(outcome, r, 10, g)
    assert rec["abstain_rate"] == 0.5 and rec["correct_rate"] == 0.25 and rec["groups"] == 2
    assert rec["zero_variance_group_share"] == 0.5
    assert json.loads((tmp_path / "s.jsonl").read_text())["window"] == 0


def test_judge_failures_are_excluded_from_rates() -> None:
    w = RateWindows(StatsConfig(window_rollouts=2), GuardConfig())
    w.add("judge_failed", 0, None, None)
    rec = w.add("abstain", 0, None, None)
    assert rec["abstain_rate"] == 1.0 and rec["judge_failed"] == 1


def test_guard_trips_on_r1_collapse_before_full_abstention() -> None:
    """R1's per-step abstain shares (correct 1 / abstain 0.5 / wrong 0): 16, 18, 28, 31, 56, 88, 100%."""
    w = RateWindows(StatsConfig(window_rollouts=100), GuardConfig(enabled=True))
    tripped_at = None
    for step, abst in enumerate((16, 18, 28, 31, 56, 88, 100), start=1):
        for i in range(100):
            w.add("abstain" if i < abst else "confident_wrong", 0, None, None)
        if w.tripped:
            tripped_at = step
            break
    assert tripped_at == 5 and "rose by 0.400" in w.tripped


def test_guard_trips_on_absolute_rate() -> None:
    w = RateWindows(StatsConfig(window_rollouts=10), GuardConfig(enabled=True, max_abstain_rate=0.8))
    for i in range(10):
        w.add("abstain" if i < 9 else "correct", 0, None, None)
    assert w.tripped and ">= max_abstain_rate" in w.tripped


async def test_tripped_guard_stops_verify() -> None:
    case = next(c for c in CASES if c["case"] == "idk")
    cfg = _config(stats=StatsConfig(window_rollouts=1), guard=GuardConfig(enabled=True, max_abstain_rate=0.5))
    server, _, _ = _server(cfg, case["judge"] * 2)
    await server.verify(_request(case))  # closes a window at 100% abstain -> trips
    with pytest.raises(GuardTripped):
        await server.verify(_request(case))


async def test_task_index_reaches_group_stats(tmp_path) -> None:
    case = next(c for c in CASES if c["case"] == "plain_correct")
    cfg = _config(stats=StatsConfig(window_rollouts=2, path=str(tmp_path / "s.jsonl")))
    server, _, _ = _server(cfg, case["judge"] * 2)
    for _ in range(2):
        await server.verify(_request(case, _ng_task_index=7))
    rec = json.loads((tmp_path / "s.jsonl").read_text())
    assert rec["groups"] == 1 and rec["zero_variance_group_share"] == 1.0


def test_strip_thinking_variants() -> None:
    from resources_servers.factuality.reply import strip_thinking

    assert strip_thinking("<think>a</think>Canberra") == "<think>a</think>Canberra"  # left to clean_reply
    assert strip_thinking("Sydney? no...</think>Canberra") == "Canberra"  # template opened the block
    assert strip_thinking("<think>Sydney? Canberra? Sydney?") == ""  # cut off mid-thought
    assert strip_thinking("Canberra <think>hmm") == "Canberra "
    assert strip_thinking("Plain answer.") == "Plain answer."
    assert grader.clean_reply(strip_thinking("<think>x</think>A")) == "A"


def test_example_rows_parse() -> None:
    from resources_servers.factuality.app import FactualityRunRequest, item_from_request

    rows = [json.loads(line) for line in (HERE.parent / "data" / "example.jsonl").read_text().splitlines()]
    kinds = []
    for r in rows:
        req = FactualityRunRequest.model_validate(r)
        assert r["agent_ref"]["name"] == "factuality_simple_agent"
        assert req.responses_create_params.input[0].role == "user"
        kinds.append(item_from_request(req).kind)
    assert kinds.count("grounded_unanswerable") == 1 and kinds.count("answerable") == 4
