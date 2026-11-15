# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import LLMResponse, OutputLengthExceededError
from harbor.models.agent.context import AgentContext

from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import NemoGymLLM
from responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym import Terminus2NemoGym


def test_attach_routed_experts_matches_by_rollout_details_not_call_order():
    llm = NemoGymLLM(model_name="test-model", api_base="http://localhost:8000/v1")
    unmatched_route = [[[1]]]
    matched_route = [[[2]]]
    llm._store_routed_experts_for_rollout_details([1], [10], [-0.1], unmatched_route)
    llm._store_routed_experts_for_rollout_details([2], [20], [-0.2], matched_route)

    metrics = SimpleNamespace(
        prompt_token_ids=[2],
        completion_token_ids=[20],
        logprobs=[-0.2],
        extra=None,
    )

    agent = Terminus2NemoGym.__new__(Terminus2NemoGym)
    agent._llm = llm
    agent._trajectory_steps = [
        SimpleNamespace(source="agent", metrics=metrics),
    ]
    agent._dump_trajectory = MagicMock()

    Terminus2NemoGym._attach_routed_experts_to_trajectory(agent)

    assert metrics.extra == {"routed_experts": matched_route}
    agent._dump_trajectory.assert_called_once()


def test_attach_routed_experts_skips_steps_without_rollout_details():
    llm = NemoGymLLM(model_name="test-model", api_base="http://localhost:8000/v1")
    llm._store_routed_experts_for_rollout_details([1], [10], [-0.1], [[[1]]])

    metrics = SimpleNamespace(extra={})

    agent = Terminus2NemoGym.__new__(Terminus2NemoGym)
    agent._llm = llm
    agent._trajectory_steps = [
        SimpleNamespace(source="agent", metrics=None),
        SimpleNamespace(source="agent", metrics=metrics),
    ]
    agent._dump_trajectory = MagicMock()

    Terminus2NemoGym._attach_routed_experts_to_trajectory(agent)

    assert metrics.extra == {}
    agent._dump_trajectory.assert_not_called()


# Jobs 5350472 and 5355345: a reply that hit max_tokens with nothing salvageable makes Harbor's Terminus2._query_llm
# append it plus "ERROR!! NONE of the actions ..." and call itself again, with no depth limit. Three trials per job
# looped ~320 times until Python's recursion limit and came back without a reward.
VALID_REPLY = '{"analysis": "a", "plan": "p", "commands": [], "task_complete": false}'


class _Runaway(BaseException):
    """Ends a test whose agent never stops asking (BaseException: Harbor's @retry does not retry it)."""


class _FakeChat:
    """Chat whose replies hit max_tokens with nothing salvageable, except calls listed in ``ok`` / ``salvageable``."""

    def __init__(self, ok=(), salvageable=(), runaway_after=50):
        self.messages = []
        self.calls = 0
        self.ok = set(ok)
        self.salvageable = set(salvageable)
        self.runaway_after = runaway_after

    async def chat(self, prompt, **kwargs):
        self.calls += 1
        if self.calls > self.runaway_after:
            raise _Runaway(f"the agent asked {self.calls} times in a row")
        if self.calls in self.ok:
            return LLMResponse(content=VALID_REPLY)
        text = "SALVAGEABLE" if self.calls in self.salvageable else '{"analysis": "a reply that never ends'
        raise OutputLengthExceededError("hit max_tokens", truncated_response=text)

    def reset_response_chain(self):
        pass


class _SalvagingParser:
    """Stands in for Harbor's XML parser, the only one with salvage_truncated_response."""

    def salvage_truncated_response(self, truncated_response):
        return (LLMResponse(content=VALID_REPLY) if truncated_response == "SALVAGEABLE" else None), False

    def parse_response(self, response):
        return SimpleNamespace(warning=None)


def _agent(tmp_path, **kwargs) -> Terminus2NemoGym:
    llm = NemoGymLLM(model_name="test-model", api_base="http://localhost:8000/v1")
    return Terminus2NemoGym(logs_dir=tmp_path, model_name="test-model", llm=llm, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs, calls", [({}, 8), ({"max_consecutive_truncations": 3}, 3)])
async def test_unsalvageable_truncations_stop_after_max_consecutive_truncations(tmp_path, kwargs, calls) -> None:
    agent = _agent(tmp_path, **kwargs)
    chat = _FakeChat()

    with pytest.raises(Exception, match=f"{calls} consecutive replies") as caught:
        await agent._query_llm(chat, "do the task")

    assert type(caught.value).__name__ == "TruncationLoopError"
    assert chat.calls == calls
    # the next turn starts from zero again
    chat.calls, chat.ok = 0, {1}
    assert (await agent._query_llm(chat, "next")).content == VALID_REPLY


@pytest.mark.asyncio
async def test_a_reply_or_a_salvaged_reply_resets_the_count(tmp_path) -> None:
    agent = _agent(tmp_path, max_consecutive_truncations=3)
    agent._parser = _SalvagingParser()
    # turn 1: 2 truncations, then a reply; turn 2: 2 truncations, then a salvageable one; turn 3: 3 truncations
    chat = _FakeChat(ok={3}, salvageable={6})

    assert (await agent._query_llm(chat, "turn 1")).content == VALID_REPLY
    assert (await agent._query_llm(chat, "turn 2")).content == VALID_REPLY
    assert chat.calls == 6
    with pytest.raises(Exception, match="3 consecutive replies"):
        await agent._query_llm(chat, "turn 3")
    assert chat.calls == 9


@pytest.mark.asyncio
async def test_none_keeps_harbors_unbounded_retries(tmp_path) -> None:
    agent = _agent(tmp_path, max_consecutive_truncations=None)
    with pytest.raises(_Runaway):
        await agent._query_llm(_FakeChat(runaway_after=20), "do the task")


@pytest.mark.asyncio
async def test_the_episode_ends_normally_so_the_verifier_runs(tmp_path, monkeypatch) -> None:
    # Harbor's trial runs the verifier only when agent.run returns (or raises AgentTimeoutError /
    # NonZeroAgentExitCodeError); run() turns TruncationLoopError into a normal return, as on max_turns.
    chat = _FakeChat()

    async def one_turn(self, instruction, environment, context):
        await self._query_llm(chat, instruction)

    monkeypatch.setattr(Terminus2, "run", one_turn)
    agent = _agent(tmp_path, max_consecutive_truncations=2)

    await agent.run("do the task", environment=MagicMock(), context=AgentContext())

    assert chat.calls == 2
    assert (tmp_path / "agent_error_flags.json").exists()


def test_preflight_checks_max_consecutive_truncations() -> None:
    Terminus2NemoGym.preflight(kwargs={"max_consecutive_truncations": 5}, env={})
    with pytest.raises(ValueError):
        Terminus2NemoGym.preflight(kwargs={"max_consecutive_truncations": 0}, env={})
