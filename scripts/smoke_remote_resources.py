# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit remote contract smoke; runs a temporary coding task on the configured service.

From the Gym checkout with Gym dependencies installed:
    python -m scripts.smoke_remote_resources

Uses the resource selected by ``GYM_REMOTE_CHECK_TASK`` (or all three when it
is unset). No model or local resource server is launched. This checks HTTP
contracts, rewards, and session cookies; it does not replace a training smoke
with a real policy model.
"""

import asyncio
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import BaseServerConfig, ServerClient, get_global_aiohttp_client, get_response_json


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("math_with_judge", "blackjack", "workspace_swe")


def response(text: str) -> dict:
    result = NeMoGymResponse(
        id="remote-port-smoke",
        created_at=0,
        model="smoke",
        object="response",
        output=[
            {
                "type": "message",
                "id": "msg",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    ).model_dump(exclude_unset=True)
    # vLLM can report these counts as unknown. The three remote
    # services currently require integers, so this exercises the explicit
    # transport compatibility setting in their remote configs.
    result["usage"] = {
        "input_tokens": 1,
        "input_tokens_details": {"cached_tokens": None},
        "output_tokens": 1,
        "output_tokens_details": {"reasoning_tokens": None},
        "total_tokens": 2,
    }
    return result


async def main() -> None:
    selected_task = os.environ.get("GYM_REMOTE_CHECK_TASK", "all")
    if selected_task == "all":
        selected_tasks = TASKS
    elif selected_task in TASKS:
        selected_tasks = (selected_task,)
    else:
        raise ValueError(f"Unknown GYM_REMOTE_CHECK_TASK: {selected_task}")

    config = OmegaConf.merge(
        *[OmegaConf.load(ROOT / f"resources_servers/{name}/configs/{name}_remote.yaml") for name in selected_tasks]
    )
    client = ServerClient(head_server_config=BaseServerConfig(host="localhost", port=11000), global_config_dict=config)

    async def call(name: str, path: str, body: dict, cookies: dict[str, str]) -> dict:
        async with asyncio.timeout(30):
            result = await client.post(name, path, json=body, cookies=cookies)
            payload = await get_response_json(result)
        if result.status != 200:
            raise RuntimeError(f"{name}{path}: HTTP {result.status}: {payload}")
        cookies.update({key: value.value for key, value in result.cookies.items()})
        print(f"{name}{path}: HTTP 200, reward={payload.get('reward', 'n/a')}", flush=True)
        return payload

    try:
        if "math_with_judge" in selected_tasks:
            params = {"input": [{"role": "user", "content": "What is 2+2?"}]}
            for answer, reward in (("4", 1), ("5", 0)):
                result = await call(
                    "math_with_judge",
                    "/verify",
                    {
                        "responses_create_params": params,
                        "question": "What is 2+2?",
                        "expected_answer": "4",
                        "response": response("\\boxed{" + answer + "}"),
                    },
                    {},
                )
                assert result["reward"] == reward, result

        if "blackjack" in selected_tasks:
            body = {"responses_create_params": {"input": [{"role": "user", "content": "Play blackjack."}]}}
            for _ in range(12):
                cookies = {}
                initial = await call("blackjack", "/reset", body, cookies)
                assert cookies and "Your hand:" in initial["observation"], initial
                result = await call(
                    "blackjack", "/step", body | {"response": response("<action>hit</action>")}, cookies
                )
                assert {"reward", "terminated", "truncated"} <= result.keys(), result
                if not result["terminated"]:
                    result = await call(
                        "blackjack", "/step", body | {"response": response("<action>stand</action>")}, cookies
                    )
                    assert result["terminated"], result
                    break
            else:
                raise RuntimeError("Blackjack smoke could not obtain a non-terminal hit in 12 attempts")

        if "workspace_swe" in selected_tasks:
            body = json.loads((ROOT / "resources_servers/workspace_swe/data/example.jsonl").read_text())
            cookies = {}
            initial = await call("workspace_swe", "/reset", body, cookies)
            assert cookies and "WORKSPACE FILES" in initial["observation"], initial
            result = await call("workspace_swe", "/step", body | {"response": response("<cmd>pwd</cmd>")}, cookies)
            assert not result["terminated"] and not result["truncated"], result
            code = '<write path="mathutils.py">def add(a,b):\n    return a+b\n\ndef divide(a,b):\n    return a/b\n</write>'
            result = await call("workspace_swe", "/step", body | {"response": response(code)}, cookies)
            assert not result["terminated"] and not result["truncated"], result
            result = await call("workspace_swe", "/step", body | {"response": response("<submit/>")}, cookies)
            assert result["terminated"] and result["reward"] == 1, result
        print("Remote resource contract smoke passed.")
    finally:
        await get_global_aiohttp_client().close()


if __name__ == "__main__":
    asyncio.run(main())
