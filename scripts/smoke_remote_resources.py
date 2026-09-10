# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit remote contract smoke; runs a temporary coding task on the configured service.

From the Gym checkout with Gym dependencies installed:
    python -m scripts.smoke_remote_resources

Uses the three *_remote.yaml configurations. No model or local resource server
is launched. This checks HTTP contracts, rewards, and session cookies; it does
not replace a training smoke with a real policy model.
"""

import asyncio
import json
from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import BaseServerConfig, ServerClient, get_global_aiohttp_client, get_response_json


ROOT = Path(__file__).resolve().parents[1]


def response(text: str) -> dict:
    return NeMoGymResponse(
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


async def main() -> None:
    config = OmegaConf.merge(
        *[
            OmegaConf.load(ROOT / f"resources_servers/{name}/configs/{name}_remote.yaml")
            for name in ("math_with_judge", "blackjack", "workspace_swe")
        ]
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

        body = {"responses_create_params": {"input": [{"role": "user", "content": "Play blackjack."}]}}
        cookies = {}
        initial = await call("blackjack", "/reset", body, cookies)
        assert cookies and "Your hand:" in initial["observation"], initial
        result = await call("blackjack", "/step", body | {"response": response("<action>stand</action>")}, cookies)
        assert result["terminated"] and result["info"]["player"] != "[]", result
        assert f"Your hand: {result['info']['player']} =" in initial["observation"], result

        body = json.loads((ROOT / "resources_servers/workspace_swe/data/example.jsonl").read_text())
        cookies = {}
        initial = await call("workspace_swe", "/reset", body, cookies)
        assert cookies and "WORKSPACE FILES" in initial["observation"], initial
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
