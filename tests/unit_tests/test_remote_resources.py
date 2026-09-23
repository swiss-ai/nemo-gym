# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf
from pydantic import BaseModel

import nemo_gym.cli.env
import nemo_gym.server_utils
from nemo_gym.cli.env import RunHelper
from nemo_gym.server_utils import (
    NEMO_GYM_MODEL_SERVER_BASE_URL_ENV_VAR_NAME,
    NEMO_GYM_MODEL_SERVER_NAME_ENV_VAR_NAME,
    BaseServerConfig,
    ServerClient,
)


def _client(*, url="https://gym.example/math/", legacy=False, server_type="resources_servers"):
    config = {"entrypoint": "app.py", "host": "localhost", "port": 9000}
    if url is not None:
        config["url"] = url
    if legacy:
        config["legacy_response_usage_details_as_zero"] = True
    return ServerClient(
        head_server_config=BaseServerConfig(host="localhost", port=8000),
        global_config_dict=OmegaConf.create({"remote": {server_type: {"math": config}}}),
    )


def _unknown_usage():
    return {
        "response": {
            "usage": {
                "input_tokens_details": {"cached_tokens": None, "other": 7},
                "output_tokens_details": {"reasoning_tokens": None, "other": 11},
            }
        }
    }


@pytest.mark.parametrize("url", ["https://gym.example/math", "https://gym.example/math/"])
async def test_remote_post_preserves_gateway_path_and_does_not_replay_503(monkeypatch, url):
    client = _client(url=url)
    response = MagicMock(status=503, cookies={"session": "updated"})
    request = AsyncMock(return_value=response)
    monkeypatch.setattr(nemo_gym.server_utils, "request", request)

    result = await client.post("remote", "/step", json={"action": "hit"}, cookies={"session": "original"})

    assert result is response
    request.assert_awaited_once_with(
        method="POST",
        url="https://gym.example/math/step",
        _internal=True,
        json={"action": "hit"},
        cookies={"session": "original"},
    )


async def test_local_server_keeps_host_and_port_routing(monkeypatch):
    request = AsyncMock()
    monkeypatch.setattr(nemo_gym.server_utils, "request", request)

    await _client(url=None).get("remote", "/health")

    assert request.await_args.kwargs["url"] == "http://localhost:9000/health"


async def test_remote_url_does_not_override_rollout_model_routing(monkeypatch):
    request = AsyncMock()
    monkeypatch.setattr(nemo_gym.server_utils, "request", request)
    monkeypatch.setenv(NEMO_GYM_MODEL_SERVER_NAME_ENV_VAR_NAME, "remote")
    monkeypatch.setenv(NEMO_GYM_MODEL_SERVER_BASE_URL_ENV_VAR_NAME, "http://model:9100/ng-rollout/7-0/")

    await _client(server_type="responses_api_models").post("remote", "/v1/responses", json={"input": "hello"})

    assert request.await_args.kwargs["url"] == "http://model:9100/ng-rollout/7-0/v1/responses"


@pytest.mark.parametrize("url_path", ["/verify", "/step", "/ng-rollout/7-0/step?mode=train"])
@pytest.mark.parametrize("as_model", [False, True])
async def test_legacy_usage_changes_only_wire_copy(monkeypatch, url_path, as_model):
    class Payload(BaseModel):
        response: dict

    payload = _unknown_usage()
    original = deepcopy(payload)
    if as_model:
        payload = Payload.model_validate(payload)
    request = AsyncMock()
    monkeypatch.setattr(nemo_gym.server_utils, "request", request)

    await _client(legacy=True).post("remote", url_path, json=payload)

    assert (payload.model_dump() if as_model else payload) == original
    wire_usage = request.await_args.kwargs["json"]["response"]["usage"]
    assert wire_usage["input_tokens_details"] == {"cached_tokens": 0, "other": 7}
    assert wire_usage["output_tokens_details"] == {"reasoning_tokens": 0, "other": 11}


@pytest.mark.parametrize(
    "method,url_path,legacy", [("POST", "/verify", False), ("POST", "/reset", True), ("GET", "/verify", True)]
)
async def test_usage_compatibility_is_opt_in_and_scoped_to_verification(monkeypatch, method, url_path, legacy):
    payload = _unknown_usage()
    request = AsyncMock()
    monkeypatch.setattr(nemo_gym.server_utils, "request", request)

    await _client(legacy=legacy).request("remote", url_path, method, json=payload)

    assert request.await_args.kwargs["json"] is payload


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"input_tokens_details": {}},
        {"input_tokens_details": {"cached_tokens": 5}, "output_tokens_details": {"reasoning_tokens": 9}},
    ],
)
async def test_legacy_usage_preserves_known_counts_and_missing_fields(monkeypatch, usage):
    payload = {"response": {"usage": usage}}
    original = deepcopy(payload)
    request = AsyncMock()
    monkeypatch.setattr(nemo_gym.server_utils, "request", request)

    await _client(legacy=True).post("remote", "/verify", json=payload)

    assert request.await_args.kwargs["json"] == original
    assert payload == original


@pytest.mark.parametrize(
    "url,server_type", [(None, "resources_servers"), ("https://model.example", "responses_api_models")]
)
async def test_legacy_usage_rejects_non_remote_resource_configuration(monkeypatch, url, server_type):
    request = AsyncMock()
    monkeypatch.setattr(nemo_gym.server_utils, "request", request)

    with pytest.raises(ValueError, match="requires a remote resources_servers configuration"):
        await _client(url=url, legacy=True, server_type=server_type).post("remote", "/verify", json=_unknown_usage())

    request.assert_not_awaited()


@pytest.mark.parametrize("prefetch_only", [False, True])
def test_remote_entrypoint_does_not_create_venv_or_process(monkeypatch, prefetch_only):
    config = OmegaConf.create(
        {
            "dry_run": True,
            "remote": {
                "resources_servers": {
                    "math": {"entrypoint": "app.py", "domain": "math", "url": "https://gym.example/math"}
                }
            },
        }
    )
    monkeypatch.setattr(nemo_gym.cli.env, "get_global_config_dict", lambda **kwargs: config)
    monkeypatch.setattr(nemo_gym.cli.env, "initialize_ray", MagicMock())
    monkeypatch.setattr(nemo_gym.cli.env, "init_telemetry", MagicMock())
    monkeypatch.setattr(nemo_gym.cli.env, "configure_telemetry_env", MagicMock())
    monkeypatch.setattr(
        nemo_gym.cli.env.HeadServer, "run_webserver", MagicMock(return_value=(None, None, MagicMock()))
    )
    client = MagicMock()
    client.return_value.poll_for_status.return_value = "success"
    monkeypatch.setattr(nemo_gym.cli.env, "ServerClient", client)
    setup = MagicMock()
    launch = MagicMock()
    monkeypatch.setattr(nemo_gym.cli.env, "setup_env_command", setup)
    monkeypatch.setattr(nemo_gym.cli.env, "run_command", launch)

    if prefetch_only:
        nemo_gym.cli.env.prefetch(None)
    else:
        runner = RunHelper()
        runner.start(None)
        assert runner._processes == {}

    setup.assert_not_called()
    launch.assert_not_called()
