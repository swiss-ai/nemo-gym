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
"""mini-swe-agent model class for a NeMo Gym model server. Runs INSIDE the task sandbox, in
mini-swe-agent's own Python (``mini --model-class nemo_gym_mini.NemoGymModel``); Gym never imports it.

MiniSweAgentNemoGym writes this file into the sandbox and passes its settings as JSON in
``NEMO_GYM_MINI``:

- ``model_kwargs`` replace those of mini's config (its default ``mini.yaml`` has ``temperature: 0.0``):
  the rollout's ``api_base``/``api_key`` and the on-policy sampling parameters.
- ``max_calls`` > 0 ends the run with mini's ``LimitsExceeded`` (a normal end, the verifier still runs)
  before that many model calls are exceeded; mini's own step limit lives in the agent config.

Messages are sent with ``role`` and ``content`` only: mini stores each assistant reply with an
``extra`` field (the raw response), and Gym's model server rejects unknown message fields with 422
(``extra_forbidden``), which failed every second call (job 5253837).

The cost lookup is disabled: the Gym model server's model is not in litellm's price list, and
mini-swe-agent re-raises every cost-calculation error, which would fail every call.
"""

import json
import os

import litellm
from minisweagent.agents.default import LimitsExceeded
from minisweagent.models.litellm_model import LitellmModel


SETTINGS = json.loads(os.environ.get("NEMO_GYM_MINI") or "{}")

litellm.cost_calculator.completion_cost = lambda *args, **kwargs: 0.0


class NemoGymModel(LitellmModel):
    def __init__(self, **kwargs):
        model_kwargs = dict(kwargs.pop("model_kwargs", None) or {})
        model_kwargs.update(SETTINGS.get("model_kwargs") or {})
        super().__init__(model_kwargs=model_kwargs, **kwargs)
        self.max_calls = int(SETTINGS.get("max_calls") or 0)

    def query(self, messages, **kwargs):
        if 0 < self.max_calls <= self.n_calls:
            raise LimitsExceeded(f"Model call limit reached ({self.max_calls}).")
        sent = [{"role": m["role"], "content": m["content"]} for m in messages]
        return super().query(sent, **kwargs)
