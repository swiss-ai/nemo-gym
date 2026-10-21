# Factuality environment

Closed-book and grounded factual QA with abstention (swiss-ai #1011). The reward is the factuality
eval grader: rubric prompt → LLM judge → strict parser → label (CORRECT / INCORRECT / NOT_ATTEMPTED)
plus a hedged flag, mapped to a reward by configuration.

## Grading core

`factuality_grading/` is a verbatim copy of `swiss-ai/factuality-sft-data` `src/evals/{grader,data}.py`
(harness ≥ 1.9); `VENDORED.json` records the source commit, harness and judge-prompt versions and file
hashes, and a test fails if the copy is edited. To change the grader, change the harness, commit, and run:

```bash
python resources_servers/factuality/scripts/sync_grader.py --src ~/scratch/factuality-sft-data
```

## verify()

| Step | Behaviour |
|---|---|
| Reply | Last assistant message only; reasoning items, earlier messages and `<think>` text are never graded |
| Truncation | `incomplete_details.reason == max_output_tokens` → `reward.truncated` (default: the minimum), no judge call. `truncation_rule: no_answer` judges replies that started an answer (the harness's eval rule) |
| Judge | `judge_model` (`responses_api_models`), temperature 0, 256 tokens; retried twice at 0.7 on errors or unparseable output |
| Judge failure | Masked: `instance_config.mask_sample = true` (NeMo RL drops it from the loss); reward = `reward.judge_failure` (default: the abstain value, neutral in the group baseline). Never scored as wrong |
| Reward | `reward.{correct, hedged_correct, abstain, hedged_wrong, confident_wrong}`; optional `length_coef`; optional `outside_answer_*` adjustments for grounded-unanswerable replies |
| Diagnostics | `outcome`, `label`, `hedged`, `truncated`, `rubric`, `reply`, `judge_output`, `judge_rationale`, `judge_error`, `outside_answer`, `ambiguity`, `grader_version`, and 0/1 indicators (`is_correct`, `is_abstain`, `is_wrong`, `is_truncated`, `judge_failed`, …) that NeMo RL averages per step |
| Windows | `stats.window_rollouts` > 0: one JSONL line per window (rates, mean reward, zero-variance group share) to `stats.path` and stderr |
| Guard | `guard.enabled`: after a window with abstain rate ≥ `max_abstain_rate`, or ≥ `max_abstain_jump` above the lowest earlier window, every further `verify` raises, so the training job stops |

Abstaining is optimal on a question exactly when P(correct) < (abstain − wrong) / (correct − wrong); with
the defaults (+1 / 0 / −1) that is 0.5. The prompt mix decides how often that happens (see
`src/rl/prompt_set.py` in factuality-sft-data).

## Data rows

```json
{"id": "...", "source": "nq_rl", "question": "...", "answers": ["..."], "answerable": true,
 "agent_ref": {"type": "responses_api_agents", "name": "factuality_simple_agent"},
 "responses_create_params": {"input": [{"role": "user", "content": "..."}], "tools": [], "parallel_tool_calls": false}}
```

Grounded rows add `documents` (`title`, `content`) and render them into the user message as in SFT v0.1;
other rubrics set `kind` (`crepe_false_premise`, `crepe_sound`, `ambiguous`) with `meta` / `gold_readings`.
Eval set names are refused as `source`. Rows are built by `src/rl/rows.py` in factuality-sft-data.

## Judge

`configs/factuality.yaml` defines `judge_model` (a `vllm_model` with thinking off) from
`judge_base_url`, `judge_api_key` and `judge_model_name`. A training config that defines its own
`judge_model` block (as NeMo RL's `env.nemo_gym` does) overrides it. Never point the judge at the
policy server: NeMo RL's vLLM rejects sampling settings that differ from training.

## Tests

```bash
pytest resources_servers/factuality/tests          # 52 hand-checked reward cases x 2 reward configs, guard, windows
python resources_servers/factuality/scripts/check_reward_cases_live.py --judge-url $(cat $SCRATCH/judge/endpoint)
```
