"""Send the hand-checked reward cases to the real judge and compare its labels with the hand labels.

The unit tests (tests/test_app.py) check everything deterministic around the judge with canned judge
outputs. This script checks the judge itself on the same replies, through the same grading core
(vendored ``grade_with``, ``strip_thinking``, final reply only), on Clariden:

    E=$(cat $SCRATCH/judge/endpoint)        # judge_server.sbatch writes it when the judge is up
    python resources_servers/factuality/scripts/check_reward_cases_live.py --judge-url $E \
        [--cases more_cases.jsonl] [--out live_check.jsonl]

Cases with hand_label TRUNCATED or SKIP (truncation, judge failures) never reach the judge and are
skipped. ``known_grader_error`` cases are expected disagreements (reported, not counted as new).
Extra cases (e.g. the eval-harness spot-check replies) use the same fields: question, answers,
answerable, documents, kind, meta, gold_readings, reply, hand_label, expect.hedged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # repo root, for `resources_servers.*`

from resources_servers.factuality.factuality_grading import VENDORED, grader  # noqa: E402
from resources_servers.factuality.factuality_grading import data as fdata  # noqa: E402
from resources_servers.factuality.reply import strip_thinking  # noqa: E402


def reply_of(case: dict) -> str:
    if case.get("output"):
        msgs = [o["text"] for o in case["output"] if o["type"] == "message"]
        return msgs[-1] if msgs else ""
    return case["reply"]


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge-url", required=True)
    ap.add_argument("--judge-model", default="qwen-judge")
    ap.add_argument("--cases", type=Path, action="append", default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=a.judge_url, api_key="EMPTY", timeout=120)
    judge = grader.JudgeConfig(client=client, model=a.judge_model)
    files = a.cases or [HERE.parent / "tests" / "reward_cases.jsonl"]
    cases = [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]
    cases = [c for c in cases if c.get("hand_label") not in ("TRUNCATED", "SKIP")]

    async def one(c: dict) -> dict:
        item = fdata.normalise_pool_row(
            {
                "id": c.get("case") or c.get("id"),
                "set": c.get("source") or "rl",
                **{
                    k: c[k]
                    for k in ("question", "answers", "answerable", "documents", "kind", "meta", "gold_readings")
                    if k in c
                },
            }
        )
        res = await grader.grade(item, strip_thinking(reply_of(c)), judge)
        exp_hedged = (c.get("expect") or {}).get("hedged")
        return {
            "case": c.get("case") or c.get("id"),
            "hand_label": c["hand_label"],
            "judge_label": res["label"],
            "label_agrees": res["label"] == c["hand_label"],
            "hand_hedged": exp_hedged,
            "judge_hedged": res["hedged"],
            "hedged_agrees": exp_hedged is None or res["hedged"] == exp_hedged,
            "known_grader_error": bool(c.get("known_grader_error")),
            "rationale": res["rationale"],
            "judge_error": res["judge_error"],
        }

    rows = await asyncio.gather(*(one(c) for c in cases))
    new = [r for r in rows if not r["label_agrees"] and not r["known_grader_error"]]
    known = [r for r in rows if not r["label_agrees"] and r["known_grader_error"]]
    print(
        f"grader {VENDORED['harness_version']} / {VENDORED['judge_prompt_version']}, judge {a.judge_model} @ {a.judge_url}"
    )
    print(
        f"{len(rows)} cases judged: label agreement {sum(r['label_agrees'] for r in rows)}/{len(rows)}, "
        f"hedged agreement {sum(r['hedged_agrees'] for r in rows)}/{len(rows)}, judge failures "
        f"{sum(r['judge_label'] is None for r in rows)}"
    )
    for r in known:
        print(
            f"  expected disagreement (known grader error): {r['case']}: judge {r['judge_label']}, hand {r['hand_label']}"
        )
    for r in new:
        print(f"  NEW disagreement: {r['case']}: judge {r['judge_label']}, hand {r['hand_label']} -- {r['rationale']}")
    for r in rows:
        if not r["hedged_agrees"]:
            print(f"  hedged differs: {r['case']}: judge {r['judge_hedged']}, hand {r['hand_hedged']}")
    if a.out:
        a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    sys.exit(1 if new else 0)


if __name__ == "__main__":
    asyncio.run(main())
