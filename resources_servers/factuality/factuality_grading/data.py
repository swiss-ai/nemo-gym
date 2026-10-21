"""Load the frozen eval suite (factuality-eval-v1) and normalise its quirks.

Quirks handled here (see the 2026-10-13 inspection):
- ``documents`` is a JSON string for SQuAD and MuSiQue; ``meta`` is a JSON string for some sets.
- ``answerable`` is null for closed-book sets (treated as answerable).
- MuSiQue unanswerable twins still carry the twin's answer; the grader ignores it.
- CREPE has no reference answers; AmbigNQ splits into multipleQAs / singleAnswer / mixed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ROOT = "/capstor/store/cscs/swissai/infra01/datasets/factuality-eval-v1"

CLOSED_BOOK = ("nq_open", "triviaqa_dev", "popqa", "simpleqa_verified", "aa_omniscience_public")
GROUNDED = ("squad2_dev", "musique_dev")
BEHAVIOUR = ("ambignq_dev", "crepe_test")
ALL_SETS = CLOSED_BOOK + GROUNDED + BEHAVIOUR


@dataclass
class Item:
    id: str
    set: str
    half: str
    question: str
    answers: list[str]
    documents: list[dict]
    meta: dict
    answerable: bool
    kind: str            # rubric: answerable | grounded_unanswerable | crepe_false_premise | crepe_sound | ambiguous
    group: str           # reporting group within a set
    gold_readings: list[dict] = field(default_factory=list)   # AmbigNQ: [{"question": str, "answers": [str]}]


def _maybe_json(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
        return json.loads(value)
    return value


def _ambig_readings(meta: dict) -> list[dict]:
    """Collect (question, answers) readings from the multipleQAs annotations."""
    ann = meta.get("annotations") or {}
    readings, seen = [], set()
    for typ, pairs in zip(ann.get("type", []), ann.get("qaPairs", [])):
        if typ != "multipleQAs":
            continue
        for q, a in zip(pairs.get("question", []), pairs.get("answer", [])):
            answers = [x for x in (a or []) if x]
            key = (q, tuple(answers))
            if q and answers and key not in seen:
                seen.add(key)
                readings.append({"question": q, "answers": answers})
    return readings


def normalise(raw: dict) -> Item:
    name = raw["set"]
    meta = _maybe_json(raw.get("meta"), {}) or {}
    docs = _maybe_json(raw.get("documents"), []) or []
    answers = [a for a in (raw.get("answers") or []) if isinstance(a, str) and a.strip()]
    answerable = raw.get("answerable")
    answerable = True if answerable is None else bool(answerable)
    readings: list[dict] = []

    if name in GROUNDED:
        kind = "answerable" if answerable else "grounded_unanswerable"
        group = "answerable" if answerable else "unanswerable"
    elif name == "crepe_test":
        fp = bool(meta.get("false_presupposition"))
        kind, group = ("crepe_false_premise", "false_premise") if fp else ("crepe_sound", "sound")
    elif name == "ambignq_dev":
        types = set((meta.get("annotations") or {}).get("type", []))
        if types == {"singleAnswer"}:
            kind, group = "answerable", "single"
        else:
            readings = _ambig_readings(meta)
            group = "multi" if types == {"multipleQAs"} else "mixed"
            kind = "ambiguous" if readings else "answerable"
    else:
        kind, group = "answerable", "all"

    return Item(id=str(raw["id"]), set=name, half=raw["half"], question=raw["question"], answers=answers,
                documents=docs, meta=meta, answerable=answerable, kind=kind, group=group, gold_readings=readings)


POOL_HALF = "pool"
KINDS = ("answerable", "grounded_unanswerable", "crepe_false_premise", "crepe_sound", "ambiguous")


def normalise_pool_row(raw: dict) -> Item:
    """A row of a non-eval question pool (RL pool, SFT pool) -> Item.

    Required: ``id``, ``set`` (the pool's source name, e.g. ``nq_rl``), ``question``, ``answers``.
    Optional: ``answerable`` (default true), ``documents`` (list or JSON string: grounded item),
    ``meta``, ``kind`` (rubric; default answerable, or grounded_unanswerable for an unanswerable
    grounded item), ``group`` (reporting group; default answerable/unanswerable for grounded
    items, ``all`` otherwise), ``gold_readings`` (ambiguous items).
    Eval set names are refused, so a pool can never be mistaken for (or mixed with) the frozen suite."""
    name = raw.get("set")
    if not name:
        raise ValueError(f"pool row {raw.get('id')!r} has no 'set'")
    if name in ALL_SETS:
        raise ValueError(f"pool row {raw.get('id')!r} uses eval set name {name!r}; pools must use their own names")
    meta = _maybe_json(raw.get("meta"), {}) or {}
    docs = _maybe_json(raw.get("documents"), []) or []
    answers = [a for a in (raw.get("answers") or []) if isinstance(a, str) and a.strip()]
    answerable = raw.get("answerable")
    answerable = True if answerable is None else bool(answerable)
    kind = raw.get("kind")
    if kind is None:
        if answerable:
            kind = "answerable"
        elif docs:
            kind = "grounded_unanswerable"
        else:
            raise ValueError(f"pool row {raw.get('id')!r}: unanswerable closed-book items need an explicit 'kind'")
    if kind not in KINDS:
        raise ValueError(f"pool row {raw.get('id')!r}: unknown kind {kind!r}")
    if kind == "answerable" and not answers:
        raise ValueError(f"pool row {raw.get('id')!r}: answerable item without answers")
    group = raw.get("group") or (("answerable" if answerable else "unanswerable") if docs else "all")
    return Item(id=str(raw["id"]), set=name, half=POOL_HALF, question=raw["question"], answers=answers,
                documents=docs, meta=meta, answerable=answerable, kind=kind, group=group,
                gold_readings=list(raw.get("gold_readings") or []))


def load_pool(path: str | Path, limit: int | None = None) -> list[Item]:
    """Load a JSONL question pool (see normalise_pool_row). ``limit`` keeps the first N rows per set."""
    path = Path(path)
    if Path(DEFAULT_ROOT) in path.resolve().parents:
        raise SystemExit(f"{path} is inside the frozen eval suite; probe mode reads pools, not eval files")
    items: list[Item] = []
    per_set: dict[str, int] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            it = normalise_pool_row(json.loads(line))
            if limit and per_set.get(it.set, 0) >= limit:
                continue
            per_set[it.set] = per_set.get(it.set, 0) + 1
            items.append(it)
    for name, n in sorted(per_set.items()):
        print(f"[data] pool {path.name} / {name}: {n} items")
    ids = [(it.set, it.id) for it in items]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate (set, id) pairs in the pool")
    return items


def load(root: str | Path, sets: list[str], half: str, allow_final: bool = False,
         limit: int | None = None) -> list[Item]:
    """Load the chosen sets and half. ``final`` needs ``allow_final`` (milestone runs only)."""
    if half not in ("dev", "final"):
        raise ValueError(f"half must be dev or final, not {half!r}")
    if half == "final" and not allow_final:
        raise SystemExit("Refusing to read final halves without --allow-final (milestone runs only).")
    root = Path(root) / "eval"
    items: list[Item] = []
    for name in sets:
        if name not in ALL_SETS:
            raise SystemExit(f"Unknown set {name!r}; known: {', '.join(ALL_SETS)}")
        path = root / name / f"{half}.jsonl"
        if not path.exists():
            print(f"[data] {name}: no {half} half, skipped")
            continue
        rows = []
        with path.open() as f:
            for line in f:
                if line.strip():
                    rows.append(normalise(json.loads(line)))
                if limit and len(rows) >= limit:
                    break
        print(f"[data] {name}/{half}: {len(rows)} items")
        items.extend(rows)
    ids = [(it.set, it.id) for it in items]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate (set, id) pairs in the loaded items")
    return items
