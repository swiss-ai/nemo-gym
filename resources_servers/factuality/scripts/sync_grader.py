"""Vendor the eval harness's grader into this resources server, verbatim, with provenance.

The reward of the ``factuality`` environment is the factuality eval grader (swiss-ai/factuality-sft-data,
``src/evals/grader.py`` and the ``Item`` loader it needs). The two files are copied unchanged into
``factuality_grading/`` and ``factuality_grading/VENDORED.json`` records where they came from: the
source commit, whether the source tree had uncommitted changes, the harness version, the judge prompt
version and a sha256 per file. ``tests/test_app.py`` fails if a vendored file no longer matches its
recorded hash, so the copy can't drift silently; to change the grader, change it in the harness and
re-run this script.

    python resources_servers/factuality/scripts/sync_grader.py --src ~/scratch/factuality-sft-data

Run it from a clean, committed harness checkout before a training run, so ``commit`` is real.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


FILES = ("src/evals/grader.py", "src/evals/data.py")
DEST = Path(__file__).resolve().parents[1] / "factuality_grading"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(src: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(src), *args], capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="factuality-sft-data checkout")
    a = ap.parse_args()
    src = a.src.resolve()
    commit = git(src, "rev-parse", "HEAD")
    dirty = bool(git(src, "status", "--porcelain", "--", *FILES))
    init = (src / "src/evals/__init__.py").read_text()
    m = re.search(r'^HARNESS_VERSION = "([^"]+)"', init, re.M)
    if not m:
        raise SystemExit("src/evals/__init__.py has no HARNESS_VERSION; is this harness >= 1.9?")
    DEST.mkdir(exist_ok=True)
    files = {}
    for rel in FILES:
        out = DEST / Path(rel).name
        shutil.copyfile(src / rel, out)
        files[out.name] = {"source_path": rel, "sha256": sha256(out)}
    # The judge prompt version is computed by the vendored grader itself (hash of its templates).
    ns: dict = {}
    grader_src = (DEST / "grader.py").read_text().replace("from .data import Item", "Item = object")
    exec(compile(grader_src, "grader.py", "exec"), ns)  # noqa: S102 - our own vendored file
    manifest = {
        "source_repo": "swiss-ai/factuality-sft-data",
        "source_commit": commit,
        "source_dirty": dirty,
        "harness_version": m.group(1),
        "judge_prompt_version": ns["JUDGE_PROMPT_VERSION"],
        "files": files,
    }
    (DEST / "VENDORED.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if dirty:
        print("WARNING: the source files have uncommitted changes; commit them and re-run before training.")


if __name__ == "__main__":
    main()
