"""Vendored copy of the factuality eval harness's grader (see VENDORED.json and scripts/sync_grader.py).

``grader.py`` and ``data.py`` are verbatim copies; don't edit them here. Change the harness and re-sync.
"""

import json
from pathlib import Path


VENDORED = json.loads((Path(__file__).parent / "VENDORED.json").read_text())
