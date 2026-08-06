"""
Persistence for evaluation batches.

Kept apart from evaluation.py so it carries no dependency on the dataset, the
vector store or the API client: writing and reading a results file is pure I/O
and can be tested without any of them.

A batch is stored as {"meta": ..., "results": [...]}. The meta block exists so a
file stays interpretable on its own — which model produced it, under what
sampling settings, from which seed, and which cases dropped out along the way.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"


def results_filename(moment: datetime | None = None) -> str:
    """
    Timestamped filename for a batch.

    The ISO timestamp lives in the payload where it is readable; the filename
    uses a compact form because a colon is not a legal filename character on
    Windows, and the project is developed there.
    """
    moment = moment or datetime.now(timezone.utc)
    return f"eval_{moment:%Y%m%d_%H%M%S}.json"


def save_results(
    results: list[dict],
    meta: dict,
    output_dir=DEFAULT_RESULTS_DIR,
    moment: datetime | None = None,
) -> Path:
    """Write a batch to a timestamped JSON file and return its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / results_filename(moment)
    path.write_text(
        json.dumps({"meta": meta, "results": results}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def load_results(path) -> tuple[list[dict], dict]:
    """
    Read a batch back as (results, meta).

    Fails with a clear message on a file that predates this format rather than
    raising a bare KeyError deep inside a metric.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "results" not in payload:
        raise ValueError(
            f"{path} is not an evaluation results file: expected a JSON object "
            f"with 'meta' and 'results' keys"
        )
    return payload["results"], payload.get("meta", {})
