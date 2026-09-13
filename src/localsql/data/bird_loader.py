"""Loading of the raw birdsql/bird23-train-filtered dataset from Hugging Face.

The dataset repo ships a single JSONL file (no pandas/pyarrow dependency is
required to read it) plus a `train_column_meaning.json` side file with
column descriptions. We download both via `huggingface_hub` and parse them
with the standard library, avoiding heavier dependencies.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from localsql.data.models import RawBirdExample

logger = logging.getLogger(__name__)

DATASET_REPO_ID = "birdsql/bird23-train-filtered"
DATASET_FILE = "data/train-00000-of-00001.jsonl"
COLUMN_MEANING_FILE = "train_column_meaning.json"


def get_dataset_revision(repo_id: str = DATASET_REPO_ID) -> str:
    """Return the current commit SHA of the dataset repo, for provenance."""
    info = HfApi().dataset_info(repo_id)
    return info.sha or "unknown"


def download_raw_jsonl(repo_id: str = DATASET_REPO_ID, filename: str = DATASET_FILE) -> Path:
    """Download (or reuse cached) raw JSONL file, returning its local path."""
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    return Path(path)


def download_column_meaning(
    repo_id: str = DATASET_REPO_ID, filename: str = COLUMN_MEANING_FILE
) -> Path:
    """Download (or reuse cached) column-meaning JSON file, returning its local path."""
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    return Path(path)


def load_raw_examples(jsonl_path: Path) -> list[RawBirdExample]:
    """Parse the raw JSONL file into typed `RawBirdExample` records."""
    examples: list[RawBirdExample] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["row_index"] = row_index
            examples.append(RawBirdExample.model_validate(record))
    return examples


def load_column_meaning(path: Path) -> dict[str, str]:
    """Load the `db_id|table_name|column_name` -> description mapping."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)
