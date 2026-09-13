"""Loading of official BIRD Mini-Dev resources (original 500, SQLite).

Two independent official sources are combined:

1. `birdsql/bird_mini_dev` on Hugging Face (the `mini_dev_sqlite` split) --
   canonical per the upstream README ("should be treated as the canonical
   version"). Question/evidence/gold-SQL/difficulty rows only; no database
   files.
2. The official `minidev.zip` archive (bird-bench.oss-cn-beijing.aliyuncs.com,
   the same host Phase 1 used for the BIRD train archive) -- contains the 11
   SQLite database files, the Spider-format `dev_tables.json` schema file,
   per-database `database_description/*.csv` column descriptions, and the
   evaluator-ready `mini_dev_sqlite_gold.sql`. The archive is ~800MB total
   (it also bundles MySQL/PostgreSQL-only assets); only the entries we need
   (~346MB compressed) are extracted via HTTP range requests, never the
   whole archive.

Neither source is the newer Mini-Dev V2 / LiveSQLBench addition
(`live_sql_bench_sqlite/`), which is explicitly excluded (see
`configs/benchmark.yaml`).
"""

from __future__ import annotations

import csv
import json
import logging
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

logger = logging.getLogger(__name__)

QUESTIONS_REPO_ID = "birdsql/bird_mini_dev"
QUESTIONS_FILE = "data/mini_dev_sqlite-00000-of-00001.json"
ARCHIVE_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
ARCHIVE_ROOT = "minidev/MINIDEV/"


def get_questions_revision(repo_id: str = QUESTIONS_REPO_ID) -> str:
    return HfApi().dataset_info(repo_id).sha or "unknown"


def download_questions_json(repo_id: str = QUESTIONS_REPO_ID, filename: str = QUESTIONS_FILE) -> Path:
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))


def load_questions(path: Path) -> list[dict]:
    """Load the 500 mini_dev_sqlite rows, preserving official row order.

    Row order matters: the official gold SQL file and evaluator are
    positionally aligned with this order.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _open_remote_zip() -> zipfile.ZipFile:
    import fsspec

    fs = fsspec.filesystem("https")
    return zipfile.ZipFile(fs.open(ARCHIVE_URL, "rb"))


def extract_archive_resources(
    raw_dir: Path,
    databases_dir: Path,
    db_ids: list[str] | None = None,
    force: bool = False,
) -> dict:
    """Extract only the needed entries from the remote minidev.zip.

    Idempotent: if a target file already exists it is skipped (unless
    `force=True`), so reruns do not re-download the ~346MB payload.
    Returns a summary of what was (re)written vs. reused.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    databases_dir.mkdir(parents=True, exist_ok=True)

    summary = {"written": [], "reused": []}

    schema_dst = raw_dir / "dev_tables.json"
    gold_dst = raw_dir / "mini_dev_sqlite_gold.sql"
    cross_check_dst = raw_dir / "mini_dev_sqlite.json"

    need_top_level = force or not (schema_dst.exists() and gold_dst.exists() and cross_check_dst.exists())

    if db_ids is None:
        # Discover db_ids from the schema file if already cached, else defer
        # to the caller after extracting the schema first.
        db_ids = []

    need_dbs = force or any(
        not (databases_dir / db_id / f"{db_id}.sqlite").exists() for db_id in db_ids
    )

    if not (need_top_level or need_dbs):
        summary["reused"] = ["dev_tables.json", "mini_dev_sqlite_gold.sql", "mini_dev_sqlite.json"] + [
            f"{db}.sqlite" for db in db_ids
        ]
        return summary

    with _open_remote_zip() as zf:
        for entry_name, dst in (
            ("dev_tables.json", schema_dst),
            ("mini_dev_sqlite_gold.sql", gold_dst),
            ("mini_dev_sqlite.json", cross_check_dst),
        ):
            if dst.exists() and not force:
                summary["reused"].append(entry_name)
                continue
            with zf.open(ARCHIVE_ROOT + entry_name) as src, dst.open("wb") as out:
                out.write(src.read())
            summary["written"].append(entry_name)

        if not db_ids:
            schema = json.loads(schema_dst.read_text(encoding="utf-8"))
            db_ids = [entry["db_id"] for entry in schema]

        for db_id in db_ids:
            db_dst = databases_dir / db_id / f"{db_id}.sqlite"
            if db_dst.exists() and not force:
                summary["reused"].append(f"{db_id}.sqlite")
            else:
                db_dst.parent.mkdir(parents=True, exist_ok=True)
                entry = f"{ARCHIVE_ROOT}dev_databases/{db_id}/{db_id}.sqlite"
                with zf.open(entry) as src, db_dst.open("wb") as out:
                    out.write(src.read())
                summary["written"].append(f"{db_id}.sqlite")

            desc_dir = databases_dir / db_id / "database_description"
            desc_prefix = f"{ARCHIVE_ROOT}dev_databases/{db_id}/database_description/"
            if desc_dir.exists() and not force:
                summary["reused"].append(f"{db_id}/database_description")
                continue
            desc_dir.mkdir(parents=True, exist_ok=True)
            wrote_any = False
            for info in zf.infolist():
                if info.filename.startswith(desc_prefix) and not info.is_dir():
                    csv_name = info.filename[len(desc_prefix):]
                    with zf.open(info) as src, (desc_dir / csv_name).open("wb") as out:
                        out.write(src.read())
                    wrote_any = True
            if wrote_any:
                summary["written"].append(f"{db_id}/database_description")

    return summary


def load_gold_sql(path: Path) -> list[tuple[str, str]]:
    """Parse `mini_dev_sqlite_gold.sql`: one `SQL\\tdb_id` pair per line."""
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sql, db_id = line.rsplit("\t", 1)
        pairs.append((sql, db_id))
    return pairs


def load_column_descriptions(databases_dir: Path, db_id: str, table_names_original: list[str]) -> dict[str, str]:
    """Load `{table}|{column}` -> description from a database's CSV files.

    CSV filenames match `table_names_original` entries exactly (BIRD
    convention). Missing files or empty descriptions are skipped -- nothing
    is fabricated.
    """
    descriptions: dict[str, str] = {}
    desc_dir = databases_dir / db_id / "database_description"
    if not desc_dir.exists():
        return descriptions

    for table_name in table_names_original:
        csv_path = desc_dir / f"{table_name}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                col = (row.get("original_column_name") or "").strip()
                desc = (row.get("column_description") or "").strip()
                if col and desc:
                    descriptions[f"{table_name}|{col}"] = desc
    return descriptions


def analyze_source_consistency(
    hf_questions: list[dict],
    archive_questions: list[dict],
    archive_gold_pairs: list[tuple[str, str]],
) -> dict:
    """Cross-check the two independent official question sources.

    Diagnostic only -- does not judge which source is "right" and does not
    change which one LocalSQL uses. Surfaced by `scripts/setup_bird_minidev.py`
    into `setup_report.json` so any future divergence between HF's
    `mini_dev_sqlite` split and the static `minidev.zip` archive is visible
    immediately rather than silently producing a mixed-source grading
    reference.

    Three checks, each comparing positionally-aligned rows:
    - `archive_internal`: archive's own question file vs its own gold.sql
      (should always be 0 -- the official evaluator's own inputs must agree
      with each other).
    - `hf_vs_archive_question`: does HF ask the same question, in the same
      position, as the archive?
    - `hf_vs_archive_sql`: does HF's `SQL` field match the archive's
      question-file `SQL` field at the same position? (Independent of
      `archive_internal`, which checks against gold.sql instead.)
    """
    n = len(hf_questions)
    if len(archive_questions) != n or len(archive_gold_pairs) != n:
        raise AssertionError(
            f"source row counts differ: hf={n} archive_questions={len(archive_questions)} "
            f"archive_gold={len(archive_gold_pairs)}"
        )

    archive_internal = [i for i in range(n) if archive_questions[i]["SQL"] != archive_gold_pairs[i][0]]
    question_mismatches = [i for i in range(n) if hf_questions[i]["question"] != archive_questions[i]["question"]]
    sql_mismatches = [i for i in range(n) if hf_questions[i]["SQL"] != archive_questions[i]["SQL"]]

    return {
        "total": n,
        "archive_internal_mismatch_count": len(archive_internal),
        "archive_internal_mismatch_indices": archive_internal,
        "hf_vs_archive_question_mismatch_count": len(question_mismatches),
        "hf_vs_archive_question_mismatch_indices": question_mismatches,
        "hf_vs_archive_sql_mismatch_count": len(sql_mismatches),
        "hf_vs_archive_sql_mismatch_indices": sql_mismatches,
    }
