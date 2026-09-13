# LocalSQL Data Contract (Phase 1)

This document defines the raw sources, internal representations, and
generated artifacts of the Phase 1 training-data pipeline.

## Raw sources

1. **Training rows**: [`birdsql/bird23-train-filtered`](https://huggingface.co/datasets/birdsql/bird23-train-filtered)
   (Hugging Face dataset, CC-BY-SA-4.0). A single JSONL file with columns
   `db_id`, `question`, `evidence`, `SQL`. 6,601 rows across 69 databases at
   the time of inspection (see `data/reports/bird_inspection_report.json`
   for the live, rerunnable numbers).
2. **Column descriptions**: `train_column_meaning.json`, shipped in the same
   HF dataset repo. Maps `"{db_id}|{table_name}|{column_name}"` to a
   human-written description. Covers ~98% of columns in the 69 databases
   used here.
3. **Database schema metadata**: the official BIRD training release,
   `train_tables.json`, inside `train.zip` published at
   `bird-bench.oss-cn-beijing.aliyuncs.com`. This is Spider-format schema
   JSON: table names, column names, column types, primary keys, and foreign
   keys for all 69 BIRD training databases. The full archive is ~8.9 GB
   (it bundles per-database SQLite files with real row data); Phase 1 only
   needs the schema file, so it is extracted directly from the remote zip's
   central directory via HTTP range requests (`localsql.data.schema_loader`)
   -- the multi-GB archive is never downloaded.

BIRD Mini-Dev is **never** loaded by this pipeline. It is reserved
exclusively for external evaluation in a later phase.

Schema is never inferred from gold SQL, and no missing metadata is
fabricated: a database with no matching entry in `train_tables.json` would
be rejected with reason `MISSING_SCHEMA` (this did not occur for any of the
69 databases in the filtered training set -- coverage is 100%).

## Canonical internal representation

Typed models live in `src/localsql/data/models.py`:

- `ColumnSchema` -- name, data type (as reported by BIRD, e.g. `INTEGER`,
  `TEXT`, `DATE`), `is_primary_key`, optional `foreign_key` (table +
  column), optional `description`.
- `TableSchema` -- name + ordered columns.
- `DatabaseSchema` -- `db_id`, `dialect` (`sqlite` for Phase 1), ordered
  tables.
- `RawBirdExample` -- the raw source row plus its `row_index` for stable
  provenance.
- `PreparedTextToSQLExample` -- the final training-ready record (see below).

## Schema serialization

`src/localsql/data/schema_serializer.py` renders a `DatabaseSchema` as a
compact, deterministic text block:

```text
customers(
  customer_id INTEGER PK,
  country TEXT,
  segment TEXT
)

orders(
  order_id INTEGER PK,
  customer_id INTEGER FK->customers.customer_id,
  order_date DATE,
  total NUMERIC
)
```

Table and column order is taken directly from the source schema, so the
same `DatabaseSchema` value always serializes to the exact same bytes.
Data types, `PK`, and `FK->table.column` annotations are included only when
present in the source metadata; column descriptions (when available) are
appended as a trailing `-- ` comment. Nothing is fabricated.

## Prompt / completion format

`src/localsql/data/prompt_builder.py` defines the single canonical template
reused by every future consumer (training, validation, baseline inference,
fine-tuned inference, production inference):

```text
SYSTEM:
You are a text-to-SQL model.
Generate exactly one read-only SQL query that answers the question using only the provided database schema.
Return SQL only.
Do not use markdown.

DIALECT:
sqlite

SCHEMA:
<canonical serialized schema>

BUSINESS CONTEXT:
<context if available>

QUESTION:
<question>
```

The `BUSINESS CONTEXT` section is omitted entirely when no context is kept
for that example (see below) -- no placeholder text is inserted.

The completion is the gold SQL with only leading/trailing whitespace
stripped (`build_completion`). Internal whitespace is preserved untouched
because collapsing it could change the meaning of a string literal embedded
in the query. No markdown fences, no `SQL:` prefix, no explanation, no
chain-of-thought.

## Business-context (evidence) treatment

BIRD's `evidence` field is oracle background knowledge a real end user may
not supply. To avoid the model becoming dependent on it, each example with
non-empty evidence has a **deterministic** keep/drop decision:

```
keep = sha256(f"{seed}:{db_id}:{row_index}")[:16 hex digits] / 16**16 < keep_probability
```

Default `keep_probability` is `0.50` (`configs/data.yaml`). This is a pure
hash of `(seed, example key)`, not a sequential RNG draw, so it is
independent of processing order and reproducible across reruns. Examples
with no evidence in the source data never receive business context. Every
prepared example records `source.evidence_available` and
`source.business_context_kept` so later work can evaluate
question+schema vs. question+schema+context.

## Database-level train/validation split

Splitting is done on **`db_id`**, never on individual examples
(`src/localsql/data/splitter.py`), so validation measures generalization to
schemas unseen during training:

1. Collect the unique `db_id`s among valid raw examples.
2. Sort them (removes input-order dependence), then shuffle with
   `random.Random(seed)` (default seed `42`).
3. Take the first `train_db_fraction` (default `0.90`) as the training set
   of database IDs; the rest go to validation.
4. Assert the two sets are disjoint (`check_split_leakage`); this is
   re-verified once more against the final prepared examples in
   `scripts/prepare_bird.py` and recorded in the validation report.

With the current 69 databases and default settings, this yields 62 training
databases / 7 validation databases (see
`data/reports/data_validation_report.json` for the live split and exact
`db_id` assignments).

## Validation and rejection reporting

`src/localsql/data/validator.py` never silently drops a row. Raw rows are
checked for empty question, empty SQL, and missing schema; prepared
examples are checked for empty serialized schema, empty prompt, a
completion that is not SQL-only (markdown fences, `SQL:` prefix, etc.), and
duplicate `example_id`s. Every rejection is recorded with a reason code in
`data/reports/data_validation_report.json`.

## Generated artifacts

Produced by `scripts/prepare_bird.py` (gitignored -- regenerate locally):

- `data/processed/train.jsonl`
- `data/processed/validation.jsonl`
- `data/reports/data_validation_report.json`

Produced by `scripts/inspect_bird.py` (gitignored):

- `data/reports/bird_inspection_report.json`

A small, committed real-data fixture (a handful of rows/schemas, licensing
permits CC-BY-SA-4.0 reuse) lives under `tests/data/sample_*` so tests never
require downloading the full dataset or the multi-GB schema archive.
