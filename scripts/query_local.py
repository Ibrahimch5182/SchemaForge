"""CLI over the SAME QueryService the API uses (no duplicate pipeline).

Model runtime comes from the environment (see configs/backend.yaml):
SCHEMAFORGE_LLAMA_EXE, SCHEMAFORGE_BASE_GGUF, SCHEMAFORGE_LORA_GGUF.

    uv run python scripts/query_local.py --database-id demo `
        --question "How many employees are there?"

Exit code 0 only when the query was generated, passed safety, and executed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pydantic import ValidationError  # noqa: E402

from localsql.backend.bootstrap import build_query_service  # noqa: E402
from localsql.backend.config import load_backend_config  # noqa: E402
from localsql.backend.errors import BackendError  # noqa: E402
from localsql.backend.models import QueryRequest  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database-id", required=True, help="Registered logical database id (never a path).")
    parser.add_argument("--question", required=True)
    parser.add_argument("--business-context", default=None)
    parser.add_argument("--config", type=Path, default=None, help="Alternative backend.yaml.")
    parser.add_argument("--list-databases", action="store_true")
    args = parser.parse_args(argv)

    try:
        service = build_query_service(load_backend_config(args.config))
        if args.list_databases:
            print(json.dumps(service.list_databases(), indent=2))
            return 0
        request = QueryRequest(
            database_id=args.database_id, question=args.question, business_context=args.business_context
        )
        response = service.query(request)
    except ValidationError as e:
        print(f"BLOCKER: invalid request: {[err['loc'] for err in e.errors()]}")
        return 2
    except BackendError as e:
        print(f"BLOCKER: {e.code}: {e.message}")
        return 1

    print(response.model_dump_json(indent=2))
    return 0 if response.status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
