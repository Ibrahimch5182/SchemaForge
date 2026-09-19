"""Phase 8: production backend (registered DB -> schema -> canonical prompt ->
model SQL -> deterministic safety -> read-only execution -> structured result).

The service layer (`QueryService`) has no FastAPI dependency; `api.py` and the
CLI are thin adapters over it. The model only ever produces SQL text --
database access, safety, limits, and errors belong to deterministic code.
"""
