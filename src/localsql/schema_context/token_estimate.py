"""Character-count-based token estimation, calibrated against real Qwen
tokenizer numbers already obtained on Kaggle (Phase 4).

This is NOT the real Qwen tokenizer -- it is a deliberately simple,
clearly-labeled proxy used only for local (offline, no-CUDA, no-model-
download) design analysis. `CALIBRATED_CHARS_PER_TOKEN` was derived by
comparing this repository's real canonical schema text length against the
REAL per-database median full-SFT token counts reported from the actual
Kaggle Phase 4 token-profile run, for the 8 long-context databases with
known real numbers (`works_cycles`, `hockey`, `movie_3`, `mondial_geo`,
`synthea`, `professional_basketball`, `donor`, `superstore`):

    mean chars/token = 3.771, stdev = 0.214 (n=8)

Validated against the REAL whole-training-set statistics from Phase 4
(not used for calibration, held out as a check): predicted median 2,723
vs. real 2,506 (+8.7%), predicted min 511 vs. real 549 (-6.9%), predicted
max 28,054 vs. real 28,082 (-0.1%). Good enough to rank/compare candidate
schema representations and estimate rough magnitudes, NOT precise enough
to replace a real tokenizer run for a final training-budget decision --
see `scripts/analyze_schema_context.py`'s `--token-profile` mode (mirrors
Phase 3/4's `--token-profile` pattern) for obtaining exact real numbers on
Kaggle.
"""

from __future__ import annotations

CALIBRATED_CHARS_PER_TOKEN = 3.771


def estimate_tokens(text: str, chars_per_token: float = CALIBRATED_CHARS_PER_TOKEN) -> float:
    """Rough, calibrated token estimate from character count. See module
    docstring for calibration methodology and known error margins."""
    return len(text) / chars_per_token
