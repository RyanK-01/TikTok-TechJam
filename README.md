# JustinAttempt — Changes from `initialPrototype`

## What Changed

Dense (semantic embedding) retrieval was previously merged into the candidate pool only on Browsing-track turns — Buying-track turns got pure BM25 keyword retrieval. This branch removes that gate so dense retrieval merges in on every ambiguous turn, regardless of track. One-line change in `starter/agent.py`'s `respond()`:

```python
# before
if track == "browsing" and self._dense_ready:
    ...
# after
if self._dense_ready:
    ...
```

## Models Used

- **Ranking/dialogue:** `qwen3.5:9b` (local, via Ollama)
- **Dense retrieval:** `qwen3-embedding:0.6b` (local, via Ollama — overrides the code's default `nomic-embed-text` via the `OLLAMA_EMBED_MODEL` env var)
- **Cost:** $0, no API key, fully local

## Benchmark Results

Stratified random 20-session sample (not the full 200-session public set):

| Metric | `initialPrototype` | `JustinAttempt` |
|---|---|---|
| Hit Rate@10 | 0.70 | **0.95** |
| MRR | 0.441 | **0.548** |
| MTTC | 5.7 | **4.2** |
| TechnicalScore | 0.588 | **0.7755** |

*Single run on one random sample, not the full public set — directional, not final.*
