# Development Notes

Internal notes for the team working on `starter/agent.py`. The competition's
own docs (`competition_specification.md`, `agent_api_contract.json`,
`evaluation_config.json`, `submission_rules.md`) are the source of truth for
rules/scoring; this file is just "how to run it and where to tweak it."

## Architecture

Per turn, `Agent.respond()` runs:

1. **SessionState** update -- turn history, disclosed constraints (keyed by
   whichever attribute we last asked about), Intent Override detection.
2. **Dual-track routing** (`_route_intent`) -- Buying (concrete signal:
   budget/material/color/size word, override, or any disclosed constraint)
   vs Browsing (still vague), re-evaluated every turn.
3. **Retrieval** -- BM25/FTS5 keyword search always runs
   (`_retrieve_candidates`). On a Browsing turn it's merged with dense
   embedding-similarity hits (`_dense_candidates`, cosine sim over a cached
   `nomic-embed-text` index) via `_merge_candidates`.
4. **LLM instruction sheet** (`_get_instruction_sheet`) -- a local Ollama
   call (`llama3.1:8b` by default) returns structured JSON: constraints,
   per-attribute weights, an optional `ask_attribute`, and a customer-facing
   `message`. Parsed/validated by `InstructionSheet.parse`. Any failure
   (unreachable, timeout, bad JSON) falls back to `InstructionSheet.fallback()`
   -- an empty sheet, which makes the executor collapse to plain retrieval
   order rather than crash the turn.
5. **Executor** (`_score_candidates`) -- deterministic, code-side. Applies
   the sheet's constraints/weights to every candidate, plus a small tie-break
   toward the original retrieval order. Sorts, returns top_k.

## Setup

Requires a local [Ollama](https://ollama.com) server with two models pulled:

```
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

No API key, no cost. Override via env vars if your setup differs:

| Var | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Chat model for the instruction sheet |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model for dense retrieval |
| `OLLAMA_TIMEOUT` | `30` (seconds) | Per-call timeout |

If Ollama isn't reachable, or the wrong models aren't pulled, the agent
**does not crash** -- after 3 consecutive LLM failures it permanently
disables the LLM step for the rest of that process and runs on plain
retrieval order only. Dense retrieval degrades the same way if numpy or the
embedding endpoint is unavailable. Both are silent-safe by design, so a
misconfigured environment shows up as *"scores look like the old baseline"*,
not an exception -- check terminal output for `[agent] ...` lines if results
look off.

## Embedding cache

First `Agent()` instantiation with no `data/embeddings.npy` present builds
one automatically: embeds all 50,000 catalog products via `nomic-embed-text`
in batches of 250, ~25 minutes one-time (GPU-dependent), then caches to
`data/embeddings.npy` + `data/embeddings_ids.json` (gitignored -- regenerate
locally, don't commit; ~150MB). Every run after that loads instantly. If the
catalog changes, delete both files to force a rebuild.

## Running it

Full public set (200 sessions, ~2 hours with a warm LLM -- run this
occasionally, not per-edit):

```
python -m evaluator.local_evaluator
```

Fast dev loop -- `data/dev_slice.jsonl` is a 20-session, scenario-balanced
slice (8 buying / 8 browsing / 3 intent_override / 1 boundary, matching the
official mix) for quick iteration (~10-15 min):

```
python -m evaluator.local_evaluator --dataset data/dev_slice.jsonl --output results_dev.json
```

Upstream evaluator unit tests (fast, no LLM/catalog needed):

```
python -m unittest tests.test_evaluator -v
```

## Where to tweak what

- **Prompt wording / instructions:** `SYSTEM_PROMPT` (top of `agent.py`).
- **Router sensitivity:** `BUDGET_RE`, `MATERIAL_WORDS`, `COLOR_WORDS`,
  `SIZE_WORDS`, and `_route_intent`.
- **Scoring formula:** `_score_candidates` -- how constraint weights and the
  retrieval-order tie-break combine.
- **Pool sizes:** `CANDIDATE_POOL_SIZE`, `CANDIDATE_SAMPLE_SIZE` (what the
  LLM actually sees in the prompt, kept small for context budget),
  `DENSE_CANDIDATE_SIZE` -- all class constants on `Agent`.
- **Personalization (`user_profile`):** stored in `SessionState.profile`,
  echoed into the prompt, but not otherwise used yet -- open pillar.

Each of the above is independently testable against `data/dev_slice.jsonl`
without waiting on a full 200-session run.
