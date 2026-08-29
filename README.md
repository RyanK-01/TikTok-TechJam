# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## Our Team's Solution (status: MVP working, accuracy tuning in progress)

Everything below the horizontal rule is the organizer's original participant
kit, unmodified except for `starter/agent.py`. This section is ours, kept up
top so anyone picking up the repo -- teammates, graders -- gets oriented
before the original instructions.

**Architecture:** per-turn pipeline of `SessionState` (conversation memory)
-> dual-track intent routing (Buying = precise BM25 keyword filter; Browsing
= BM25 + dense-embedding hybrid) -> a local LLM "instruction sheet" call
(structured constraints + per-attribute weights + an optional clarifying
question) -> a deterministic, code-side executor that scores/ranks
candidates and enforces the contract (valid IDs only, deduped, capped at
`top_k`). Full architecture detail, setup steps, and tuning knobs live in
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) -- read that before changing
`starter/agent.py`.

**Model choice & cost:** local [Ollama](https://ollama.com) only --
`llama3.1:8b` for the ranking/dialogue step, `nomic-embed-text` for dense
retrieval. **$0 cost, no API key.** Requires a local Ollama server reachable
over HTTP (`http://localhost:11434` by default, override via `OLLAMA_HOST`)
-- this is loopback traffic to a process on the same machine, not an
internet or cloud API dependency. Typical measured latency is ~5-15s/turn
once the models are warm; a full 200-session public-set run takes on the
order of 2-3 hours end to end, so iterate against `data/dev_slice.jsonl`
(20 sessions, ~15 min) instead of the full set while tuning.

**Network dependency and offline fallback** (per `docs/submission_rules.md`,
which requires this be stated explicitly): the agent calls a local Ollama
server over HTTP for both the LLM ranking step and dense retrieval. If
Ollama is unreachable, times out, or the required models aren't pulled, the
agent does **not** crash or raise -- after 3 consecutive failures it
permanently disables the affected component (LLM step, dense retrieval, or
both, independently) for the remainder of that process and falls back to
plain BM25 keyword retrieval for every remaining turn. This is automatic,
requires no configuration, and is what the agent effectively runs as if
graded in a network-restricted environment with no Ollama installed.

**Current results:**

| Metric | Weak BM25 baseline (public set, 200) | Current (public set, 200) |
|---|---|---|
| Hit Rate@10 | 0.125 | see `results.json` -- full run in progress, will be updated here |
| MRR | 0.068 | see `results.json` |
| MTTC | 9.81 | see `results.json` |

A 20-session scenario-balanced slice (`data/dev_slice.jsonl`) scored Hit
Rate@10 `0.45`, MRR `0.32`, MTTC `7.4` -- small sample, noisy, but directionally
consistent with the full run.

**Known limitations / open work** (see `docs/DEVELOPMENT.md` for exactly
where each of these lives in code):
- `SYSTEM_PROMPT` is a first draft, not yet tuned -- the single biggest
  remaining accuracy lever.
- `user_profile` is stored per session but not yet used to influence
  ranking -- personalization pillar is still open.
- The intent router, executor scoring weights, and retrieval pool sizes are
  first-pass heuristics, not measured/tuned against the public set.
- A one-time ~25-minute embedding index build is required on any machine
  that hasn't run the agent before (cached afterward to `data/embeddings.npy`,
  gitignored, rebuilds automatically if missing).

**Team contributions:** _fill in before final submission._

---

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
