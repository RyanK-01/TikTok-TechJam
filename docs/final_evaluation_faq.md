# Track 4 Final Evaluation FAQ

This document is the participant-facing clarification for final evaluation in
the TechJam Conversational E-Commerce Search Challenge.

## 1. Final Evaluation Process

The 800 final evaluation sessions will remain private and will not be released to
participants. This policy supersedes the previously announced plan to distribute
the final evaluation package after the Devpost submission deadline.

The review committee will run each team's submitted repository under the same
environment and evaluation conditions.

- The submitted Git commit is the frozen version of the solution.
- The review committee will evaluate that frozen commit; later changes will not
  be included in the final run.
- The organizer will use the same official evaluator and evaluation conditions
  for every submission.
- Final test inputs, intent cards, ground-truth labels, and per-session evaluation
  data will remain private.

The final evaluation uses the same input schema, Agent interface, metric formula,
stopping rule, invalid-output handling, deterministic customer-message templates,
and `ask_attribute` response policy as the released official evaluator. No
undisclosed natural-language paraphrases are introduced.

## 2. Network, Models, APIs, and Credentials

LLM usage is optional. Teams may use an external API, a local model, or a non-LLM
approach. LLM semantic ranking is an allowed technical direction, not a mandatory
requirement.

Submissions may use an external API, a local model, or a non-LLM approach, but
every required dependency must be fully documented so the review committee can
reproduce the submission. Teams must:

- network access and external API calls are allowed
- teams manage their own credentials, usage limits, service availability, and
  costs
- API key values must not be shared with the organizer or committed to the
  public repository
- API keys should be passed through environment variables
- the README should document only required environment-variable names and setup
  instructions; no separate credential manifest is required
- an offline fallback is not mandatory, but all external dependencies and any
  fallback behavior must be disclosed

The organizer does not provide, inject, or reimburse model API keys, tokens, or
credits.

## 3. Hardware, Runtime, and Timeouts

Teams should disclose the Python version, hardware, dependencies, runtime,
latency, token usage, and estimated model cost used to produce their results.
The review committee will apply uniform evaluation conditions across all teams.

## 4. Data, Catalog, and Derived Artifacts

The frozen 50,000-product catalog is the official retrieval and scoring space.
All recommendations must use valid `parent_asin` values from this catalog. Teams
must not introduce, replace, or fabricate ASINs.

Offline preprocessing is allowed, including:

- catalog-derived embeddings and local indexes
- derived attributes, labels, or summaries
- local sidecar files
- legally usable pretrained embedding, reranking, and language models

Precomputed local artifacts do not need to be rebuilt in memory at startup.
Large assets should be supplied through documented and reproducible download
instructions rather than committed directly to the repository. There is
currently no track-specific package-size limit.

Teams may use legally accessible upstream Amazon Reviews 2023 data or other
public corpora for preprocessing if their sources and usage are disclosed.
External data must not be used to reconstruct unreleased evaluation labels.
Heavy externally deployed industrial vector-database infrastructure remains out
of scope.

Each `parent_asin` represents a parent product rather than a specific color/size
SKU variant. Intent cards are derived from the same frozen catalog metadata
available to participants, together with the predefined scenario policy; they do
not use additional hidden variant-level product attributes.

The Agent does not receive raw conversation histories, user identifiers, raw
purchase histories, review text, or timestamps. It receives the published
`reset(session_id, user_profile)` and
`respond(session_id, user_message, turn, top_k)` inputs and maintains its own
conversation state.

## 5. Agent Interface and Evaluator Behavior

- An Agent may ask a clarification question and return recommendations in the
  same turn.
- `ask_attribute` must contain one allowed attribute or `null`.
- The simulator responds according to structured `ask_attribute`; it does not
  infer the requested attribute from the natural-language `message`.
- Recommendation order determines ranking. Optional numeric recommendation
  scores are ignored.
- Duplicate and invalid ASINs are removed, and only the first 10 valid unique
  recommendations are scored.
- The session stops automatically after the first valid target hit; no trigger
  word is required.
- An Intent Override session cannot record a hit before the changed intent is
  revealed.
- `reset()` is called once per session. Teams may share immutable indexes, but
  conversational state must remain isolated between sessions.
- The current official evaluator processes sessions sequentially; concurrent
  execution is not required.

## 6. Judging and TechnicalScore

For Track 4, Section 4.6 of the problem statement is the applicable
track-specific rubric.

For the online submission round, the criteria retain their published weights and
are assessed out of 90 points:

- Technical Execution: 35%
- Innovation & Problem Insight: 20%
- Impact & Relevance: 20%
- Feasibility & Practicality: 15%

They are not equally weighted or renormalized. Presentation & Communication
contributes the remaining 10% at the Final Event.

`TechnicalScore` is an objective input to the Technical Execution assessment. It
is not a separate judging criterion and does not represent the entire Technical
Execution score. Judges also consider code quality, architecture, reliability,
technical complexity, and effective use of models or APIs.

Latency, token usage, and estimated cost do not directly change the core
`TechnicalScore`, but may be considered under Feasibility & Practicality.

## 7. Submission and Demonstration

- Teams may replace the starter Agent as long as the required `Agent` interface
  is preserved.
- Teams may modify the evaluator for local experimentation, but reported final
  results must be produced using the unmodified official evaluator.
- A UI is optional. It may be used to improve the demonstration, but it is not a
  separately assessed technical component and does not replace the runnable
  Python Agent.
- The demonstration should show at least one complete multi-turn session.
- LLM-based systems should report token usage when available. Non-LLM systems may
  omit `usage` or report zero usage.
- The repository must include setup instructions, dependency requirements,
  required environment-variable names, and a clear command for running the Agent
  with the official evaluator.
