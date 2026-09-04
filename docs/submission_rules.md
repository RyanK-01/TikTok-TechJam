# Submission Rules

This document defines the participant submission requirements for the
TechJam Conversational E-Commerce Search Challenge.

## What Teams Must Submit

Each team must submit:

- one Python agent entry file exporting `Agent`
- any required local helper modules
- setup instructions
- a short report describing method, model choice, and limitations
- a disclosure of latency, token usage, and estimated model cost

## Required Interface

Your submission must export:

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [{"parent_asin": "B000..."}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

## Allowed Submission Contents

You may include:

- Python source files
- small local config files
- lightweight local assets required by your agent
- dependency manifest and install instructions

## Disallowed Submission Contents

Do not include:

- unreleased final evaluation data or labels
- copied organizer-only files
- API keys or secrets
- code that requires privileged host access
- code that modifies evaluator files
- code that depends on undeclared external services

## Model Policy

Teams may use any legally accessible LLM API, local model, or non-LLM approach.
LLM usage is optional.

The 800 final test samples will remain private. The review committee will run
each team's frozen submitted repository under uniform evaluation conditions.
Network access and external API calls are allowed when they are fully documented
and reproducible in the evaluation environment. Teams must:

- manage their own credentials, usage limits, service availability, and costs
- pass API keys through environment variables and never commit secret values
- document the required environment-variable names and setup instructions
- disclose model choice, network dependencies, latency, token usage, estimated
  cost, and any fallback behavior

An offline fallback is not mandatory.

## Output Rules

Your `respond(...)` output must follow these rules:

- `message` must be a string
- `ask_attribute` must be one allowed attribute or `null`
- `recommendations` must be ordered best to worst
- only the first 10 valid unique `parent_asin` values are scored
- `usage` should report non-negative token counts when available

## Reproducibility Requirements

Your submission package must contain:

- exact Python version requirement if non-default
- dependency installation steps
- one command to run the agent in the official harness
- any non-obvious environment variables

If your code cannot be reproduced from the submitted bundle and instructions,
the organizer may treat the run as invalid.

## Final Evaluation and Code Freeze

- The repository commit submitted by the Devpost deadline is the frozen version
  of the solution.
- The review committee will evaluate that frozen commit; later changes will not
  be included in the final run.
- The organizer will run the same official evaluator under the same evaluation
  conditions for every submission.
- The final test samples, labels, and per-session evaluation data will not be
  distributed to participants.

## Recommended File Layout

```text
submission/
  agent.py
  requirements.txt
  README.md
  src/
```

## Final Notes

- The review committee will apply uniform evaluation conditions across all teams.
- Final results will be generated with the official evaluator against the private
  final test set.
- See `docs/final_evaluation_faq.md` for the complete final evaluation policy.
