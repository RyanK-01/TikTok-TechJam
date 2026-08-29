from __future__ import annotations

import json
import os
import re
import sqlite3
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

# Simple, deliberately narrow signal for "the customer just replaced an
# earlier preference" (Intent Override sessions). It only clears
# bookkeeping -- it never drops earlier turns from the search text (see
# SessionState.query_text) since the underlying category usually still
# holds even when one constraint changes.
OVERRIDE_RE = re.compile(r"\bignore\b.*\b(earlier|previous)\b", re.IGNORECASE)

CATALOG_COLUMNS = ("parent_asin", "title", "categories", "features", "details", "store", "description")

ALLOWED_ATTRIBUTES = frozenset({
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
})

SYSTEM_PROMPT = (
    "You are the ranking assistant inside a shopping agent for an Amazon "
    "Clothing/Shoes/Jewelry catalog. Each turn you receive the conversation "
    "so far, what the customer has already told us, and a sample of "
    "currently-matching products. Output ONE JSON object describing how to "
    "filter and rank the full candidate pool -- you do not pick products "
    "directly, and you never see full product IDs.\n\n"
    "Rules:\n"
    "- Use only the allowed attribute names below.\n"
    "- If the product sample still looks broad or mixed after applying "
    "known constraints, set ask_attribute to the single most useful thing "
    "to learn next. Otherwise set it to null.\n"
    "- Never ask about an attribute already present in \"Known constraints "
    "so far\".\n"
    "- Respond with JSON only, matching the schema exactly. No prose, no "
    "markdown fences.\n\n"
    "Allowed ask_attribute values: " + ", ".join(sorted(ALLOWED_ATTRIBUTES)) + ", or null.\n\n"
    "Schema:\n"
    "{\n"
    '  "constraints": {"<attribute>": "<value>", ...},\n'
    '  "weights": {"<attribute>": <0.0-1.0>, ...},\n'
    '  "ask_attribute": "<allowed value or null>",\n'
    '  "message": "<one short, natural sentence to the customer>"\n'
    "}"
)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _as_float(value: object) -> float | None:
    """Best-effort numeric parse -- used for catalog price/rating and for
    reading a budget value like "under $80" out of free text."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
    return None


@dataclass
class SessionState:
    """Per-session memory. Everything the pipeline knows about one customer."""

    profile: dict
    turn: int = 0
    messages: list[str] = field(default_factory=list)
    agent_messages: list[str] = field(default_factory=list)
    ask_log: list[str | None] = field(default_factory=list)
    # attribute -> the reply text that answered it (assumed to answer
    # whichever attribute we last asked about)
    disclosed: dict[str, str] = field(default_factory=dict)
    last_asked: str | None = None
    override_seen: bool = False

    def record_message(self, message: str, turn: int) -> None:
        self.turn = turn
        if OVERRIDE_RE.search(message):
            self.override_seen = True
            self.disclosed.clear()
        elif self.last_asked:
            self.disclosed[self.last_asked] = message
        self.messages.append(message)

    def record_turn_result(self, ask_attribute: str | None, message: str) -> None:
        self.last_asked = ask_attribute
        self.ask_log.append(ask_attribute)
        self.agent_messages.append(message)

    def query_text(self) -> str:
        # Full turn history, not just the latest message: this is what makes
        # retrieval "remember" constraints revealed several turns ago.
        return " ".join(self.messages)


@dataclass
class InstructionSheet:
    """The LLM's answer for one turn: how to filter/rank candidates, and
    whether to ask a clarifying question instead."""

    constraints: dict[str, str] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    ask_attribute: str | None = None
    message: str = ""
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})

    @classmethod
    def parse(cls, raw: str) -> "InstructionSheet":
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("instruction sheet must be a JSON object")

        constraints = {
            str(key): str(value)
            for key, value in (data.get("constraints") or {}).items()
            if value not in (None, "", [])
        }

        weights: dict[str, float] = {}
        for key, value in (data.get("weights") or {}).items():
            try:
                weights[str(key)] = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                continue

        ask_attribute = data.get("ask_attribute")
        if ask_attribute is not None:
            ask_attribute = str(ask_attribute).strip().lower()
            if ask_attribute in ("", "null", "none"):
                ask_attribute = None
            elif ask_attribute not in ALLOWED_ATTRIBUTES:
                ask_attribute = "other"

        message = data.get("message")
        message = str(message).strip() if message else ""

        return cls(constraints=constraints, weights=weights, ask_attribute=ask_attribute, message=message)

    @classmethod
    def fallback(cls) -> "InstructionSheet":
        # Empty constraints -> the executor's scoring collapses to the
        # original retrieval order, and respond() fills in a generic message.
        return cls()


class Agent:
    """Retrieval -> LLM instruction sheet -> deterministic execution.

    Pipeline per turn: update SessionState -> retrieve a candidate pool from
    the catalog using everything known so far -> ask a local LLM (via
    Ollama) for a small JSON "instruction sheet" (constraints + weights +
    an optional clarifying question) -> apply that sheet to the candidates
    deterministically -> return top_k. Any LLM failure (unreachable,
    malformed output, timeout) falls back to the plain retrieval order
    rather than raising.
    """

    MAX_TURNS = 10
    # Candidates scored by the executor. Kept independent of top_k (fixed
    # at 10 by the contract) so the LLM's weights have real room to rerank.
    CANDIDATE_POOL_SIZE = 100
    # Candidates actually shown to the LLM in the prompt -- kept small to
    # fit Ollama's default 4096-token context alongside conversation history.
    CANDIDATE_SAMPLE_SIZE = 18
    LLM_FAILURE_THRESHOLD = 3

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._meta: dict[str, dict] = {}
        self._build_index()

        self.ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        self.llm_timeout = float(os.environ.get("OLLAMA_TIMEOUT", "30"))
        self._llm_failures = 0
        self._llm_disabled = False

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                self._meta[parent_asin] = {
                    "price": _as_float(product.get("price")),
                    "average_rating": _as_float(product.get("average_rating")),
                }
                batch.append(
                    (
                        parent_asin,
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(profile=user_profile)

    # ---- Stage 1: cheap retrieval -----------------------------------

    def _retrieve_candidates(self, state: SessionState, limit: int) -> list[dict]:
        unique_terms = list(dict.fromkeys(_terms(state.query_text())))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin, title, categories, features, details, store, description "
            "FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [dict(zip(CATALOG_COLUMNS, row)) for row in rows]

    # ---- Stage 2: LLM instruction sheet ------------------------------

    def _summarize_candidate(self, candidate: dict) -> str:
        title = (candidate.get("title") or "product").strip()
        if len(title) > 70:
            title = title[:67].rstrip() + "..."
        price = self._meta.get(candidate["parent_asin"], {}).get("price")
        price_text = f"${price:.2f}" if isinstance(price, float) else "price n/a"
        categories = " ".join((candidate.get("categories") or "").split()[:6])
        return f'"{title}" | {price_text} | {categories}'.strip()

    def _build_user_prompt(self, state: SessionState, candidates: list[dict]) -> str:
        profile = state.profile or {}
        lines = [
            f"Turn {state.turn} of {self.MAX_TURNS}.",
            "Customer profile: "
            f"purchase_frequency={profile.get('purchase_frequency', 'unknown')}, "
            f"rating_style={profile.get('rating_style', 'unknown')}, "
            f"preference_tags={profile.get('preference_tags', [])}, "
            f'summary="{profile.get("summary", "")}"',
            "",
            "Conversation so far:",
        ]
        for index, customer_message in enumerate(state.messages):
            lines.append(f'{index + 1}. Customer: "{customer_message}"')
            if index < len(state.agent_messages):
                asked = state.ask_log[index] if index < len(state.ask_log) else None
                suffix = f" (asked: {asked})" if asked else ""
                lines.append(f'   Agent: "{state.agent_messages[index]}"{suffix}')
        lines.append("")
        lines.append(
            "Known constraints so far: " + (json.dumps(state.disclosed) if state.disclosed else "none yet")
        )
        lines.append("")
        sample = candidates[: self.CANDIDATE_SAMPLE_SIZE]
        lines.append(f"Sample of currently-matching products ({len(candidates)} total, showing {len(sample)}):")
        for candidate in sample:
            lines.append(f"- {self._summarize_candidate(candidate)}")
        lines.append("")
        lines.append("Return the JSON instruction sheet now.")
        return "\n".join(lines)

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        payload = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2, "num_ctx": 4096},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.ollama_host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.llm_timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body.get("message", {}).get("content", "")
        usage = {
            "prompt_tokens": int(body.get("prompt_eval_count") or 0),
            "completion_tokens": int(body.get("eval_count") or 0),
        }
        return content, usage

    def _get_instruction_sheet(self, state: SessionState, candidates: list[dict]) -> InstructionSheet:
        if self._llm_disabled:
            return InstructionSheet.fallback()
        user_prompt = self._build_user_prompt(state, candidates)
        try:
            content, usage = self._call_ollama(SYSTEM_PROMPT, user_prompt)
            sheet = InstructionSheet.parse(content)
            sheet.usage = usage
            self._llm_failures = 0
            return sheet
        except Exception:
            self._llm_failures += 1
            if self._llm_failures >= self.LLM_FAILURE_THRESHOLD:
                self._llm_disabled = True
            return InstructionSheet.fallback()

    # ---- Stage 3: deterministic execution ----------------------------

    def _score_candidates(self, candidates: list[dict], sheet: InstructionSheet) -> list[str]:
        scored: list[tuple[float, int, str]] = []
        for rank, candidate in enumerate(candidates):
            parent_asin = candidate["parent_asin"]
            meta = self._meta.get(parent_asin, {})
            combined_text = " ".join(
                candidate.get(field_name, "") for field_name in CATALOG_COLUMNS[1:]
            ).lower()

            score = 0.0
            for attribute, value in sheet.constraints.items():
                weight = sheet.weights.get(attribute, 0.5)
                if attribute == "budget":
                    budget_max = _as_float(value)
                    price = meta.get("price")
                    if budget_max is not None and price is not None and price <= budget_max:
                        score += weight
                elif value.lower() in combined_text:
                    score += weight
            # Tie-break toward the original BM25 order so an empty/failed
            # instruction sheet degrades to plain retrieval, not a shuffle.
            score += 1.0 / (1 + rank) * 0.1
            scored.append((score, rank, parent_asin))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [parent_asin for _, _, parent_asin in scored]

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        state = self._sessions[session_id]
        state.record_message(user_message, turn)

        candidates = self._retrieve_candidates(state, self.CANDIDATE_POOL_SIZE)
        if not candidates:
            message = "Tell me a bit more about what you're looking for."
            state.record_turn_result(None, message)
            return {
                "message": message,
                "ask_attribute": None,
                "recommendations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

        sheet = self._get_instruction_sheet(state, candidates)
        ranked_ids = self._score_candidates(candidates, sheet)[:top_k]
        message = sheet.message or "Here are the closest matches I found."

        state.record_turn_result(sheet.ask_attribute, message)
        return {
            "message": message,
            "ask_attribute": sheet.ask_attribute,
            "recommendations": [{"parent_asin": pid} for pid in ranked_ids],
            "usage": sheet.usage,
        }
