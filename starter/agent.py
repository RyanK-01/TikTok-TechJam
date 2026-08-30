from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without numpy
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False


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

# A reply that declines to give a preference ("I don't have a preference
# for X") is not a disclosed value -- it must not be stored as one, or the
# agent (a) thinks it has real info to act on and (b) keeps re-asking the
# same attribute because nothing ever looks "already answered".
NO_PREFERENCE_RE = re.compile(r"don't have (?:an? )?(?:additional )?preference|no preference", re.IGNORECASE)

# Router signals: presence of any of these in a message (or an already
# non-empty disclosed-constraints dict) is treated as "the customer has
# given us something concrete" -> Buying track. Their absence -> Browsing.
BUDGET_RE = re.compile(r"\$\s?\d|\bunder\b|\bbudget\b|\bless than\b|\bcheap\b", re.IGNORECASE)
MATERIAL_WORDS = {
    "alloy", "cotton", "denim", "fabric", "gold", "leather", "modal",
    "nylon", "polyester", "rayon", "rubber", "silk", "silver", "spandex",
    "steel", "sterling", "suede", "textile", "wool",
}
COLOR_WORDS = {"black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange", "navy", "beige"}
SIZE_WORDS = {"small", "medium", "large", "size", "xl", "xs", "wide", "narrow", "petite", "tall"}
CONSTRAINT_INTRO_RE = re.compile(
    r"(?:key requirement is|what matters is|what i need is):\s*([^.\n]+)",
    re.IGNORECASE,
)
GENERIC_CONSTRAINT_TERMS = {
    "additional", "arch", "around", "approximately", "call", "closure",
    "earlier", "from", "have", "imported", "judgment", "material", "need",
    "preference", "prioritize", "use", "your",
}
CATEGORY_RE = re.compile(r"\blooking for\s+(.+?)(?:[.,]|\bbut\b|$)", re.IGNORECASE)
CATEGORY_FILLER_TERMS = {"clothing", "item", "items", "shoes", "jewelry"}
QUERY_TERM_SYNONYMS = {
    "boys": ("boy",),
    "girls": ("girl",),
    "men": ("mens", "male"),
    "mens": ("men", "male"),
    "pants": ("pant", "trousers"),
    "shirts": ("shirt", "tee", "tshirt"),
    "shoes": ("shoe", "sneakers"),
    "shorts": ("short",),
    "sneakers": ("shoes", "shoe"),
    "tshirts": ("shirt", "tee", "tshirt"),
    "women": ("womens", "female"),
    "womens": ("women", "female"),
}

CATALOG_COLUMNS = ("parent_asin", "title", "categories", "features", "details", "store", "description")

ALLOWED_ATTRIBUTES = frozenset({
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
})

SYSTEM_PROMPT = (
    "You are the ranking assistant inside a shopping agent for an Amazon "
    "Clothing/Shoes/Jewelry catalog. Each turn you receive the conversation "
    "so far, what the customer has already told us, a detected Buying/"
    "Browsing track, and a sample of currently-matching products. Output "
    "ONE JSON object describing how to filter and rank the full candidate "
    "pool -- you do not pick products directly, and you never see full "
    "product IDs.\n\n"
    "Rules:\n"
    "- Use only the allowed attribute names below.\n"
    "- On a BUYING turn, favor fewer, higher-confidence constraints (the "
    "customer has told us something concrete -- lock onto it).\n"
    "- On a BROWSING turn, the candidate pool is more varied by design; "
    "favor asking a clarifying question unless one option is clearly "
    "strongest.\n"
    "- If the product sample still looks broad or mixed after applying "
    "known constraints, set ask_attribute to the single most useful thing "
    "to learn next. Otherwise set it to null.\n"
    "- Never ask about an attribute already present in \"Known constraints "
    "so far\".\n"
    "- Keep message under 12 words.\n"
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


def _constraint_terms(value: str) -> list[str]:
    return [
        term
        for term in _terms(value)
        if term not in GENERIC_CONSTRAINT_TERMS
    ]


def _clean_constraint(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n\"'")


def _constraint_attribute(key: str) -> str:
    for attribute in sorted(ALLOWED_ATTRIBUTES, key=len, reverse=True):
        if key == attribute or key.startswith(f"{attribute}_"):
            return attribute
    return key.split("_", 1)[0]


def _expand_query_terms(terms: list[str], limit: int) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for term in terms:
        for candidate in (term, *QUERY_TERM_SYNONYMS.get(term, ())):
            if candidate and candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
                if len(expanded) >= limit:
                    return expanded
    return expanded


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
    # attributes the customer explicitly has no preference for -- distinct
    # from disclosed (there's no value here, just "don't ask again")
    declined: set[str] = field(default_factory=set)
    last_asked: str | None = None
    override_seen: bool = False
    override_start: int | None = None
    override_turn: int | None = None

    def record_message(self, message: str, turn: int) -> None:
        self.turn = turn
        if OVERRIDE_RE.search(message):
            self.override_seen = True
            self.override_start = len(self.messages)
            self.override_turn = turn
            self.disclosed.clear()
            self.declined.clear()  # a new intent may reopen previously-declined attributes
        elif self.last_asked:
            if NO_PREFERENCE_RE.search(message):
                self.declined.add(self.last_asked)
            else:
                self.disclosed[self.last_asked] = message
        self.messages.append(message)

    def record_turn_result(self, ask_attribute: str | None, message: str) -> None:
        self.last_asked = ask_attribute
        self.ask_log.append(ask_attribute)
        self.agent_messages.append(message)

    def query_text(self) -> str:
        # Full turn history, not just the latest message: this is what makes
        # retrieval "remember" constraints revealed several turns ago.
        messages = [message for message in self.messages if not NO_PREFERENCE_RE.search(message)]
        if self.override_start is not None and self.messages:
            after_override = [
                message
                for message in self.messages[self.override_start :]
                if not NO_PREFERENCE_RE.search(message)
            ]
            return " ".join([self.messages[0], *after_override])
        return " ".join(messages)


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
    """Dual-track retrieval -> LLM instruction sheet -> deterministic execution.

    Pipeline per turn: update SessionState -> route the turn to a Buying
    (precise keyword/filter) or Browsing (keyword + dense-embedding hybrid)
    track -> ask a local LLM (via Ollama) for a small JSON "instruction
    sheet" (constraints + weights + an optional clarifying question) ->
    apply that sheet to the candidates deterministically -> return top_k.
    Any LLM/embedding failure falls back to the plain retrieval order
    rather than raising.
    """

    MAX_TURNS = 10
    # Candidates scored by the executor. Kept independent of top_k (fixed
    # at 10 by the contract) so the LLM's weights have real room to rerank.
    CANDIDATE_POOL_SIZE = 180
    # Candidates actually shown to the LLM in the prompt -- kept small to
    # fit Ollama's default 4096-token context alongside conversation history.
    CANDIDATE_SAMPLE_SIZE = 6
    # How many dense-similarity hits to pull in on a Browsing turn, merged
    # with the keyword results before scoring.
    DENSE_CANDIDATE_SIZE = 100
    LLM_FAILURE_THRESHOLD = 3
    LLM_GUIDANCE_TURNS = frozenset({10})
    ASK_SEQUENCE = ("material", "feature", "color", "size", "style", "use_case", "brand", "budget", "other")
    QUESTION_BY_ATTRIBUTE = {
        "brand": "Do you have a brand preference?",
        "budget": "What budget range should I stay within?",
        "category": "Which product category should I focus on?",
        "color": "Do you have a color preference?",
        "feature": "What feature matters most for this item?",
        "material": "Do you have a material preference?",
        "other": "Is there one more requirement I should prioritize?",
        "size": "What size or fit should I look for?",
        "style": "What style should I prioritize?",
        "use_case": "How do you plan to use this item?",
    }

    EMBED_BATCH_SIZE = 50
    EMBED_CHECKPOINT_EVERY = 1000
    EMBED_MAX_CHARS = 600
    EMBEDDINGS_PATH = Path("data/embeddings.npy")
    EMBEDDING_IDS_PATH = Path("data/embeddings_ids.json")
    PARTIAL_EMBEDDINGS_PATH = Path("data/embeddings_partial.npy")
    PARTIAL_EMBEDDING_IDS_PATH = Path("data/embeddings_partial_ids.json")

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._products: dict[str, dict] = {}
        self._build_index()

        self.ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        self.embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.llm_timeout = float(os.environ.get("OLLAMA_TIMEOUT", "30"))
        self._llm_failures = 0
        self._llm_disabled = False
        self._dense_failures = 0
        self._dense_disabled = False

        self._embedding_matrix = None
        self._embedding_ids: list[str] = []
        self._dense_cache: dict[tuple[str, int], list[dict]] = {}
        self._dense_ready = False
        self._ensure_embeddings()

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
                fields = {
                    "parent_asin": parent_asin,
                    "title": _text(product.get("title")),
                    "categories": _text(product.get("categories")),
                    "features": _text(product.get("features")),
                    "details": _text(product.get("details")),
                    "store": _text(product.get("store")),
                    "description": _text(product.get("description")),
                    "price": _as_float(product.get("price")),
                    "average_rating": _as_float(product.get("average_rating")),
                    "rating_number": _as_float(product.get("rating_number")),
                }
                self._products[parent_asin] = fields
                batch.append(tuple(fields[column] for column in CATALOG_COLUMNS))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(profile=user_profile)

    # ---- Intent routing ------------------------------------------------

    def _route_intent(self, state: SessionState, user_message: str) -> str:
        """"buying" (precise) or "browsing" (exploratory), decided fresh
        each turn -- a session can move from browsing to buying as the
        customer reveals concrete constraints."""
        if OVERRIDE_RE.search(user_message):
            return "buying"  # a replaced preference is still a hard constraint
        if BUDGET_RE.search(user_message):
            return "buying"
        message_terms = set(_terms(user_message))
        if message_terms & (MATERIAL_WORDS | COLOR_WORDS | SIZE_WORDS):
            return "buying"
        if state.disclosed:
            return "buying"
        return "browsing"

    # ---- Stage 1a: cheap keyword retrieval -----------------------------

    def _category_terms(self, state: SessionState) -> list[str]:
        if not state.messages:
            return []
        match = CATEGORY_RE.search(state.messages[0])
        if not match:
            return []
        terms = [
            term
            for term in _terms(match.group(1))
            if term not in CATEGORY_FILLER_TERMS
        ]
        return _expand_query_terms(list(dict.fromkeys(terms)), 12)

    @staticmethod
    def _fts_or_terms(terms: list[str]) -> str:
        return " OR ".join(f'"{term}"' for term in terms)

    def _bm25_query_expressions(self, state: SessionState, sheet: InstructionSheet) -> list[str]:
        category_terms = self._category_terms(state)
        constraint_terms: list[str] = []
        for key, value in sheet.constraints.items():
            if _constraint_attribute(key) == "budget":
                continue
            constraint_terms.extend(_constraint_terms(value))
        constraint_terms = _expand_query_terms(list(dict.fromkeys(constraint_terms)), 24)
        broad_terms = _expand_query_terms(list(dict.fromkeys(_terms(state.query_text()))), 45)

        expressions: list[str] = []
        if category_terms and constraint_terms:
            category_group = self._fts_or_terms(category_terms)
            constraint_group = self._fts_or_terms(constraint_terms)
            expressions.append(f"({category_group}) AND ({constraint_group})")
        if category_terms:
            expressions.append(self._fts_or_terms(category_terms))
        if constraint_terms:
            expressions.append(self._fts_or_terms(constraint_terms))
        if broad_terms:
            expressions.append(self._fts_or_terms(broad_terms))
        return list(dict.fromkeys(expression for expression in expressions if expression))

    def _retrieve_candidates(self, state: SessionState, limit: int, sheet: InstructionSheet) -> list[dict]:
        seen: set[str] = set()
        candidates: list[dict] = []
        for expression in self._bm25_query_expressions(state, sheet):
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, limit),
            ).fetchall()
            for row in rows:
                parent_asin = row[0]
                if parent_asin in seen or parent_asin not in self._products:
                    continue
                seen.add(parent_asin)
                candidates.append(self._products[parent_asin])
                if len(candidates) >= limit:
                    return candidates
        return candidates

    # ---- Stage 1b: dense/embedding retrieval (Browsing track) ----------

    def _embedding_text(self, product: dict) -> str:
        parts = [product.get(field_name, "") for field_name in ("title", "categories", "features", "description")]
        combined = " ".join(part for part in parts if part).strip()
        return (combined or product.get("title") or "product")[: self.EMBED_MAX_CHARS]

    def _embed_batch(self, texts: list[str], attempts: int = 3) -> list[list[float]]:
        payload = json.dumps({"model": self.embed_model, "input": texts}).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(attempts):
            request = urllib.request.Request(
                f"{self.ollama_host}/api/embed",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=max(60.0, self.llm_timeout)) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return body["embeddings"]
            except Exception as exc:  # transient Ollama/GPU hiccups do happen mid-batch
                last_error = exc
                if attempt < attempts - 1:
                    time.sleep(1.5 * (attempt + 1))
        raise last_error  # type: ignore[misc]

    def _build_embeddings(self) -> None:
        parent_asins = list(self._products.keys())
        total = len(parent_asins)
        eta_minutes = total * 0.03 / 60
        print(f"[agent] building embedding index for {total} products (one-time, ~{eta_minutes:.0f} min)...")

        completed_matrix = None
        completed_count = 0
        if self.PARTIAL_EMBEDDINGS_PATH.exists() and self.PARTIAL_EMBEDDING_IDS_PATH.exists():
            try:
                partial_ids = json.loads(self.PARTIAL_EMBEDDING_IDS_PATH.read_text(encoding="utf-8"))
                partial_matrix = np.load(self.PARTIAL_EMBEDDINGS_PATH)
                if (
                    isinstance(partial_ids, list)
                    and len(partial_ids) == partial_matrix.shape[0]
                    and parent_asins[: len(partial_ids)] == partial_ids
                ):
                    completed_matrix = partial_matrix.astype(np.float32, copy=False)
                    completed_count = len(partial_ids)
                    print(f"[agent] resuming embedding index from {completed_count}/{total} ({100 * completed_count / total:.1f}%)")
                else:
                    print("[agent] ignoring stale partial embedding cache -- catalog order does not match.")
            except Exception as exc:
                print(f"[agent] ignoring unreadable partial embedding cache ({exc}).")

        vectors: list[list[float]] = []
        for start in range(completed_count, total, self.EMBED_BATCH_SIZE):
            batch_ids = parent_asins[start : start + self.EMBED_BATCH_SIZE]
            batch_texts = [self._embedding_text(self._products[pid]) for pid in batch_ids]
            vectors.extend(self._embed_batch(batch_texts))
            done = start + len(batch_ids)
            print(f"[agent]   {done}/{total} ({100 * done / total:.1f}%)", flush=True)
            if done == total or done % self.EMBED_CHECKPOINT_EVERY == 0:
                completed_matrix = self._write_embedding_checkpoint(parent_asins[:done], completed_matrix, vectors)
                completed_count = done
                vectors.clear()

        matrix = self._combine_embedding_chunks(completed_matrix, vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms

        self.EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.EMBEDDINGS_PATH, matrix)
        self.EMBEDDING_IDS_PATH.write_text(json.dumps(parent_asins), encoding="utf-8")
        self.PARTIAL_EMBEDDINGS_PATH.unlink(missing_ok=True)
        self.PARTIAL_EMBEDDING_IDS_PATH.unlink(missing_ok=True)

        self._embedding_matrix = matrix
        self._embedding_ids = parent_asins
        self._dense_ready = True
        print(f"[agent] embedding index ready: {matrix.shape}")

    def _combine_embedding_chunks(self, completed_matrix: object, vectors: list[list[float]]) -> object:
        pending_matrix = np.array(vectors, dtype=np.float32)
        if completed_matrix is None:
            return pending_matrix
        if not len(vectors):
            return completed_matrix
        return np.concatenate((completed_matrix, pending_matrix), axis=0)

    def _write_embedding_checkpoint(
        self,
        parent_asins: list[str],
        completed_matrix: object,
        vectors: list[list[float]],
    ) -> object:
        matrix = self._combine_embedding_chunks(completed_matrix, vectors)
        self.PARTIAL_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_embeddings_path = self.PARTIAL_EMBEDDINGS_PATH.with_suffix(".tmp.npy")
        tmp_ids_path = self.PARTIAL_EMBEDDING_IDS_PATH.with_suffix(".tmp.json")
        np.save(tmp_embeddings_path, matrix)
        tmp_ids_path.write_text(json.dumps(parent_asins), encoding="utf-8")
        tmp_embeddings_path.replace(self.PARTIAL_EMBEDDINGS_PATH)
        tmp_ids_path.replace(self.PARTIAL_EMBEDDING_IDS_PATH)
        return matrix

    def _ensure_embeddings(self) -> None:
        if not _NUMPY_AVAILABLE:
            print("[agent] numpy not available -- dense retrieval disabled, Browsing falls back to keyword search.")
            return
        build_requested = os.environ.get("BUILD_EMBEDDINGS", "").lower() in {"1", "true", "yes", "on"}
        try:
            if self.EMBEDDINGS_PATH.exists() and self.EMBEDDING_IDS_PATH.exists():
                ids = json.loads(self.EMBEDDING_IDS_PATH.read_text(encoding="utf-8"))
                matrix = np.load(self.EMBEDDINGS_PATH)
                if len(ids) == matrix.shape[0] == len(self._products):
                    self._embedding_matrix = matrix
                    self._embedding_ids = ids
                    self._dense_ready = True
                    return
                print("[agent] cached embeddings don't match the current catalog -- rebuilding.")
            if not build_requested:
                print("[agent] embedding cache missing -- set BUILD_EMBEDDINGS=1 to build it; Browsing falls back to keyword search.")
                return
            self._build_embeddings()
        except Exception as exc:
            print(f"[agent] dense retrieval unavailable ({exc}) -- Browsing falls back to keyword search.")

    def _dense_candidates(self, query_text: str, limit: int) -> list[dict]:
        if not self._dense_ready or self._dense_disabled or not query_text.strip():
            return []
        cache_key = (query_text[: self.EMBED_MAX_CHARS], limit)
        cached = self._dense_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            vectors = self._embed_batch([cache_key[0]], attempts=1)
            vector = np.array(vectors[0], dtype=np.float32)
            norm = np.linalg.norm(vector)
            if not norm:
                return []
            vector = vector / norm
            self._dense_failures = 0
        except Exception:
            self._dense_failures += 1
            if self._dense_failures >= self.LLM_FAILURE_THRESHOLD:
                self._dense_disabled = True
            return []

        scores = self._embedding_matrix @ vector
        k = min(limit, scores.shape[0])
        top_indices = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        candidates = [self._products[self._embedding_ids[i]] for i in top_indices]
        if len(self._dense_cache) > 512:
            self._dense_cache.clear()
        self._dense_cache[cache_key] = candidates
        return candidates

    @staticmethod
    def _merge_candidates(primary: list[dict], secondary: list[dict], limit: int) -> list[dict]:
        seen: set[str] = set()
        merged: list[dict] = []
        for group in (primary, secondary):
            for candidate in group:
                parent_asin = candidate["parent_asin"]
                if parent_asin in seen:
                    continue
                seen.add(parent_asin)
                merged.append(candidate)
                if len(merged) >= limit:
                    return merged
        return merged

    # ---- Code-side guidance --------------------------------------------

    def _add_constraint(
        self,
        constraints: dict[str, str],
        weights: dict[str, float],
        attribute: str,
        value: str,
        weight: float,
    ) -> None:
        value = _clean_constraint(value)
        if not value or NO_PREFERENCE_RE.search(value):
            return
        meaningful_terms = _constraint_terms(value)
        if not meaningful_terms:
            return

        key = attribute
        if key in constraints and constraints[key].lower() != value.lower():
            suffix = 2
            while f"{attribute}_{suffix}" in constraints:
                suffix += 1
            key = f"{attribute}_{suffix}"
        constraints[key] = value
        weights[key] = max(weights.get(key, 0.0), weight)

    def _infer_attribute_for_value(self, value: str) -> str:
        lowered_terms = set(_terms(value))
        if BUDGET_RE.search(value):
            return "budget"
        if lowered_terms & MATERIAL_WORDS:
            return "material"
        if lowered_terms & COLOR_WORDS:
            return "color"
        if lowered_terms & SIZE_WORDS:
            return "size"
        if lowered_terms & {"hiking", "running", "gym", "winter", "outdoor", "work", "walking"}:
            return "use_case"
        if lowered_terms & {"fit", "neck", "sleeve", "style", "torso"}:
            return "style"
        return "feature"

    def _extract_constraint_phrases(self, text: str) -> list[str]:
        phrases: list[str] = []
        for match in CONSTRAINT_INTRO_RE.finditer(text):
            for item in match.group(1).split(";"):
                cleaned = _clean_constraint(item)
                if cleaned:
                    phrases.append(cleaned)
        return phrases

    def _deterministic_instruction_sheet(self, state: SessionState) -> InstructionSheet:
        constraints: dict[str, str] = {}
        weights: dict[str, float] = {}
        query_text = state.query_text()

        for phrase in self._extract_constraint_phrases(query_text):
            self._add_constraint(
                constraints,
                weights,
                self._infer_attribute_for_value(phrase),
                phrase,
                0.9,
            )

        for attribute, value in state.disclosed.items():
            for phrase in self._extract_constraint_phrases(value) or [value]:
                inferred = attribute if attribute in ALLOWED_ATTRIBUTES else self._infer_attribute_for_value(phrase)
                if inferred == "other":
                    inferred = self._infer_attribute_for_value(phrase)
                self._add_constraint(constraints, weights, inferred, phrase, 0.85)

        terms = set(_terms(query_text))
        for material in sorted(terms & MATERIAL_WORDS):
            self._add_constraint(constraints, weights, "material", material, 0.7)
        for color in sorted(terms & COLOR_WORDS):
            self._add_constraint(constraints, weights, "color", color, 0.7)

        budget = _as_float(query_text) if BUDGET_RE.search(query_text) else None
        if budget is not None:
            self._add_constraint(constraints, weights, "budget", str(budget), 0.8)

        return InstructionSheet(constraints=constraints, weights=weights)

    def _merge_instruction_sheets(self, llm_sheet: InstructionSheet, deterministic_sheet: InstructionSheet) -> InstructionSheet:
        constraints = dict(llm_sheet.constraints)
        weights = dict(llm_sheet.weights)
        for key, value in deterministic_sheet.constraints.items():
            constraints[key] = value
            weights[key] = max(weights.get(key, 0.0), deterministic_sheet.weights.get(key, 0.7))
        return InstructionSheet(
            constraints=constraints,
            weights=weights,
            ask_attribute=llm_sheet.ask_attribute,
            message=llm_sheet.message,
            usage=llm_sheet.usage,
        )

    def _known_or_spent_attributes(self, state: SessionState, sheet: InstructionSheet) -> set[str]:
        spent = {
            attribute
            for attribute in [*state.ask_log, *state.declined, *state.disclosed.keys()]
            if attribute
        }
        if self._category_terms(state):
            spent.add("category")
        for key in sheet.constraints:
            attribute = _constraint_attribute(key)
            if attribute in ALLOWED_ATTRIBUTES:
                spent.add(attribute)
        return spent

    def _choose_ask_attribute(self, state: SessionState, sheet: InstructionSheet) -> str | None:
        if state.turn >= self.MAX_TURNS:
            return None
        spent = self._known_or_spent_attributes(state, sheet)
        if sheet.ask_attribute and sheet.ask_attribute not in spent:
            return sheet.ask_attribute
        for attribute in self.ASK_SEQUENCE:
            if attribute not in spent:
                return attribute
        return None

    def _candidate_certainty(self, state: SessionState, sheet: InstructionSheet, candidates: list[dict]) -> float:
        if not candidates:
            return 0.0
        top_candidates = candidates[: min(10, len(candidates))]
        category_terms = self._category_terms(state)
        category_hits = 0
        if category_terms:
            for candidate in top_candidates:
                category_text = " ".join(
                    str(candidate.get(field_name, ""))
                    for field_name in ("title", "categories")
                ).lower()
                if any(term in category_text for term in category_terms):
                    category_hits += 1
        category_score = category_hits / len(top_candidates) if category_terms else 0.0

        weighted_total = 0.0
        weighted_possible = 0.0
        for candidate in top_candidates[:5]:
            combined_text = " ".join(
                candidate.get(field_name, "") for field_name in CATALOG_COLUMNS[1:]
            ).lower()
            for key, value in sheet.constraints.items():
                attribute = _constraint_attribute(key)
                if attribute == "budget":
                    continue
                weight = sheet.weights.get(key, 0.5)
                weighted_total += weight * self._constraint_match_score(attribute, value, candidate, combined_text)
                weighted_possible += weight
        constraint_score = weighted_total / weighted_possible if weighted_possible else 0.0

        if weighted_possible:
            return 0.7 * constraint_score + 0.3 * category_score
        return category_score

    # ---- Stage 2: LLM instruction sheet ---------------------------------

    def _summarize_candidate(self, candidate: dict) -> str:
        title = (candidate.get("title") or "product").strip()
        if len(title) > 70:
            title = title[:67].rstrip() + "..."
        price = candidate.get("price")
        price_text = f"${price:.2f}" if isinstance(price, float) else "price n/a"
        categories = " ".join((candidate.get("categories") or "").split()[:6])
        return f'"{title}" | {price_text} | {categories}'.strip()

    def _build_user_prompt(self, state: SessionState, candidates: list[dict], track: str) -> str:
        profile = state.profile or {}
        track_note = (
            "Detected track: BUYING -- the customer has given something concrete; "
            "prioritize precision and lock in hard constraints."
            if track == "buying"
            else "Detected track: BROWSING -- the customer is still exploring; the "
            "candidate pool is broader by design, favor a clarifying question "
            "unless one option is clearly strongest."
        )
        lines = [
            f"Turn {state.turn} of {self.MAX_TURNS}.",
            track_note,
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
        if state.declined:
            lines.append(
                "Customer has NO preference for these -- do not ask again: "
                + ", ".join(sorted(state.declined))
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
                "options": {"temperature": 0.2, "num_ctx": 4096, "num_predict": 140},
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
            "prompt_tokens": max(0, int(body.get("prompt_eval_count") or 0)),
            "completion_tokens": max(0, int(body.get("eval_count") or 0)),
        }
        return content, usage

    def _get_instruction_sheet(self, state: SessionState, candidates: list[dict], track: str) -> InstructionSheet:
        if self._llm_disabled:
            return InstructionSheet.fallback()
        user_prompt = self._build_user_prompt(state, candidates, track)
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

    def _should_call_llm(
        self,
        state: SessionState,
        deterministic_sheet: InstructionSheet,
        candidates: list[dict],
    ) -> bool:
        if self._llm_disabled:
            return False
        certainty = self._candidate_certainty(state, deterministic_sheet, candidates)
        if state.override_seen and state.turn == state.override_turn:
            return True
        if state.override_seen and state.turn == self.MAX_TURNS and certainty < 0.65:
            return True
        if not deterministic_sheet.constraints:
            return state.turn <= 2 or (
                state.turn in self.LLM_GUIDANCE_TURNS and certainty < 0.75
            )
        if (
            state.turn in self.LLM_GUIDANCE_TURNS
            and len(deterministic_sheet.constraints) < 3
            and certainty < 0.75
        ):
            return True
        return False

    # ---- Stage 3: deterministic execution -------------------------------

    def _constraint_match_score(self, attribute: str, value: str, candidate: dict, combined_text: str) -> float:
        if attribute == "budget":
            budget_max = _as_float(value)
            price = candidate.get("price")
            if budget_max is not None and price is not None and price <= budget_max:
                return 1.0
            return 0.0

        lowered_value = value.lower().strip()
        if lowered_value and lowered_value in combined_text:
            return 1.0

        terms = list(dict.fromkeys(_constraint_terms(value)))
        if not terms:
            return 0.0
        matched = sum(1 for term in terms if term in combined_text)
        if matched == len(terms):
            return 0.9
        if matched >= 2:
            return 0.35 + 0.45 * (matched / len(terms))
        if len(terms) == 1 and matched == 1:
            return 0.75
        return 0.0

    def _profile_score(self, profile: dict, candidate: dict, combined_text: str) -> float:
        score = 0.0
        for tag in profile.get("preference_tags") or []:
            tag_text = str(tag).lower()
            if tag_text and tag_text in combined_text:
                score += 0.025
        rating = candidate.get("average_rating")
        if isinstance(rating, float):
            if rating >= 4.6:
                score += 0.035
            elif rating >= 4.3:
                score += 0.02
        rating_number = candidate.get("rating_number")
        if isinstance(rating_number, float) and rating_number > 0:
            score += min(0.09, 0.018 * len(str(int(rating_number))))
        return min(score, 0.12)

    def _score_candidates(self, candidates: list[dict], sheet: InstructionSheet, state: SessionState) -> list[str]:
        scored: list[tuple[float, int, str]] = []
        for rank, candidate in enumerate(candidates):
            combined_text = " ".join(
                candidate.get(field_name, "") for field_name in CATALOG_COLUMNS[1:]
            ).lower()

            score = 0.0
            for attribute, value in sheet.constraints.items():
                weight = sheet.weights.get(attribute, 0.5)
                score += weight * self._constraint_match_score(
                    _constraint_attribute(attribute),
                    value,
                    candidate,
                    combined_text,
                )
            score += self._profile_score(state.profile, candidate, combined_text)
            # Tie-break toward the original retrieval order so an empty/
            # failed instruction sheet degrades to plain retrieval, not a
            # shuffle.
            score += 1.0 / (1 + rank) * 0.1
            scored.append((score, rank, candidate["parent_asin"]))

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

        # Everything below touches retrieval/scoring/LLM logic that we own;
        # the LLM and embedding calls already catch their own failures, but
        # this is a last line of defense so a stray bug in our code still
        # returns a valid, contract-shaped response instead of raising and
        # counting the turn (or the whole session) as a hard miss.
        try:
            state.record_message(user_message, turn)

            track = self._route_intent(state, user_message)
            deterministic_sheet = self._deterministic_instruction_sheet(state)
            candidates = self._retrieve_candidates(state, self.CANDIDATE_POOL_SIZE, deterministic_sheet)
            if self._dense_ready and self._candidate_certainty(state, deterministic_sheet, candidates) < 0.85:
                dense_hits = self._dense_candidates(state.query_text(), self.DENSE_CANDIDATE_SIZE)
                candidates = self._merge_candidates(candidates, dense_hits, self.CANDIDATE_POOL_SIZE)

            if not candidates:
                message = "Tell me a bit more about what you're looking for."
                state.record_turn_result(None, message)
                return {
                    "message": message,
                    "ask_attribute": None,
                    "recommendations": [],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                }

            llm_sheet = (
                self._get_instruction_sheet(state, candidates, track)
                if self._should_call_llm(state, deterministic_sheet, candidates)
                else InstructionSheet.fallback()
            )
            sheet = self._merge_instruction_sheets(llm_sheet, deterministic_sheet)
            ranked_ids = self._score_candidates(candidates, sheet, state)[:top_k]

            # Hard enforcement, not just a prompt instruction: never re-ask
            # an attribute the customer already answered or declined,
            # regardless of what the LLM returned.
            ask_attribute = self._choose_ask_attribute(state, sheet)
            message = (
                self.QUESTION_BY_ATTRIBUTE[ask_attribute]
                if ask_attribute
                else sheet.message or "Here are the closest matches I found."
            )

            state.record_turn_result(ask_attribute, message)
            return {
                "message": message,
                "ask_attribute": ask_attribute,
                "recommendations": [{"parent_asin": pid} for pid in ranked_ids],
                "usage": sheet.usage,
            }
        except Exception:
            return {
                "message": "Sorry, could you tell me more about what you're looking for?",
                "ask_attribute": None,
                "recommendations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
