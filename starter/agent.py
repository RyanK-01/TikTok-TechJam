from __future__ import annotations

import json
import math
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
    # Conversational filler: closed-class words above miss these, and
    # because retail copy doesn't talk like a customer chats, plain
    # catalog-rarity (see _term_rarity) actually scores them *higher* than
    # real attribute words like "pants" or "shirt" -- rare in product text,
    # but for the wrong reason. Left in, they'd out-rank genuine content
    # words for the query-term budget.
    "actually", "maybe", "really", "very", "just", "so", "well", "yeah",
    "yes", "okay", "ok", "hi", "hello", "hey", "thanks", "thank", "am",
    "was", "were", "do", "does", "did", "can", "could", "should", "shall",
    "may", "might", "must", "kind", "sort", "basically", "literally",
    "totally", "pretty", "quite", "rather", "still", "also", "even",
    "much", "many", "more", "most", "think", "guess", "know", "mean",
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
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon",
    "fabric", "denim", "suede", "linen", "cashmere", "velvet", "satin", "canvas",
    "mesh", "fleece", "corduroy", "tweed", "chiffon", "lace", "faux", "vegan",
    "synthetic", "elastane", "lycra", "polyurethane", "microfiber", "sherpa",
    "flannel", "jersey", "knit", "acrylic", "viscose", "modal", "bamboo", "hemp",
}
COLOR_WORDS = {
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "navy", "beige", "tan", "khaki", "maroon",
    "teal", "turquoise", "gold", "silver", "ivory", "cream", "burgundy", "olive",
    "mustard", "lavender", "coral", "charcoal", "multicolor", "multicolour",
    "rose", "magenta", "cyan", "indigo", "plum", "mint", "peach", "camel", "rust",
}
SIZE_WORDS = {
    "small", "medium", "large", "size", "xl", "xs", "xxl", "xxs", "xxxl", "wide",
    "narrow", "petite", "tall", "short", "long", "slim", "loose", "relaxed",
    "oversized", "regular", "plus", "big", "junior", "toddler", "youth",
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
    # attributes the customer explicitly has no preference for -- distinct
    # from disclosed (there's no value here, just "don't ask again")
    declined: set[str] = field(default_factory=set)
    last_asked: str | None = None
    override_seen: bool = False
    # Salient terms from the customer's very first message (see
    # Agent._capture_category) -- captured once and never cleared, even
    # across an Intent Override, on the same reasoning OVERRIDE_RE already
    # uses above: the broad category ("running shoes") almost always
    # survives a change to one constraint ("actually, make it black"),
    # so it's worth anchoring retrieval + scoring on regardless of what
    # else the customer revises later.
    category_terms: list[str] = field(default_factory=list)
    # The last successfully-scored ranking. Re-served (via
    # Agent._ensure_top_k) if a later turn comes up short -- e.g. a retrieval
    # miss or empty candidate pool -- so a transient failure never forces an
    # empty/short response when we already had a decent guess on hand.
    last_ranked: list[str] = field(default_factory=list)

    def record_message(self, message: str, turn: int) -> None:
        self.turn = turn
        if OVERRIDE_RE.search(message):
            self.override_seen = True
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
    # Candidates scored by the executor on a Buying turn. Kept independent of
    # top_k (fixed at 10 by the contract) so the LLM's weights have real room
    # to rerank. 100 was measured too shallow: a real match can sit well
    # outside the top 100 by plain BM25/dense order and only surface once
    # exact-constraint scoring re-ranks it -- but scoring can't rescue a
    # candidate that retrieval already dropped.
    CANDIDATE_POOL_SIZE = 400
    # Browsing turns have no disclosed constraints yet, so the query is
    # necessarily broader and recall matters more than precision -- pull a
    # deeper pool than Buying's.
    BROWSING_POOL_SIZE = 600
    # Candidates actually shown to the LLM in the prompt -- kept small to
    # fit Ollama's default 4096-token context alongside conversation history.
    # Independent of the pool sizes above: the LLM only needs a
    # representative sample to infer constraints/weights from, not the full
    # pool (that's scored deterministically afterward).
    CANDIDATE_SAMPLE_SIZE = 18
    # How many dense-similarity hits to pull in on a Browsing turn, merged
    # with the keyword results before scoring.
    DENSE_CANDIDATE_SIZE = 60
    # Max terms fed into the FTS MATCH query (see _ranked_query_terms). This
    # only sizes a SQLite OR-query, not the Ollama prompt -- CANDIDATE_SAMPLE_SIZE
    # and conversation length are what matter for the LLM's context budget.
    QUERY_TERM_LIMIT = 70
    LLM_FAILURE_THRESHOLD = 3

    EMBED_BATCH_SIZE = 250
    EMBED_MAX_CHARS = 600
    EMBEDDINGS_PATH = Path("data/embeddings.npy")
    EMBEDDING_IDS_PATH = Path("data/embeddings_ids.json")

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._products: dict[str, dict] = {}
        self._catalog_size = 0
        self._term_rarity_cache: dict[str, float] = {}
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
        popularity: list[tuple[float, str]] = []
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
                }
                self._products[parent_asin] = fields
                # rating_number (review count), not average_rating, as the
                # popularity signal -- a handful of 5-star reviews shouldn't
                # outrank a well-established, widely-bought item.
                popularity.append((_as_float(product.get("rating_number")) or 0.0, parent_asin))
                batch.append(tuple(fields[column] for column in CATALOG_COLUMNS))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        # 'row' vocab: one row per term with the number of documents (products)
        # it appears in. Backed by the same FTS index, so this is an
        # index-lookup per term, not a new corpus scan -- used to weight
        # query terms by how discriminative they are (see _term_rarity).
        cursor.execute("CREATE VIRTUAL TABLE products_vocab USING fts5vocab('products', 'row')")
        self.connection.commit()
        self._catalog_size = len(self._products)

        # Absolute last-resort recommendations (see _ensure_top_k): if a turn
        # ever comes up short on real candidates, top up with the catalog's
        # most-reviewed products rather than returning fewer than top_k. A
        # blind popular-item guess scores the same as an empty slot (0) on a
        # miss, but never worse, and occasionally hits by sheer popularity.
        popularity.sort(key=lambda item: -item[0])
        self._fallback_ids: list[str] = [parent_asin for _, parent_asin in popularity[:10]]

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

    def _term_rarity(self, term: str) -> float:
        """How discriminative `term` is across the catalog -- a classic idf.
        A common word ("shirt") barely narrows the candidate pool; a rare
        one ("corduroy") does most of the work, so it should outweigh a
        recently-said but generic word when both are competing for the
        query-term budget. Backed by SQLite's own fts5vocab table (an
        index lookup, not a corpus scan), and cached since the catalog is
        static for the life of the process."""
        cached = self._term_rarity_cache.get(term)
        if cached is not None:
            return cached
        row = self.connection.execute(
            "SELECT doc FROM products_vocab WHERE term = ?", (term,)
        ).fetchone()
        doc_freq = row[0] if row else 0
        # +1 smoothing on both sides: keeps the score finite and positive
        # even for a term absent from the catalog entirely (a typo, a brand
        # name outside the index) instead of dividing by zero or blowing up.
        rarity = math.log((self._catalog_size + 1) / (doc_freq + 1)) + 1.0
        self._term_rarity_cache[term] = rarity
        return rarity

    def _ranked_query_terms(self, state: SessionState, limit: int) -> list[str]:
        """Pick up to `limit` search terms from the full turn history,
        favoring terms that are both recent and catalog-rare over ones that
        are merely recent-and-generic. A term confirmed via a direct
        clarifying-question answer (state.disclosed) or captured as a
        category anchor (state.category_terms, see _capture_category)
        always survives the cut, regardless of score -- neither is
        incidental word choice."""
        total_turns = len(state.messages)
        if not total_turns:
            return []

        # Most recent turn each term appeared in (1-indexed) -- a term
        # mentioned again later "refreshes" its recency.
        last_seen_turn: dict[str, int] = {}
        for turn_index, message in enumerate(state.messages, start=1):
            for term in _terms(message):
                last_seen_turn[term] = turn_index

        guaranteed_terms: set[str] = set(state.category_terms)
        for value in state.disclosed.values():
            guaranteed_terms.update(_terms(value))

        def score(term: str, turn_index: int) -> float:
            recency_weight = turn_index / total_turns  # (0, 1], most recent turn == 1.0
            return recency_weight * self._term_rarity(term)

        ranked = sorted(last_seen_turn, key=lambda term: -score(term, last_seen_turn[term]))
        guaranteed = [term for term in ranked if term in guaranteed_terms]
        rest = [term for term in ranked if term not in guaranteed_terms]
        return (guaranteed + rest)[:limit]

    def _retrieve_candidates(self, state: SessionState, limit: int) -> list[dict]:
        unique_terms = self._ranked_query_terms(state, self.QUERY_TERM_LIMIT)
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, limit),
        ).fetchall()
        # Look up the full record (incl. price/rating, which the FTS table
        # itself doesn't carry) instead of rebuilding a partial dict from
        # the SQL row -- self._products is the single source of truth.
        return [self._products[row[0]] for row in rows if row[0] in self._products]

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
        vectors: list[list[float]] = []
        for start in range(0, total, self.EMBED_BATCH_SIZE):
            batch_ids = parent_asins[start : start + self.EMBED_BATCH_SIZE]
            batch_texts = [self._embedding_text(self._products[pid]) for pid in batch_ids]
            vectors.extend(self._embed_batch(batch_texts))
            done = start + len(batch_ids)
            print(f"[agent]   {done}/{total} ({100 * done / total:.1f}%)", flush=True)

        matrix = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms

        self.EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.EMBEDDINGS_PATH, matrix)
        self.EMBEDDING_IDS_PATH.write_text(json.dumps(parent_asins), encoding="utf-8")

        self._embedding_matrix = matrix
        self._embedding_ids = parent_asins
        self._dense_ready = True
        print(f"[agent] embedding index ready: {matrix.shape}")

    def _ensure_embeddings(self) -> None:
        if not _NUMPY_AVAILABLE:
            print("[agent] numpy not available -- dense retrieval disabled, Browsing falls back to keyword search.")
            return
        try:
            if self.EMBEDDINGS_PATH.exists() and self.EMBEDDING_IDS_PATH.exists():
                ids = json.loads(self.EMBEDDING_IDS_PATH.read_text(encoding="utf-8"))
                matrix = np.load(self.EMBEDDINGS_PATH)
                if len(ids) == matrix.shape[0] == len(self._products):
                    # Row count matching the catalog doesn't guarantee the
                    # cache was built with the embedding model we'll use for
                    # live queries -- a stale cache from a different model
                    # (different vector width) would otherwise pass this
                    # check, then hard-crash every dense lookup later deep
                    # inside a matmul (see _dense_candidates). One cheap
                    # live probe call catches that up front, at cost of a
                    # single extra embed call per process start.
                    live_dim = self._probe_embedding_dim()
                    if live_dim is not None and live_dim != matrix.shape[1]:
                        print(
                            f"[agent] cached embeddings are {matrix.shape[1]}-dim but the live "
                            f"'{self.embed_model}' model produces {live_dim}-dim vectors -- "
                            "dense retrieval disabled until the cache is rebuilt (delete "
                            f"{self.EMBEDDINGS_PATH} and {self.EMBEDDING_IDS_PATH}, or set "
                            "OLLAMA_EMBED_MODEL to match the model that built the cache)."
                        )
                        return
                    self._embedding_matrix = matrix
                    self._embedding_ids = ids
                    self._dense_ready = True
                    return
                print("[agent] cached embeddings don't match the current catalog -- rebuilding.")
            self._build_embeddings()
        except Exception as exc:
            print(f"[agent] dense retrieval unavailable ({exc}) -- Browsing falls back to keyword search.")

    def _probe_embedding_dim(self) -> int | None:
        """One throwaway live embed call, used only to size-check the cache
        against whatever model is actually configured right now. Returns
        None (rather than raising) if Ollama isn't reachable at startup --
        that's a separate, already-handled failure mode, not evidence of a
        cache mismatch, so it shouldn't block trusting an otherwise-valid
        cache."""
        try:
            vectors = self._embed_batch(["dimension probe"], attempts=1)
            return len(vectors[0])
        except Exception:
            return None

    def _dense_candidates(self, query_text: str, limit: int) -> list[dict]:
        if not self._dense_ready or self._dense_disabled or not query_text.strip():
            return []
        try:
            vectors = self._embed_batch([query_text[: self.EMBED_MAX_CHARS]], attempts=1)
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

        # Belt-and-suspenders: _ensure_embeddings probes for this at startup,
        # but a live query vector of the wrong width must never reach the
        # matmul below -- that raises ValueError uncaught by any try/except
        # in this method, which used to propagate all the way up through
        # respond()'s outer catch-all and silently blank out the entire
        # turn (empty recommendations, no LLM call) instead of just
        # skipping the dense contribution for this one query.
        if vector.shape[0] != self._embedding_matrix.shape[1]:
            self._dense_disabled = True
            print(
                f"[agent] live query embedding is {vector.shape[0]}-dim but the cached matrix "
                f"is {self._embedding_matrix.shape[1]}-dim -- disabling dense retrieval."
            )
            return []

        scores = self._embedding_matrix @ vector
        k = min(limit, scores.shape[0])
        top_indices = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [self._products[self._embedding_ids[i]] for i in top_indices]

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

    # ---- Stage 3: deterministic execution -------------------------------

    # Per-term category bonus and its cap. Deliberately small and capped:
    # the category anchor is a lower-confidence, secondary signal (it's just
    # whatever stood out in the opening message) and must never be able to
    # outweigh an actual disclosed constraint, which is why this is a fixed
    # ceiling rather than summed unbounded -- a candidate sharing several
    # generic category words but matching none of the customer's real
    # constraints should still lose to one that matches a real constraint.
    CATEGORY_TERM_BONUS = 0.15
    CATEGORY_BONUS_CAP = 0.5

    def _score_candidates(self, state: SessionState, candidates: list[dict], sheet: InstructionSheet) -> list[str]:
        scored: list[tuple[float, int, str]] = []
        for rank, candidate in enumerate(candidates):
            combined_text = " ".join(
                candidate.get(field_name, "") for field_name in CATALOG_COLUMNS[1:]
            ).lower()
            # Token set, not raw text -- lets constraint matching below survive
            # word order/spacing differences from a plain substring check
            # (e.g. constraint "leather boots" now matches "boots made of
            # leather"). Plurals/stems still won't collide (a real stemmer
            # would be the next step); MATERIAL_WORDS/COLOR_WORDS/SIZE_WORDS
            # cover the common variant spellings instead.
            combined_terms = set(_terms(combined_text))

            score = 0.0
            for attribute, value in sheet.constraints.items():
                weight = sheet.weights.get(attribute, 0.5)
                if attribute == "budget":
                    budget_max = _as_float(value)
                    price = candidate.get("price")
                    if budget_max is not None and price is not None and price <= budget_max:
                        score += weight
                else:
                    value_terms = set(_terms(value))
                    if value_terms and value_terms <= combined_terms:
                        score += weight

            if state.category_terms:
                category_hits = sum(1 for term in state.category_terms if term in combined_terms)
                score += min(category_hits * self.CATEGORY_TERM_BONUS, self.CATEGORY_BONUS_CAP)

            # Tie-break toward the original retrieval order so an empty/
            # failed instruction sheet degrades to plain retrieval, not a
            # shuffle.
            score += 1.0 / (1 + rank) * 0.1
            scored.append((score, rank, candidate["parent_asin"]))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [parent_asin for _, _, parent_asin in scored]

    def _capture_category(self, state: SessionState, message: str) -> None:
        """Anchor on the opening message's most salient terms once, on turn
        1 -- reuses the same recency x rarity scoring as _ranked_query_terms
        so "category" here means whatever's genuinely discriminative in the
        opener, not just its first few words. See the category_terms field
        comment on SessionState for why this is never refreshed later."""
        if state.category_terms:
            return
        candidates = list(dict.fromkeys(_terms(message)))
        if not candidates:
            return
        ranked = sorted(candidates, key=lambda term: -self._term_rarity(term))
        state.category_terms = ranked[:6]

    def _ensure_top_k(self, state: SessionState, ranked_ids: list[str], top_k: int) -> list[str]:
        """Top up a short ranking rather than ever returning fewer than
        top_k: first with the last turn's good ranking (still plausible --
        the target rarely moves far turn to turn), then with the catalog's
        most-reviewed products (a popular-item guess scores the same as an
        empty slot on a miss, but never worse, and occasionally hits)."""
        out: list[str] = []
        seen: set[str] = set()
        for source in (ranked_ids, state.last_ranked, self._fallback_ids):
            for parent_asin in source:
                if parent_asin in seen or parent_asin not in self._products:
                    continue
                seen.add(parent_asin)
                out.append(parent_asin)
                if len(out) >= top_k:
                    return out
        return out

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
            self._capture_category(state, user_message)

            track = self._route_intent(state, user_message)
            pool_size = self.BROWSING_POOL_SIZE if track == "browsing" else self.CANDIDATE_POOL_SIZE
            if track == "browsing" and self._dense_ready:
                # Reserve room for the dense merge below: retrieving a full
                # pool_size of keyword hits first would satisfy
                # _merge_candidates' limit before it ever looks at the dense
                # list (it fills from the first group up to `limit` and
                # stops there), silently dropping every dense candidate on
                # any query popular enough to return pool_size keyword hits
                # on its own -- which is most of them at this catalog size.
                keyword_limit = max(1, pool_size - self.DENSE_CANDIDATE_SIZE)
                candidates = self._retrieve_candidates(state, keyword_limit)
                dense_hits = self._dense_candidates(state.query_text(), self.DENSE_CANDIDATE_SIZE)
                candidates = self._merge_candidates(candidates, dense_hits, pool_size)
            else:
                candidates = self._retrieve_candidates(state, pool_size)

            if not candidates:
                ask_attribute = None
                message = "Tell me a bit more about what you're looking for."
                state.record_turn_result(ask_attribute, message)
                # Never forfeit a scoreable turn to an empty candidate pool:
                # fall back to the last good ranking, then popularity.
                return {
                    "message": message,
                    "ask_attribute": ask_attribute,
                    "recommendations": [
                        {"parent_asin": pid} for pid in self._ensure_top_k(state, [], top_k)
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                }

            sheet = self._get_instruction_sheet(state, candidates, track)
            ranked_ids = self._ensure_top_k(state, self._score_candidates(state, candidates, sheet)[:top_k], top_k)
            state.last_ranked = ranked_ids
            message = sheet.message or "Here are the closest matches I found."

            # Hard enforcement, not just a prompt instruction: never re-ask
            # an attribute the customer already said they have no
            # preference for, regardless of what the LLM returned.
            ask_attribute = sheet.ask_attribute
            if ask_attribute in state.declined:
                ask_attribute = None

            state.record_turn_result(ask_attribute, message)
            return {
                "message": message,
                "ask_attribute": ask_attribute,
                "recommendations": [{"parent_asin": pid} for pid in ranked_ids],
                "usage": sheet.usage,
            }
        except Exception:
            # Last line of defense: still try to serve the last good ranking
            # or a popularity guess rather than an empty list, which forfeits
            # the turn outright. self._fallback_ids is precomputed at index
            # build time, so nothing here can raise a second time.
            try:
                ranked_ids = self._ensure_top_k(state, [], top_k)
            except Exception:
                ranked_ids = list(self._fallback_ids)
            return {
                "message": "Sorry, could you tell me more about what you're looking for?",
                "ask_attribute": None,
                "recommendations": [{"parent_asin": pid} for pid in ranked_ids],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
