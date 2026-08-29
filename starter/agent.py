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

# Canonical, single-word vocabulary used two ways: (1) routing signal
# (presence in the customer's message), and (2) structured per-product
# attribute extraction at catalog-index time (see _extract_attrs). Sourced
# from a scan of the real catalog's `details.Material`/`details.Color`
# values, not guessed -- see docs/DEVELOPMENT.md.
MATERIAL_WORDS = {
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon",
    "fabric", "denim", "suede", "metal", "steel", "plastic", "synthetic", "wood",
    "silicone", "crystal", "glass", "gemstone", "rubber", "stone", "vinyl", "brass",
    "rhinestone", "alloy", "acrylic", "neoprene", "canvas", "mesh", "velvet", "satin",
    "chiffon", "lace", "microfiber", "polyurethane", "viscose", "elastane", "lycra",
    "cashmere", "fleece", "linen", "bamboo", "gold", "silver", "platinum", "titanium",
    "copper", "zinc", "pearl", "diamond", "resin",
}
COLOR_WORDS = {
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "navy", "beige", "tan", "ivory", "cream", "khaki",
    "maroon", "teal", "turquoise", "lavender", "coral", "mint", "burgundy", "charcoal",
    "olive", "mustard", "multicolor",
}
SIZE_WORDS = {
    "small", "medium", "large", "size", "xl", "xs", "xxl", "xxs", "wide", "narrow",
    "petite", "tall", "regular", "plus",
}
STYLE_WORDS = {
    "casual", "formal", "athletic", "vintage", "classic", "modern", "bohemian",
    "sporty", "elegant", "minimalist", "streetwear", "business", "chic",
}
# Multi-word canonical terms that single-token matching misses -- checked
# via substring, kept short/high-value rather than exhaustive.
MATERIAL_PHRASES = {"faux leather", "sterling silver", "stainless steel", "genuine leather"}
COLOR_PHRASES = {"rose gold"}
SIZE_PHRASES = {"one size", "plus size"}

# Where a product's raw `details` dict already carries a clean attribute,
# prefer it outright over vocabulary-matching free text.
DETAIL_ATTR_MAP = {
    "material": ("Material", "Fabric Type"),
    "color": ("Color",),
    "brand": ("Brand", "Brand Name"),
    "style": ("Style", "Fit Type", "Pattern", "Closure Type"),
    "size": ("Size",),
}
STRUCTURED_ATTRIBUTES = frozenset(DETAIL_ATTR_MAP.keys())

CATALOG_COLUMNS = ("parent_asin", "title", "categories", "features", "details", "store", "description")

ALLOWED_ATTRIBUTES = frozenset({
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
})

SYSTEM_PROMPT = (
    "ROLE\n"
    "You are the sole language-understanding component of a shopping agent "
    "for an Amazon Clothing/Shoes/Jewelry catalog. Your only job is to read "
    "the customer's full conversation and turn it into one structured "
    "instruction sheet. You never see any product data -- not a title, not "
    "an ID, nothing. All catalog interaction (retrieval, filtering, "
    "ranking) is handled by deterministic code downstream using exactly "
    "what you output. Put your full attention into understanding the "
    "customer, not into guessing about products.\n\n"
    "WHAT YOU RECEIVE EACH TURN\n"
    "- The complete conversation so far (every customer message and every "
    "agent reply/question) -- always use all of it, not just the latest "
    "message; earlier turns often carry constraints that still apply.\n"
    "- Known constraints already disclosed, and attributes the customer "
    "explicitly has no preference for (never ask about those again).\n"
    "- Attributes not yet explored -- prefer asking about one of these "
    "over anything already disclosed or declined.\n"
    "- A detected BUYING or BROWSING track for this turn.\n"
    "- A one-line summary of the current candidate pool (count, price "
    "range, category spread) -- for calibrating breadth only, not for "
    "picking or reasoning about individual products.\n"
    "- The customer's aggregate profile (purchase history summary, rating "
    "style, preference tags).\n\n"
    "OUTPUT: constraints and weights\n"
    "- `constraints`: one entry per attribute the conversation has "
    "actually established, using ONLY the allowed attribute names below "
    "as keys -- never invent a new key (e.g. write \"material\", not "
    "\"band\" or \"fabric_type\"); an invented key is silently discarded "
    "downstream.\n"
    "- If the customer states what they're looking for up front (e.g. "
    "\"I'm looking for X\"), that X is almost always the `category` -- "
    "capture it as a constraint immediately, weight 0.8-1.0, even before "
    "any question has been asked. Do not leave `constraints` empty when "
    "the opening message already names what's being shopped for.\n"
    "- For material, color, size, style, and brand, prefer a single "
    "common canonical word (e.g. material: cotton, leather, polyester, "
    "wool, denim, silk, metal, alloy; color: black, white, blue, red, "
    "green, brown, gray, navy; size: small, medium, large, xl, xs) over "
    "a paraphrase -- these values are matched against catalog data "
    "automatically, so exact, ordinary wording matters more than "
    "elegant phrasing.\n"
    "- For budget, extract a plain number (e.g. \"80\" for \"under $80\").\n"
    "- `weights`: 0.0-1.0 per constraint, reflecting how confident/"
    "important that signal is. On a BUYING turn, the customer gave "
    "something concrete -- weight it high (0.7-1.0). On a BROWSING turn, "
    "signals are softer -- weight 0.3-0.6 unless clearly stated.\n\n"
    "OUTPUT: ask_attribute\n"
    "- If the pool summary still looks broad (a high count, or a wide "
    "price/category spread) relative to what's been disclosed, set "
    "ask_attribute to the single most useful attribute to learn next -- "
    "prefer one from \"attributes not yet explored\". Otherwise set it to "
    "null and let the ranking stand.\n"
    "- Never set ask_attribute to something already disclosed or "
    "declined -- pick a genuinely new one, or null.\n"
    "- If the customer says to use their judgment on something, treat "
    "that the same as a decline: do not ask about it again.\n\n"
    "OUTPUT: message\n"
    "- One short, natural, customer-facing sentence. If ask_attribute is "
    "set, phrase it as that question. If null, briefly acknowledge you're "
    "showing results.\n\n"
    "FORMAT\n"
    "Respond with JSON only, matching the schema exactly. No prose, no "
    "markdown fences, no explanation outside the JSON object.\n\n"
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


def _extract_attrs(product: dict) -> dict[str, set[str]]:
    """Structured, canonical attributes for one catalog product -- built
    once at index time so the executor can match an instruction sheet's
    constraint values against clean data instead of raw text search.
    `details` dict values (when present) are the most reliable source and
    are preferred outright; vocabulary matching over title/features/
    description fills in the rest.
    """
    attrs: dict[str, set[str]] = {name: set() for name in STRUCTURED_ATTRIBUTES}

    details = product.get("details")
    if isinstance(details, dict):
        for attr, keys in DETAIL_ATTR_MAP.items():
            for key in keys:
                value = details.get(key)
                if value:
                    attrs[attr].add(str(value).strip().lower())

    store = product.get("store")
    if store:
        attrs["brand"].add(str(store).strip().lower())

    text_blob = " ".join(
        [str(product.get("title") or ""), _text(product.get("features")), _text(product.get("description"))]
    ).lower()
    terms = set(_terms(text_blob))
    attrs["material"] |= terms & MATERIAL_WORDS
    attrs["color"] |= terms & COLOR_WORDS
    attrs["size"] |= terms & SIZE_WORDS
    attrs["style"] |= terms & STYLE_WORDS
    attrs["material"] |= {phrase for phrase in MATERIAL_PHRASES if phrase in text_blob}
    attrs["color"] |= {phrase for phrase in COLOR_PHRASES if phrase in text_blob}
    attrs["size"] |= {phrase for phrase in SIZE_PHRASES if phrase in text_blob}

    return attrs


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

    def record_message(self, message: str, turn: int) -> None:
        self.turn = turn
        if OVERRIDE_RE.search(message):
            self.override_seen = True
            self.disclosed.clear()
            self.declined.clear()  # a new intent may reopen previously-declined attributes
            # The override's own content is the single most important thing
            # in it -- without this it was only visible on the turn it was
            # said (relying on the LLM re-noticing it in the raw transcript
            # every turn after, which it did not do reliably in testing).
            self.disclosed["override"] = message
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
        # Built from genuinely informative content only: the opening
        # message (always carries the base intent/category) plus every
        # real disclosed value. NOT the full raw transcript -- that fills
        # up with boilerplate ("I don't have an additional preference for
        # X") which BM25 can actually over-weight, since such phrasing is
        # rare in a product-catalog corpus and so looks "distinctive" to it.
        parts = [self.messages[0]] if self.messages else []
        parts.extend(self.disclosed.values())
        return " ".join(parts)


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

        # Only accept keys from the allowed attribute vocabulary. An
        # invented key (seen in testing: "band" instead of "material")
        # would otherwise silently skip structured-attribute matching and
        # fall back to weaker raw-text search for no good reason -- ask_attribute
        # was already validated this way, constraints/weights were not.
        constraints = {
            str(key): str(value)
            for key, value in (data.get("constraints") or {}).items()
            if str(key) in ALLOWED_ATTRIBUTES and value not in (None, "", [])
        }

        weights: dict[str, float] = {}
        for key, value in (data.get("weights") or {}).items():
            if str(key) not in ALLOWED_ATTRIBUTES:
                continue
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
    # Was 100 -- raised after finding real targets sitting at BM25 rank
    # ~103 and ~284 in post-mortem testing. Cheap to raise now that the LLM
    # never sees the candidate list (only aggregate stats), so this only
    # costs a bit of our own retrieval/scoring compute, not prompt tokens.
    CANDIDATE_POOL_SIZE = 300
    # How many dense-similarity hits to pull in on a Browsing turn, merged
    # with the keyword results before scoring.
    DENSE_CANDIDATE_SIZE = 60
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
                    "attrs": _extract_attrs(product),
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

    def _retrieve_candidates(self, state: SessionState, limit: int) -> list[dict]:
        unique_terms = list(dict.fromkeys(_terms(state.query_text())))[:40]
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
                    self._embedding_matrix = matrix
                    self._embedding_ids = ids
                    self._dense_ready = True
                    return
                print("[agent] cached embeddings don't match the current catalog -- rebuilding.")
            self._build_embeddings()
        except Exception as exc:
            print(f"[agent] dense retrieval unavailable ({exc}) -- Browsing falls back to keyword search.")

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

    @staticmethod
    def _pool_summary(candidates: list[dict]) -> str:
        """A cheap, aggregate description of the current candidate pool --
        count, price range, category spread. This is deliberately NOT a
        product list: the LLM gets just enough signal to judge whether the
        pool is still broad (and so whether to ask a clarifying question),
        without ever seeing product data. Computed from data already in
        memory, no extra retrieval cost."""
        if not candidates:
            return "No products currently match."
        prices = [c["price"] for c in candidates if isinstance(c.get("price"), float)]
        price_range = f"${min(prices):.0f}-${max(prices):.0f}" if prices else "unknown"
        category_counts: dict[str, int] = {}
        excluded = {"clothing", "shoes & jewelry", "clothing, shoes & jewelry"}
        for candidate in candidates:
            for part in (candidate.get("categories") or "").split(","):
                part = part.strip()
                if part and part.lower() not in excluded:
                    category_counts[part] = category_counts.get(part, 0) + 1
        top_categories = sorted(category_counts.items(), key=lambda item: -item[1])[:5]
        spread = ", ".join(name for name, _ in top_categories) or "mixed"
        return f"{len(candidates)} products currently match. Price range: {price_range}. Category spread: {spread}."

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
        remaining = sorted(ALLOWED_ATTRIBUTES - set(state.disclosed) - state.declined)
        lines.append("Attributes not yet explored (prefer these for ask_attribute): " + ", ".join(remaining))
        lines.append("")
        lines.append("Candidate pool: " + self._pool_summary(candidates))
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

    def _score_candidates(self, candidates: list[dict], sheet: InstructionSheet) -> list[str]:
        scored: list[tuple[float, int, str]] = []
        for rank, candidate in enumerate(candidates):
            combined_text = None  # built lazily -- only needed for the raw-text fallback path
            attrs = candidate.get("attrs") or {}

            score = 0.0
            for attribute, value in sheet.constraints.items():
                weight = sheet.weights.get(attribute, 0.5)
                lowered_value = value.lower()
                if attribute == "budget":
                    budget_max = _as_float(value)
                    price = candidate.get("price")
                    if budget_max is not None and price is not None and price <= budget_max:
                        score += weight
                elif attribute in STRUCTURED_ATTRIBUTES and attrs.get(attribute):
                    # Structured match against clean, canonicalized product
                    # data instead of raw text -- substring-safe both ways
                    # so "gold" still matches a stored "rose gold".
                    if any(lowered_value in term or term in lowered_value for term in attrs[attribute]):
                        score += weight
                elif attribute == "category":
                    if lowered_value in (candidate.get("categories") or "").lower():
                        score += weight
                else:
                    if combined_text is None:
                        combined_text = " ".join(
                            candidate.get(field_name, "") for field_name in CATALOG_COLUMNS[1:]
                        ).lower()
                    if lowered_value in combined_text:
                        score += weight
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
            if track == "browsing" and self._dense_ready:
                # Reserve room for dense hits up front -- _merge_candidates
                # fills from BM25 first, so if BM25's own limit were the
                # full pool size it could fill every slot itself and leave
                # zero room for dense candidates to ever get merged in
                # (this was a real regression, found via testing).
                bm25_limit = max(self.CANDIDATE_POOL_SIZE - self.DENSE_CANDIDATE_SIZE, self.DENSE_CANDIDATE_SIZE)
                candidates = self._retrieve_candidates(state, bm25_limit)
                dense_hits = self._dense_candidates(state.query_text(), self.DENSE_CANDIDATE_SIZE)
                candidates = self._merge_candidates(candidates, dense_hits, self.CANDIDATE_POOL_SIZE)
            else:
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

            sheet = self._get_instruction_sheet(state, candidates, track)
            ranked_ids = self._score_candidates(candidates, sheet)[:top_k]
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
            return {
                "message": "Sorry, could you tell me more about what you're looking for?",
                "ask_attribute": None,
                "recommendations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
