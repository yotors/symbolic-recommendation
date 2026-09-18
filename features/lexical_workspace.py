"""Non-neural, causal lexical evidence for symbolic recommendation workspaces.

Only ``title`` and ``abstract`` strings are read from article metadata.  Fit the
IDF model once on unique *training* articles; scoring never fits or updates it.
The caller must pass only articles preceding the candidate impression, ordered
oldest to newest.  No user/item identifier, outcome, entity embedding, or model
prediction enters the evidence.

For candidate token set Q and a previous document d, the bounded BM25 match is

    sum(IDF(t) * tf(t,d) / (tf(t,d) + k1*(1-b+b*len(d)/avg_train_len)))
    ----------------------------------------------------------------------
                              sum(IDF(t), t in Q)

where the numerator includes only matched query terms.  This is ordinary BM25
divided by its query-specific theoretical upper bound.  It preserves the term
frequency/length normalization while putting all emitted values in [0, 1].
Peak match keeps a minority interest visible; top-three mean measures repeated
support; coverage measures candidate concepts represented across the history.
These are evidence predicates for mining, not a replacement recommendation
score.  The implementation is deliberately stdlib-only.
"""

from __future__ import annotations

import functools
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


LEXICAL_FEATURES = (
    "lexical_history_coverage",
    "lexical_recent_coverage",
    "lexical_peak_match",
    "lexical_top3_match",
)
_TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)
_STOPWORDS = frozenset(
    "a an and are as at be been being but by can could did do does for from "
    "had has have he her his how i if in into is it its may more my no not of "
    "on or our she so than that the their them there these they this those to "
    "was we were what when where which who will with would you your".split()
)
_TEXT_FIELDS = ("title", "abstract")


@dataclass(frozen=True)
class LexicalWorkspaceConfig:
    """Fixed resource bounds and corpus-independent retrieval parameters."""

    max_history: int = 200
    max_tokens: int = 512
    max_field_chars: int = 20_000
    recent_window: int = 5
    k1: float = 1.2
    b: float = 0.75

    def __post_init__(self) -> None:
        for name in ("max_history", "max_tokens", "max_field_chars", "recent_window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("k1 must be finite and positive")
        if not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise ValueError("b must lie in [0, 1]")


DEFAULT_CONFIG = LexicalWorkspaceConfig()


@functools.lru_cache(maxsize=32_768)
def _tokens(text: str, max_tokens: int) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    result: list[str] = []
    for match in _TOKEN_RE.finditer(normalized):
        token = match.group()
        if len(token) > 2 and token not in _STOPWORDS:
            result.append(token)
            if len(result) >= max_tokens:
                break
    return tuple(result)


def lexical_tokens(
    article: Mapping[str, Any], *, config: LexicalWorkspaceConfig = DEFAULT_CONFIG,
) -> tuple[str, ...]:
    """Tokenize explicit text fields; all other metadata is ignored.

    Adapters for other platforms should map their headline/summary fields onto
    title/abstract.  Missing or non-string values are absent text, not stringified
    objects (which could accidentally expose labels or identifiers).
    """

    text = " ".join(
        value[:config.max_field_chars]
        for field in _TEXT_FIELDS
        if isinstance(value := article.get(field), str)
    )
    return _tokens(text, config.max_tokens)


def fit_lexical_idf_model(
    training_articles: Iterable[Mapping[str, Any]], *,
    config: LexicalWorkspaceConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Fit JSON-safe IDF/length statistics from training text only.

    The caller supplies one record per training article.  Repeated records with
    identical token multisets are deduplicated without consulting IDs, making
    corpus fitting invariant to record ordering and exposure multiplicity.
    No validation/serving text is added automatically.
    """

    document_frequency: Counter[str] = Counter()
    signatures: set[tuple[tuple[str, int], ...]] = set()
    total_length = 0
    for article in training_articles:
        counts = Counter(lexical_tokens(article, config=config))
        if not counts:
            continue
        signature = tuple(sorted(counts.items()))
        if signature in signatures:
            continue
        signatures.add(signature)
        document_frequency.update(counts.keys())
        total_length += counts.total()
    document_count = len(signatures)
    return {
        "version": "lexical-bm25-v1",
        "document_count": document_count,
        "average_document_length": total_length / document_count if document_count else None,
        "idf": {
            token: math.log1p((document_count - count + 0.5) / (count + 0.5))
            for token, count in sorted(document_frequency.items())
        },
        # Unseen terms receive a fixed training-derived value.  Scoring does
        # not update document count, document frequency, or this OOV prior.
        "default_idf": math.log1p((document_count + 0.5) / 0.5) if document_count else None,
        "max_tokens": config.max_tokens,
        "max_field_chars": config.max_field_chars,
    }


def build_lexical_workspace_facts(
    candidate: Mapping[str, Any],
    history_articles: Iterable[Mapping[str, Any]],
    model: Mapping[str, Any] | None,
    *, config: LexicalWorkspaceConfig = DEFAULT_CONFIG,
) -> dict[str, float | None]:
    """Emit four bounded facts using frozen training stats and prior history.

    ``None`` distinguishes missing history/text/model from a real measured zero
    overlap.  Only the latest ``max_history`` entries are retained.  Empty
    history entries still occupy a recency position.  Full-history features are
    order invariant inside that bound; recent coverage intentionally is not.
    """

    missing = dict.fromkeys(LEXICAL_FEATURES)
    if not model or not model.get("document_count"):
        return missing
    if any(
        model.get(name, getattr(config, name)) != getattr(config, name)
        for name in ("max_tokens", "max_field_chars")
    ):
        raise ValueError("Lexical scoring token bounds must match the fitted model")
    weights = model.get("idf")
    default_weight = model.get("default_idf")
    avg_length = model.get("average_document_length")
    if not isinstance(weights, Mapping) or not all(
        isinstance(value, (float, int)) and not isinstance(value, bool)
        and math.isfinite(value) and value > 0
        for value in (default_weight, avg_length)
    ):
        return missing
    query = sorted(set(lexical_tokens(candidate, config=config)))
    if not query:
        return missing
    query_weights = {token: weights.get(token, default_weight) for token in query}
    if not all(
        isinstance(value, (float, int)) and not isinstance(value, bool)
        and math.isfinite(value) and value > 0 for value in query_weights.values()
    ):
        return missing
    denominator = math.fsum(query_weights.values())

    # deque limits iterable memory without losing the last entries or consuming
    # any external corpus beyond the explicitly supplied preceding history.
    from collections import deque

    preceding = deque(history_articles, maxlen=config.max_history)
    history_counts = [Counter(lexical_tokens(article, config=config)) for article in preceding]
    usable = [counts for counts in history_counts if counts]
    if not usable:
        return missing
    recent = [counts for counts in history_counts[-config.recent_window:] if counts]

    def coverage(documents: list[Counter[str]]) -> float | None:
        if not documents:
            return None
        present = set().union(*(counts.keys() for counts in documents))
        return math.fsum(query_weights[token] for token in query if token in present) / denominator

    matches: list[float] = []
    for counts in usable:
        length_penalty = config.k1 * (1.0 - config.b + config.b * counts.total() / avg_length)
        matches.append(math.fsum(
            query_weights[token] * counts[token] / (counts[token] + length_penalty)
            for token in query if counts[token]
        ) / denominator)
    top = sorted(matches, reverse=True)[:3]
    result = {
        "lexical_history_coverage": coverage(usable),
        "lexical_recent_coverage": coverage(recent),
        "lexical_peak_match": top[0],
        "lexical_top3_match": math.fsum(top) / len(top),
    }
    return {
        name: None if value is None else round(max(0.0, min(1.0, value)), 8)
        for name, value in result.items()
    }
