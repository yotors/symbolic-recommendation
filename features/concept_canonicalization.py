"""Portable, label-free canonical IDs for extracted article descriptors.

This module is deliberately narrower than a semantic ranker.  It turns noisy
LLM strings into repeatable symbolic identifiers before workspace construction.
It may use explicit entity identifiers/aliases attached to an article, but it
never reads user histories, impressions, outcomes, ranks, scores, or vectors.

The transformation is conservative:

* Unicode, whitespace and punctuation are normalized. Open-vocabulary plural
  folding is intentionally not attempted because it can merge named entities
  (for example ``Nationals Park``) with unrelated common nouns.
* format and communicative-intent variants are mapped to a small, documented
  vocabulary already used by the extractor contract.
* an acronym is expanded only when at least two input article records contain
  the same unambiguous acronym/full-form co-occurrence.
* an entity alias is used only when metadata binds it to exactly one stable
  entity identifier (globally, or locally within the article).

The returned ``annotations`` mapping has the same five descriptor fields as an
``llm_article_facts`` record and can therefore be passed to the LLM projection
after integration.  The registry is JSON data and can be frozen after training
then reused for validation and live articles.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence


SCHEMA_VERSION = "mindplex-canonical-article-annotations-v1"
REGISTRY_SCHEMA = "mindplex-content-canonical-registry-v1"
ALGORITHM_VERSION = "portable-symbolic-canonicalizer-v3"

_MULTI_FIELDS = ("concepts", "event_types", "intents", "audiences")
_FIELDS = (*_MULTI_FIELDS, "format")
_PREFIX = {
    "concepts": "concept",
    "event_types": "event",
    "intents": "intent",
    "audiences": "audience",
    "format": "format",
}
_ENTITY_FIELDS = (
    "title_entities", "abstract_entities", "entities", "entity_ids",
    "named_entities",
)
_ENTITY_ID_KEYS = (
    "wikidataid", "wikidata_id", "entity_id", "entityid", "@id", "id",
    "uri",
)
_ENTITY_LABEL_KEYS = ("label", "name", "surface", "text")
_ENTITY_ALIAS_KEYS = ("aliases", "alias", "surface_forms", "surfaceforms")
_BEHAVIOR_KEYS = frozenset({
    "action", "actions", "click", "clicked", "clicks", "engagement",
    "impression", "impressions", "label", "labels", "rank", "ranking",
    "relevant", "relevance", "reward", "score", "scores", "skip",
    "skipped", "target", "targets", "user", "users",
})
_INITIALISM_STOPWORDS = frozenset({"a", "an", "and", "for", "of", "or", "the", "to"})

# These are field semantics, not a dataset taxonomy.  Concepts, event types and
# audiences intentionally have no hand-authored synonym table.
_RAW_ALIASES = {
    "format": {
        "news report": "report", "news story": "report", "reporting": "report",
        "factual report": "report", "opinion piece": "opinion", "op ed": "opinion",
        "editorial": "opinion", "explanatory article": "explainer",
        "explainer article": "explainer", "explanation": "explainer",
        "listicle": "list", "ranked list": "list", "product review": "review",
        "critical review": "review", "analytical article": "analysis",
        "analytic article": "analysis", "photo gallery": "gallery",
        "image gallery": "gallery", "slideshow": "gallery",
        "video report": "video", "video article": "video",
    },
    "intents": {
        "informative": "inform", "informing": "inform", "report facts": "inform",
        "explanation": "explain", "explanatory": "explain", "explaining": "explain",
        "comparison": "compare", "comparative": "compare", "comparing": "compare",
        "advice": "advise", "advisory": "advise", "guidance": "advise",
        "evaluation": "review", "evaluate": "review", "evaluating": "review",
    },
}


class CanonicalizationError(ValueError):
    """The content inputs or a supplied frozen registry are invalid."""


def _json_hash(value):
    try:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise CanonicalizationError("canonicalization inputs must be finite JSON data") from None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _surface(value):
    if not isinstance(value, str) or not value.strip():
        raise CanonicalizationError("descriptor values must be nonempty strings")
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("\u2019", "'").replace("\u2018", "'")
    value = re.sub(r"(?<=\w)'s\b", "", value)
    pieces = []
    for character in value:
        if character.isalnum() or character in "+#":
            pieces.append(character)
        elif character == "&":
            pieces.append(" and ")
        else:
            pieces.append(" ")
    normalized = " ".join("".join(pieces).split())
    if not normalized:
        raise CanonicalizationError("descriptor values must contain letters or numbers")
    if len(normalized) > 160:
        raise CanonicalizationError("normalized descriptor values may not exceed 160 characters")
    return normalized


def lexical_form(value, field):
    """Return the field-aware, human-readable canonical lexical form."""
    if field not in _FIELDS:
        raise CanonicalizationError(f"unknown descriptor field: {field}")
    normalized = _surface(value)
    aliases = _RAW_ALIASES.get(field, {})
    # Alias keys pass through the same normalization below at module load time;
    # the direct lookup is retained for the common already-normalized path.
    return aliases.get(normalized, normalized)


def _slug(value):
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value.casefold()).strip("-")[:48]
    return slug or "unicode"


def _canonical_from_lexical(field, lexical):
    digest = hashlib.sha256(lexical.encode("utf-8")).hexdigest()
    return f"{_PREFIX[field]}:{_slug(lexical)}~{digest}"


def _behavior_key_present(value):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key.casefold() in _BEHAVIOR_KEYS:
                return True
            if _behavior_key_present(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_behavior_key_present(item) for item in value)
    return False


def _annotation_mapping(value):
    if not isinstance(value, Mapping):
        raise CanonicalizationError("annotations must be a mapping")
    if "extraction_records" in value:
        if value.get("schema") != "mindplex-llm-article-facts-v1":
            raise CanonicalizationError("only the article-facts extraction envelope is supported")
        records = value.get("extraction_records")
        if not isinstance(records, Mapping):
            raise CanonicalizationError("extraction_records must be a mapping")
        source = {key: copy.deepcopy(item) for key, item in value.items()
                  if key != "extraction_records"}
        return records, source
    return value, {"kind": "direct-article-mapping"}


def _article_mapping(value):
    if isinstance(value, Mapping) and isinstance(value.get("articles"), list):
        source = value["articles"]
    elif isinstance(value, Mapping):
        source = []
        for identifier, article in value.items():
            if not isinstance(article, Mapping):
                raise CanonicalizationError("article mappings must contain article records")
            materialized = dict(article)
            materialized.setdefault("id", identifier)
            source.append(materialized)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        source = value
    else:
        raise CanonicalizationError("articles must be a sequence, ID mapping, or dataset mapping")
    result = {}
    for article in source:
        if not isinstance(article, Mapping):
            raise CanonicalizationError("each article must be a mapping")
        identifier = article.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise CanonicalizationError("each article requires a nonempty string ID")
        if identifier in result:
            raise CanonicalizationError(f"duplicate article ID: {identifier}")
        result[identifier] = article
    return result


def _validate_record(identifier, record, article):
    if not isinstance(identifier, str) or not identifier or not isinstance(record, Mapping):
        raise CanonicalizationError("annotation IDs and records must be valid mappings")
    if set(record) - {*_FIELDS, "provenance"}:
        raise CanonicalizationError(f"annotation {identifier} has non-content fields")
    if _behavior_key_present(record):
        raise CanonicalizationError(f"annotation {identifier} contains behavioral fields")
    for field in _MULTI_FIELDS:
        values = record.get(field)
        if not isinstance(values, list):
            raise CanonicalizationError(f"annotation {identifier} {field} must be a list")
        for value in values:
            _surface(value)
    if record.get("format") is not None:
        _surface(record["format"])
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        raise CanonicalizationError(f"annotation {identifier} lacks provenance")
    source_id = provenance.get("source_id")
    if source_id is not None and source_id != article.get("source_id", identifier):
        raise CanonicalizationError(f"annotation source ID mismatch: {identifier}")
    content_hash = provenance.get("article_content_sha256")
    if content_hash is not None:
        # Keep this recipe in lockstep with llm_article_facts/text_embeddings.
        from .text_embeddings import article_text
        expected = hashlib.sha256(article_text(article.get("title"), article.get("abstract")).encode()).hexdigest()
        if content_hash != expected:
            raise CanonicalizationError(f"stale annotation content hash: {identifier}")


def _identifier_anchor(value, namespace=None):
    if not isinstance(value, str) or not value.strip():
        return None
    raw = unicodedata.normalize("NFKC", value).strip()
    if re.fullmatch(r"[Qq][1-9][0-9]*", raw):
        return f"concept:entity/wikidata/{raw.casefold()}"
    if raw.startswith(("http://", "https://", "urn:")):
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"concept:entity/uri/{digest}"
    if namespace:
        namespace = _slug(_surface(str(namespace)))
    else:
        namespace = "opaque"
    normalized = _surface(raw)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"concept:entity/{namespace}/{_slug(normalized)}~{digest}"


def _entity_descriptor(value):
    """Return ``(anchor, aliases)`` for supported structured metadata."""
    if isinstance(value, str):
        if (re.fullmatch(r"[Qq][1-9][0-9]*", value.strip())
                or value.strip().startswith(("http://", "https://", "urn:"))):
            return _identifier_anchor(value), set()
        lexical = lexical_form(value, "concepts")
        return _canonical_from_lexical("concepts", lexical), {lexical}
    if not isinstance(value, Mapping):
        return None, set()
    lowered = {str(key).casefold(): item for key, item in value.items()}
    identifier = next((lowered[key] for key in _ENTITY_ID_KEYS
                       if key in lowered and isinstance(lowered[key], str) and lowered[key].strip()), None)
    namespace = lowered.get("namespace") or lowered.get("source")
    if "wikidataid" in lowered or "wikidata_id" in lowered:
        namespace = "wikidata"
    surfaces = []
    for key in _ENTITY_LABEL_KEYS:
        candidate = lowered.get(key)
        if isinstance(candidate, str) and candidate.strip():
            surfaces.append(candidate)
    for key in _ENTITY_ALIAS_KEYS:
        candidate = lowered.get(key)
        if isinstance(candidate, str):
            surfaces.append(candidate)
        elif isinstance(candidate, (list, tuple)):
            surfaces.extend(item for item in candidate if isinstance(item, str))
    aliases = {lexical_form(item, "concepts") for item in surfaces if item.strip()}
    anchor = _identifier_anchor(identifier, namespace) if identifier is not None else None
    if anchor is None and aliases:
        lexical = min(aliases)
        anchor = _canonical_from_lexical("concepts", lexical)
    return anchor, aliases


def _article_entities(article):
    result = []
    for field in _ENTITY_FIELDS:
        values = article.get(field)
        if values is None:
            continue
        if isinstance(values, Mapping) or isinstance(values, str):
            values = [values]
        if not isinstance(values, (list, tuple)):
            continue
        for position, value in enumerate(values):
            anchor, aliases = _entity_descriptor(value)
            if anchor is not None:
                result.append({"anchor": anchor, "aliases": aliases,
                               "source_field": field, "source_position": position})
    return result


def _initialism(phrase):
    words = [word for word in phrase.split()
             if word not in _INITIALISM_STOPWORDS and word and word[0].isalnum()]
    return "".join(word[0] for word in words)


def _learn_acronyms(records):
    # One extractor mistake must not redefine a corpus-wide identifier.  Two
    # independent article records are the minimum evidence for an observed
    # acronym alias; ambiguity still rejects every competing expansion.
    candidates = defaultdict(lambda: defaultdict(int))
    for record in records.values():
        forms = {lexical_form(value, "concepts") for value in record.get("concepts", [])}
        acronyms = {form for form in forms if " " not in form and 2 <= len(form) <= 10
                    and form.isascii() and form.isalnum()}
        phrases = {form for form in forms if " " in form}
        for acronym in acronyms:
            for phrase in phrases:
                if _initialism(phrase) == acronym:
                    candidates[acronym][phrase] += 1
    return {
        key: next(iter(values)) for key, values in candidates.items()
        if len(values) == 1 and next(iter(values.values())) >= 2
    }


def _registry_components(records, articles):
    entity_alias_targets = defaultdict(set)
    for article in articles.values():
        for entity in _article_entities(article):
            for alias in entity["aliases"]:
                entity_alias_targets[alias].add(entity["anchor"])
    entity_aliases = {alias: next(iter(targets)) for alias, targets in entity_alias_targets.items()
                      if len(targets) == 1}
    ambiguous = sorted(alias for alias, targets in entity_alias_targets.items() if len(targets) > 1)
    return _learn_acronyms(records), entity_aliases, ambiguous


def _validate_registry(registry):
    if not isinstance(registry, Mapping) or registry.get("schema") != REGISTRY_SCHEMA:
        raise CanonicalizationError("registry has an unsupported schema")
    allowed = {"schema", "acronym_aliases", "entity_aliases", "ambiguous_entity_aliases"}
    if set(registry) != allowed:
        raise CanonicalizationError("registry fields do not match its schema")
    for field in ("acronym_aliases", "entity_aliases"):
        values = registry[field]
        if not isinstance(values, Mapping):
            raise CanonicalizationError(f"registry {field} must be a mapping")
        for key, value in values.items():
            if not isinstance(key, str) or not isinstance(value, str) or _surface(key) != key:
                raise CanonicalizationError(f"registry {field} contains invalid values")
    if (not isinstance(registry["ambiguous_entity_aliases"], list)
            or any(not isinstance(item, str) or _surface(item) != item
                   for item in registry["ambiguous_entity_aliases"])):
        raise CanonicalizationError("registry ambiguous aliases are invalid")
    return copy.deepcopy(dict(registry))


def build_canonical_registry(annotations, articles):
    """Learn only unambiguous content aliases; no behavioral data are accepted."""
    records, _source = _annotation_mapping(annotations)
    article_map = _article_mapping(articles)
    for identifier, record in records.items():
        if identifier not in article_map:
            raise CanonicalizationError(f"annotation article ID absent from source: {identifier}")
        _validate_record(identifier, record, article_map[identifier])
    acronym_aliases, entity_aliases, ambiguous = _registry_components(records, article_map)
    return {
        "schema": REGISTRY_SCHEMA,
        "acronym_aliases": {key: acronym_aliases[key] for key in sorted(acronym_aliases)},
        "entity_aliases": {key: entity_aliases[key] for key in sorted(entity_aliases)},
        "ambiguous_entity_aliases": ambiguous,
    }


def _article_evidence(article_map):
    fields = ("id", "source_id", "title", "abstract", *_ENTITY_FIELDS)
    return [{field: copy.deepcopy(article[field]) for field in fields if field in article}
            for _identifier, article in sorted(article_map.items())]


def canonicalize_annotations(annotations, articles, *, registry=None,
                             include_entity_concepts=True, max_entity_concepts=8):
    """Return canonical annotations plus complete transformation provenance.

    ``registry=None`` learns conservative aliases from this content batch.  For
    a causal benchmark or online serving, call :func:`build_canonical_registry`
    on the permitted training corpus, freeze the returned JSON, and pass it as
    ``registry`` for every subsequent split.
    """
    if not isinstance(include_entity_concepts, bool):
        raise CanonicalizationError("include_entity_concepts must be boolean")
    if (isinstance(max_entity_concepts, bool) or not isinstance(max_entity_concepts, int)
            or not 0 <= max_entity_concepts <= 64):
        raise CanonicalizationError("max_entity_concepts must be an integer from 0 to 64")
    records, annotation_source = _annotation_mapping(annotations)
    article_map = _article_mapping(articles)
    for identifier, record in records.items():
        if identifier not in article_map:
            raise CanonicalizationError(f"annotation article ID absent from source: {identifier}")
        _validate_record(identifier, record, article_map[identifier])
    selected_registry = (build_canonical_registry(records, article_map) if registry is None
                         else _validate_registry(registry))
    acronym_aliases = selected_registry["acronym_aliases"]
    global_entity_aliases = selected_registry["entity_aliases"]
    output = {}
    statistics = defaultdict(int)

    for identifier in sorted(records):
        record, article = records[identifier], article_map[identifier]
        entity_rows = _article_entities(article)
        local_targets = defaultdict(set)
        for entity in entity_rows:
            for alias in entity["aliases"]:
                local_targets[alias].add(entity["anchor"])
        local_aliases = {alias: next(iter(targets)) for alias, targets in local_targets.items()
                         if len(targets) == 1}
        mappings = {field: [] for field in _FIELDS}
        canonical = {field: [] for field in _MULTI_FIELDS}
        canonical["format"] = None

        for field in _FIELDS:
            raw_values = ([record[field]] if field == "format" and record.get(field) is not None
                          else record.get(field, []) if field != "format" else [])
            seen = set()
            for raw in raw_values:
                lexical = lexical_form(raw, field)
                basis = "lexical"
                if field == "concepts":
                    expanded = acronym_aliases.get(lexical)
                    if expanded is not None:
                        lexical, basis = expanded, "observed-acronym"
                    target = local_aliases.get(lexical) or global_entity_aliases.get(lexical)
                    if target is not None:
                        value = target
                        basis = "explicit-entity-alias-local" if lexical in local_aliases else "explicit-entity-alias-registry"
                    else:
                        value = _canonical_from_lexical(field, lexical)
                else:
                    value = _canonical_from_lexical(field, lexical)
                mappings[field].append({"raw": raw, "lexical": lexical,
                                        "canonical_id": value, "basis": basis})
                statistics["raw_descriptor_values"] += 1
                if value in seen:
                    statistics["descriptor_values_merged"] += 1
                    continue
                seen.add(value)
                if field == "format":
                    if canonical[field] is not None and canonical[field] != value:
                        raise CanonicalizationError(f"annotation {identifier} has conflicting formats")
                    canonical[field] = value
                else:
                    canonical[field].append(value)

        added_entities = []
        if include_entity_concepts:
            existing = set(canonical["concepts"])
            for entity in entity_rows:
                if entity["anchor"] in existing:
                    continue
                if len(added_entities) >= max_entity_concepts:
                    statistics["entity_concepts_dropped_by_limit"] += 1
                    continue
                existing.add(entity["anchor"])
                canonical["concepts"].append(entity["anchor"])
                added_entities.append({key: entity[key] for key in
                                       ("anchor", "source_field", "source_position")})
                statistics["entity_concepts_added"] += 1
        for field in _MULTI_FIELDS:
            canonical[field].sort()
        statistics["canonical_descriptor_values"] += sum(len(canonical[field]) for field in _MULTI_FIELDS)
        statistics["canonical_descriptor_values"] += int(canonical["format"] is not None)
        source_provenance = copy.deepcopy(record["provenance"])
        source_provenance["canonicalization"] = {
            "schema": SCHEMA_VERSION,
            "algorithm": ALGORITHM_VERSION,
            "source_record_sha256": _json_hash(record),
            "registry_sha256": _json_hash(selected_registry),
            "mappings": mappings,
            "entity_concepts_added": added_entities,
        }
        canonical["provenance"] = source_provenance
        output[identifier] = canonical

    result = {
        "schema": SCHEMA_VERSION,
        "annotations": output,
        "provenance": {
            "algorithm": ALGORITHM_VERSION,
            "annotation_source": annotation_source,
            "input_annotation_records_sha256": _json_hash(records),
            "article_content_and_entity_evidence_sha256": _json_hash(_article_evidence(article_map)),
            "registry": selected_registry,
            "registry_sha256": _json_hash(selected_registry),
            "registry_source": "content-batch" if registry is None else "supplied-frozen",
            "output_annotations_sha256": _json_hash(output),
            "configuration": {
                "include_entity_concepts": include_entity_concepts,
                "max_entity_concepts": max_entity_concepts,
                "vectors_used": False,
                "behavioral_fields_used": [],
            },
            "statistics": {key: statistics[key] for key in sorted(statistics)},
        },
    }
    return result


# Normalize the compact controlled aliases through the same pipeline once.
for _field, _aliases in tuple(_RAW_ALIASES.items()):
    normalized_aliases = {}
    for _alias, _target in _aliases.items():
        normalized_alias = _surface(_alias)
        normalized_target = _surface(_target)
        normalized_aliases[normalized_alias] = normalized_target
    _RAW_ALIASES[_field] = normalized_aliases


__all__ = [
    "ALGORITHM_VERSION", "CanonicalizationError", "REGISTRY_SCHEMA",
    "SCHEMA_VERSION", "build_canonical_registry",
    "canonicalize_annotations", "lexical_form",
]
