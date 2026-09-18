"""Label-free plans and proof reduction for entity-continuity evidence.

The source relation is exact identity of a MIND Wikidata entity in a candidate
and an article from the user's explicit, preceding click history.  Python only
constructs observations and validates returned proof dependencies.  A
``recent`` or ``older`` conclusion is emitted only after PeTTaChainer proves
the structural chain. Canonical concepts additionally require a proof through
their original named ``HasConcept`` anchor and an explicit canonical mapping.

History positions are oldest to newest.  The final five original positions
are recent.  Repeated article IDs remain distinct interaction origins, while
multiple entity paths from the same origin are deliberately one observation.
No outcome, label, score, mined rule, or latest-user profile is accepted here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any


RELATIONAL_WORKSPACE_SCHEMA = "mindplex-relational-continuity-proofs-v3"
RELATIONAL_PROJECTION_SCHEMA = "mindplex-preserved-relational-projection-v3"
REL_ENTITY_CONTINUITY_SCOPE = "rel_entity_continuity_scope"
REL_ENTITY_CONTINUITY_PROOF_IDS = "rel_entity_continuity_proof_ids"
REL_CONCEPT_CONTINUITY_SCOPE = "rel_concept_continuity_scope"
REL_CONCEPT_CONTINUITY_PROOF_IDS = "rel_concept_continuity_proof_ids"
RELATIONAL_WORKSPACE_FEATURES = (
    REL_ENTITY_CONTINUITY_SCOPE, REL_CONCEPT_CONTINUITY_SCOPE,
)
RELATIONAL_SCOPE_VALUES = ("unknown", "none", "older", "recent")
RECENT_HISTORY_POSITIONS = 5

ENGAGED_ENTITY_RULE_ID = "rel_v1_derive_engaged_entity_origin"
ENTITY_CONTINUITY_RULE_ID = "rel_v1_derive_entity_continuity"
ENGAGED_CONCEPT_RULE_ID = "rel_v1_derive_engaged_concept_origin"
CONCEPT_CONTINUITY_RULE_ID = "rel_v1_derive_concept_continuity"
CANONICAL_CONCEPT_BRIDGE_RULE_ID = "rel_v2_ground_canonical_concept"
ENTITY_RELATIONAL_STRUCTURAL_RULES = (
    (
        f"(: {ENGAGED_ENTITY_RULE_ID} "
        "(Implication "
        "(And (RelObservedClick $scope $origin $user $history) "
        "(RelMentionsEntity $history $entity)) "
        "(RelEngagedEntityOrigin $scope $user $entity $origin)) "
        "(CTV (STV 1.0 1.0) (STV 0.0 1.0)))"
    ),
    (
        f"(: {ENTITY_CONTINUITY_RULE_ID} "
        "(Implication "
        "(And (RelCaseCandidate $case $scope $user $candidate) "
        "(RelEngagedEntityOrigin $scope $user $entity $origin) "
        "(RelMentionsEntity $candidate $entity)) "
        "(RelEntityContinuity $case $origin $entity)) "
        "(CTV (STV 1.0 1.0) (STV 0.0 1.0)))"
    ),
)
CONCEPT_RELATIONAL_STRUCTURAL_RULES = (
    (
        f"(: {CANONICAL_CONCEPT_BRIDGE_RULE_ID} "
        "(Implication "
        "(And (HasConcept $source $lexical) "
        "(RelCanonicalConceptMapping $source $lexical $article $concept)) "
        "(RelHasCanonicalConcept $article $concept)) "
        "(CTV (STV 1.0 1.0) (STV 0.0 1.0)))"
    ),
    (
        f"(: {ENGAGED_CONCEPT_RULE_ID} "
        "(Implication "
        "(And (RelConceptObservedClick $scope $origin $user $history) "
        "(RelHasCanonicalConcept $history $concept)) "
        "(RelEngagedConceptOrigin $scope $user $concept $origin)) "
        "(CTV (STV 1.0 1.0) (STV 0.0 1.0)))"
    ),
    (
        f"(: {CONCEPT_CONTINUITY_RULE_ID} "
        "(Implication "
        "(And (RelConceptCaseCandidate $case $scope $user $candidate) "
        "(RelEngagedConceptOrigin $scope $user $concept $origin) "
        "(RelHasCanonicalConcept $candidate $concept)) "
        "(RelConceptContinuity $case $origin $concept)) "
        "(CTV (STV 1.0 1.0) (STV 0.0 1.0)))"
    ),
)
RELATIONAL_STRUCTURAL_RULES = (
    *ENTITY_RELATIONAL_STRUCTURAL_RULES,
    *CONCEPT_RELATIONAL_STRUCTURAL_RULES,
)

_WIKIDATA_ENTITY = re.compile(r"^Q[1-9][0-9]*$", re.IGNORECASE)
_CANONICAL_CONCEPT = re.compile(r"^concept:[^~\s]+~[0-9a-f]{64}$")
_JSON_STRING = r'"(?:\\.|[^"\\])*"'
_CERTAIN_CONCEPT_ANCHOR = re.compile(
    rf"^\(:\s+([A-Za-z][A-Za-z0-9_]*)\s+"
    rf"\(HasConcept\s+({_JSON_STRING})\s+({_JSON_STRING})\)\s+"
    r"\(STV\s+1(?:\.0+)?\s+1(?:\.0+)?\)\s*\)$"
)
_SAFE_PROOF_ATOM = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_FORBIDDEN_PROVENANCE_KEYS = frozenset(
    {"action", "click", "engagement", "label", "labels", "outcome", "relevant", "score"}
)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _symbol(prefix: str, *parts: Any) -> str:
    return f"{prefix}_{_canonical_hash(parts)[:24]}"


def _wikidata_identifier(value: object) -> str | None:
    if isinstance(value, Mapping):
        for key in ("WikidataId", "WikidataID", "WdId", "EntityId"):
            if key in value:
                return _wikidata_identifier(value.get(key))
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized if _WIKIDATA_ENTITY.fullmatch(normalized) else None


def mind_wikidata_entities(article: object) -> tuple[str, ...] | None:
    """Return exact stable entity IDs, or ``None`` when evidence is unknown.

    The two MIND entity columns may be explicit empty lists, which means the
    article has no usable Wikidata observation.  Missing/malformed columns are
    unknown.  Human-readable labels are never substituted for stable IDs.
    """

    if not isinstance(article, Mapping):
        return None
    fields = []
    for name in ("title_entities", "abstract_entities"):
        if name not in article:
            continue
        raw = article[name]
        if not isinstance(raw, (list, tuple)):
            return None
        fields.append(raw)
    if not fields:
        return None
    result: set[str] = set()
    for raw in fields:
        for item in raw:
            identifier = _wikidata_identifier(item)
            if identifier is not None:
                result.add(identifier)
            elif item not in (None, ""):
                # A nonempty but ungrounded annotation does not justify a
                # closed-world "none" conclusion.
                return None
    return tuple(sorted(result))


def canonical_annotation_concepts(annotation: object) -> tuple[str, ...] | None:
    """Return canonical concept IDs only when their source record is anchored."""

    if not isinstance(annotation, Mapping):
        return None
    raw = annotation.get("concepts")
    provenance = annotation.get("provenance")
    if not isinstance(raw, list) or not isinstance(provenance, Mapping):
        return None
    if not isinstance(provenance.get("article_id"), str) or not provenance["article_id"]:
        return None
    concepts = []
    for value in raw:
        if not isinstance(value, str) or not _CANONICAL_CONCEPT.fullmatch(value):
            return None
        concepts.append(value)
    anchored = provenance.get("anchored_statements")
    canonicalization = provenance.get("canonicalization")
    mappings = canonicalization.get("mappings") if isinstance(canonicalization, Mapping) else None
    concept_mappings = mappings.get("concepts") if isinstance(mappings, Mapping) else None
    if not isinstance(anchored, list) or any(not isinstance(item, str) for item in anchored):
        return None
    if not isinstance(concept_mappings, list):
        return None
    mapped = {
        item.get("canonical_id")
        for item in concept_mappings if isinstance(item, Mapping)
    }
    if any(concept not in mapped for concept in concepts):
        return None
    if any(not _concept_anchor_fact_ids(annotation, concept) for concept in concepts):
        return None
    return tuple(sorted(set(concepts)))


def _parse_concept_anchor(statement: object) -> tuple[str, str, str, str] | None:
    """Parse the deliberately narrow, certain ``HasConcept`` source form."""

    if not isinstance(statement, str):
        return None
    source = statement.strip()
    match = _CERTAIN_CONCEPT_ANCHOR.fullmatch(source)
    if match is None:
        return None
    try:
        article_id = json.loads(match.group(2))
        lexical = json.loads(match.group(3))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(article_id, str) or not article_id:
        return None
    if not isinstance(lexical, str) or not lexical:
        return None
    return match.group(1), article_id, lexical, source


def _concept_anchor_facts(
    annotation: Mapping[str, object], concept_id: str,
) -> tuple[tuple[str, str, str, str], ...]:
    """Return ``(fact ID, article ID, lexical, source)`` anchor records."""

    try:
        provenance = annotation["provenance"]
        mappings = provenance["canonicalization"]["mappings"]["concepts"]
        statements = provenance["anchored_statements"]
    except (KeyError, TypeError):
        return ()
    expected_article_id = provenance.get("article_id")
    if not isinstance(expected_article_id, str) or not expected_article_id:
        return ()
    source_values = {
        value
        for item in mappings
        if isinstance(item, Mapping) and item.get("canonical_id") == concept_id
        for value in (item.get("raw"), item.get("lexical"))
        if isinstance(value, str) and value
    }
    parsed = [
        anchor for statement in statements
        if (anchor := _parse_concept_anchor(statement)) is not None
        and anchor[1] == expected_article_id
        and anchor[2] in source_values
    ]
    # One name must denote one immutable statement.  Ambiguous named anchors
    # are unavailable evidence rather than something the bridge may choose.
    names: dict[str, str] = {}
    for fact_id, _article_id, _lexical, source in parsed:
        if fact_id in names and names[fact_id] != source:
            return ()
        names[fact_id] = source
    return tuple(sorted(set(parsed)))


def _concept_anchor_fact_ids(annotation: Mapping[str, object], concept_id: str) -> tuple[str, ...]:
    return tuple(sorted({item[0] for item in _concept_anchor_facts(annotation, concept_id)}))


@dataclass(frozen=True, slots=True)
class RelationalOriginPlan:
    origin_id: str
    dependency_key: str
    causal_history_id: str
    history_article_id: str
    history_article_atom: str
    history_position: int
    recency: str
    observed_click_fact_id: str
    matched_entity_ids: tuple[str, ...]
    history_entity_fact_ids: tuple[str, ...]
    candidate_entity_fact_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RelationalProofRoot:
    """One deterministic, label-free proof obligation."""

    origin_id: str
    matched_value_id: str
    matched_value_atom: str
    query: str


@dataclass(frozen=True, slots=True)
class RelationalProofPlan:
    case_id: str
    scope_id: str
    user_id: str
    candidate_id: str
    candidate_article_atom: str | None
    history_ids: tuple[str, ...]
    query: str | None
    statements: tuple[str, ...]
    case_candidate_fact_id: str | None
    origins: tuple[RelationalOriginPlan, ...]
    complete_entity_evidence: bool
    complete_recent_entity_evidence: bool

    @property
    def requires_query(self) -> bool:
        return self.query is not None

    @property
    def proof_roots(self) -> tuple[RelationalProofRoot, ...]:
        return tuple(
            RelationalProofRoot(
                origin_id=origin.origin_id,
                matched_value_id=entity_id,
                matched_value_atom=f"rel_entity_{entity_id.lower()}",
                query=(
                    f"(: $prf (RelEntityContinuity {self.case_id} "
                    f"{origin.origin_id} rel_entity_{entity_id.lower()}) $tv)"
                ),
            )
            for origin in self.origins
            for entity_id in origin.matched_entity_ids
        )

    @property
    def proof_queries(self) -> tuple[str, ...]:
        return tuple(root.query for root in self.proof_roots)


@dataclass(frozen=True, slots=True)
class ConceptBridgePath:
    canonical_concept_id: str
    concept_atom: str
    annotation_anchor_fact_id: str
    canonical_mapping_fact_id: str


@dataclass(frozen=True, slots=True)
class ConceptRelationalOriginPlan:
    origin_id: str
    dependency_key: str
    causal_history_id: str
    history_article_id: str
    history_article_atom: str
    history_position: int
    recency: str
    observed_click_fact_id: str
    matched_concept_ids: tuple[str, ...]
    history_concept_fact_ids: tuple[str, ...]
    candidate_concept_fact_ids: tuple[str, ...]
    annotation_anchor_fact_ids: tuple[str, ...]
    history_bridge_paths: tuple[ConceptBridgePath, ...]
    candidate_bridge_paths: tuple[ConceptBridgePath, ...]


@dataclass(frozen=True, slots=True)
class ConceptRelationalProofPlan:
    case_id: str
    scope_id: str
    user_id: str
    candidate_id: str
    candidate_article_atom: str | None
    history_ids: tuple[str, ...]
    query: str | None
    statements: tuple[str, ...]
    case_candidate_fact_id: str | None
    origins: tuple[ConceptRelationalOriginPlan, ...]
    complete_concept_evidence: bool
    complete_recent_concept_evidence: bool

    @property
    def requires_query(self) -> bool:
        return self.query is not None

    @property
    def proof_roots(self) -> tuple[RelationalProofRoot, ...]:
        return tuple(
            RelationalProofRoot(
                origin_id=origin.origin_id,
                matched_value_id=concept_id,
                matched_value_atom=f"rel_concept_{_canonical_hash(concept_id)[:24]}",
                query=(
                    f"(: $prf (RelConceptContinuity {self.case_id} "
                    f"{origin.origin_id} "
                    f"rel_concept_{_canonical_hash(concept_id)[:24]}) $tv)"
                ),
            )
            for origin in self.origins
            for concept_id in origin.matched_concept_ids
        )

    @property
    def proof_queries(self) -> tuple[str, ...]:
        return tuple(root.query for root in self.proof_roots)


def _article_observation(article_id: str, articles: Mapping[str, object]):
    article = articles.get(article_id)
    entities = mind_wikidata_entities(article)
    if entities is None:
        return None, None, ()
    article_atom = _symbol("rel_article", article_id, entities)
    facts = tuple(
        (
            _symbol("fact_rel_mentions", article_atom, entity),
            f"(: {_symbol('fact_rel_mentions', article_atom, entity)} "
            f"(RelMentionsEntity {article_atom} rel_entity_{entity.lower()}) "
            "(STV 1.0 1.0))",
        )
        for entity in entities
    )
    return article_atom, entities, facts


def build_relational_proof_plan(
    candidate_id: str,
    ordered_history_ids: Iterable[str],
    articles: Mapping[str, object],
    *,
    user_id: str,
    observation_cache: dict[str, tuple[object, ...]] | None = None,
) -> RelationalProofPlan:
    """Build a causal source-fact plan without consulting engagement labels."""

    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate_id must be a nonempty string")
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("user_id must be a nonempty string")
    if not isinstance(articles, Mapping):
        raise TypeError("articles must map article IDs to article records")
    if isinstance(ordered_history_ids, (str, bytes)):
        raise TypeError("ordered_history_ids must be an iterable of article IDs")
    history = tuple(ordered_history_ids)
    if any(not isinstance(identifier, str) or not identifier for identifier in history):
        raise ValueError("history must contain nonempty article ID strings")

    def observe(identifier):
        if observation_cache is None:
            return _article_observation(identifier, articles)
        if identifier not in observation_cache:
            observation_cache[identifier] = _article_observation(identifier, articles)
        return observation_cache[identifier]

    candidate_atom, candidate_entities, candidate_facts = observe(candidate_id)
    history_observations = [
        observe(identifier) for identifier in history
    ]
    complete = candidate_entities is not None and all(
        entities is not None for _, entities, _ in history_observations
    )
    recent_start = max(0, len(history) - RECENT_HISTORY_POSITIONS)
    complete_recent = candidate_entities is not None and all(
        entities is not None
        for _, entities, _ in history_observations[recent_start:]
    )
    versioned_history = tuple(
        article_atom if article_atom is not None else _symbol("rel_unknown_article", identifier)
        for identifier, (article_atom, _, _) in zip(history, history_observations)
    )
    user_atom = _symbol("rel_user", user_id)
    causal_history_id = _symbol("rel_causal_history", user_id, history)
    scope_id = _symbol("rel_scope", user_atom, versioned_history)
    case_id = _symbol(
        "rel_case", scope_id,
        candidate_atom if candidate_atom is not None else ("unknown", candidate_id),
    )

    statements = {statement for _, statement in candidate_facts}
    origins: list[RelationalOriginPlan] = []
    candidate_fact_ids = {
        entity: fact_id for entity, (fact_id, _) in zip(candidate_entities or (), candidate_facts)
    }
    for position, (history_id, observation) in enumerate(zip(history, history_observations)):
        article_atom, entities, entity_facts = observation
        statements.update(statement for _, statement in entity_facts)
        origin_id = _symbol("rel_origin", scope_id, position, versioned_history[position])
        dependency_key = _symbol(
            "rel_interaction_origin", causal_history_id, position, history_id,
        )
        observed_id = _symbol("fact_rel_observed", scope_id, origin_id)
        history_atom = versioned_history[position]
        statements.add(
            f"(: {observed_id} "
            f"(RelObservedClick {scope_id} {origin_id} {user_atom} {history_atom}) "
            "(STV 1.0 1.0))"
        )
        history_fact_ids = {
            entity: fact_id for entity, (fact_id, _) in zip(entities or (), entity_facts)
        }
        matched = tuple(sorted(set(candidate_entities or ()) & set(entities or ())))
        origins.append(RelationalOriginPlan(
            origin_id=origin_id,
            dependency_key=dependency_key,
            causal_history_id=causal_history_id,
            history_article_id=history_id,
            history_article_atom=history_atom,
            history_position=position,
            recency="recent" if position >= recent_start else "older",
            observed_click_fact_id=observed_id,
            matched_entity_ids=matched,
            history_entity_fact_ids=tuple(history_fact_ids[value] for value in matched),
            candidate_entity_fact_ids=tuple(candidate_fact_ids[value] for value in matched),
        ))

    case_candidate_id = None
    query = None
    if candidate_atom is not None:
        case_candidate_id = _symbol("fact_rel_candidate", case_id)
        statements.add(
            f"(: {case_candidate_id} "
            f"(RelCaseCandidate {case_id} {scope_id} {user_atom} {candidate_atom}) "
            "(STV 1.0 1.0))"
        )
        # A query is necessary only when a structural path can exist.  This is
        # a label-free prefilter; it never promotes a relationship to a proof.
        if any(origin.matched_entity_ids for origin in origins):
            query = f"(: $prf (RelEntityContinuity {case_id} $origin $entity) $tv)"

    return RelationalProofPlan(
        case_id=case_id,
        scope_id=scope_id,
        user_id=user_id,
        candidate_id=candidate_id,
        candidate_article_atom=candidate_atom,
        history_ids=history,
        query=query,
        statements=tuple(sorted(statements)),
        case_candidate_fact_id=case_candidate_id,
        origins=tuple(origins),
        complete_entity_evidence=complete,
        complete_recent_entity_evidence=complete_recent,
    )


# Cheap serving alias: this creates only source facts and a query plan.  It
# intentionally cannot manufacture a proof-derived feature value.


def _concept_observation(article_id: str, annotations: Mapping[str, object]):
    annotation = annotations.get(article_id)
    concepts = canonical_annotation_concepts(annotation)
    if concepts is None:
        return None, None, (), {}
    provenance = annotation["provenance"]
    if provenance.get("article_id") != article_id:
        return None, None, (), {}
    anchors = {
        concept: _concept_anchor_facts(annotation, concept) for concept in concepts
    }
    # A canonical ID without its exact named content anchor is unavailable
    # evidence, not a free-standing semantic assertion.
    if any(not values for values in anchors.values()):
        return None, None, (), {}
    provenance = annotation["provenance"]
    version = {
        "concepts": concepts,
        "article_content_sha256": provenance.get("article_content_sha256"),
        "registry_sha256": provenance.get("canonicalization", {}).get("registry_sha256"),
        "anchors": {
            concept: tuple((fact_id, lexical, source) for fact_id, _, lexical, source in values)
            for concept, values in anchors.items()
        },
    }
    article_atom = _symbol("rel_concept_article", article_id, version)
    facts: dict[str, str] = {}
    bridge_paths: dict[str, tuple[ConceptBridgePath, ...]] = {}
    for concept in concepts:
        concept_atom = f"rel_concept_{_canonical_hash(concept)[:24]}"
        paths = []
        for anchor_id, source_article_id, lexical, source_statement in anchors[concept]:
            if anchor_id in facts and facts[anchor_id] != source_statement:
                return None, None, (), {}
            facts[anchor_id] = source_statement
            mapping_id = _symbol(
                "fact_rel_canonical_mapping", article_atom, concept, anchor_id,
            )
            facts[mapping_id] = (
                f"(: {mapping_id} "
                "(RelCanonicalConceptMapping "
                f"{json.dumps(source_article_id, ensure_ascii=False)} "
                f"{json.dumps(lexical, ensure_ascii=False)} "
                f"{article_atom} {concept_atom}) "
                "(STV 1.0 1.0))"
            )
            paths.append(ConceptBridgePath(
                canonical_concept_id=concept,
                concept_atom=concept_atom,
                annotation_anchor_fact_id=anchor_id,
                canonical_mapping_fact_id=mapping_id,
            ))
        bridge_paths[concept] = tuple(sorted(
            paths,
            key=lambda path: (
                path.annotation_anchor_fact_id, path.canonical_mapping_fact_id,
            ),
        ))
    return article_atom, concepts, tuple(sorted(facts.items())), bridge_paths


def build_concept_relational_proof_plan(
    candidate_id: str,
    ordered_history_ids: Iterable[str],
    annotations: Mapping[str, object],
    *,
    user_id: str,
    observation_cache: dict[str, tuple[object, ...]] | None = None,
) -> ConceptRelationalProofPlan:
    """Plan canonical-concept continuity from provenance-anchored records."""

    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate_id must be a nonempty string")
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("user_id must be a nonempty string")
    if not isinstance(annotations, Mapping):
        raise TypeError("annotations must map article IDs to canonical records")
    if isinstance(ordered_history_ids, (str, bytes)):
        raise TypeError("ordered_history_ids must be an iterable of article IDs")
    history = tuple(ordered_history_ids)
    if any(not isinstance(identifier, str) or not identifier for identifier in history):
        raise ValueError("history must contain nonempty article ID strings")

    def observe(identifier):
        if observation_cache is None:
            return _concept_observation(identifier, annotations)
        if identifier not in observation_cache:
            observation_cache[identifier] = _concept_observation(identifier, annotations)
        return observation_cache[identifier]

    candidate_atom, candidate_concepts, candidate_facts, candidate_paths = observe(candidate_id)
    history_observations = [
        observe(identifier) for identifier in history
    ]
    complete = candidate_concepts is not None and all(
        concepts is not None for _, concepts, _, _ in history_observations
    )
    recent_start = max(0, len(history) - RECENT_HISTORY_POSITIONS)
    complete_recent = candidate_concepts is not None and all(
        concepts is not None
        for _, concepts, _, _ in history_observations[recent_start:]
    )
    versioned_history = tuple(
        article_atom if article_atom is not None else _symbol("rel_unknown_concept_article", identifier)
        for identifier, (article_atom, _, _, _) in zip(history, history_observations)
    )
    user_atom = _symbol("rel_concept_user", user_id)
    causal_history_id = _symbol("rel_causal_history", user_id, history)
    scope_id = _symbol("rel_concept_scope", user_atom, versioned_history)
    case_id = _symbol(
        "rel_concept_case", scope_id,
        candidate_atom if candidate_atom is not None else ("unknown", candidate_id),
    )
    statements = {statement for _, statement in candidate_facts}
    origins = []
    for position, (history_id, observation) in enumerate(zip(history, history_observations)):
        article_atom, concepts, concept_facts, history_paths = observation
        statements.update(statement for _, statement in concept_facts)
        history_atom = versioned_history[position]
        origin_id = _symbol("rel_concept_origin", scope_id, position, history_atom)
        dependency_key = _symbol(
            "rel_interaction_origin", causal_history_id, position, history_id,
        )
        observed_id = _symbol("fact_rel_concept_observed", scope_id, origin_id)
        statements.add(
            f"(: {observed_id} "
            f"(RelConceptObservedClick {scope_id} {origin_id} {user_atom} {history_atom}) "
            "(STV 1.0 1.0))"
        )
        matched = tuple(sorted(set(candidate_concepts or ()) & set(concepts or ())))
        matched_candidate_paths = tuple(
            path for concept in matched for path in candidate_paths.get(concept, ())
        )
        matched_history_paths = tuple(
            path for concept in matched for path in history_paths.get(concept, ())
        )
        candidate_mapping_ids = tuple(
            path.canonical_mapping_fact_id for path in matched_candidate_paths
        )
        history_mapping_ids = tuple(
            path.canonical_mapping_fact_id for path in matched_history_paths
        )
        anchors = {
            *(path.annotation_anchor_fact_id for path in matched_candidate_paths),
            *(path.annotation_anchor_fact_id for path in matched_history_paths),
        }
        origins.append(ConceptRelationalOriginPlan(
            origin_id=origin_id,
            dependency_key=dependency_key,
            causal_history_id=causal_history_id,
            history_article_id=history_id,
            history_article_atom=history_atom,
            history_position=position,
            recency="recent" if position >= recent_start else "older",
            observed_click_fact_id=observed_id,
            matched_concept_ids=matched,
            history_concept_fact_ids=history_mapping_ids,
            candidate_concept_fact_ids=candidate_mapping_ids,
            annotation_anchor_fact_ids=tuple(sorted(anchors)),
            history_bridge_paths=matched_history_paths,
            candidate_bridge_paths=matched_candidate_paths,
        ))

    case_candidate_id = None
    query = None
    if candidate_atom is not None:
        case_candidate_id = _symbol("fact_rel_concept_candidate", case_id)
        statements.add(
            f"(: {case_candidate_id} "
            f"(RelConceptCaseCandidate {case_id} {scope_id} {user_atom} {candidate_atom}) "
            "(STV 1.0 1.0))"
        )
        if any(origin.matched_concept_ids for origin in origins):
            query = f"(: $prf (RelConceptContinuity {case_id} $origin $concept) $tv)"
    return ConceptRelationalProofPlan(
        case_id=case_id,
        scope_id=scope_id,
        user_id=user_id,
        candidate_id=candidate_id,
        candidate_article_atom=candidate_atom,
        history_ids=history,
        query=query,
        statements=tuple(sorted(statements)),
        case_candidate_fact_id=case_candidate_id,
        origins=tuple(origins),
        complete_concept_evidence=complete,
        complete_recent_concept_evidence=complete_recent,
    )




def build_relational_plans(
    candidate_id: str,
    ordered_history_ids: Iterable[str],
    articles: Mapping[str, object],
    annotations: Mapping[str, object],
    *,
    user_id: str,
    entity_observation_cache: dict[str, tuple[object, ...]] | None = None,
    concept_observation_cache: dict[str, tuple[object, ...]] | None = None,
) -> tuple[RelationalProofPlan, ConceptRelationalProofPlan]:
    """Return entity and canonical-concept source plans for one candidate."""

    history = tuple(ordered_history_ids)
    return (
        build_relational_proof_plan(
            candidate_id, history, articles, user_id=user_id,
            observation_cache=entity_observation_cache,
        ),
        build_concept_relational_proof_plan(
            candidate_id, history, annotations, user_id=user_id,
            observation_cache=concept_observation_cache,
        ),
    )


def _proof_origins(plan: RelationalProofPlan, proofs: Iterable[str]):
    pattern = re.compile(
        rf"\(RelEntityContinuity\s+{re.escape(plan.case_id)}\s+"
        rf"([A-Za-z][A-Za-z0-9_]*)\s+([A-Za-z][A-Za-z0-9_]*)\)"
    )
    by_origin: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {}
    allowed = {origin.origin_id: origin for origin in plan.origins}
    for proof in proofs:
        if not isinstance(proof, str) or not proof.strip():
            raise ValueError("PeTTa proof results must be nonempty strings")
        roots = set(pattern.findall(proof))
        if len(roots) != 1:
            raise ValueError("proof result does not contain exactly one requested root")
        origin_id, root_entity_atom = roots.pop()
        if not _SAFE_PROOF_ATOM.fullmatch(origin_id) or origin_id not in allowed:
            raise ValueError("proof returned an origin outside the causal history")
        origin = allowed[origin_id]
        if not origin.matched_entity_ids:
            raise ValueError("proof returned an origin without shared Wikidata evidence")
        # A real two-hop proof must retain both rule names and all root source
        # facts.  This prevents a direct asserted conclusion from passing as a
        # chained derivation and makes the dependency ledger meaningful.
        required = (
            ENGAGED_ENTITY_RULE_ID,
            ENTITY_CONTINUITY_RULE_ID,
            plan.case_candidate_fact_id,
            origin.observed_click_fact_id,
        )
        if any(value is None or str(value) not in proof for value in required):
            raise ValueError("proof is missing a required two-hop dependency")
        matched_paths = [
            (entity_id, candidate_fact, history_fact)
            for entity_id, candidate_fact, history_fact in zip(
                origin.matched_entity_ids,
                origin.candidate_entity_fact_ids,
                origin.history_entity_fact_ids,
            )
            if root_entity_atom == f"rel_entity_{entity_id.lower()}"
            and candidate_fact in proof and history_fact in proof
        ]
        if not matched_paths:
            raise ValueError("proof is missing matching candidate/history entity facts")
        source_ids = {
            str(plan.case_candidate_fact_id), origin.observed_click_fact_id,
            *(candidate for _, candidate, _ in matched_paths),
            *(history for _, _, history in matched_paths),
        }
        normalized = proof.strip()
        by_origin.setdefault(origin_id, {})[normalized] = {
            "source_fact_ids": tuple(sorted(source_ids)),
            "matched_value_ids": tuple(sorted({value for value, _, _ in matched_paths})),
            "annotation_anchor_fact_ids": (),
            "canonical_mapping_fact_ids": (),
        }
    return by_origin


def _serialized_proof_alternatives(
    alternatives: Mapping[str, Mapping[str, tuple[str, ...]]],
) -> list[dict[str, object]]:
    result = []
    for proof in sorted(alternatives):
        details = alternatives[proof]
        result.append({
            "proof_metta": proof,
            "proof_sha256": hashlib.sha256(proof.encode("utf-8")).hexdigest(),
            "source_fact_ids": list(details["source_fact_ids"]),
            "matched_value_ids": list(details["matched_value_ids"]),
            "annotation_anchor_fact_ids": list(details["annotation_anchor_fact_ids"]),
            "canonical_mapping_fact_ids": list(details["canonical_mapping_fact_ids"]),
        })
    return result


def _require_complete_proof_roots(
    proof_roots: Sequence[RelationalProofRoot],
    by_origin: Mapping[str, Mapping[str, Mapping[str, tuple[str, ...]]]],
    *,
    family: str,
) -> None:
    expected = {
        (root.origin_id, root.matched_value_id) for root in proof_roots
    }
    actual = {
        (origin_id, value_id)
        for origin_id, alternatives in by_origin.items()
        for details in alternatives.values()
        for value_id in details["matched_value_ids"]
    }
    if actual != expected:
        missing = len(expected - actual)
        extra = len(actual - expected)
        raise ValueError(
            f"{family} proof coverage is incomplete or unexpected "
            f"(expected={len(expected)}, actual={len(actual)}, "
            f"missing={missing}, extra={extra})"
        )


def _require_reported_root_coverage(
    proof_strings: Sequence[str], proof_roots: Sequence[RelationalProofRoot],
    *,
    case_id: str,
    predicate: str,
    family: str,
) -> None:
    pattern = re.compile(
        rf"\({predicate}\s+{re.escape(case_id)}\s+"
        rf"([A-Za-z][A-Za-z0-9_]*)\s+([A-Za-z][A-Za-z0-9_]*)\)"
    )
    expected = {
        (root.origin_id, root.matched_value_atom) for root in proof_roots
    }
    actual = {
        root
        for proof in proof_strings
        if isinstance(proof, str)
        for root in pattern.findall(proof)
    }
    if actual != expected:
        raise ValueError(
            f"{family} proof coverage is incomplete or unexpected "
            f"(expected={len(expected)}, actual={len(actual)}, "
            f"missing={len(expected - actual)}, extra={len(actual - expected)})"
        )


def reduce_relational_proofs(
    plan: RelationalProofPlan,
    proof_strings: Iterable[str],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Reduce proof paths to one dependency-aware observation per click origin."""

    materialized = tuple(proof_strings)
    if plan.query is None:
        if materialized:
            raise ValueError("a non-queryable relational plan cannot have proofs")
        by_origin = {}
    else:
        _require_reported_root_coverage(
            materialized, plan.proof_roots, case_id=plan.case_id,
            predicate="RelEntityContinuity", family="entity-continuity",
        )
        by_origin = _proof_origins(plan, materialized)
    _require_complete_proof_roots(
        plan.proof_roots, by_origin, family="entity-continuity",
    )
    origin_by_id = {origin.origin_id: origin for origin in plan.origins}
    proven = [origin_by_id[origin_id] for origin_id in sorted(by_origin)]
    if any(origin.recency == "recent" for origin in proven):
        scope = "recent"
    elif proven and plan.complete_recent_entity_evidence:
        scope = "older"
    elif proven:
        # An older proof establishes continuity, but an unavailable recent
        # observation means the stronger ``recent`` category has not been
        # ruled out. The fixed four-state serving vocabulary must abstain.
        scope = "unknown"
    elif plan.complete_entity_evidence:
        scope = "none"
    else:
        scope = "unknown"

    ledger: dict[str, dict[str, object]] = {}
    proof_ids = []
    retained_origins = proven if scope in {"recent", "older"} else ()
    for origin in retained_origins:
        proof_id = _symbol("rel_proof", plan.case_id, origin.origin_id)
        alternatives = _serialized_proof_alternatives(by_origin[origin.origin_id])
        representative = alternatives[0]["proof_metta"]
        dependencies = sorted({
            item for alternative in alternatives
            for item in alternative["source_fact_ids"]
        })
        matched_values = sorted({
            item for alternative in alternatives
            for item in alternative["matched_value_ids"]
        })
        ledger[proof_id] = {
            "schema": RELATIONAL_WORKSPACE_SCHEMA,
            "relation_family": "wikidata_entity_continuity",
            "case_id": plan.case_id,
            "scope_id": plan.scope_id,
            "origin_id": origin.origin_id,
            "dependency_key": origin.dependency_key,
            "causal_history_id": origin.causal_history_id,
            "user_id": plan.user_id,
            "candidate_id": plan.candidate_id,
            "history_article_id": origin.history_article_id,
            "history_position": origin.history_position,
            "recency": origin.recency,
            "matched_wikidata_entity_ids": matched_values,
            "source_fact_ids": dependencies,
            "rule_ids": [ENGAGED_ENTITY_RULE_ID, ENTITY_CONTINUITY_RULE_ID],
            "proof_alternative_count": len(alternatives),
            "proof_alternatives": alternatives,
            "proof_metta": representative,
            "proof_sha256": hashlib.sha256(representative.encode("utf-8")).hexdigest(),
        }
        proof_ids.append(proof_id)
    return {
        REL_ENTITY_CONTINUITY_SCOPE: scope,
        REL_ENTITY_CONTINUITY_PROOF_IDS: proof_ids,
    }, ledger


def _concept_proof_origins(plan: ConceptRelationalProofPlan, proofs: Iterable[str]):
    pattern = re.compile(
        rf"\(RelConceptContinuity\s+{re.escape(plan.case_id)}\s+"
        rf"([A-Za-z][A-Za-z0-9_]*)\s+([A-Za-z][A-Za-z0-9_]*)\)"
    )
    by_origin: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {}
    allowed = {origin.origin_id: origin for origin in plan.origins}
    for proof in proofs:
        if not isinstance(proof, str) or not proof.strip():
            raise ValueError("PeTTa concept-proof results must be nonempty strings")
        roots = set(pattern.findall(proof))
        if len(roots) != 1:
            raise ValueError("concept proof does not contain exactly one requested root")
        origin_id, root_concept_atom = roots.pop()
        if not _SAFE_PROOF_ATOM.fullmatch(origin_id) or origin_id not in allowed:
            raise ValueError("concept proof returned an origin outside the causal history")
        origin = allowed[origin_id]
        if not origin.matched_concept_ids:
            raise ValueError("concept proof returned an origin without shared canonical evidence")
        required = (
            CANONICAL_CONCEPT_BRIDGE_RULE_ID,
            ENGAGED_CONCEPT_RULE_ID,
            CONCEPT_CONTINUITY_RULE_ID,
            plan.case_candidate_fact_id,
            origin.observed_click_fact_id,
        )
        if any(value is None or str(value) not in proof for value in required):
            raise ValueError(
                "concept proof is missing a required anchored three-rule dependency"
            )
        matched_values = set()
        participating_candidate_paths = set()
        participating_history_paths = set()
        for concept_id in origin.matched_concept_ids:
            expected_concept_atom = f"rel_concept_{_canonical_hash(concept_id)[:24]}"
            if root_concept_atom != expected_concept_atom:
                continue
            candidate_paths = [
                path for path in origin.candidate_bridge_paths
                if path.canonical_concept_id == concept_id
                and path.annotation_anchor_fact_id in proof
                and path.canonical_mapping_fact_id in proof
            ]
            history_paths = [
                path for path in origin.history_bridge_paths
                if path.canonical_concept_id == concept_id
                and path.annotation_anchor_fact_id in proof
                and path.canonical_mapping_fact_id in proof
            ]
            if candidate_paths and history_paths:
                matched_values.add(concept_id)
                participating_candidate_paths.update(candidate_paths)
                participating_history_paths.update(history_paths)
        if not matched_values:
            raise ValueError(
                "concept proof is missing matching anchored canonicalization paths"
            )
        paths = participating_candidate_paths | participating_history_paths
        anchor_ids = {path.annotation_anchor_fact_id for path in paths}
        mapping_ids = {path.canonical_mapping_fact_id for path in paths}
        source_ids = {
            str(plan.case_candidate_fact_id), origin.observed_click_fact_id,
            *anchor_ids, *mapping_ids,
        }
        normalized = proof.strip()
        by_origin.setdefault(origin_id, {})[normalized] = {
            "source_fact_ids": tuple(sorted(source_ids)),
            "matched_value_ids": tuple(sorted(matched_values)),
            "annotation_anchor_fact_ids": tuple(sorted(anchor_ids)),
            "canonical_mapping_fact_ids": tuple(sorted(mapping_ids)),
        }
    return by_origin


def reduce_concept_relational_proofs(
    plan: ConceptRelationalProofPlan,
    proof_strings: Iterable[str],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Reduce canonical-concept paths to one observation per click origin."""

    materialized = tuple(proof_strings)
    if plan.query is None:
        if materialized:
            raise ValueError("a non-queryable concept plan cannot have proofs")
        by_origin = {}
    else:
        _require_reported_root_coverage(
            materialized, plan.proof_roots, case_id=plan.case_id,
            predicate="RelConceptContinuity", family="concept-continuity",
        )
        by_origin = _concept_proof_origins(plan, materialized)
    _require_complete_proof_roots(
        plan.proof_roots, by_origin, family="concept-continuity",
    )
    origin_by_id = {origin.origin_id: origin for origin in plan.origins}
    proven = [origin_by_id[origin_id] for origin_id in sorted(by_origin)]
    if any(origin.recency == "recent" for origin in proven):
        scope = "recent"
    elif proven and plan.complete_recent_concept_evidence:
        scope = "older"
    elif proven:
        # Do not turn an older path plus missing recent annotations into a
        # false recency ordering. The proof was checked for completeness, but
        # its ambiguous categorical observation deliberately abstains.
        scope = "unknown"
    elif plan.complete_concept_evidence:
        scope = "none"
    else:
        scope = "unknown"

    ledger = {}
    proof_ids = []
    retained_origins = proven if scope in {"recent", "older"} else ()
    for origin in retained_origins:
        proof_id = _symbol("rel_concept_proof", plan.case_id, origin.origin_id)
        alternatives = _serialized_proof_alternatives(by_origin[origin.origin_id])
        representative = alternatives[0]["proof_metta"]
        dependencies = sorted({
            item for alternative in alternatives
            for item in alternative["source_fact_ids"]
        })
        anchors = sorted({
            item for alternative in alternatives
            for item in alternative["annotation_anchor_fact_ids"]
        })
        mappings = sorted({
            item for alternative in alternatives
            for item in alternative["canonical_mapping_fact_ids"]
        })
        matched_values = sorted({
            item for alternative in alternatives
            for item in alternative["matched_value_ids"]
        })
        ledger[proof_id] = {
            "schema": RELATIONAL_WORKSPACE_SCHEMA,
            "relation_family": "canonical_concept_continuity",
            "case_id": plan.case_id,
            "scope_id": plan.scope_id,
            "origin_id": origin.origin_id,
            "dependency_key": origin.dependency_key,
            "causal_history_id": origin.causal_history_id,
            "user_id": plan.user_id,
            "candidate_id": plan.candidate_id,
            "history_article_id": origin.history_article_id,
            "history_position": origin.history_position,
            "recency": origin.recency,
            "matched_canonical_concept_ids": matched_values,
            "source_fact_ids": dependencies,
            "annotation_anchor_fact_ids": anchors,
            "canonical_mapping_fact_ids": mappings,
            "rule_ids": [
                CANONICAL_CONCEPT_BRIDGE_RULE_ID,
                ENGAGED_CONCEPT_RULE_ID,
                CONCEPT_CONTINUITY_RULE_ID,
            ],
            "proof_alternative_count": len(alternatives),
            "proof_alternatives": alternatives,
            "proof_metta": representative,
            "proof_sha256": hashlib.sha256(representative.encode("utf-8")).hexdigest(),
        }
        proof_ids.append(proof_id)
    return {
        REL_CONCEPT_CONTINUITY_SCOPE: scope,
        REL_CONCEPT_CONTINUITY_PROOF_IDS: proof_ids,
    }, ledger


def _explicit_history(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{label} lacks an explicit ordered history")
    return tuple(value)


def _context_bindings(data: Mapping[str, object]):
    """Yield each projected context with its immutable containing identity."""

    events = data.get("events")
    if not isinstance(events, list):
        raise ValueError("relational projection events must be a list")
    for index, context in enumerate(events):
        if not isinstance(context, Mapping):
            raise ValueError(f"training context {index} must be a mapping")
        user_id, candidate_id = context.get("user"), context.get("article")
        if not isinstance(user_id, str) or not user_id:
            raise ValueError(f"training context {index} lacks a user ID")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"training context {index} lacks a candidate ID")
        history = _explicit_history(context.get("history"), f"training context {index}")
        yield ("training", index), context, user_id, candidate_id, history

    evaluation_keys = [
        key for key in ("eval_impressions", "evaluation", "tests", "impressions")
        if isinstance(data.get(key), list)
    ]
    if len(evaluation_keys) != 1:
        raise ValueError("relational projection requires exactly one evaluation list")
    key = evaluation_keys[0]
    for case_index, case in enumerate(data[key]):
        if not isinstance(case, Mapping):
            raise ValueError(f"evaluation impression {case_index} must be a mapping")
        user_id = case.get("user")
        candidates = case.get("candidates")
        contexts = case.get("candidate_context")
        if not isinstance(user_id, str) or not user_id:
            raise ValueError(f"evaluation impression {case_index} lacks a user ID")
        history = _explicit_history(
            case.get("history"), f"evaluation impression {case_index}",
        )
        if not isinstance(candidates, list) or not isinstance(contexts, Mapping):
            raise ValueError(f"evaluation impression {case_index} lacks candidate contexts")
        for candidate_index, candidate_id in enumerate(candidates):
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError("evaluation candidate IDs must be nonempty strings")
            context = contexts.get(candidate_id)
            if not isinstance(context, Mapping):
                raise ValueError("each evaluation candidate needs a context mapping")
            yield (
                key, case_index, candidate_index,
            ), context, user_id, candidate_id, history


def relational_context_observations_sha256(data: Mapping[str, object]) -> str:
    """Hash projected fields together with their exact containing identities."""

    observations = []
    for location, context, user_id, candidate_id, history in _context_bindings(data):
        observations.append({
            "location": location,
            "user_id": user_id,
            "candidate_id": candidate_id,
            "ordered_history_ids": history,
            "entity_scope": context.get(REL_ENTITY_CONTINUITY_SCOPE),
            "entity_proofs": context.get(REL_ENTITY_CONTINUITY_PROOF_IDS),
            "concept_scope": context.get(REL_CONCEPT_CONTINUITY_SCOPE),
            "concept_proofs": context.get(REL_CONCEPT_CONTINUITY_PROOF_IDS),
        })
    return _canonical_hash(observations)


def _expected_projection_roots(data: Mapping[str, object]) -> dict[str, int]:
    """Rebuild label-free proof obligations from the preserved source data."""

    raw_articles = data.get("articles")
    if not isinstance(raw_articles, list):
        raise ValueError("relational projection articles must be a list")
    articles: dict[str, Mapping[str, object]] = {}
    for index, article in enumerate(raw_articles):
        if not isinstance(article, Mapping):
            raise ValueError(f"relational projection article {index} is invalid")
        article_id = article.get("id")
        if not isinstance(article_id, str) or not article_id:
            raise ValueError(f"relational projection article {index} lacks an ID")
        if article_id in articles:
            raise ValueError(f"duplicate relational projection article ID: {article_id}")
        articles[article_id] = article
    annotations = data.get("llm_article_annotations") or {}
    if not isinstance(annotations, Mapping):
        raise ValueError("relational projection annotations must be a mapping")

    planned = {}
    entity_cache: dict[str, tuple[object, ...]] = {}
    concept_cache: dict[str, tuple[object, ...]] = {}
    for _location, _context, user_id, candidate_id, history in _context_bindings(data):
        key = (user_id, candidate_id, history)
        if key not in planned:
            planned[key] = build_relational_plans(
                candidate_id, history, articles, annotations, user_id=user_id,
                entity_observation_cache=entity_cache,
                concept_observation_cache=concept_cache,
            )
    entity = sum(
        len(entity_plan.proof_roots)
        for entity_plan, _concept_plan in planned.values()
    )
    concept = sum(
        len(concept_plan.proof_roots)
        for _entity_plan, concept_plan in planned.values()
    )
    queryable = sum(
        int(plan.requires_query)
        for pair in planned.values() for plan in pair
    )
    return {
        "unique_candidate_history_plans": len(planned),
        "queryable_plans": queryable,
        "expected_entity_proof_roots": entity,
        "expected_concept_proof_roots": concept,
        "expected_proof_roots": entity + concept,
    }


def _record_matches_context(
    record: Mapping[str, object], user_id: str, candidate_id: str,
    history: tuple[str, ...],
) -> bool:
    position = record.get("history_position")
    if isinstance(position, bool) or not isinstance(position, int):
        return False
    if not 0 <= position < len(history):
        return False
    causal_history_id = _symbol("rel_causal_history", user_id, history)
    expected_dependency = _symbol(
        "rel_interaction_origin", causal_history_id, position, history[position],
    )
    expected_recency = (
        "recent" if position >= max(0, len(history) - RECENT_HISTORY_POSITIONS)
        else "older"
    )
    return all((
        record.get("user_id") == user_id,
        record.get("candidate_id") == candidate_id,
        record.get("causal_history_id") == causal_history_id,
        record.get("history_article_id") == history[position],
        record.get("dependency_key") == expected_dependency,
        record.get("recency") == expected_recency,
    ))


def relational_safety_audit(data: Mapping[str, object]) -> dict[str, int]:
    """Compute, rather than assert, the two published causal-safety counts."""

    ledger = data.get("relational_proof_ledger")
    if not isinstance(ledger, Mapping):
        raise ValueError("relational proof ledger must be a mapping")
    duplicate_origin_contributions = 0
    future_history_references = 0
    for _location, context, user_id, candidate_id, history in _context_bindings(data):
        for proof_field, family in (
            (REL_ENTITY_CONTINUITY_PROOF_IDS, "wikidata_entity_continuity"),
            (REL_CONCEPT_CONTINUITY_PROOF_IDS, "canonical_concept_continuity"),
        ):
            references = context.get(proof_field)
            if not isinstance(references, list):
                continue
            dependency_keys = []
            for reference in references:
                record = ledger.get(reference)
                if not isinstance(record, Mapping):
                    future_history_references += 1
                    continue
                if record.get("relation_family") != family or not _record_matches_context(
                    record, user_id, candidate_id, history,
                ):
                    future_history_references += 1
                dependency_keys.append(record.get("dependency_key"))
            valid_keys = [key for key in dependency_keys if isinstance(key, str)]
            duplicate_origin_contributions += len(valid_keys) - len(set(valid_keys))
    return {
        "duplicate_origin_contributions": duplicate_origin_contributions,
        "future_history_references": future_history_references,
    }


def _proof_contains_atom(proof: str, atom: str) -> bool:
    if not isinstance(atom, str) or not _SAFE_PROOF_ATOM.fullmatch(atom):
        return False
    return re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(atom)}(?![A-Za-z0-9_])", proof,
    ) is not None


def _validated_proof_alternatives(
    proof_id: str, record: Mapping[str, object], family: str,
) -> tuple[set[str], set[str], set[str], set[str]]:
    specifications = {
        "wikidata_entity_continuity": (
            (ENGAGED_ENTITY_RULE_ID, ENTITY_CONTINUITY_RULE_ID),
            "RelEntityContinuity", "rel_proof", "matched_wikidata_entity_ids",
            _WIKIDATA_ENTITY,
        ),
        "canonical_concept_continuity": (
            (
                CANONICAL_CONCEPT_BRIDGE_RULE_ID,
                ENGAGED_CONCEPT_RULE_ID,
                CONCEPT_CONTINUITY_RULE_ID,
            ),
            "RelConceptContinuity", "rel_concept_proof",
            "matched_canonical_concept_ids", _CANONICAL_CONCEPT,
        ),
    }
    rules, predicate, proof_prefix, matched_field, value_pattern = specifications[family]
    if record.get("schema") != RELATIONAL_WORKSPACE_SCHEMA:
        raise ValueError("relational proof record has the wrong schema")
    case_id, origin_id = record.get("case_id"), record.get("origin_id")
    if not isinstance(case_id, str) or not _SAFE_PROOF_ATOM.fullmatch(case_id):
        raise ValueError("relational proof has an invalid case ID")
    if not isinstance(origin_id, str) or not _SAFE_PROOF_ATOM.fullmatch(origin_id):
        raise ValueError("relational proof has an invalid origin ID")
    if proof_id != _symbol(proof_prefix, case_id, origin_id):
        raise ValueError("relational proof ID disagrees with its case and origin")
    if record.get("rule_ids") != list(rules):
        raise ValueError("relational proof rule IDs are incomplete or reordered")
    alternatives = record.get("proof_alternatives")
    if not isinstance(alternatives, list) or not alternatives:
        raise ValueError("relational proof does not retain its alternatives")
    if record.get("proof_alternative_count") != len(alternatives):
        raise ValueError("relational proof alternative count is inconsistent")

    ordered_proofs = []
    all_sources: set[str] = set()
    all_values: set[str] = set()
    all_anchors: set[str] = set()
    all_mappings: set[str] = set()
    root = re.compile(
        rf"\({predicate}\s+{re.escape(case_id)}\s+{re.escape(origin_id)}\s+"
        rf"([A-Za-z][A-Za-z0-9_]*)\)"
    )
    for alternative in alternatives:
        if not isinstance(alternative, Mapping):
            raise ValueError("invalid retained relational proof alternative")
        proof = alternative.get("proof_metta")
        if not isinstance(proof, str) or not proof.strip() or proof != proof.strip():
            raise ValueError("retained relational proof text is invalid")
        if hashlib.sha256(proof.encode("utf-8")).hexdigest() != alternative.get("proof_sha256"):
            raise ValueError("retained relational proof text disagrees with its hash")
        root_values = set(root.findall(proof))
        if len(root_values) != 1:
            raise ValueError("retained proof does not contain its requested root")
        if any(not _proof_contains_atom(proof, rule) for rule in rules):
            raise ValueError("retained proof does not contain every structural rule")
        sources = alternative.get("source_fact_ids")
        values = alternative.get("matched_value_ids")
        anchors = alternative.get("annotation_anchor_fact_ids")
        mappings = alternative.get("canonical_mapping_fact_ids")
        for name, values_list in (
            ("source facts", sources), ("matched values", values),
            ("annotation anchors", anchors), ("canonical mappings", mappings),
        ):
            if not isinstance(values_list, list) or len(values_list) != len(set(values_list)):
                raise ValueError(f"retained proof has invalid {name}")
            if any(not isinstance(item, str) for item in values_list):
                raise ValueError(f"retained proof has invalid {name}")
        if not sources or not values:
            raise ValueError("retained proof lacks a source path or matched value")
        if any(not _proof_contains_atom(proof, source) for source in sources):
            raise ValueError("retained proof claims a dependency absent from its proof")
        if any(value_pattern.fullmatch(value) is None for value in values):
            raise ValueError("retained proof has an invalid matched value")
        expected_root_values = {
            f"rel_entity_{value.lower()}"
            if family == "wikidata_entity_continuity"
            else f"rel_concept_{_canonical_hash(value)[:24]}"
            for value in values
        }
        if root_values != expected_root_values:
            raise ValueError("retained proof root disagrees with its matched value")
        if family == "canonical_concept_continuity":
            if not anchors or not mappings:
                raise ValueError("canonical proof lacks anchored bridge dependencies")
            if not (set(anchors) | set(mappings)) <= set(sources):
                raise ValueError("canonical bridge dependencies are not source facts")
            if any(not _proof_contains_atom(proof, item) for item in (*anchors, *mappings)):
                raise ValueError("canonical bridge dependency is absent from its proof")
        elif anchors or mappings:
            raise ValueError("entity proof unexpectedly contains canonical bridge metadata")
        ordered_proofs.append(proof)
        all_sources.update(sources)
        all_values.update(values)
        all_anchors.update(anchors)
        all_mappings.update(mappings)
    if ordered_proofs != sorted(set(ordered_proofs)):
        raise ValueError("relational proof alternatives are duplicated or unordered")

    representative = ordered_proofs[0]
    if record.get("proof_metta") != representative:
        raise ValueError("relational representative proof is not the first retained alternative")
    if hashlib.sha256(representative.encode("utf-8")).hexdigest() != record.get("proof_sha256"):
        raise ValueError("relational representative proof disagrees with its hash")
    top_sources = record.get("source_fact_ids")
    top_values = record.get(matched_field)
    if top_sources != sorted(all_sources) or top_values != sorted(all_values):
        raise ValueError("relational proof summary does not match retained alternatives")
    if family == "canonical_concept_continuity" and (
        record.get("annotation_anchor_fact_ids") != sorted(all_anchors)
        or record.get("canonical_mapping_fact_ids") != sorted(all_mappings)
    ):
        raise ValueError("canonical proof summary does not match retained bridge paths")
    return all_sources, all_values, all_anchors, all_mappings


def validate_relational_projection(data: Mapping[str, object]) -> None:
    """Fail closed when a projected snapshot or dependency ledger is corrupt."""

    if not isinstance(data, Mapping):
        raise ValueError("relational projection must be a mapping")
    metadata = data.get("metadata")
    audit = metadata.get("relational_workspace") if isinstance(metadata, Mapping) else None
    ledger = data.get("relational_proof_ledger")
    if not isinstance(audit, Mapping) or audit.get("schema") != RELATIONAL_WORKSPACE_SCHEMA:
        raise ValueError("missing or invalid relational workspace metadata")
    if not isinstance(ledger, Mapping):
        raise ValueError("relational proof ledger must be a mapping")
    expected_hash = _canonical_hash(ledger)
    if audit.get("proof_ledger_sha256") != expected_hash:
        raise ValueError("relational proof ledger disagrees with its metadata hash")
    if audit.get("context_observations_sha256") != relational_context_observations_sha256(data):
        raise ValueError("relational context observations disagree with their metadata hash")
    if audit.get("structural_rules_sha256") != _canonical_hash(RELATIONAL_STRUCTURAL_RULES):
        raise ValueError("relational structural rules disagree with their metadata hash")
    if audit.get("projection_schema") != RELATIONAL_PROJECTION_SCHEMA:
        raise ValueError("relational projection has an unsupported projection schema")

    expected_roots = _expected_projection_roots(data)

    def count(name: str) -> int:
        value = audit.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"relational projection has an invalid {name}")
        return value

    for name, expected in expected_roots.items():
        if count(name) != expected:
            raise ValueError(
                f"relational projection {name} disagrees with label-free proof plans"
            )
    root_count = expected_roots["expected_proof_roots"]
    if count("wildcard_queries_submitted") != 0:
        raise ValueError("relational projection used wildcard proof queries")
    if count("queries_submitted") != root_count:
        raise ValueError("relational projection did not query every exact proof root")
    if count("complete_proof_roots") != root_count:
        raise ValueError("relational projection has incomplete exact proof roots")
    if count("proof_rows_returned") < root_count:
        raise ValueError("relational projection returned fewer proofs than exact roots")
    if count("proof_origins") != len(ledger):
        raise ValueError("relational projection proof-origin count is inconsistent")
    steps = count("query_steps_per_root")
    batch_size = count("query_batch_size")
    if steps < 1 or batch_size < 1 or count("query_steps") != steps:
        raise ValueError("relational projection has an invalid per-root query budget")
    if audit.get("query_step_budget_policy") != (
        "per-root budget multiplied by roots in each query_many batch"
    ):
        raise ValueError("relational projection has an unsupported query-budget policy")
    if count("total_query_step_budget") != steps * root_count:
        raise ValueError("relational projection total query budget is inconsistent")
    expected_maximum = steps * min(batch_size, root_count) if root_count else 0
    if count("maximum_batch_query_step_budget") != expected_maximum:
        raise ValueError("relational projection batch query budget is inconsistent")
    expected_batches = (root_count + batch_size - 1) // batch_size
    if count("query_batches") != expected_batches:
        raise ValueError("relational projection query-batch count is inconsistent")
    safety = audit.get("safety")
    computed_safety = relational_safety_audit(data)
    if not isinstance(safety, Mapping) or any(
        safety.get(name) != value for name, value in computed_safety.items()
    ):
        raise ValueError("reported relational safety audit was not computed from the projection")
    if any(computed_safety.values()):
        raise ValueError("computed relational safety audit is nonzero")

    dependency_groups = {}
    for proof_id, record in ledger.items():
        if not isinstance(proof_id, str) or not isinstance(record, Mapping):
            raise ValueError("invalid relational proof-ledger record")
        if _FORBIDDEN_PROVENANCE_KEYS & set(record):
            raise ValueError("relational proof ledger contains an outcome field")
        family = record.get("relation_family")
        if family not in {
            "wikidata_entity_continuity", "canonical_concept_continuity",
        }:
            raise ValueError("relational proof has an unknown relation family")
        _validated_proof_alternatives(proof_id, record, family)
        position = record.get("history_position")
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise ValueError("relational proof has an invalid history position")
        group = (
            record.get("user_id"), record.get("candidate_id"),
            record.get("causal_history_id"), record.get("history_article_id"), position,
        )
        dependency_groups.setdefault(group, set()).add(record.get("dependency_key"))
    if any(len(keys) != 1 or None in keys for keys in dependency_groups.values()):
        raise ValueError("relation families disagree about a shared click dependency")

    referenced_proofs = set()
    for _location, context, user_id, candidate_id, history in _context_bindings(data):
        for scope_field, proof_field, family in (
            (
                REL_ENTITY_CONTINUITY_SCOPE,
                REL_ENTITY_CONTINUITY_PROOF_IDS,
                "wikidata_entity_continuity",
            ),
            (
                REL_CONCEPT_CONTINUITY_SCOPE,
                REL_CONCEPT_CONTINUITY_PROOF_IDS,
                "canonical_concept_continuity",
            ),
        ):
            scope = context.get(scope_field)
            references = context.get(proof_field)
            if scope not in RELATIONAL_SCOPE_VALUES or not isinstance(references, list):
                raise ValueError("candidate context lacks a valid relational observation")
            if len(references) != len(set(references)) or any(reference not in ledger for reference in references):
                raise ValueError("candidate context has invalid relational proof references")
            if any(ledger[reference].get("relation_family") != family for reference in references):
                raise ValueError("candidate context references the wrong relation family")
            if any(
                not _record_matches_context(
                    ledger[reference], user_id, candidate_id, history,
                )
                for reference in references
            ):
                raise ValueError("candidate context references a proof from another causal context")
            if scope in {"none", "unknown"} and references:
                raise ValueError("non-positive relational scope cannot reference a proof")
            if scope in {"recent", "older"} and not references:
                raise ValueError("positive relational scope requires a proof")
            recencies = {ledger[reference]["recency"] for reference in references}
            expected = "recent" if "recent" in recencies else ("older" if recencies else scope)
            if expected != scope:
                raise ValueError("relational scope disagrees with referenced origin proofs")
            referenced_proofs.update(references)
    if referenced_proofs != set(ledger):
        raise ValueError("relational proof ledger contains unreferenced or missing records")


__all__ = [
    "CANONICAL_CONCEPT_BRIDGE_RULE_ID",
    "CONCEPT_CONTINUITY_RULE_ID",
    "CONCEPT_RELATIONAL_STRUCTURAL_RULES",
    "ConceptBridgePath",
    "ConceptRelationalOriginPlan",
    "ConceptRelationalProofPlan",
    "ENGAGED_ENTITY_RULE_ID",
    "ENGAGED_CONCEPT_RULE_ID",
    "ENTITY_RELATIONAL_STRUCTURAL_RULES",
    "ENTITY_CONTINUITY_RULE_ID",
    "RECENT_HISTORY_POSITIONS",
    "RELATIONAL_PROJECTION_SCHEMA",
    "RELATIONAL_SCOPE_VALUES",
    "RELATIONAL_STRUCTURAL_RULES",
    "RELATIONAL_WORKSPACE_FEATURES",
    "RELATIONAL_WORKSPACE_SCHEMA",
    "REL_CONCEPT_CONTINUITY_PROOF_IDS",
    "REL_CONCEPT_CONTINUITY_SCOPE",
    "REL_ENTITY_CONTINUITY_PROOF_IDS",
    "REL_ENTITY_CONTINUITY_SCOPE",
    "RelationalOriginPlan",
    "RelationalProofRoot",
    "RelationalProofPlan",
    "build_concept_relational_proof_plan",
    "build_relational_proof_plan",
    "build_relational_plans",
    "mind_wikidata_entities",
    "relational_context_observations_sha256",
    "relational_safety_audit",
    "canonical_annotation_concepts",
    "reduce_concept_relational_proofs",
    "reduce_relational_proofs",
    "validate_relational_projection",
]
