"""Label-free observations over grounded article annotations and prior history.

These numbers describe the extractor's annotations, not calibrated truth about
an article or a reader. Only the real miner learns whether they predict an
outcome. No raw text, click labels, latest profiles, or ranking enter this API.
Missing annotations remain absent evidence; they are never a synthetic zero.
"""

from collections.abc import Mapping
import math
import unicodedata


LLM_WORKSPACE_SCHEMA = "mindplex-llm-history-observations-v1"
LLM_NUMERIC_FEATURES = (
    "llm_concept_affinity", "llm_recent_concept_affinity",
    "llm_concept_peak_overlap", "llm_concept_novelty",
    "llm_format_affinity", "llm_event_affinity",
    "llm_intent_affinity", "llm_audience_affinity",
)
LLM_WORKSPACE_FEATURES = (*LLM_NUMERIC_FEATURES, "llm_history_coverage", "llm_format")


def _identifier(value):
    return value.get("id") if isinstance(value, Mapping) else value


def _label(value):
    if not isinstance(value, str) or not value.strip():
        return None
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _values(record, field):
    if not isinstance(record, Mapping):
        return set()
    raw = record.get(field)
    if field == "format":
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return set()
    return {label for item in raw if (label := _label(item)) is not None}


def build_llm_workspace_facts(candidate_id, ordered_history_ids, annotations):
    """Describe candidate-to-history annotation matches, oldest to newest.

    Multi-value affinity is the average fraction of candidate values present
    in each annotated history item. Repeated reads retain multiplicity. A
    history item without values for that particular field is excluded from its
    denominator. Recent history means the last five *original* positions, not
    the five latest successfully annotated items. Novelty is the fraction of
    candidate concepts absent from the union of annotated history concepts.
    """
    if not isinstance(annotations, Mapping):
        raise ValueError("article annotations must be a mapping")

    def resolve(identifier):
        try:
            record = annotations.get(_identifier(identifier))
        except TypeError:
            return None
        return record if isinstance(record, Mapping) else None

    candidate = resolve(candidate_id)
    history = [resolve(item) for item in ordered_history_ids]
    facts = {feature: None for feature in LLM_WORKSPACE_FEATURES}
    facts["llm_history_coverage"] = (
        sum(item is not None for item in history) / len(history) if history else 0.0
    )
    formats = sorted(_values(candidate, "format"))
    facts["llm_format"] = formats[0] if formats else None

    def overlaps(field, prior):
        values = _values(candidate, field)
        if not values:
            return []
        return [len(values & other) / len(values) for record in prior
                if (other := _values(record, field))]

    for field, feature in (
        ("concepts", "llm_concept_affinity"),
        ("format", "llm_format_affinity"),
        ("event_types", "llm_event_affinity"),
        ("intents", "llm_intent_affinity"),
        ("audiences", "llm_audience_affinity"),
    ):
        observations = overlaps(field, history)
        if observations:
            facts[feature] = math.fsum(observations) / len(observations)
    concepts = overlaps("concepts", history)
    if concepts:
        facts["llm_concept_peak_overlap"] = max(concepts)
        candidate_concepts = _values(candidate, "concepts")
        seen = set().union(*(_values(record, "concepts") for record in history))
        facts["llm_concept_novelty"] = len(candidate_concepts - seen) / len(candidate_concepts)
    recent = overlaps("concepts", history[-5:])
    if recent:
        facts["llm_recent_concept_affinity"] = math.fsum(recent) / len(recent)
    return {key: round(value, 8) if isinstance(value, float) else value
            for key, value in facts.items()}
