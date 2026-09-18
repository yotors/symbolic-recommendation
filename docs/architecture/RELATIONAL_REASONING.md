# Bounded relational reasoning

## Purpose and boundary

This extension tests whether an explicit, proof-derived relationship between a
candidate and a user's preceding positive history improves the existing
symbolic ranker. It deliberately leaves the selected `balanced_rank`
aggregation, candidate slates, evaluation labels, and ordinary feature
families unchanged. `balanced_rank` is an ordinal evidence-family combiner;
it is selected for ranking accuracy and is not presented as a calibrated click
probability.

The implementation has two separate kinds of rules:

- fixed structural rules establish that a candidate is related to a known
  history item;
- fpMiner-discovered behavioral rules learn whether that derived relation,
  alone or in conjunction with other causal features, predicts engagement or
  pair preference.

Consequently, a relation proof is evidence available to the learned model. It
does not assert that the user will click.

## Relational observations

Two bounded relation families are materialized.

### Exact entity continuity

An article's stable entity set is read from grounded Wikidata identifiers in
its title and abstract metadata. Human-readable labels are not substituted for
identifiers. A missing or malformed annotation is `unknown`; an explicit,
valid empty annotation is known evidence that contains no usable entity.

For a user `u`, candidate `a`, preceding positive-history article `h`, entity
`e`, history scope `s`, case `c`, and interaction origin `o`, the source facts
have this shape:

```metta
(: fact_history_entity
   (RelMentionsEntity history_article entity_q42)
   (STV 1.0 1.0))

(: fact_candidate_entity
   (RelMentionsEntity candidate_article entity_q42)
   (STV 1.0 1.0))

(: fact_observed_click
   (RelObservedClick scope origin user history_article)
   (STV 1.0 1.0))

(: fact_candidate
   (RelCaseCandidate case scope user candidate_article)
   (STV 1.0 1.0))
```

PeTTaChainer applies two structural implications:

```metta
RelObservedClick(s,o,u,h) AND RelMentionsEntity(h,e)
    -> RelEngagedEntityOrigin(s,u,e,o)

RelCaseCandidate(c,s,u,a) AND RelEngagedEntityOrigin(s,u,e,o)
    AND RelMentionsEntity(a,e)
    -> RelEntityContinuity(c,o,e)
```

### Canonical concept continuity

The second family uses exact equality of canonical concept IDs from the frozen
content-annotation workspace. A concept is accepted only when its canonical
mapping and named source statement are present in the provenance record. A
free text label without that anchor is unavailable evidence.

The corresponding source facts and implications are:

```metta
(: original_annotation_anchor
   (HasConcept "history article text" "source concept label")
   (STV 1.0 1.0))

(: canonical_mapping
   (RelCanonicalConceptMapping
      "history article text" "source concept label"
      history_article concept_x)
   (STV 1.0 1.0))

HasConcept(source,label)
    AND RelCanonicalConceptMapping(source,label,article,x)
    -> RelHasCanonicalConcept(article,x)

RelConceptObservedClick(s,o,u,h) AND RelHasCanonicalConcept(h,x)
    -> RelEngagedConceptOrigin(s,u,x,o)

RelConceptCaseCandidate(c,s,u,a)
    AND RelEngagedConceptOrigin(s,u,x,o)
    AND RelHasCanonicalConcept(a,x)
    -> RelConceptContinuity(c,o,x)
```

Source facts and structural implications use certain STVs because they encode
observed identity and deterministic composition, not engagement probability.
The later mined behavioral rule receives an evidence-derived CTV estimated
from engagement cases.

No event-type equality is treated as story continuity. A genuine story-update
relation requires a stable event/story identifier from content ingestion;
MIND does not provide one with the required semantics.

## Causal projection and provenance

Offline projection uses the ordered history stored on each historical row. It
is the history available before that impression and before its outcome. The
current candidate's label is retained for later behavioral training, but it is
never supplied to relational planning or PeTTaChainer. Candidates in the same
impression therefore use the same pre-outcome history snapshot.

Every positive proof must contain:

- exactly one requested conclusion root;
- both entity structural rule IDs, or all three concept structural rule IDs;
- the candidate and observed-history facts;
- matching candidate/history entity or concept facts; and
- an origin that belongs to the supplied causal history.

The reducer maps validated proofs to one categorical observation per family:

| Value | Meaning |
| --- | --- |
| `recent` | At least one proven origin is among the final five positions of the original history. |
| `older` | At least one origin is proven, but none is recent. |
| `none` | All required annotations are known and no origin is proven. |
| `unknown` | Required annotation evidence is incomplete, so absence cannot be concluded. |

The proof root includes the matched entity or concept ID. This prevents the
reasoner's evidence revision from collapsing distinct matching values before
validation. Every expected `(origin, value)` root must return at least one
witnessed proof. Alternative paths that PeTTaChainer returns for a root are
retained with their hashes, but the bounded query is not claimed to enumerate
every possible proof path. All returned paths from one history occurrence are
then deliberately collapsed to one categorical origin contribution.

Entity and concept records derived from the same interaction share a
dependency key, preserving their common source for provenance and future
source-aware fusion. The fixed challenger activates only canonical-concept
continuity, so an entity and a concept proof from one click cannot receive two
top-level relational-family votes in this experiment. Proof IDs remain trace
references rather than independent statistical samples: CTV evidence is
counted over historical engagement cases, never over the number of proof
paths.

Each retained origin has a ledger record containing the user, candidate,
history article and position, matched stable IDs, source fact IDs, structural
rule IDs, representative proof, and proof hash. The immutable projection also
hashes its source observations, structural rules, AtomSpace statements,
projected contexts, and complete ledger. Validation fails closed for missing
dependencies, duplicate references, future-history references, label-bearing
provenance, or hash disagreement.

## Mining, proof execution, and ranking

The relational projector computes article observations once, reuses each
`(user, candidate, ordered history)` plan, inserts the union of named source
facts and structural rules, and submits only queryable cases in batches. A
host-side identifier intersection is only a query-planning optimization: a
positive feature is emitted only when PeTTaChainer returns the complete
two-implication entity proof or three-implication anchored-concept proof.

The resulting candidate fields are
`rel_entity_continuity_scope` and
`rel_concept_continuity_scope`. In the fixed challenger profile, canonical
concept continuity is added to the point and pair feature sets; exact entity
continuity remains projected and provenance-audited for separate controlled
experiments. Candidate values are computed once. The general pair encoder then
derives `left`, `right`, `equal`, or known/unknown comparison states, avoiding
a second relational proof for every candidate pair.

Proof multiplicity is provenance, not scoring multiplicity. All concept paths
from one historical interaction are collapsed into that candidate's single
`none`/`older`/`recent` observation before mining, and proof IDs are never
miner predicates. Arms A and B activate neither relational field; arm C
activates canonical-concept continuity only. Thus the selected A/B/C design
cannot give an entity proof and a concept proof from the same click two
independent votes. Other profiles are also prevented from activating the two
relational families together until final point and pair fusion can consume
their shared interaction-origin dependency explicitly.

fpMiner receives closed historical cases containing the selected ordinary and
relational categorical features plus the engagement target. It discovers the
behavioral implications and conjunctions supported by training cases. The
full training population then supplies the positive/negative evidence used to
estimate their CTVs. PeTTaChainer loads the compiled point rules, pair rules,
and fixed structural rules in one isolated scorer worker and produces the
proof-backed signals consumed by the unchanged ranking pipeline.

Responsibility is therefore explicit:

| Component | Responsibility |
| --- | --- |
| Content ingestion | Versioned entity/concept observations and their anchors. |
| PeTTaChainer | Bounded structural composition and serving-time rule proofs. |
| fpMiner | Discovery of behavioral associations involving direct or derived evidence. |
| CTV estimation | Strength and confidence from historical positive/negative evidence. |
| Ranking | Existing dependency/family-aware point and pair ordering. |

## Live update and cache contract

Historical replay fields, including `unknown`, are authoritative and are never
recomputed from a later user profile. For a live feed, the active scorer uses
at most the latest 50 known positive-history article IDs and builds versioned
entity and concept plans for all candidates in one batch.

The cache key contains the relation family and case ID. The case ID commits to
the user, candidate observation version, and complete ordered history version.
Therefore:

- a click, like, or completion extends positive history and naturally creates
  new scopes and cache keys before the remaining queue is reranked;
- a skip affects the separate transient negative-feedback proof path, but does
  not claim positive entity/concept continuity;
- changed article annotations create a new article atom and case identity; and
- atomic model promotion installs a fresh scorer worker and clears loaded
  relation statements and derived-feature caches. Proof-ledger records still
  referenced by an event or a card eligible for feedback are retained; all
  unreferenced live records are discarded.

Only previously unseen named statements are inserted. Queries are batched and
bounded; an expected path that produces no PeTTaChainer proof is an error, not
a synthetic `none`. `unknown` abstains from ranking rather than becoming an
annotation-coverage signal. Cache size, history length, query batches, and
reasoning steps are bounded.

An interaction reranks the unserved portion of that feed session immediately
using updated private user facts. It does not synchronously rediscover global
behavioral rules. Events accumulate until the configured event threshold,
then a background worker mines and compiles an immutable snapshot while the
old scorer continues serving. Promotion is atomic; a stale or failed build is
discarded. The current implementation uses the event-count trigger. A wall
clock trigger can be added as an operational policy without changing the
workspace contract.

## Fixed three-arm evaluation

The ablation runs three fresh processes on identical enriched dataset bytes,
logged candidate slates, labels, seeds, aggregation, and selected base
configuration. Every arm starts with a cold proof cache.

| Arm | Relation mode | Purpose |
| --- | --- | --- |
| A — existing evidence | `disabled` | Reference ranking with the existing point and pair profiles. |
| B — configuration-inertness control | `facts_only` | Change only the mode marker while filtering relation fields through the same profiles; this checks fresh-process determinism, not added knowledge. |
| C — bounded chained conclusion | `chained` | Add canonical-concept continuity to the point and pair profiles, then mine and prove the resulting behavioral rules. |

A and B are required to have byte-identical rule hashes, score records, and
ranked outputs. This is a configuration-inertness and fresh-process
reproducibility check; it is not presented as an additional-knowledge arm. C
may differ only in activation mode and the declared point/pair profiles.
Comparisons use paired impression and user-cluster bootstrap intervals and
retain rule, cohort, ranking, provenance, timing, CPU, and memory audits.

The C-minus-A comparison measures the complete relational
projection/mining/reasoning addition. It does not by itself isolate a unique
accuracy contribution of PeTTaChainer from the added knowledge or behavioral
feature. Numerical results belong to the generated experiment artifact and
are not asserted by this architecture specification.
