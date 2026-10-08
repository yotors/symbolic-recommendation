"""Mining, model compilation and atomic promotion methods."""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import random
import threading
import time
from collections import Counter, OrderedDict

from ..app.async_mining import MiningSnapshot
from ..app.reasoner import IsolatedPeTTaChainer
from ..core.ctv_calibration import (
    DEFAULT_EVIDENCE_K as CTV_EVIDENCE_K_DEFAULT,
    CTVObservation, calibrate_ctv, reencode_ctv_confidence,
)
from ..core.symbolic import QuantileNumericEvidence
from ..features.llm_workspace import LLM_NUMERIC_FEATURES
from .conditional_llm_mining import (
    ConditionalMiningConfig, FpMinerUnary, mine_conditional_llm_patterns,
)
from .retention import retain_complete_units
from .rule_parser import (
    POSITIVE, parse_rules,
)


def make_model_lifecycle_mixin(
    *,
    BACKGROUND_MINING_BUILD_MODE,
    BACKGROUND_MODEL_FIELDS,
    CONDITIONAL_LLM_CONTEXT_PREDICATES,
    FEATURE_PROFILES,
    INTERACTION_PAIRS,
    INTERACTION_TRIPLES,
    LIVE_NEGATIVE_RULE_SOURCES,
    LLM_PAIR_PREDICATES,
    MINER_LOCK,
    MINING_WORKSPACE_MODE,
    MINING_WORKSPACE_SCHEMA_VERSION,
    NUMERIC_PAIR_EVIDENCE,
    PAIR_CATEGORICAL_SIDE_FAMILIES,
    PAIR_CATEGORICAL_SIDE_PREDICATES,
    PAIR_EVIDENCE_ALIASES,
    PAIR_FEATURES,
    PAIR_FEATURE_PROFILES,
    PAIR_INTERACTIONS,
    PAIR_ORDERS,
    PAIR_REDUNDANT_INTEREST,
    PAIR_STABLE_DOMINANCE_SOURCES,
    RELATIONAL_PROOF_FIELDS,
    SERVING_MODEL_SCHEMA,
    _pair_feature_execution_plan,
):
    class ModelLifecycleMixin:
        @staticmethod
        def _mining_workspace_plan(kind,selected_features,depth,case_policy):
            """Return the stable identity of one exact AtomSpace projection."""
            return json.dumps({
                "schema":MINING_WORKSPACE_SCHEMA_VERSION,
                "kind":str(kind),
                "features":list(selected_features),
                "depth":int(depth),
                "case_policy":str(case_policy),
            },sort_keys=True,separators=(",",":"),ensure_ascii=True)

        def _retained_mining_population(self):
            """Freeze the newest complete causal units used by every miner stage.

            A source impression is atomic.  Records without an impression are
            already closed singleton interactions and receive a stable event unit.
            The returned source indices remain unchanged when the window appends,
            which preserves exact delta case identities until a unit expires.
            """
            records=[]
            for event_index,event in enumerate(self.data.get("events",[])):
                impression=event.get("impression")
                unit=(f"impression:{impression}" if impression
                      else f"event:{event_index}")
                records.append((unit,(event_index,event)))
            return retain_complete_units(
                records,
                max_units=int(self.config["mining_retention_max_units"]),
                max_cases=int(self.config["mining_retention_max_cases"]),
            )

        def _prune_mining_workspace_cache(self):
            """Best-effort cleanup after a complete rule snapshot is usable."""
            try:
                with MINER_LOCK:
                    removed=self._mining_workspaces.prune(
                        keep=self._active_mining_workspace_plans
                    )
                    support_removed=self._incremental_fpminer.prune(
                        keep=self._active_mining_workspace_plans
                    )
                return {"removed":list(removed),
                        "support_removed":list(support_removed),"error":None}
            except Exception as exc:
                # Cache maintenance must never roll back a scorer that compiled
                # successfully. A failed clear is marked dirty by the cache and the
                # next synchronization will rebuild that space before trusting it.
                return {
                    "removed":[],
                    "support_removed":[],
                    "error":f"{type(exc).__name__}: {exc}"[:1000],
                }

        def _mine_incremental_workspace(
            self, synced, *, plan, cases, features, depth, min_support,
            retention_units, retention_audit, workspace_kind,
        ):
            """Mine one synced space and release a rejected non-active stage."""
            try:
                return self._incremental_fpminer.mine(
                    plan=plan,full_space=synced.space,cases=cases,
                    features=features,depth=depth,min_support=min_support,
                    evidence_k=float(self.config["ctv_evidence_k"]),
                    retention_units=retention_units,
                    retention_audit=retention_audit,
                )
            except BaseException as exc:
                # Active spaces remain aligned with the last usable scorer and can
                # safely retry a failed transactional support update. A new or
                # otherwise non-active stage has no usable support snapshot, so it
                # must not consume a workspace-cache slot after rejection.
                if plan not in self._active_mining_workspace_plans:
                    try:
                        self._mining_workspaces.rollback(synced)
                    except BaseException as rollback_exc:
                        if hasattr(exc,"add_note"):
                            exc.add_note(
                                f"failed to roll back rejected {workspace_kind} "
                                "mining workspace: "
                                f"{type(rollback_exc).__name__}: {rollback_exc}"
                            )
                raise

        def _mining_event_cases(self,indexed_events=None):
            """Sample point cases while retaining their append-stable source IDs."""
            indexed=(list(indexed_events) if indexed_events is not None else
                     list(enumerate(self.data["events"])))
            ratio=int(self.config["negative_ratio"])
            if ratio<=0 or not any(event.get("impression")
                                   for _index,event in indexed):
                return [(f"point_case_{index}",event) for index,event in indexed]
            groups={}
            for index,event in indexed:
                key=event.get("impression") or f'live_{event.get("user","unknown")}'
                groups.setdefault(str(key),[]).append((index,event))
            selected=set()
            for key,items in groups.items():
                positives=[item for item in items if item[1].get("action") in POSITIVE]
                negatives=[item for item in items if item[1].get("action") not in POSITIVE]
                selected.update(index for index,_event in positives)
                if not positives:
                    # Logged MIND impressions without a click remain excluded by
                    # the historical click-conditioned sampler.  A newly observed
                    # live impression is different: dropping its only skip would
                    # make the online outcome visible to calibration but invisible
                    # to fpMiner discovery.  Retain the first bounded negatives so
                    # later appends cannot replace an earlier selected case and
                    # force the incremental workspace to rebuild.
                    is_online_live=(key.startswith("live_") and any(
                        index>=self._offline_event_count for index,_event in items
                    ))
                    if is_online_live and negatives:
                        online_negatives=[
                            item for item in negatives
                            if item[0]>=self._offline_event_count
                        ]
                        selected.update(
                            index for index,_event in online_negatives[:ratio]
                        )
                    continue
                if not negatives: continue
                count=min(len(negatives),ratio*len(positives))
                stable=f'{self.config["random_seed"]}:{key}'.encode()
                local=random.Random(int.from_bytes(hashlib.sha256(stable).digest()[:8],"big"))
                selected.update(index for index,_event in local.sample(negatives,count))
            return [(f"point_case_{index}",event)
                    for index,event in indexed if index in selected]

        def _online_negative_sampling_audit(self,training_cases,indexed_events=None):
            """Describe bounded live-negative retention in a mining snapshot."""
            retained_indices={
                int(case_id.rsplit("_",1)[-1]) for case_id,_event in training_cases
            }
            groups={}
            indexed=(list(indexed_events) if indexed_events is not None else
                     list(enumerate(self.data["events"])))
            for index,event in indexed:
                key=event.get("impression") or f'live_{event.get("user","unknown")}'
                groups.setdefault(str(key),[]).append((index,event))
            eligible=set()
            for key,items in groups.items():
                if (not key.startswith("live_")
                        or not any(index>=self._offline_event_count
                                   for index,_event in items)
                        or any(event.get("action") in POSITIVE
                               for _index,event in items)):
                    continue
                eligible.update(
                    index for index,event in items
                    if (index>=self._offline_event_count
                        and event.get("action") not in POSITIVE)
                )
            return {
                "policy":"first_observed_append_stable",
                "cap_per_all_negative_live_impression":int(
                    self.config["negative_ratio"]
                ),
                "eligible":len(eligible),
                "retained":len(eligible & retained_indices),
            }

        @staticmethod
        def _relative_value(left, right, order):
            """Encode a comparison without exposing either training outcome."""
            left=str(left or "unknown"); right=str(right or "unknown")
            if left==right: return "equal"
            ranks={value:index for index,value in enumerate(order)}
            if left not in ranks and right not in ranks: return "incomparable"
            if left not in ranks: return "right_known"
            if right not in ranks: return "left_known"
            return "left" if ranks[left]>ranks[right] else "right"

        @staticmethod
        def _relative_numeric_value(left, right):
            """Bound a numeric comparison to six stable symbolic outcomes."""
            def finite(value):
                try:
                    number=float(value)
                except (TypeError,ValueError):
                    return None
                return number if math.isfinite(number) else None
            left_number=finite(left); right_number=finite(right)
            if left_number is None and right_number is None: return "unknown"
            if left_number is None: return "right_known"
            if right_number is None: return "left_known"
            if math.isclose(left_number,right_number,rel_tol=1e-12,abs_tol=1e-12):
                return "equal"
            return "left" if left_number>right_number else "right"

        @staticmethod
        def _numeric_missing(value):
            """Return whether numeric observation is absent or non-finite."""
            if value is None or isinstance(value,bool): return True
            try: number=float(value)
            except (TypeError,ValueError,OverflowError): return True
            return not math.isfinite(number)

        def _pair_features(self,left_attrs,right_attrs,left_article,right_article,
                           needed=None):
            """Project one ordered candidate pair into champion predicates."""
            needed=set(needed) if needed is not None else None
            wants=lambda predicate: needed is None or predicate in needed
            plan=_pair_feature_execution_plan(
                None if needed is None else tuple(sorted(needed))
            )
            attrs={}
            left_scope=left_attrs.get("history_size_bucket","unknown")
            right_scope=right_attrs.get("history_size_bucket","unknown")
            # This is a context shared by both pair orientations, not an identity
            # or outcome. Unary support is neutral; only a mined conjunction can
            # establish that a preference depends on the amount of prior history.
            if wants("pair_history_scope"):
                attrs["pair_history_scope"]=(
                    left_scope if left_scope==right_scope else "unknown"
                )
            for output,source in plan["ordered"]:
                if source in left_attrs or source in right_attrs:
                    attrs[output]=self._relative_value(
                        left_attrs.get(source),right_attrs.get(source),PAIR_ORDERS[source]
                    )
            for output,source in plan["direct_numeric"]:
                attrs[output]=self._relative_numeric_value(
                    left_attrs.get(source),right_attrs.get(source)
                )
            for output,source in plan["llm"]:
                if (self._numeric_missing(left_attrs.get(source))
                        or self._numeric_missing(right_attrs.get(source))):
                    attrs[output]="incomparable"
                    continue
                attrs[output]=self._relative_numeric_value(
                    left_attrs.get(source),right_attrs.get(source)
                )
            for source,predicate in plan["quantile"]:
                if (self._numeric_missing(left_attrs.get(source))
                        or self._numeric_missing(right_attrs.get(source))):
                    # Annotation absence is not directional evidence. This is
                    # stricter than the generic encoder's useful left_known /
                    # right_known states because LLM coverage may be budget- or
                    # provider-dependent and must never decide the ranking.
                    attrs[predicate]="incomparable"
                    continue
                encoder=self._numeric_pair_encoders.get(source)
                if encoder is not None:
                    attrs[predicate]=encoder.encode_pair(
                        left_attrs.get(source),right_attrs.get(source)
                    )
            left_topic=str(left_article.get("topic","unknown"))
            right_topic=str(right_article.get("topic","unknown"))
            left_subcategory=str(left_article.get("subcategory","unknown"))
            right_subcategory=str(right_article.get("subcategory","unknown"))
            if wants("pair_same_topic"):
                attrs["pair_same_topic"]=(
                    "same" if left_topic==right_topic else "different"
                )
            if wants("pair_same_subcategory"):
                attrs["pair_same_subcategory"]=(
                    "same" if left_subcategory==right_subcategory else "different"
                )
            return attrs

        def _bounded_pair_features(self,attrs):
            bounded={}
            for predicate,value in attrs.items():
                allowed=self._pair_feature_vocabulary.get(predicate)
                if allowed:
                    if value in allowed:
                        bounded[predicate]=value
                    else:
                        bounded[predicate]="other"
            return bounded

        def _fit_pair_numeric_encoders(self,indexed_events=None):
            """Freeze dataset-scale-free numeric symbols from training facts only."""
            active_profile=set(PAIR_FEATURE_PROFILES[
                self.config["pair_feature_profile"]
            ])
            enabled={
                source:predicate for source,predicate in NUMERIC_PAIR_EVIDENCE.items()
                if predicate in active_profile
            }
            if not enabled:
                self._numeric_pair_encoders={}
                return
            groups={}
            indexed=(list(indexed_events) if indexed_events is not None else
                     list(enumerate(self.data.get("events",[]))))
            for _event_index,event in indexed:
                impression=event.get("impression")
                if impression: groups.setdefault(str(impression),[]).append(event)
            values={source:[] for source in enabled}
            pairs={source:[] for source in enabled}
            pair_weights={source:[] for source in enabled}

            def finite(value):
                if value is None or isinstance(value,bool): return None
                try: number=float(value)
                except (TypeError,ValueError,OverflowError): return None
                return number if math.isfinite(number) else None

            for events in groups.values():
                contextual=[(event,self.event_features(event)) for event in events]
                for _event,attrs in contextual:
                    for source in values: values[source].append(attrs.get(source))
                # Magnitude bins are a label-free representation choice: use all
                # unordered candidates in the logged slate, never click-vs-skip
                # membership. Per-feature, each impression contributes total
                # mass one, preventing large slates from defining every cut.
                for source in pairs:
                    candidates=[finite(attrs.get(source)) for _event,attrs in contextual]
                    complete=[
                        (left,right)
                        for left_index,left in enumerate(candidates)
                        if left is not None
                        for right in candidates[left_index+1:]
                        if right is not None and left != right
                    ]
                    if not complete: continue
                    weight=1.0/len(complete)
                    pairs[source].extend(complete)
                    pair_weights[source].extend([weight]*len(complete))
            bins=int(self.config["pair_numeric_bins"])
            self._numeric_pair_encoders={
                source:QuantileNumericEvidence.fit(
                    source,values[source],pairs[source],
                    training_pair_weights=pair_weights[source],bins=bins
                )
                for source in enabled
            }

        @staticmethod
        def _pair_mining_case_id(impression,left_event_index,right_event_index):
            """Name an oriented pair by its stable source events, not batch order."""
            signature=json.dumps({
                "schema":MINING_WORKSPACE_SCHEMA_VERSION,
                "impression":str(impression),
                "left_event_index":int(left_event_index),
                "right_event_index":int(right_event_index),
            },sort_keys=True,separators=(",",":"),ensure_ascii=True)
            digest=hashlib.sha256(signature.encode("utf-8")).hexdigest()[:32]
            return f"pair_case_{digest}"

        def _pair_training_cases(self,negative_ratio=None,indexed_events=None):
            """Create orientation-balanced clicked-vs-exposed pair cases.

            ``search_weight`` gives every closed impression total mass one.  This
            aligns target-aware discovery with the benchmark's macro-per-impression
            AUC even when impressions have very different slate sizes.  The legacy
            fpMiner path consumes raw cases.  Final CTV estimation always revisits
            the complete retained pair population; ``pair_ctv_mode`` determines
            whether those rows are counted directly or normalized by impression.
            """
            groups={}
            indexed=(list(indexed_events) if indexed_events is not None else
                     list(enumerate(self.data.get("events",[]))))
            for event_index,event in indexed:
                impression=event.get("impression")
                if impression:
                    groups.setdefault(str(impression),[]).append((event_index,event))
            ratio=int(self.config["pair_negative_ratio"] if negative_ratio is None else negative_ratio)
            cases=[]; group_count=max(1,len(groups))
            for group_index,(impression,events) in enumerate(groups.items()):
                positives=[item for item in events if item[1].get("action") in POSITIVE]
                negatives=[item for item in events if item[1].get("action") not in POSITIVE]
                if not positives or not negatives: continue
                impression_cases=[]
                for positive_index,(positive_source,positive) in enumerate(positives):
                    selected=negatives
                    if ratio>0 and len(selected)>ratio:
                        stable=f'{self.config["random_seed"]}:{impression}:{positive_index}'.encode()
                        local=random.Random(int.from_bytes(hashlib.sha256(stable).digest()[:8],"big"))
                        selected=local.sample(selected,ratio)
                    for negative_source,negative in selected:
                        # Emit both orientations. This makes the pair target exactly
                        # balanced, prevents a random left/right artifact, and gives
                        # every learned predicate an explicit anti-symmetric view.
                        for left_source,left,right_source,right,left_wins in (
                            (positive_source,positive,negative_source,negative,True),
                            (negative_source,negative,positive_source,positive,False),
                        ):
                            left_article=self.article(left["article"])
                            right_article=self.article(right["article"])
                            attrs=self._pair_features(
                                self.event_features(left),self.event_features(right),
                                left_article,right_article
                            )
                            impression_cases.append({
                                "case_id":self._pair_mining_case_id(
                                    impression,left_source,right_source
                                ),
                                "impression":impression,"attrs":attrs,
                                "positive":left_wins,
                                "temporal_fold":min(2,3*group_index//group_count),
                            })
                search_weight=1.0/len(impression_cases)
                for case in impression_cases:
                    case["search_weight"]=search_weight
                cases.extend(impression_cases)
            return cases

        @staticmethod
        def _pair_evidence_lineage(premises):
            """Canonical raw-evidence lineage for proof dependency grouping."""
            def canonical(predicate):
                seen=set()
                while predicate in PAIR_EVIDENCE_ALIASES and predicate not in seen:
                    seen.add(predicate)
                    predicate=PAIR_EVIDENCE_ALIASES[predicate]
                return predicate
            features={canonical(predicate) for predicate,_value in premises}
            return frozenset(
                "topic_interest" if feature in PAIR_REDUNDANT_INTEREST else feature
                for feature in features
            )

        @classmethod
        def _pair_dependency_lineage(cls,premises):
            """Directional evidence that may connect two proof dependencies.

            Orientation-invariant predicates can condition a useful rule, but they
            cannot identify which candidate should win. Excluding those gates here
            prevents unrelated text and taxonomy rules from becoming one dependency
            merely because both apply to the same history scope. The complete
            evidence lineage remains available to the miner above.
            """
            return cls._pair_evidence_lineage(premises).difference(
                CONDITIONAL_LLM_CONTEXT_PREDICATES
            )

        @classmethod
        def _pair_rules_share_dependency(cls,left_rule,right_rule):
            """Whether two selected rules must share one PeTTa proof dependency."""
            left_lineage=cls._pair_dependency_lineage(left_rule["premises"])
            right_lineage=cls._pair_dependency_lineage(right_rule["premises"])
            left_owner=left_rule.get("dependency_owner")
            right_owner=right_rule.get("dependency_owner")
            # An explicit owner is authoritative even when a conditional variant's
            # other premises have a distinct lineage.
            if ((left_owner and left_owner in right_lineage)
                    or (right_owner and right_owner in left_lineage)
                    or (left_owner and right_owner and left_owner==right_owner)):
                return True
            # Gate-only rules have no directional evidence and must not all collapse
            # into one dependency through empty-lineage equality.
            if not left_lineage or not right_lineage:
                return False
            if left_lineage==right_lineage:
                return True
            if not left_lineage.intersection(right_lineage):
                return False
            smaller=min(len(left_rule["coverage"]),len(right_rule["coverage"]))
            overlap=(len(left_rule["coverage"]&right_rule["coverage"])/smaller
                     if smaller else 1.0)
            return overlap>=0.9

        def _mine_pairwise(self,indexed_events=None,retention=None):
            """Mine the champion conditional pair rules and compile PeTTa proofs.

            Real fpMiner unary semantic rules seed conditional expansion with
            history scope. The seed itself is then removed so correlated text
            evidence cannot vote twice. Final CTVs use equal impression mass.
            """
            if retention is None:
                retention=self._retained_mining_population()
            if indexed_events is None:
                indexed_events=retention.records
            indexed_events=tuple(indexed_events)
            retention_audit=retention.audit.as_dict()
            started=time.perf_counter(); self._fit_pair_numeric_encoders(indexed_events)
            self._pair_categorical_labels={}
            discovery_cases=self._pair_training_cases(indexed_events=indexed_events)
            if not discovery_cases:
                self._active_pair_rule_ids=set(); self.pair_rules=[]
                self._pair_rule_sources=[]
                self.last_pair_mining={
                    "rules":0,"cases":0,
                    "reason":"no closed positive/negative impressions",
                    "retention":retention_audit,
                    "workspace_mode":MINING_WORKSPACE_MODE,
                    "workspace_sync":[],
                    "full_structure_research":False,
                }
                return self.last_pair_mining
            # Negative sampling is only a search-time optimisation for fpMiner.
            # Recalibrate every discovered structure over every within-impression
            # click-vs-skip pair, otherwise the sampled opponents make CTVs and
            # temporal-stability decisions depend on an arbitrary mining budget.
            population_cases=(discovery_cases if int(self.config["pair_negative_ratio"])==0
                              else self._pair_training_cases(
                                  negative_ratio=0,indexed_events=indexed_events
                              ))
            discovery_search_weight=math.fsum(
                case["search_weight"] for case in discovery_cases
            )
            # The two miners consume different support units. fpMiner counts raw
            # oriented rows in its MeTTa scratch space; target-aware search consumes
            # case weights whose total is one per usable impression. Never pass a
            # weighted threshold off as though fpMiner itself understood weights.
            fpminer_min_support=max(
                int(self.config["pair_min_support"]),
                math.ceil(len(discovery_cases)*0.001),
            )
            target_min_support=max(
                int(self.config["pair_min_support"]),
                math.ceil(discovery_search_weight*0.001),
            )
            vocabulary_min_support=target_min_support
            calibration_min_support=max(
                int(self.config["pair_min_support"]),math.ceil(len(population_cases)*0.001)
            )
            value_counts={feature:Counter() for feature in PAIR_FEATURES}
            for case in discovery_cases:
                for feature in PAIR_FEATURES:
                    if feature in case["attrs"]:
                        value_counts[feature][case["attrs"][feature]] += (
                            case["search_weight"]
                        )
            max_values=max(2,int(self.config["max_feature_values"])); vocabulary={}
            for feature,counts in value_counts.items():
                if feature in PAIR_CATEGORICAL_SIDE_PREDICATES:
                    continue
                ranked=[value for value,count in counts.most_common()
                        if value is not None and count>=vocabulary_min_support]
                if not ranked: continue
                keep=set(ranked[:max_values])
                if any(value not in keep for value in counts): keep.add("other")
                vocabulary[feature]=keep
            # Left/right categorical roles use one pooled, training-only
            # vocabulary. Mirrored cases therefore cannot keep one orientation
            # while mapping its reverse to a different symbol.
            for _family,(left_predicate,right_predicate) in (
                    PAIR_CATEGORICAL_SIDE_FAMILIES.items()):
                if (left_predicate not in value_counts
                        or right_predicate not in value_counts):
                    continue
                pooled=value_counts[left_predicate]+value_counts[right_predicate]
                ranked=[
                    value for value,count in pooled.most_common()
                    if value is not None and count>=2*vocabulary_min_support
                ]
                if not ranked: continue
                keep=set(ranked[:max_values])
                vocabulary[left_predicate]=set(keep)
                vocabulary[right_predicate]=set(keep)
            self._pair_feature_vocabulary=vocabulary
            bounded_discovery=[{**case,"attrs":self._bounded_pair_features(case["attrs"])}
                               for case in discovery_cases]
            bounded_population=[{**case,"attrs":self._bounded_pair_features(case["attrs"])}
                                for case in population_cases]
            predicate_comparability={}
            for feature in PAIR_FEATURES:
                comparable=sum(
                    str(case["attrs"].get(feature,"")).lower() in {"left","right"}
                    or str(case["attrs"].get(feature,"")).lower().startswith(("left_q","right_q"))
                    for case in bounded_population
                )
                if comparable:
                    predicate_comparability[feature]=comparable/len(bounded_population)

            workspace_sync=[]; workspace_plans=set()
            def sync_space(selected_features,depth,cases,case_policy):
                workspace_cases={}
                for case in cases:
                    case_id=case["case_id"]
                    facts=[]
                    for predicate in selected_features:
                        if predicate in case["attrs"]:
                            value=case["attrs"][predicate]
                            if (predicate in LLM_PAIR_PREDICATES
                                    and str(value).lower() in {
                                        "unknown","incomparable",
                                        "left_known","right_known",
                                    }):
                                # Annotation availability is not a preference and
                                # cannot become a real-fpMiner discovery seed.
                                continue
                            facts.append(
                                f'({predicate} {case_id} '
                                f'{json.dumps(case["attrs"][predicate])})'
                            )
                    outcome="click" if case["positive"] else "skip"
                    facts.append(f'(engagement {case_id} {json.dumps(outcome)})')
                    if case_id in workspace_cases:
                        raise RuntimeError(f"duplicate stable pair case ID: {case_id}")
                    workspace_cases[case_id]=tuple(facts)
                plan_key=self._mining_workspace_plan(
                    "pair",selected_features,depth,case_policy
                )
                synced=self._mining_workspaces.sync(plan_key,workspace_cases)
                workspace_plans.add(plan_key); workspace_sync.append(synced.as_dict())
                return synced,plan_key,workspace_cases

            active=[feature for feature in PAIR_FEATURE_PROFILES[
                self.config["pair_feature_profile"]
            ] if feature in vocabulary]
            # The real MeTTa miner always owns unary discovery.  Only the legacy
            # strategy enumerates the curated deeper feature combinations.
            unary_active=tuple(
                feature for feature in active
                if feature not in PAIR_CATEGORICAL_SIDE_PREDICATES
            )
            plan=[(unary_active,2,bounded_discovery,"sampled_discovery")]
            if not unary_active: plan=[]
            scoped_categorical_plans=[]
            if (int(self.config["pair_conjunctions"])>=3
                    and "pair_history_scope" in active):
                scoped_categorical_plans=[
                    (predicate,"pair_history_scope")
                    for predicate in sorted(PAIR_CATEGORICAL_SIDE_PREDICATES)
                    if predicate in active
                ]
                # Accuracy-first categorical discovery sees the exhaustive closed
                # pair population. Opponent sampling may accelerate unrelated
                # semantic discovery, but it must not erase a global cold-start
                # category before impression-macro validation can examine it.
                plan.extend((features,3,bounded_population,"full_population")
                            for features in scoped_categorical_plans)
            raw=[]; fpminer_incremental=[]
            for selected_features,depth,cases,case_policy in plan:
                synced,plan_key,workspace_cases=sync_space(
                    selected_features,depth,cases,case_policy
                )
                mined=self._mine_incremental_workspace(
                    synced,plan=plan_key,cases=workspace_cases,
                    features=selected_features,depth=depth,
                    min_support=fpminer_min_support,
                    retention_units=retention.unit_ids,
                    retention_audit=retention_audit,workspace_kind="pair",
                )
                raw.extend(mined.output)
                fpminer_incremental.append(mined.audit.as_dict())
            rules=parse_rules([str(value) for value in raw],features=PAIR_FEATURES)
            categorical_emitted=sum(
                any(predicate in PAIR_CATEGORICAL_SIDE_PREDICATES
                    for predicate,_value in rule["premises"])
                for rule in rules
            )
            filtered_rules=[]; categorical_eligible=[]
            for rule in rules:
                side_premises=[
                    (predicate,value) for predicate,value in rule["premises"]
                    if predicate in PAIR_CATEGORICAL_SIDE_PREDICATES
                ]
                if not side_premises:
                    filtered_rules.append(rule); continue
                scopes=[value for predicate,value in rule["premises"]
                        if predicate=="pair_history_scope"]
                # Categorical priors are allowed only as actual fpMiner-emitted,
                # scope-conditioned rules. Side unaries and cross-family joins
                # are discovery artifacts, not serving rules.
                if (len(rule["premises"])!=2 or len(side_premises)!=1
                        or len(scopes)!=1
                        or str(scopes[0]).lower() in {"unknown","incomparable"}):
                    continue
                predicate,_value=side_premises[0]
                family=next(
                    name for name,predicates in PAIR_CATEGORICAL_SIDE_FAMILIES.items()
                    if predicate in predicates
                )
                rule.update(
                    categorical_fact_family=family,
                    scoped_categorical_prior=True,
                    evidence_relationship="dependent_variant_not_independent_vote",
                    dependency_owner=(
                        "pair_text_semantic_top3_mean"
                        if family=="llm_format" else
                        "pair_subcategory_affinity"
                        if family=="subcategory" else "topic_interest"
                    ),
                )
                categorical_eligible.append(rule)
                filtered_rules.append(rule)
            rules=filtered_rules
            categorical_search={
                "actual_miner":"recommendation/miner/fpMiner.metta",
                "plans":[list(plan) for plan in scoped_categorical_plans],
                "plan_population":"full exhaustive closed pair population",
                "miner_depth":3,
                "emitted_with_side_predicate":categorical_emitted,
                "eligible_scoped_rules":len(categorical_eligible),
                "eligible_scoped_rule_records":[{
                    "premises":[list(item) for item in rule["premises"]],
                    "source":rule["source"],
                    "discovery_ctv":rule.get("discovery_ctv"),
                    "support":rule.get("support"),
                } for rule in categorical_eligible],
                "rejected_side_rules":categorical_emitted-len(categorical_eligible),
                "scope_policy":"all known causal history-size buckets",
                "vocabulary":"shared train-only left/right; serving OOV abstains",
                "host_generated_rules":0,
            }
            target_search=None
            semantic=tuple(feature for feature in active
                           if feature in LLM_PAIR_PREDICATES)
            context=tuple(feature for feature in active
                          if feature in CONDITIONAL_LLM_CONTEXT_PREDICATES)
            semantic_seeds=[
                FpMinerUnary(
                    rule_id=rule["id"],predicate=rule["premises"][0][0],
                    value=rule["premises"][0][1],target=rule["target"],
                    source=rule["source"],
                )
                for rule in rules
                if len(rule["premises"])==1
                and rule["premises"][0][0] in semantic
            ]
            if semantic and context and semantic_seeds:
                # Search the complete population with equal impression mass.
                # fpMiner's real unary rules are mandatory provenance seeds;
                # conditional children must improve them in stable time folds.
                conditional_result=mine_conditional_llm_patterns(
                    ({feature:case["attrs"][feature]
                      for feature in (*semantic,*context)
                      if feature in case["attrs"]
                      and str(case["attrs"][feature]).lower() not in {
                          "unknown","incomparable","left_known","right_known"
                      }}
                     for case in bounded_population),
                    (case["positive"] for case in bounded_population),
                    weights=(case["search_weight"] for case in bounded_population),
                    folds=(case["temporal_fold"] for case in bounded_population),
                    fpminer_unaries=semantic_seeds,positive_target="click",
                    semantic_predicates=semantic,context_predicates=context,
                    predicate_lineages={
                        feature:"+".join(sorted(
                            self._pair_evidence_lineage(((feature,"value"),))
                        ))
                        for feature in (*semantic,*context)
                    },
                    config=ConditionalMiningConfig(
                        min_support=target_min_support,
                        fold_min_support=max(1.0,target_min_support/3.0),
                        max_depth=max(2,int(self.config["pair_conjunctions"])-1),
                        top_k=max(256,int(self.config["pair_max_rules"])*24),
                        min_usable_folds=2,
                        min_effect=float(self.config["pair_min_effect"]),
                        min_incremental_effect=0.0,
                        max_values_per_predicate=max_values,
                    ),
                )
                existing={(rule["premises"],rule["target"]) for rule in rules}
                for pattern in conditional_result.patterns:
                    premises=tuple(
                        (atom.predicate,atom.value) for atom in pattern.premises
                    )
                    key=(premises,"click")
                    if key in existing:
                        continue
                    rules.append({
                        "premises":premises,"target":"click",
                        "support":pattern.counts.tp,
                        "strength":pattern.precision,"confidence":0.0,
                        "conditional_incremental_effect":
                            pattern.incremental_effect,
                        "conditional_stable_incremental_effect":
                            pattern.stable_incremental_effect,
                        "conditional_incremental_wracc":
                            pattern.incremental_wracc,
                        "conditional_robust_incremental_wracc":
                            pattern.robust_incremental_wracc,
                        "conditional_evidence_lineages":
                            pattern.evidence_lineages,
                        "conditional_lineage_signature":
                            pattern.lineage_signature,
                        "conditional_variant_signature":
                            pattern.variant_signature,
                        "conditional_fpminer_seed_rule_ids":
                            pattern.fpminer_seed_rule_ids,
                        "dependency_owner":"pair_text_semantic_top3_mean",
                        "target_weighted_support":pattern.weighted.support,
                        "target_weighted_contingency":
                            pattern.weighted.as_dict(),
                        "target_count_contingency":pattern.counts.as_dict(),
                        "target_fold_statistics":[
                            fold.as_dict()
                            for fold in pattern.fold_statistics
                        ],
                        "source":
                            "recommendation/mining/conditional_llm_mining.py",
                    })
                    existing.add(key)
                target_search={
                    "kind":"conditional_llm_seed_only",
                    "requires_real_fpminer_seed":True,
                    "semantic_predicates":list(semantic),
                    "context_predicates":list(context),
                    "fpminer_semantic_seeds":len(semantic_seeds),
                    "config":dict(conditional_result.config.__dict__),
                    "audit":conditional_result.audit.as_dict(),
                    "candidate_patterns":len(conditional_result.patterns),
                    "deeper_candidate_patterns":len(
                        conditional_result.patterns
                    ),
                }
            else:
                target_search={
                    "kind":"conditional_llm_seed_only",
                    "requires_real_fpminer_seed":True,
                    "semantic_predicates":list(semantic),
                    "context_predicates":list(context),
                    "fpminer_semantic_seeds":len(semantic_seeds),
                    "candidate_patterns":0,
                    "deeper_candidate_patterns":0,
                    "reason":(
                        "no active semantic/context vocabulary or real "
                        "fpMiner semantic seed"
                    ),
                }
            # Unary semantic rules seed discovery but do not also vote.  Their
            # conditional children and non-LLM fpMiner rules form the model.
            semantic_seed_ids={seed.rule_id for seed in semantic_seeds}
            removed=[
                rule for rule in rules
                if rule.get("id") in semantic_seed_ids
                and len(rule.get("premises",()))==1
                and rule["premises"][0][0] in LLM_PAIR_PREDICATES
            ]
            removed_ids={id(rule) for rule in removed}
            rules=[rule for rule in rules if id(rule) not in removed_ids]
            target_search.update({
                "semantic_seed_policy":"discovery_only",
                "discovery_only_seed_rules":[{
                    "rule_id":rule["id"],
                    "premises":[list(item) for item in rule["premises"]],
                    "source":rule["source"],
                    "support":rule["support"],
                    "discovery_ctv":rule["discovery_ctv"],
                } for rule in removed],
                "discovery_only_seed_count":len(removed),
                "candidate_conditional_children":sum(
                    rule.get("source","").endswith(
                        "conditional_llm_mining.py"
                    ) for rule in rules
                ),
                "backoff_policy":"non-LLM mined rules remain active",
            })
            n=len(bounded_population); wins=sum(case["positive"] for case in bounded_population)
            base_rate=wins/n if n else 0.5
            impression_macro=True
            unavailable_evidence={"incomparable","unknown","left_known","right_known"}
            def rule_applicable(case,premises):
                """True only when every premise family is observed for this pair."""
                return all(
                    case["attrs"].get(predicate) is not None
                    and str(case["attrs"].get(predicate)).lower()
                        not in unavailable_evidence
                    for predicate,_value in premises
                )
            calibrated=[]
            for rule in rules:
                matched={index for index,case in enumerate(bounded_population)
                         if all(case["attrs"].get(predicate)==value
                                for predicate,value in rule["premises"])}
                antecedent_total=len(matched)
                joint=sum(bounded_population[index]["positive"] for index in matched)
                outside=n-antecedent_total; outside_joint=wins-joint
                raw_pair_strength=(joint/antecedent_total
                                   if antecedent_total else 0.0)
                strength=raw_pair_strength
                raw_selection_k=float(self.config["pair_rule_selection_k"])
                selection_count_confidence=(
                    antecedent_total/(antecedent_total+raw_selection_k)
                    if antecedent_total else 0.0
                )
                # PeTTa's tv_formulas.metta decodes every confidence with K=800.
                # Encode actual evidence using that same invariant at the boundary.
                count_confidence=(
                    antecedent_total/(antecedent_total+CTV_EVIDENCE_K_DEFAULT)
                    if antecedent_total else 0.0
                )
                # In an orientation-balanced pair workspace a directional premise
                # can match at most one of the two orientations.  Its comparable
                # coverage is therefore 2n_r/n.  Treat that portability/reliability
                # as host-side selection reliability: a rule that only decides a
                # small exceptional slice must not displace an equally accurate
                # rule that orders most production candidates. Coverage must not be
                # folded into the CTV confidence that PeTTa decodes as evidence.
                directional_predicates=[
                    predicate for predicate,_value in rule["premises"]
                    if predicate in predicate_comparability
                ]
                if directional_predicates:
                    jointly_comparable=sum(
                        all(
                            str(case["attrs"].get(predicate,"")).lower()
                            in {"left","right"}
                            or str(case["attrs"].get(predicate,"")).lower().startswith(
                                ("left_q","right_q")
                            )
                            for predicate in directional_predicates
                        )
                        for case in bounded_population
                    )
                    activation_coverage=jointly_comparable/n if n else 0.0
                else:
                    activation_coverage=(
                                     (min(1.0,2.0*antecedent_total/n) if n else 0.0))
                confidence=count_confidence
                selection_confidence=(
                    selection_count_confidence*activation_coverage
                )
                negative_strength=outside_joint/outside if outside else base_rate
                negative_confidence=(outside/(outside+CTV_EVIDENCE_K_DEFAULT)
                                     if outside else 0.0)
                calibration=None
                petta_calibration=None
                calibrated_base_rate=base_rate
                is_scoped_categorical=rule.get("scoped_categorical_prior") is True
                calibration=calibrate_ctv((
                    CTVObservation(
                        impression_id=case["impression"],
                        matched=index in matched,
                        target=bool(case["positive"]),
                        applicable=rule_applicable(case,rule["premises"]),
                        # ``population_cases`` is exhaustive even when the
                        # discovery pass sampled opponents.  The stored weight
                        # is label-independent and sums to one per impression.
                        weight=float(case["search_weight"]),
                    )
                    for index,case in enumerate(bounded_population)
                ), evidence_k=float(self.config["pair_rule_selection_k"]),
                   rule_kind="pair_preference")
                petta_calibration=reencode_ctv_confidence(
                    calibration,evidence_k=CTV_EVIDENCE_K_DEFAULT
                )
                strength=calibration.positive.strength
                negative_strength=calibration.negative.strength
                calibrated_base_rate=calibration.applicable_target_base_rate
                selection_confidence=calibration.positive.confidence
                confidence=petta_calibration.positive.confidence
                negative_confidence=petta_calibration.negative.confidence
                activation_coverage=calibration.activation.weighted_fraction

                effect=strength-calibrated_base_rate; specificity=len(rule["premises"])
                fold_effects=[]
                fold_supports=[]
                fold_effective_impressions=[]
                for fold in range(3):
                    fold_population=[index for index,case in enumerate(bounded_population)
                                     if case["temporal_fold"]==fold]
                    fold_matches=matched.intersection(fold_population)
                    if not fold_population: continue
                    fold_calibration=calibrate_ctv((
                        CTVObservation(
                            impression_id=bounded_population[index]["impression"],
                            matched=index in fold_matches,
                            target=bool(bounded_population[index]["positive"]),
                            applicable=rule_applicable(
                                bounded_population[index],rule["premises"]
                            ),
                            weight=float(
                                bounded_population[index]["search_weight"]
                            ),
                        )
                        for index in fold_population
                    ), evidence_k=float(self.config["pair_rule_selection_k"]),
                       rule_kind="pair_preference_fold")
                    fold_support=fold_calibration.positive.weighted_support
                    fold_effective=fold_calibration.positive.effective_impressions
                    macro_fold_min=target_min_support/3.0
                    if fold_support<max(1.0,macro_fold_min):
                        continue
                    fold_base=fold_calibration.applicable_target_base_rate
                    fold_strength=fold_calibration.positive.strength

                    fold_effects.append(fold_strength-fold_base)
                    fold_supports.append(fold_support)
                    fold_effective_impressions.append(fold_effective)
                required_folds=3 if is_scoped_categorical else 2
                stable_effect=(min(fold_effects)
                               if len(fold_effects)>=required_folds else -1.0)
                uses_impression_support=True
                calibrated_support=(
                    calibration.positive.weighted_support
                    if uses_impression_support and calibration is not None
                    else antecedent_total
                )
                required_support=target_min_support
                mined_support=rule["support"]
                ctv_calibration_audit=None
                selection_calibration_audit=None
                if calibration is not None and petta_calibration is not None:
                    selection_calibration_audit=calibration.as_dict()
                    selection_calibration_audit["confidence_role"]=(
                        "host_rule_selection_only_not_petta_stv"
                    )
                    ctv_calibration_audit=petta_calibration.as_dict()
                    # Some experimental modes preserve raw conditional strengths
                    # or a raw unmatched branch. Record the exact CTV compiled for
                    # PeTTa, not merely the population object it was derived from.
                    ctv_calibration_audit["positive"]["strength"]=strength
                    ctv_calibration_audit["positive"]["confidence"]=confidence
                    ctv_calibration_audit["negative"]["strength"]=negative_strength
                    ctv_calibration_audit["negative"]["confidence"]=negative_confidence
                    ctv_calibration_audit["confidence_role"]=(
                        "petta_stv_evidence_encoded_with_fixed_k800"
                    )
                rule.update(
                    strength=strength,confidence=confidence,
                    selection_confidence=selection_confidence,
                    raw_pair_strength=raw_pair_strength,
                    count_confidence=count_confidence,
                    selection_count_confidence=selection_count_confidence,
                    activation_coverage=activation_coverage,support=joint,
                    mined_support=mined_support,
                    antecedent_support=antecedent_total,joint_support=joint,
                    negative_strength=negative_strength,negative_confidence=negative_confidence,
                    lift=(strength/calibrated_base_rate if calibrated_base_rate else 0.0),effect=effect,
                    ctv_calibration=ctv_calibration_audit,
                    selection_calibration=selection_calibration_audit,
                    petta_evidence_k=CTV_EVIDENCE_K_DEFAULT,
                    pair_rule_selection_k=float(
                        self.config["pair_rule_selection_k"]
                    ),
                    ctv_estimation_mode="impression_macro_kish",
                    confidence_basis=(
                        "petta_kish_effective_impressions_k800"
                    ),
                    selection_confidence_basis=(
                        "kish_effective_impressions_with_pair_rule_selection_k"
                        if calibration is not None else
                        "raw_pair_count_with_pair_rule_selection_k_times_activation_coverage"
                    ),
                    calibration_base_rate=calibrated_base_rate,
                    temporal_fold_effects=[round(value,8) for value in fold_effects],
                    temporal_fold_weighted_supports=[round(value,8)
                                                     for value in fold_supports],
                    temporal_fold_kish_effective_impressions=[round(value,8)
                                                              for value in fold_effective_impressions],
                    required_temporal_folds=required_folds,
                    calibrated_support=calibrated_support,
                    ctv_support=calibrated_support,
                    calibrated_support_unit=(
                        "equal_impression_weighted_activation_mass"
                        if uses_impression_support else "raw_oriented_pair_case"
                    ),
                    ctv_support_unit=(
                        "equal_impression_weighted_activation_mass"
                        if uses_impression_support else "raw_oriented_pair_case"
                    ),
                    required_calibrated_support=required_support,
                    stable_effect=stable_effect,
                    # Balanced orientations make 2p-1 an anti-symmetric learned
                    # vote margin.  It is later counted only when PeTTaChainer
                    # proves this mined rule for the grounded pair.
                    vote_weight=max(0.0,2.0*strength-1.0),
                    specificity=specificity,coverage=matched,
                    quality=max(0.0,stable_effect)*selection_confidence
                            *math.log1p(calibrated_support)
                            *(1+0.2*(specificity-1)),
                )
                if (calibrated_support>=required_support and effect>0
                        and stable_effect>=float(self.config["pair_min_effect"])):
                    calibrated.append(rule)
            def selection_quality(rule):
                return rule["quality"]
            calibrated.sort(key=lambda rule:(-selection_quality(rule),
                                             -rule["specificity"],
                                             -rule["antecedent_support"],rule["premises"]))
            selected=[]
            rule_cap=int(self.config["pair_max_rules"])
            def retain_candidates(candidates,limit=None,*,within_family=False):
                retained=0
                for rule in candidates:
                    redundant=False
                    for previous in selected:
                        if (within_family
                                and (previous["specificity"]==1)
                                != (rule["specificity"]==1)):
                            continue
                        union_cases=rule["coverage"]|previous["coverage"]
                        intersection_cases=rule["coverage"]&previous["coverage"]
                        if (impression_macro
                                or rule.get("scoped_categorical_prior") is True
                                or previous.get("scoped_categorical_prior") is True):
                            # Categorical priors are calibrated per impression;
                            # redundancy must use the same sample unit. Otherwise
                            # one large slate can erase a distinct portable rule.
                            union=math.fsum(
                                bounded_population[index]["search_weight"]
                                for index in union_cases
                            )
                            intersection=math.fsum(
                                bounded_population[index]["search_weight"]
                                for index in intersection_cases
                            )
                        else:
                            union=len(union_cases)
                            intersection=len(intersection_cases)
                        overlap=(intersection/union if union else 1.0)
                        if overlap>=0.98 and abs(rule["strength"]-previous["strength"])<0.02:
                            redundant=True; break
                    if not redundant:
                        selected.append(rule); retained+=1
                    if len(selected)>=rule_cap or (limit is not None and retained>=limit):
                        return True
                return False
            # A shared cap must not let either half of the hybrid starve the
            # other. Reserve deterministic quotas for the real fpMiner unary
            # backoff and target-expanded structures, then backfill unused slots
            # by the common quality order. During the reserved pass, redundancy
            # is evaluated within each family so at least one deeper rule can be
            # exercised even when it closely specializes a unary rule.
            unary=[rule for rule in calibrated if rule["specificity"]==1]
            deeper=[rule for rule in calibrated if rule["specificity"]>1]
            if unary and deeper:
                if rule_cap>=2:
                    unary_quota=(rule_cap+1)//2
                    deeper_quota=rule_cap-unary_quota
                    retain_candidates(iter(unary),unary_quota,within_family=True)
                    retain_candidates(iter(deeper),deeper_quota,within_family=True)
                    if len(selected)<rule_cap:
                        retain_candidates(iter(calibrated))
                else:
                    # A one-rule artifact cannot contain both families; retain
                    # the auditable real-miner backoff instead of silently
                    # presenting one host-expanded rule as the whole hybrid.
                    retain_candidates(iter(unary))
            else:
                retain_candidates(iter(calibrated))

            # Preserve useful specific variants, but make strongly nested evidence
            # dependent in PeTTa instead of confidence-inflating it. The residual
            # challenger keeps each conjunction as a separate evidence hyperedge
            # and compiles only its log-odds increment beyond selected parents.
            parents=list(range(len(selected)))
            def find(index):
                while parents[index]!=index:
                    parents[index]=parents[parents[index]]; index=parents[index]
                return index
            def union(left,right):
                left_root,right_root=find(left),find(right)
                if left_root!=right_root: parents[right_root]=left_root
            roots={}
            variants=Counter()
            residual_audit=None
            for left in range(len(selected)):
                for right in range(left+1,len(selected)):
                    if self._pair_rules_share_dependency(
                            selected[left],selected[right]):
                        union(left,right)
            for rule_index,rule in enumerate(selected):
                root=find(rule_index)
                cluster_index=roots.setdefault(root,len(roots)+1)
                dependency_id=f"pair_mined_cluster_{cluster_index}"
                variants[dependency_id]+=1
                rule.update(
                    id=dependency_id,dependency_id=dependency_id,
                    variant_id=f"{dependency_id}_v{variants[dependency_id]}",
                )
            if target_search is not None:
                compiled_children=[
                    rule for rule in selected
                    if rule.get("source","").endswith(
                        "conditional_llm_mining.py")
                ]
                target_search.update({
                    "compiled_conditional_children":len(compiled_children),
                    "compiled_conditional_premises":[
                        [list(item) for item in rule["premises"]]
                        for rule in compiled_children
                    ],
                })
            calibrated_categorical=[
                rule for rule in calibrated
                if rule.get("scoped_categorical_prior") is True
            ]
            selected_categorical=[
                rule for rule in selected
                if rule.get("scoped_categorical_prior") is True
            ]
            categorical_search.update({
                "calibration":"mandatory equal-impression CTV regardless of global pair_ctv_mode",
                "temporal_gate":"all three source-order folds",
                "support_gate":(
                    "equal-impression weighted activation mass >= target_min_support; "
                    "Kish effective impressions determine confidence only"
                ),
                "redundancy":"equal-impression weighted activation Jaccard",
                "calibrated_scoped_rules":len(calibrated_categorical),
                "selected_scoped_rules":len(selected_categorical),
                "selected_scoped_premises":[
                    [list(item) for item in rule["premises"]]
                    for rule in selected_categorical
                ],
                "dependency_policy":"dependent alternative inside existing evidence owner; never another vote",
                "category_symbol_labels":{
                    symbol:self._pair_categorical_labels[symbol]
                    for symbol in sorted({
                        value for rule in categorical_eligible
                        for predicate,value in rule["premises"]
                        if predicate in PAIR_CATEGORICAL_SIDE_PREDICATES
                    })
                    if symbol in self._pair_categorical_labels
                },
            })
            used_features={predicate for rule in selected for predicate,_value in rule["premises"]}
            for _family,(left_predicate,right_predicate) in (
                    PAIR_CATEGORICAL_SIDE_FAMILIES.items()):
                if {left_predicate,right_predicate}.intersection(used_features):
                    # A one-sided rule still needs both observed values to enforce
                    # whole-family OOV/missing abstention at serving time.
                    used_features.update((left_predicate,right_predicate))
            self._pair_feature_vocabulary={feature:values for feature,values in vocabulary.items()
                                           if feature in used_features}
            sources=[]
            # Correlated variants are alternatives, not independent observations
            # to revise into one conclusion.  Give each variant an isolated PeTTa
            # proof channel in proof-margin mode, while retaining its logical
            # dependency as a separate argument.  The scorer will still take at
            # most one inferred margin per dependency.
            isolate_variants=True
            for rule in selected:
                rule.setdefault("source","recommendation/miner/fpMiner.metta")
                terms=[f'({predicate.title()} $pair {json.dumps(value)})'
                       for predicate,value in rule["premises"]]
                premise=terms[0] if len(terms)==1 else f"(And {' '.join(terms)})"
                proof_strength=rule.get("proof_strength",rule["strength"])
                proof_confidence=rule.get("proof_confidence",rule["confidence"])
                negative_strength=(0.5 if "proof_strength" in rule
                                   else rule["negative_strength"])
                negative_confidence=(0.0 if "proof_strength" in rule
                                     else rule["negative_confidence"])
                ctv=(f'(CTV (STV {proof_strength} {proof_confidence}) '
                     f'(STV {negative_strength} {negative_confidence}))')
                proof_channel=(rule["variant_id"] if isolate_variants
                               else rule["dependency_id"])
                rule["proof_channel_id"]=proof_channel
                dependency=json.dumps(rule["dependency_id"])
                channel=json.dumps(proof_channel)
                conclusion=(f'(MinedPairPreference $pair {dependency} {channel})'
                            if isolate_variants else
                            f'(MinedPairPreference $pair {dependency})')
                variant_source=(
                    f'(: {rule["variant_id"]} (Implication {premise} '
                    f'{conclusion}) {ctv})'
                )
                sources.append(variant_source)
                if isolate_variants:
                    # This compiler-issued contract is checked before applying the
                    # alpha-normalized proof optimization.  Hand-written or future
                    # nonlocal rules therefore fail closed instead of being
                    # silently treated as extensional channel templates.
                    rule["proof_factorization"]={
                        "schema":"isolated_extensional_pair_channel_v1",
                        "case_variable":"$pair",
                        "premise_tv":[1.0,1.0],
                        "single_channel_producer":True,
                        "dependency_id":rule["dependency_id"],
                        "proof_channel_id":proof_channel,
                        "premises":[list(item) for item in rule["premises"]],
                        "rule_source_sha256":hashlib.sha256(
                            variant_source.encode("utf-8")
                        ).hexdigest(),
                    }
                rule.pop("coverage",None)
            merge_ids=set()
            if selected:
                proof_roots=(
                    sorted({(rule["dependency_id"],rule["proof_channel_id"])
                            for rule in selected})
                    if isolate_variants else
                    [(dependency_id,dependency_id) for dependency_id in sorted(
                        {rule["dependency_id"] for rule in selected}
                    )]
                )
                for root_index,(dependency_id,proof_channel) in enumerate(proof_roots,1):
                    dependency=json.dumps(dependency_id)
                    channel=json.dumps(proof_channel)
                    pair_signal=(f'(PairSignal $pair {dependency} {channel})'
                                 if isolate_variants else
                                 f'(PairSignal $pair {dependency})')
                    mined_signal=(f'(MinedPairPreference $pair {dependency} {channel})'
                                  if isolate_variants else
                                  f'(MinedPairPreference $pair {dependency})')
                    merge_id=f"pair_merge_rule_{root_index}"
                    merge_ids.add(merge_id)
                    decision_id=f"pair_decision_rule_{root_index}"
                    merge_ids.add(decision_id)
                    sources.extend((
                        f'(: {decision_id} (Implication '
                        f'{mined_signal} {pair_signal}) '
                        '(CTV (STV 1.0 1.0) (STV 0.0 1.0)))',
                        # This is a one-way decision adapter, not a claim that
                        # every family is necessary for PairWin. Bayesian inversion
                        # would feed other families back through the shared target
                        # and invent evidence in an otherwise absent channel.
                        f'(: (no_inverse {merge_id}) (Implication {pair_signal} '
                        '(PairWin $pair)) (CTV (STV 1.0 1.0) (STV 0.0 1.0)))',
                    ))
            self._pair_rule_sources=sources
            self._active_pair_rule_ids={
                identifier for rule in selected
                for identifier in (rule["dependency_id"],rule["variant_id"])
            }
            if selected: self._active_pair_rule_ids.update(merge_ids)
            self.pair_rules=selected
            self.last_pair_mining={
                "rules":len(selected),"cases":len(discovery_cases),"source_cases":n,
                "wins":wins,"losses":n-wins,
                "pair_target_semantics":{
                    "training_unit":"one closed observed training impression",
                    "eligible_impression":(
                        "contains at least one observed clicked and one observed "
                        "nonclicked candidate"
                    ),
                    "pair_construction":(
                        "every observed clicked-versus-nonclicked candidate pair; "
                        "both left/right orientations"
                    ),
                    "positive_target":(
                        "left candidate is the observed clicked candidate and right "
                        "candidate is the observed nonclicked candidate"
                    ),
                    "interpretation":(
                        "conditional within-impression pair-ordering target; not a "
                        "universal item-relevance claim"
                    ),
                },
                "evidence_clusters":len(roots),
                "proof_channels":len(proof_roots) if selected else 0,
                "proof_channel_mode":(
                    "variant_isolated" if isolate_variants else "shared_dependency"
                ),
                "proof_factorization":{
                    "enabled":bool(selected and isolate_variants),
                    "mode":"alpha_normalized_isolated_channels",
                    "safety_contract":"isolated_extensional_pair_channel_v1",
                    "maximum_templates":len(proof_roots) if selected else 0,
                    "host_applicability":"exact categorical premise join",
                    "reasoner_role":"two-hop channel proof and inferred STV",
                },
                "base_rate":round(base_rate,8),
                "min_support":target_min_support,
                "min_support_unit":"equal_impression_mass_target_search_only",
                "fpminer_min_support":fpminer_min_support,
                "fpminer_support_unit":"raw_oriented_pair_case",
                "target_min_support":target_min_support,
                "target_support_unit":"equal_impression_mass",
                "calibration_min_support":calibration_min_support,
                "calibration_min_support_unit":"raw_oriented_pair_case",
                "selected_ctv_min_support":target_min_support,
                "selected_ctv_min_support_unit":
                    "equal_impression_weighted_activation_mass",
                "selected_ctv_support_policy":{
                    "ordinary_rules":
                        "equal_impression_weighted_activation_mass",
                    "scoped_categorical_rules":(
                        "equal_impression_weighted_activation_mass"
                    ),
                },
                "miner_calls":len(plan),
                "fpminer_query_calls":sum(
                    item["miner_calls"] for item in fpminer_incremental
                ),
                "fpminer_incremental":fpminer_incremental,
                "workspace_mode":MINING_WORKSPACE_MODE,
                "workspace_sync":workspace_sync,
                "workspace_plans":len(workspace_plans),
                "retention":retention_audit,
                "full_structure_research":any(
                    item.get("full_structure_research",False)
                    for item in fpminer_incremental
                ),
                # No held-out outcomes participate in this training-population
                # estimate.
                "full_population_ctv_estimation":True,
                "discovery_sample":{
                    "purpose":"rule_structure_discovery_only",
                    "oriented_pair_cases":len(bounded_discovery),
                    "complete_impressions":len({
                        case["impression"] for case in bounded_discovery
                    }),
                    "negative_opponents_per_positive":int(
                        self.config["pair_negative_ratio"]
                    ),
                    "sampling_policy":(
                        "all negative opponents"
                        if int(self.config["pair_negative_ratio"])==0 else
                        "deterministic capped negative opponents per positive"
                    ),
                    "held_out_evaluation_labels_used":False,
                    "discovery_statistics_reused_for_final_ctv_estimation":False,
                },
                "ctv_estimation_population":{
                    "purpose":"CTV estimation, temporal gates and rule selection",
                    "oriented_pair_cases":len(bounded_population),
                    "complete_impressions":len({
                        case["impression"] for case in bounded_population
                    }),
                    "opponent_sampling":"none",
                    "source":(
                        "all clicked-versus-nonclicked oriented pairs from complete "
                        "retained closed training impressions"
                    ),
                    "held_out_evaluation_labels_used":False,
                    "ordinary_rule_weighting":
                        "one total mass per impression with Kish effective support",
                    "scoped_categorical_weighting":(
                        "one total mass per impression with Kish effective support"
                    ),
                },
                "miner_strategy":"conditional_llm_seed_only",
                "ctv_evidence_k":float(self.config["ctv_evidence_k"]),
                "pair_ctv_mode":self.config["pair_ctv_mode"],
                "petta_ctv_evidence_k":CTV_EVIDENCE_K_DEFAULT,
                "pair_rule_selection_k":float(
                    self.config["pair_rule_selection_k"]
                ),
                "pair_ctv_evidence_k":float(self.config["pair_ctv_evidence_k"]),
                "pair_ctv_evidence_k_status":(
                    "deprecated alias of pair_rule_selection_k; never used to "
                    "encode PeTTa STV confidence"
                ),
                "confidence_contract":(
                    "PeTTa STV confidence = evidence/(evidence+800); configurable "
                    "selection K affects host-side rule selection only"
                ),
                "pair_dependency_mode":"clustered",
                "residual_hypergraph":residual_audit,
                "target_search":target_search,
                "categorical_search":categorical_search,
                "search_weight_unit":"equal_impression_mass_target_search_only",
                "search_weight_total":round(discovery_search_weight,8),
                "feature_profile":self.config["pair_feature_profile"],
                "rules_by_premises":dict(sorted(Counter(rule["specificity"] for rule in selected).items())),
                "feature_vocabulary":{feature:len(values) for feature,values
                                      in self._pair_feature_vocabulary.items()},
                "numeric_evidence":{
                    source:encoder.to_metadata()
                    for source,encoder in self._numeric_pair_encoders.items()
                },
                "numeric_evidence_fitting":{
                    "split":"training_only",
                    "pair_population":"all_unordered_same_impression_candidates",
                    "target_labels_used":False,
                    "weight_unit":"equal_impression_mass_per_feature",
                    "delta_direction":"restored_only_after_magnitude_bin_transform",
                },
                "seconds":round(time.perf_counter()-started,3),
            }
            return self.last_pair_mining

        @staticmethod
        def _serving_model_digest(payload):
            encoded=json.dumps(
                payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        def serving_model(self):
            """Return a content-addressed immutable model for scorer startup.

            The artifact contains learned symbolic vocabularies, calibrated rules,
            numeric encoders and the exact MeTTa compiler output. It intentionally
            excludes user/session state, proof caches and training events so every
            serving replica starts clean from one auditable model version.
            """
            payload={
                "schema":SERVING_MODEL_SCHEMA,
                "source_rule_version":int(self.version),
                "symbolic_only":bool(self.symbolic_only),
                "config":copy.deepcopy(self.config),
                "feature_vocabulary":{
                    predicate:sorted(values)
                    for predicate,values in self._feature_vocabulary.items()
                },
                "pair_feature_vocabulary":{
                    predicate:sorted(values)
                    for predicate,values in self._pair_feature_vocabulary.items()
                },
                "pair_categorical_labels":dict(
                    sorted(self._pair_categorical_labels.items())
                ),
                "numeric_pair_encoders":{
                    source:encoder.to_metadata()
                    for source,encoder in self._numeric_pair_encoders.items()
                },
                "point_rules":copy.deepcopy(self.mined_rules),
                "pair_rules":copy.deepcopy(self.pair_rules),
                "compiled_point_rules":list(self._point_rule_sources),
                "compiled_point_channels":list(self._point_channel_sources),
                "compiled_pair_rules":list(self._pair_rule_sources),
                "active_point_rule_ids":sorted(self._active_rule_ids),
                "active_pair_rule_ids":sorted(self._active_pair_rule_ids),
                "click_base_rate":float(self._click_base_rate),
                "tie_break_stats":copy.deepcopy(self._tie_break_stats),
            }
            return {
                **payload,
                "model_sha256":self._serving_model_digest(payload),
            }

        def _load_serving_model(self,model):
            """Validate and install a frozen model without rerunning fpMiner."""
            if not isinstance(model,dict) or model.get("schema")!=SERVING_MODEL_SCHEMA:
                raise ValueError("unsupported serving model schema")
            payload={key:value for key,value in model.items()
                     if key!="model_sha256"}
            expected=self._serving_model_digest(payload)
            if not hmac.compare_digest(
                    str(model.get("model_sha256","")),expected):
                raise ValueError("serving model digest does not match its content")
            if bool(model.get("symbolic_only"))!=self.symbolic_only:
                raise ValueError("serving model evidence mode does not match scorer")

            def vocabulary(field):
                raw=model.get(field)
                if not isinstance(raw,dict):
                    raise ValueError(f"serving model has no valid {field}")
                parsed={}
                for predicate,values in raw.items():
                    if (not isinstance(predicate,str)
                            or not isinstance(values,list)
                            or any(not isinstance(value,str) for value in values)):
                        raise ValueError(f"serving model has invalid {field}")
                    parsed[predicate]=set(values)
                return parsed

            point_rules=copy.deepcopy(model.get("point_rules"))
            pair_rules=copy.deepcopy(model.get("pair_rules"))
            if not isinstance(point_rules,list) or not isinstance(pair_rules,list):
                raise ValueError("serving model rules must be lists")
            for rule in (*point_rules,*pair_rules):
                if not isinstance(rule,dict) or not isinstance(
                        rule.get("premises"),list):
                    raise ValueError("serving model contains an invalid rule")
                rule["premises"]=[tuple(item) for item in rule["premises"]]
            source_fields=(
                "compiled_point_rules","compiled_point_channels",
                "compiled_pair_rules",
            )
            sources={field:model.get(field) for field in source_fields}
            if any(not isinstance(value,list)
                   or any(not isinstance(item,str) for item in value)
                   for value in sources.values()):
                raise ValueError("serving model compiled rules must be string lists")

            self._feature_vocabulary=vocabulary("feature_vocabulary")
            self._pair_feature_vocabulary=vocabulary("pair_feature_vocabulary")
            labels=model.get("pair_categorical_labels")
            if (not isinstance(labels,dict)
                    or any(not isinstance(key,str) or not isinstance(value,str)
                           for key,value in labels.items())):
                raise ValueError("serving model categorical labels are invalid")
            self._pair_categorical_labels=dict(labels)
            encoders=model.get("numeric_pair_encoders")
            if not isinstance(encoders,dict):
                raise ValueError("serving model numeric encoders are invalid")
            self._numeric_pair_encoders={
                source:QuantileNumericEvidence.from_metadata(metadata)
                for source,metadata in encoders.items()
            }
            self.mined_rules=point_rules
            self.mined_output=[]
            self.pair_rules=pair_rules
            self._point_rule_sources=list(sources["compiled_point_rules"])
            self._point_channel_sources=list(sources["compiled_point_channels"])
            self._pair_rule_sources=list(sources["compiled_pair_rules"])
            self._active_rule_ids=set(model.get("active_point_rule_ids") or ())
            self._active_pair_rule_ids=set(
                model.get("active_pair_rule_ids") or ()
            )
            self._click_base_rate=float(model.get("click_base_rate"))
            if not 0.0<=self._click_base_rate<=1.0:
                raise ValueError("serving model click base rate is invalid")
            tie_break_stats=model.get("tie_break_stats")
            if not isinstance(tie_break_stats,dict):
                raise ValueError("serving model tie-break statistics are invalid")
            self._tie_break_stats=copy.deepcopy(tie_break_stats)

            # Fail before accepting traffic if compiler metadata and rule objects
            # disagree. PeTTa then receives exactly the validated source snapshot.
            self._point_channel_topology()
            self._compile_pair_rule_matcher()
            self.engine.replace([
                *self._point_rule_sources,*self._point_channel_sources,
                *self._pair_rule_sources,*LIVE_NEGATIVE_RULE_SOURCES,
            ])
            self.version=max(1,int(model.get("source_rule_version",1)))
            self.pending_events=0
            self._last_mined_event_sequence=self._event_sequence
            self.last_mined_at=time.time()
            self.last_pair_mining={
                "execution":"serving_model_load",
                "rules":len(self.pair_rules),
                "model_sha256":expected,
            }
            self.last_mining={
                "execution":"serving_model_load",
                "rules":len(self.mined_rules),
                "pair_rules":len(self.pair_rules),
                "model_sha256":expected,
                "version":self.version,
            }
            self._prewarm_serving_channels()
            self._startup_mode="serving_model"
            self._serving_model_sha256=expected

        def _prewarm_serving_channels(self):
            """Resolve model-invariant proof templates before readiness.

            Isolated point/pair channel proofs depend on their compiled rule and
            certain premise facts, not on a user's case identifier. Loading one
            alpha-normalized template per rule removes the first request's query
            penalty. Case-level caches are then cleared so no synthetic warm-up
            identity can be served as user evidence.
            """
            started=time.perf_counter()
            content_cache=self._prewarm_candidate_content_cache()
            point_specs=[]
            for index,rule in enumerate(self.mined_rules):
                attrs=dict(rule["premises"])
                point_specs.append((
                    f"warm_point_article_{index}",f"warm_point_case_{index}",
                    attrs,attrs,
                ))
            if point_specs:
                # Shared-target modes consume grounded candidate atoms. Weighted
                # mode is factorized and intentionally skips this mutation.
                if self.config.get("aggregation")!="weighted":
                    self._ensure_candidate_specs(point_specs)
                self._proofs_for_specs(point_specs)

            pair_specs=[]
            if (self.config.get("ranking_mode")=="pairwise"
                    and self.pair_rules):
                pair_specs=[
                    (f"warm_pair_case_{index}",dict(rule["premises"]))
                    for index,rule in enumerate(self.pair_rules)
                ]
                self._ensure_pair_specs(pair_specs)
                self._proofs_for_pair_specs(pair_specs)

            audit={
                "status":"complete",
                "seconds":round(time.perf_counter()-started,6),
                "point_channels":len(self._point_channel_proof_cache),
                "pair_channels":len(self._pair_channel_proof_cache),
                "point_reasoner_query_calls":self._point_query_calls,
                "pair_reasoner_query_calls":self._pair_query_calls,
                "content_cache":content_cache,
            }
            self._proof_cache.clear()
            self._pair_proof_cache.clear()
            self._pair_case_attrs.clear()
            self._pair_margin_cache.clear()
            self._point_query_calls=0
            self._point_query_roots=0
            self._point_pruned_query_roots=0
            self._point_channel_activations=0
            self._point_reused_channel_activations=0
            self._pair_query_calls=0
            self._pair_query_roots=0
            self._pair_pruned_query_roots=0
            self._pair_channel_activations=0
            self._pair_reused_channel_activations=0
            self._startup_prewarm=audit
            return audit

        def _rebuild_tie_break_stats(self,indexed_events=None):
            """Build safe, training-only priors used only for exact proof ties.

            The symbolic score remains the decision signal.  MIND's bounded
            predicates intentionally collapse many articles to the same proof, so
            a small empirical editorial prior makes the order of an exact tie
            useful without allowing metadata to override a proof with a different
            score.  Beta(1, 9) smoothing matches the dataset CTR bucket prior.
            """
            totals={"topic":Counter(),"format":Counter(),"subcategory":Counter()}
            positives={"topic":Counter(),"format":Counter(),"subcategory":Counter()}
            indexed=(list(indexed_events) if indexed_events is not None else
                     list(enumerate(self.data.get("events",[]))))
            for _event_index,event in indexed:
                try:
                    article=self.article(event["article"])
                except (KeyError,ValueError):
                    continue
                for key in totals:
                    value=str(article.get(key,"unknown"))
                    totals[key][value]+=1
                    positives[key][value]+=event.get("action") in POSITIVE
            self._tie_break_stats={
                key:{value:(positives[key][value]+1)/(count+10.0)
                     for value,count in totals[key].items() if count}
                for key in totals
            }

        def _tie_break(self, article):
            return {
                "topic_prior":self._tie_break_stats.get("topic",{}).get(
                    str(article.get("topic",article.get("category","unknown"))),0.0),
                "format_prior":self._tie_break_stats.get("format",{}).get(
                    str(article.get("format","unknown")),0.0),
                "subcategory_prior":self._tie_break_stats.get("subcategory",{}).get(
                    str(article.get("subcategory","unknown")),0.0),
            }

        @staticmethod
        def _proof_ranking_signature(row):
            """Ranking evidence derived exclusively from PeTTa-backed signals."""
            stv=row.get("stv",{})
            return (
                float(row.get("ranking_score",row.get("score",0.0))),
                float(row.get("pairwise_score",0.0) or 0.0),
                float(row.get("score",0.0)),
                float(stv.get("strength",0.0)),
                float(stv.get("confidence",0.0)),
            )

        @staticmethod
        def _ranking_signature(row):
            """Comparable signal used by serving and replay AUC.

            The first three components are proof-derived. The final three are
            consulted only when those proof values are equal, matching the sort
            order in :meth:`_rank` without turning a calibrated score into noisy
            pseudo-precision.
            """
            tie=row.get("tie_break",{})
            stv=row.get("stv",{})
            return (*ModelLifecycleMixin._proof_ranking_signature(row),
                float(tie.get("topic_prior",0.0)),
                float(tie.get("format_prior",0.0)),
                float(tie.get("subcategory_prior",0.0)),
            )

        @staticmethod
        def _pointwise_signature(row):
            tie=row.get("tie_break",{}); stv=row.get("stv",{})
            return (float(row.get("score",0.0)),float(stv.get("strength",0.0)),
                    float(stv.get("confidence",0.0)),float(tie.get("topic_prior",0.0)),
                    float(tie.get("format_prior",0.0)),
                    float(tie.get("subcategory_prior",0.0)))

        def _capture_background_mining(self):
            """Capture one causal training snapshot without copying static corpora."""
            with self.lock:
                if self._closed:
                    raise RuntimeError("recommendation Lab is closed")
                retained_relational_proofs=self._retained_live_relational_proofs()
                snapshot_data=dict(self.data)
                # These are the only source structures mutated by live feedback.
                # Articles, vectors, adapters and frozen workspaces are immutable and
                # intentionally shared to keep the short capture phase bounded.
                snapshot_data["events"]=copy.deepcopy(self.data.get("events",[]))
                snapshot_data["users"]=copy.deepcopy(self.data.get("users",{}))
                snapshot_config=self.config.copy()
                sequence=self._event_sequence; base_version=self.version

                staged=copy.copy(self)
                staged.instance_id=f"{self.instance_id}-staged-{sequence}"
                staged.lock=threading.RLock()
                staged.data=snapshot_data
                staged.config=snapshot_config
                staged.engine=IsolatedPeTTaChainer()
                # The long-lived PeTTa runtime (and, when enabled, its persistent
                # workspace cache) remains shared. MINER_LOCK serializes every use.
                staged.petta=self.petta
                staged._closed=False; staged._background_mining=None
                staged._staged_background_build=True
                staged._event_sequence=sequence
                staged._last_mined_event_sequence=self._last_mined_event_sequence
                staged.pending_events=0
                staged.runs=[]; staged.feed_cache=OrderedDict()
                staged._feed_rank_cache_hits=0; staged._feed_rank_cache_misses=0
                staged._feed_sessions={}
                staged._online_events=[]
                staged._popularity=Counter(
                    event["article"] for event in snapshot_data["events"]
                    if event.get("action") in POSITIVE
                )
                staged._tie_break_stats={}
                staged._loaded_candidates=set(); staged._proof_cache={}
                staged._loaded_point_channels=set()
                staged._point_channel_proof_cache={}
                staged._point_channel_templates={}
                staged._point_query_calls=0; staged._point_query_roots=0
                staged._point_pruned_query_roots=0
                staged._point_channel_activations=0
                staged._point_reused_channel_activations=0
                staged._last_point_completeness={}
                staged._last_point_cache_stats={}
                staged._loaded_pairs=set(); staged._pair_proof_cache={}
                staged._loaded_pair_channels=set()
                staged._pair_channel_proof_cache={}
                staged._pair_channel_templates={}; staged._pair_proof_origins={}
                staged._pair_case_attrs={}; staged._pair_margin_cache={}
                staged._pair_query_calls=0; staged._pair_query_roots=0
                staged._pair_pruned_query_roots=0
                staged._pair_channel_activations=0
                staged._pair_reused_channel_activations=0
                staged._relational_feature_cache={}
                staged._loaded_relational_statements=set()
                # Event proof references are immutable training provenance. Keep
                # their records in the staged snapshot even though derived-feature
                # and worker caches always start cold.
                staged._live_relational_proof_ledger=retained_relational_proofs
                staged._candidate_relational_proof_refs={}
                staged._relational_query_calls=0
                staged._relational_query_roots=0
                return MiningSnapshot(sequence,base_version,staged)

        def _build_background_mining(self,snapshot):
            """Fully rebuild rules on a staged object while the old scorer serves."""
            staged=snapshot.payload
            try:
                staged.mine()
                return staged
            except BaseException:
                staged.engine.close()
                staged.engine=None
                raise

        @staticmethod
        def _dispose_background_mining(staged):
            engine=getattr(staged,"engine",None)
            if engine is not None:
                engine.close()
                staged.engine=None

        def _promote_background_mining(self,snapshot,staged):
            """Adopt one coherent staged rule/worker snapshot if it is still current."""
            old_engine=None
            with self.lock:
                if (self._closed or self.version!=snapshot.base_version
                        or self.config!=staged.config):
                    return {"promoted":False,"reason":"stale_source"}
                if staged.engine is None or staged.engine.pid is None:
                    raise RuntimeError("staged PeTTaChainer worker is not ready")
                expected_version=snapshot.base_version+1
                if staged.version!=expected_version:
                    raise RuntimeError("staged mining version is inconsistent")
                # The build may overlap newly served cards or accepted feedback.
                # Preserve only proof records that remain reachable from the
                # current event/served-context ledgers before retiring old caches.
                retained_relational_proofs=self._retained_live_relational_proofs(
                    getattr(staged,"_live_relational_proof_ledger",{})
                )

                old_engine=self.engine
                self.engine=staged.engine; staged.engine=None
                for field in BACKGROUND_MODEL_FIELDS:
                    setattr(self,field,getattr(staged,field))
                self.version=expected_version
                self._last_mined_event_sequence=snapshot.event_sequence
                self.pending_events=max(
                    0,self._event_sequence-self._last_mined_event_sequence
                )
                self.last_mined_at=time.time()
                workspace_prune={
                    **self._prune_mining_workspace_cache(),"deferred":False,
                }
                self.last_mining={
                    **(staged.last_mining or {}),
                    "version":self.version,
                    "execution":"background",
                    "build_mode":BACKGROUND_MINING_BUILD_MODE,
                    "event_sequence":snapshot.event_sequence,
                    "workspace_prune":workspace_prune,
                }
                self._loaded_candidates.clear(); self._loaded_pairs.clear()
                self._loaded_point_channels.clear()
                self._point_channel_proof_cache.clear()
                self._point_channel_templates.clear()
                self._point_query_calls=0; self._point_query_roots=0
                self._point_pruned_query_roots=0
                self._point_channel_activations=0
                self._point_reused_channel_activations=0
                self._last_point_completeness={}
                self._last_point_cache_stats={}
                self._loaded_pair_channels.clear()
                self._proof_cache.clear(); self._pair_proof_cache.clear()
                self._pair_channel_proof_cache.clear()
                self._pair_channel_templates.clear(); self._pair_case_attrs.clear()
                self._pair_margin_cache.clear(); self._pair_proof_origins.clear()
                self._relational_feature_cache.clear()
                self._loaded_relational_statements.clear()
                self._live_relational_proof_ledger=retained_relational_proofs
                self._candidate_relational_proof_refs.clear()
                self._relational_query_calls=0
                self._relational_query_roots=0
                # Preserve the bounded served-impression ledger so feedback from a
                # card rendered immediately before promotion remains valid. Its
                # unserved queue is upgraded through the new scorer on that event;
                # an ordinary next-page request receives normal version-reset
                # semantics instead.
                self.feed_cache.clear()
                outcome={
                    "promoted":True,"version":self.version,
                    "event_sequence":snapshot.event_sequence,
                    "pending_events":self.pending_events,
                }
            # No request can still hold the previous engine after the Lab lock was
            # acquired above. Retiring it outside that lock keeps new feeds available.
            try:
                old_engine.close()
            except BaseException:
                pass
            return outcome

        def _background_remine_due(self):
            with self.lock:
                return (not self._closed
                        and self.pending_events>=int(self.config["mine_interval"]))

        def close(self):
            """Cancel future promotions and release the active proof worker."""
            with self.lock:
                self._closed=True
                coordinator=self._background_mining
                engine=self.engine
                candidate_executor=getattr(
                    self,"_candidate_feature_executor",None
                )
                self._candidate_feature_executor=None
            def release_workspaces():
                if coordinator is not None:
                    coordinator.wait()
                try:
                    with MINER_LOCK:
                        self._mining_workspaces.prune()
                        self._incremental_fpminer.prune()
                except BaseException:
                    # Process shutdown/dataset replacement must still release the
                    # scorer. A dirty cache cannot be reused after close.
                    pass
            if coordinator is not None:
                stopped=coordinator.close(wait=False)
            else:
                stopped=True
            if stopped:
                release_workspaces()
            else:
                threading.Thread(
                    target=release_workspaces,
                    name="recommendation-mining-workspace-cleanup",
                    daemon=True,
                ).start()
            engine.close()
            if candidate_executor is not None:
                candidate_executor.shutdown(wait=True,cancel_futures=True)

        def mine(self):
            if getattr(self,"serving_only",False):
                raise ValueError(
                    "serving-only scorers cannot mine; publish a new frozen model"
                )
            # Both PeTTa miner instances share the named recommendation scratch
            # space. Keep a metadata snapshot as well: if mining or clean-worker
            # compilation fails, the still-live old scorer and its model metadata
            # must remain one coherent version.
            with MINER_LOCK:
                snapshot={
                    "feature_vocabulary":self._feature_vocabulary,
                    "pair_feature_vocabulary":self._pair_feature_vocabulary,
                    "pair_categorical_labels":self._pair_categorical_labels,
                    "numeric_pair_encoders":self._numeric_pair_encoders,
                    "mined_rules":self.mined_rules,
                    "mined_output":self.mined_output,
                    "point_rule_sources":self._point_rule_sources,
                    "point_channel_sources":self._point_channel_sources,
                    "pair_rules":self.pair_rules,
                    "pair_rule_sources":self._pair_rule_sources,
                    "active_rule_ids":self._active_rule_ids,
                    "active_pair_rule_ids":self._active_pair_rule_ids,
                    "active_mining_workspace_plans":self._active_mining_workspace_plans,
                    "last_pair_mining":self.last_pair_mining,
                    "click_base_rate":self._click_base_rate,
                    "tie_break_stats":self._tie_break_stats,
                }
                try:
                    return self._mine_once()
                except BaseException:
                    self._feature_vocabulary=snapshot["feature_vocabulary"]
                    self._pair_feature_vocabulary=snapshot["pair_feature_vocabulary"]
                    self._pair_categorical_labels=snapshot["pair_categorical_labels"]
                    self._numeric_pair_encoders=snapshot["numeric_pair_encoders"]
                    self.mined_rules=snapshot["mined_rules"]
                    self.mined_output=snapshot["mined_output"]
                    self._point_rule_sources=snapshot["point_rule_sources"]
                    self._point_channel_sources=snapshot["point_channel_sources"]
                    self.pair_rules=snapshot["pair_rules"]
                    self._pair_rule_sources=snapshot["pair_rule_sources"]
                    self._active_rule_ids=snapshot["active_rule_ids"]
                    self._active_pair_rule_ids=snapshot["active_pair_rule_ids"]
                    self._active_mining_workspace_plans=snapshot[
                        "active_mining_workspace_plans"
                    ]
                    self.last_pair_mining=snapshot["last_pair_mining"]
                    self._click_base_rate=snapshot["click_base_rate"]
                    self._tie_break_stats=snapshot["tie_break_stats"]
                    raise

        def _mine_once(self):
            started=time.perf_counter()
            retention=self._retained_mining_population()
            indexed_events=retention.records
            retention_audit=retention.audit.as_dict()
            self._rebuild_tie_break_stats(indexed_events)
            training_cases=self._mining_event_cases(indexed_events)
            if not training_cases: raise RuntimeError("negative sampling left no mining cases")
            training_events=[event for _case_id,event in training_cases]
            raw_rows=[(case_id,event,self.event_features(event))
                      for case_id,event in training_cases]
            selected_profile=FEATURE_PROFILES[self.config["feature_profile"]]
            value_counts={feature:Counter(attrs.get(feature)
                                          for _case_id,_event,attrs in raw_rows
                                          if feature in attrs)
                          for feature in selected_profile}
            max_values=max(2,int(self.config["max_feature_values"]))
            vocabulary={}
            for feature,counts in value_counts.items():
                ranked=[value for value,count in counts.most_common()
                        if value is not None and count>=int(self.config["min_support"])]
                if not ranked: continue
                keep=set(ranked[:max_values])
                if any(value not in keep for value in counts): keep.add("other")
                vocabulary[feature]=keep
            self._feature_vocabulary=vocabulary
            feature_rows=[(case_id,event,self._bounded_features(attrs))
                          for case_id,event,attrs in raw_rows]
            workspace_sync=[]; workspace_plans=set()
            def sync_space(selected_features,depth):
                workspace_cases={}
                for case,event,attrs in feature_rows:
                    facts=[]
                    outcome="click" if event["action"] in POSITIVE else "skip"
                    for predicate in selected_features:
                        if predicate in attrs:
                            facts.append(
                                f'({predicate} {case} {json.dumps(attrs[predicate])})'
                            )
                    facts.append(f'(engagement {case} {json.dumps(outcome)})')
                    workspace_cases[case]=tuple(facts)
                plan_key=self._mining_workspace_plan(
                    "point",selected_features,depth,"sampled_discovery"
                )
                synced=self._mining_workspaces.sync(plan_key,workspace_cases)
                workspace_plans.add(plan_key); workspace_sync.append(synced.as_dict())
                return synced,plan_key,workspace_cases

            active_features=[feature for feature in selected_profile if feature in vocabulary]
            plan=[(tuple(active_features),2)] if active_features else []
            raw=[]; fpminer_incremental=[]
            for selected_features,depth in plan:
                synced,plan_key,workspace_cases=sync_space(selected_features,depth)
                mined=self._mine_incremental_workspace(
                    synced,plan=plan_key,cases=workspace_cases,
                    features=selected_features,depth=depth,
                    min_support=int(self.config["min_support"]),
                    retention_units=retention.unit_ids,
                    retention_audit=retention_audit,workspace_kind="point",
                )
                raw.extend(mined.output)
                fpminer_incremental.append(mined.audit.as_dict())
            rules=parse_rules([str(v) for v in raw])
            target_search=None
            if not rules: raise RuntimeError("MeTTa miner produced no usable rules")
            self.mined_output=[str(v) for v in raw]
            # Negative sampling is a search-time optimization only.  Re-estimate
            # every mined rule on the complete event population before compiling
            # its CTV, otherwise the sampled click prevalence would bias both the
            # rule strength and PeTTaChainer ranking.
            population_rows=[(event,self._bounded_features(self.event_features(event)))
                             for _event_index,event in indexed_events]
            n=len(population_rows)
            target_total=sum(event["action"] in POSITIVE for event,_attrs in population_rows)
            sample_target_total=sum(event["action"] in POSITIVE for event in training_events)
            base_rate=target_total/n if n else 0.0
            for rule in rules:
                antecedent_total=joint=0
                for event,attrs in population_rows:
                    matches=all(attrs.get(p)==v for p,v in rule["premises"])
                    antecedent_total+=matches; joint+=matches and event["action"] in POSITIVE
                outside=n-antecedent_total; outside_joint=target_total-joint
                strength=joint/antecedent_total if antecedent_total else 0.0
                evidence_k=float(self.config["ctv_evidence_k"])
                confidence=(antecedent_total/(antecedent_total+evidence_k)
                            if antecedent_total else 0.0)
                negative_strength=outside_joint/outside if outside else 0.0
                negative_confidence=(outside/(outside+evidence_k)
                                     if outside else 0.0)
                specificity=len(rule["premises"]); mined_support=rule["support"]
                rule.update(strength=strength,confidence=confidence,support=joint,
                            mined_support=mined_support,
                            negative_strength=negative_strength,negative_confidence=negative_confidence,
                            antecedent_support=antecedent_total,joint_support=joint,
                            lift=(strength/base_rate if base_rate else 0.0),specificity=specificity,
                            # Both positive- and negative-lift click rules carry
                            # ranking information.  Their posterior is later
                            # shrunk toward this same empirical base rate.
                            quality=confidence*abs(strength-base_rate)
                                    *(1+0.25*(specificity-1))*math.log1p(antecedent_total))
            def rule_order(rule):
                return (
                    -rule["quality"],-rule["specificity"],
                    -rule["support"],rule["premises"],
                )
            rules.sort(key=rule_order)
            max_rules=int(self.config["max_rules"])
            if len(rules)>max_rules:
                # When the budget can represent them, keep a backoff layer from
                # every mined depth before filling by global quality order.
                depths=sorted({rule["specificity"] for rule in rules})
                quota,extra=divmod(max_rules,len(depths))
                selected=[]; selected_keys=set()
                for index,depth in enumerate(depths):
                    take=quota+(index<extra)
                    for rule in (candidate for candidate in rules if candidate["specificity"]==depth):
                        if take<=0: break
                        key=(rule["premises"],rule["target"])
                        selected.append(rule); selected_keys.add(key); take-=1
                for rule in rules:
                    if len(selected)>=max_rules: break
                    key=(rule["premises"],rule["target"])
                    if key not in selected_keys:
                        selected.append(rule); selected_keys.add(key)
                rules=sorted(selected,key=rule_order)
            rule_sources=[]; point_channel_sources=[]; point_compiled_ids=set()
            for index,rule in enumerate(rules,1):
                rule["id"]=f"mined_{index}"
                terms=[f'({p.title()} $case {json.dumps(v)})' for p,v in rule["premises"]]
                premise=terms[0] if len(terms)==1 else f"(And {' '.join(terms)})"
                ctv=f'(CTV (STV {rule["strength"]} {rule["confidence"]}) (STV {rule["negative_strength"]} {rule["negative_confidence"]}))'
                shared_source=(
                    f'(: {rule["id"]} (Implication {premise} '
                    f'(Engagement $case {json.dumps(rule["target"])})) {ctv})'
                )
                rule_sources.append(shared_source)

                # Weighted aggregation needs every applicable mined rule, whereas
                # one bounded query against the shared Engagement target returns
                # only the revision representative reached before its search budget
                # expires. Compile an additional isolated channel for each rule.
                # Max/hybrid continue to query the unchanged shared target above.
                channel=f'point_{rule["id"]}'
                variant_id=f"point_variant_{index}"
                decision_id=f"point_decision_rule_{index}"
                encoded_rule=json.dumps(rule["id"])
                encoded_channel=json.dumps(channel)
                mined_signal=(
                    f'(MinedPointPreference $case {encoded_rule} '
                    f'{encoded_channel})'
                )
                point_signal=(
                    f'(PointSignal $case {encoded_rule} {encoded_channel})'
                )
                variant_source=(
                    f'(: {variant_id} (Implication {premise} {mined_signal}) '
                    f'{ctv})'
                )
                decision_source=(
                    f'(: {decision_id} (Implication {mined_signal} '
                    f'{point_signal}) '
                    '(CTV (STV 1.0 1.0) (STV 0.0 1.0)))'
                )
                point_channel_sources.extend((variant_source,decision_source))
                point_compiled_ids.update((rule["id"],variant_id,decision_id))
                rule["point_variant_id"]=variant_id
                rule["point_decision_id"]=decision_id
                rule["point_proof_channel_id"]=channel
                rule["point_proof_factorization"]={
                    "schema":"isolated_extensional_point_channel_v1",
                    "case_variable":"$case",
                    "premise_tv":[1.0,1.0],
                    "single_channel_producer":True,
                    "rule_id":rule["id"],
                    "variant_id":variant_id,
                    "decision_id":decision_id,
                    "proof_channel_id":channel,
                    "premises":[list(item) for item in rule["premises"]],
                    "rule_source_sha256":hashlib.sha256(
                        variant_source.encode("utf-8")
                    ).hexdigest(),
                    "decision_source_sha256":hashlib.sha256(
                        decision_source.encode("utf-8")
                    ).hexdigest(),
                }
            pair_mining=self._mine_pairwise(
                indexed_events=indexed_events,retention=retention
            )
            pair_workspace_plans={
                item["plan"] for item in pair_mining.get("workspace_sync",[])
            }
            active_workspace_plans=frozenset(
                workspace_plans | pair_workspace_plans
            )
            retained_relational_proofs=self._retained_live_relational_proofs()
            # Retraction by proof name is process-global in the MeTTa runtime and a
            # second KB still shares all compiled rule indexes. Install one complete
            # point+pair snapshot in a fresh process instead. The old worker stays
            # live until this compilation succeeds, so a failed remine cannot leave
            # the scorer half-retracted.
            self.engine.replace([
                *rule_sources,*point_channel_sources,
                *self._pair_rule_sources,*LIVE_NEGATIVE_RULE_SOURCES,
            ])
            self._point_rule_sources=rule_sources
            self._point_channel_sources=point_channel_sources
            self._active_rule_ids=point_compiled_ids; self.mined_rules=rules
            self._active_mining_workspace_plans=active_workspace_plans
            self._click_base_rate=base_rate
            self._loaded_candidates.clear(); self._loaded_pairs.clear()
            self._loaded_point_channels.clear()
            self._point_channel_proof_cache.clear()
            self._point_channel_templates.clear()
            self._point_query_calls=0; self._point_query_roots=0
            self._point_pruned_query_roots=0
            self._point_channel_activations=0
            self._point_reused_channel_activations=0
            self._last_point_completeness={}
            self._last_point_cache_stats={}
            self._loaded_pair_channels.clear(); self._pair_channel_proof_cache.clear()
            self._pair_channel_templates.clear()
            self._pair_case_attrs.clear(); self._pair_proof_cache.clear(); self._pair_margin_cache.clear()
            self._pair_proof_origins.clear()
            self._relational_feature_cache.clear()
            self._loaded_relational_statements.clear()
            self._live_relational_proof_ledger=retained_relational_proofs
            self._candidate_relational_proof_refs.clear()
            self._relational_query_calls=0
            self._relational_query_roots=0
            self.pending_events=0; self.version+=1; self.feed_cache.clear(); self._proof_cache.clear(); self.last_mined_at=time.time()
            if self._background_mining is not None:
                # Explicit/configuration mines cover every event visible to this ModelLifecycleMixin.
                # A concurrently staged background artifact will fail its version
                # check and be disposed rather than overwriting this newer snapshot.
                self._last_mined_event_sequence=self._event_sequence
            workspace_prune={"removed":[],"error":None,"deferred":True}
            if not self._staged_background_build:
                workspace_prune={
                    **self._prune_mining_workspace_cache(),"deferred":False,
                }
            serving_prewarm=self._prewarm_serving_channels()
            result={"rules":len(rules),"cases":len(training_events),"source_cases":n,
                    "positives":sample_target_total,"negatives":len(training_events)-sample_target_total,
                    "source_positives":target_total,"source_negatives":n-target_total,
                    "online_negative_sampling":self._online_negative_sampling_audit(
                        training_cases,indexed_events
                    ),
                    "retention":retention_audit,
                    "sample_base_rate":round(sample_target_total/len(training_events),8),
                    "base_rate":round(base_rate,8),"depths":list(range(2,int(self.config["conjunctions"])+1)),
                    "feature_vocabulary":{feature:len(values) for feature,values in vocabulary.items()},
                    "rules_by_premises":dict(sorted(Counter(rule["specificity"] for rule in rules).items())),
                    "point_proof_channels":len(rules),
                    "point_proof_factorization":{
                        "enabled":True,
                        "serving_mode":"weighted_aggregation",
                        "mode":"alpha_normalized_isolated_channels",
                        "safety_contract":"isolated_extensional_point_channel_v1",
                        "maximum_templates":len(rules),
                        "host_applicability":"exact categorical premise join",
                        "reasoner_role":"two-hop channel proof and inferred STV",
                    },
                    "miner_calls":len(plan),"miner_strategy":"conditional_llm_seed_only",
                    "fpminer_query_calls":sum(
                        item["miner_calls"] for item in fpminer_incremental
                    ),
                    "fpminer_incremental":fpminer_incremental,
                    "workspace_mode":MINING_WORKSPACE_MODE,
                    "workspace_sync":workspace_sync,
                    "workspace_plans":len(active_workspace_plans),
                    "workspace_prune":workspace_prune,
                    "full_structure_research":any(
                        item.get("full_structure_research",False)
                        for item in fpminer_incremental
                    ),
                    "full_population_ctv_estimation":True,
                    "ctv_evidence_k":float(self.config["ctv_evidence_k"]),
                    "fpminer_min_support":int(self.config["min_support"]),
                    "fpminer_support_unit":"raw_point_case",
                    "target_min_support":None,
                    "target_support_unit":None,
                    "target_search":target_search,"pairwise":pair_mining,
                    "serving_prewarm":serving_prewarm,
                    "seconds":round(time.perf_counter()-started,3),"version":self.version}
            self.last_mining=result
            return result


    return ModelLifecycleMixin
