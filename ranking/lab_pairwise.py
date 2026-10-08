"""PeTTaChainer pair-proof preparation and ranking methods."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter

from ..features.relational_workspace import (
    REL_CONCEPT_CONTINUITY_PROOF_IDS,
    REL_CONCEPT_CONTINUITY_SCOPE,
    REL_ENTITY_CONTINUITY_PROOF_IDS,
    REL_ENTITY_CONTINUITY_SCOPE,
)
from ..mining.rule_parser import proof_tv


def make_pairwise_ranking_mixin(
    *,
    CONTEXT_FEATURES,
    DEFAULT_MAX_PROOF_CACHE_ENTRIES,
    LIVE_NEGATIVE_FEATURE,
    LIVE_NEGATIVE_RULE_IDS,
    PAIR_CATEGORICAL_SIDE_PREDICATES,
    PAIR_SIDE_PREDICATE_SWAP,
    RELATIONAL_PROOF_FIELDS,
):
    class PairwiseRankingMixin:
        @staticmethod
        def pair_candidate_case(attrs):
            signature=json.dumps(attrs,sort_keys=True,separators=(",",":"),ensure_ascii=False)
            digest=hashlib.blake2s(signature.encode("utf-8"),digest_size=10).hexdigest()
            return f"pair_candidate_{digest}"

        def _compile_pair_rule_matcher(self):
            """Compile an exact inverted join for the active pair-rule premises.

            Pair-rule applicability used to rescan every premise of every rule for
            both orientations of every candidate comparison.  The compiled index
            maps each observed ``(predicate, value)`` token to the rules containing
            it. A rule activates only when all of its indexed premises matched, so
            this changes traversal cost without changing applicability or order.

            The immutable signature also notices tests, staged promotion, or other
            callers that replace or mutate ``pair_rules`` directly.
            """
            premises=tuple(
                tuple(tuple(item) for item in rule["premises"])
                for rule in self.pair_rules
            )
            if premises==getattr(self,"_pair_matcher_signature",None):
                return
            inverted={}
            for rule_index,rule_premises in enumerate(premises):
                for token in rule_premises:
                    inverted.setdefault(token,[]).append(rule_index)
            self._pair_matcher_signature=premises
            self._pair_matcher_inverted={
                token:tuple(rule_indices)
                for token,rule_indices in inverted.items()
            }
            self._pair_matcher_required=tuple(map(len,premises))
            active_predicates={predicate for rule in premises
                               for predicate,_value in rule}
            # Categorical sides are grounded as an inseparable observed pair.
            # Asking the feature builder for only the referenced side would make
            # the bounding step abstain because its companion side was absent.
            # Recover categorical pairs from the supplied reverse map without
            # coupling this module to the server's family table.
            for predicate in tuple(active_predicates):
                companion=PAIR_SIDE_PREDICATE_SWAP.get(predicate)
                if companion is not None:
                    active_predicates.add(companion)
            self._pair_active_predicates=frozenset(active_predicates)

        def _matching_pair_rule_premises(self,attrs):
            """Return active premises in original rule order via the compiled join."""
            match_counts=[0]*len(self._pair_matcher_signature)
            inverted=self._pair_matcher_inverted
            for token in attrs.items():
                for rule_index in inverted.get(token,()):
                    match_counts[rule_index]+=1
            signature=self._pair_matcher_signature
            return tuple(
                signature[rule_index]
                for rule_index,required in enumerate(self._pair_matcher_required)
                if match_counts[rule_index]==required
            )

        @staticmethod
        def _reverse_pair_features(attrs):
            """Reverse one pair projection without recomputing candidate evidence.

            Every directional pair encoder is antisymmetric. Categorical side
            predicates swap names; directional values swap their ``left`` and
            ``right`` prefixes; shared/equal/incomparable values remain unchanged.
            This operates on the unbounded projection so the normal vocabulary
            filter is still applied independently to the reversed orientation.
            """
            reversed_attrs={}
            for predicate,value in attrs.items():
                reverse_predicate=PAIR_SIDE_PREDICATE_SWAP.get(predicate,predicate)
                if predicate in PAIR_CATEGORICAL_SIDE_PREDICATES:
                    # Category symbols are opaque: a real category may legally be
                    # named "left", "right_known" or "left_q1". Only the side
                    # predicate swaps for categorical facts.
                    reverse_value=value
                elif value=="left": reverse_value="right"
                elif value=="right": reverse_value="left"
                elif value=="left_known": reverse_value="right_known"
                elif value=="right_known": reverse_value="left_known"
                elif isinstance(value,str) and value.startswith("left_q"):
                    reverse_value="right_"+value[len("left_"):]
                elif isinstance(value,str) and value.startswith("right_q"):
                    reverse_value="left_"+value[len("right_"):]
                else: reverse_value=value
                reversed_attrs[reverse_predicate]=reverse_value
            return reversed_attrs

        def _oriented_pair_specs(self,left,right,case_cache=None):
            """Build both orientations from one candidate-feature comparison."""
            left_article,left_attrs=left
            right_article,right_attrs=right
            started=time.perf_counter()
            forward_raw=self._pair_features(
                left_attrs,right_attrs,left_article,right_article,
                needed=self._pair_active_predicates,
            )
            feature_seconds=time.perf_counter()-started
            started=time.perf_counter()
            reverse_raw=self._reverse_pair_features(forward_raw)
            reverse_seconds=time.perf_counter()-started
            started=time.perf_counter()
            forward_attrs=self._bounded_pair_features(forward_raw)
            reverse_attrs=self._bounded_pair_features(reverse_raw)
            bounding_seconds=time.perf_counter()-started
            started=time.perf_counter()
            forward_activation=self._matching_pair_rule_premises(forward_attrs)
            reverse_activation=self._matching_pair_rule_premises(reverse_attrs)
            activation_seconds=time.perf_counter()-started
            case_cache={} if case_cache is None else case_cache
            missing=object(); cache_seconds=0.0; serialization_seconds=0.0
            def activation_case(activation):
                nonlocal cache_seconds,serialization_seconds
                started=time.perf_counter()
                # The compiled matcher already returns a canonical immutable tuple
                # in model rule order. Re-normalizing it for every orientation was
                # pure allocation in the serving hot path.
                key=activation
                case=case_cache.get(key,missing)
                cache_seconds+=time.perf_counter()-started
                if case is missing:
                    started=time.perf_counter()
                    case=self.pair_candidate_case({"activation":activation})
                    serialization_seconds+=time.perf_counter()-started
                    case_cache[key]=case
                return case
            forward_case=activation_case(forward_activation)
            reverse_case=activation_case(reverse_activation)
            return (
                (forward_case,forward_attrs),(reverse_case,reverse_attrs),
                {
                    "pair_feature_derivation_seconds":feature_seconds,
                    "reverse_orientation_derivation_seconds":reverse_seconds,
                    "pair_feature_bounding_seconds":bounding_seconds,
                    "pair_activation_join_seconds":activation_seconds,
                    "pair_activation_case_cache_seconds":cache_seconds,
                    "pair_case_serialization_seconds":serialization_seconds,
                },
            )

        @staticmethod
        def pair_channel_case(dependency,channel,premises):
            """Return the alpha-normalized proof case for one isolated rule.

            Pair rules are universally quantified over ``$pair`` and their input
            facts are certain, extensional statements.  Consequently, a channel's
            proof and inferred STV are invariant under a collision-resistant renaming
            of the pair atom.  This stable identity lets proof-margin mode ask
            PeTTa once per mined channel instead of once per observed joint rule
            activation vector.
            """
            signature=json.dumps({
                "dependency":str(dependency),"channel":str(channel),
                "premises":[list(item) for item in premises],
            },sort_keys=True,separators=(",",":"),ensure_ascii=False)
            digest=hashlib.blake2s(signature.encode("utf-8"),digest_size=20).hexdigest()
            return f"pair_channel_{digest}"

        def _pair_spec(self,left,right):
            self._compile_pair_rule_matcher()
            left_article,left_attrs=left
            right_article,right_attrs=right
            attrs=self._bounded_pair_features(
                self._pair_features(
                    left_attrs,right_attrs,left_article,right_article,
                    needed=self._pair_active_predicates,
                )
            )
            # Active rules can only distinguish their own premise match vector.
            # Reusing one grounded context for an identical activation vector is
            # semantically exact and bounds serving identities by 2^rule-count,
            # rather than by every irrelevant combination of raw feature values.
            activation=self._matching_pair_rule_premises(attrs)
            signature={"activation":activation}
            return self.pair_candidate_case(signature),attrs

        def _ensure_pair_specs(self,specs,*,timeout_sec=None):
            """Index exact activation cases; proof channels are grounded lazily."""
            started=time.perf_counter()
            cache_limit=int(self.config.get(
                "max_proof_cache_entries",DEFAULT_MAX_PROOF_CACHE_ENTRIES
            ))
            new_cases={
                case for case,_attrs in specs
                if case not in self._pair_case_attrs
            }
            if len(self._pair_case_attrs)+len(new_cases)>cache_limit:
                raise RuntimeError(
                    "pair case cache limit exceeded; promote a fresh rule snapshot"
                )
            for case,attrs in specs:
                # A case is its selected-rule activation vector.  Unused raw
                # values cannot change any proof, so one representative is exact.
                self._pair_case_attrs.setdefault(case,attrs)
            cache_seconds=time.perf_counter()-started
            self._last_pair_materialization_profile={
                "pair_case_cache_index_seconds":cache_seconds,
                "pair_atom_serialization_seconds":0.0,
                "pair_atomspace_insertion_seconds":0.0,
                "candidate_pair_atoms_inserted":0,
                "factorized_channels":True,
                "total_seconds":cache_seconds,
            }

        def _proofs_for_pair_specs(self,specs,*,timeout_sec=None):
            runtime_started=time.perf_counter()
            timeout_sec=(self._serving_reasoner_timeout()
                         if timeout_sec is None else float(timeout_sec))
            batch=max(1,int(self.config["query_batch_size"])); calls=0
            proof_roots=tuple(sorted({(
                rule.get("dependency_id",rule["id"]),
                rule.get("proof_channel_id",rule.get("dependency_id",rule["id"])),
            ) for rule in self.pair_rules}))
            inference_key=(
                self.version,int(self.config["pair_chain_steps"]),batch,
                "proof_margin",proof_roots,
            )
            unique=list(dict.fromkeys(case for case,_attrs in specs))
            missing=[case for case in unique if (*inference_key,case) not in self._pair_proof_cache]
            cache_limit=int(self.config.get(
                "max_proof_cache_entries",DEFAULT_MAX_PROOF_CACHE_ENTRIES
            ))
            if len(self._pair_proof_cache)+len(missing)>cache_limit:
                raise RuntimeError(
                    "pair proof cache limit exceeded; promote a fresh rule snapshot"
                )
            pair_stats={
                "case_root_requests":len(unique),
                "case_root_hits":len(unique)-len(missing),
                "case_root_misses":len(missing),
                "channel_root_requests":0,
                "channel_root_hits":0,
                "channel_root_misses":0,
            }
            topology_seconds=0.0; activation_join_seconds=0.0
            template_serialization_seconds=0.0; atomspace_insertion_seconds=0.0
            query_seconds=0.0; inserted_template_atoms=0
            topology_started=time.perf_counter()
            supplied_attrs={case:attrs for case,attrs in specs}
            rules_by_root={}
            for rule in self.pair_rules:
                root=(
                    rule.get("dependency_id",rule["id"]),
                    rule.get("proof_channel_id",rule.get(
                        "dependency_id",rule["id"]
                    )),
                )
                rules_by_root.setdefault(root,[]).append(rule)
            # Exact factorization is deliberately fail-closed.  Every queried
            # root must have one producer, an isolated variant channel, and a
            # conjunction of keyed same-case extensional facts.  Shared-target
            # posterior mode follows the unfactorized branch below.
            topology={}
            compiled_source_hashes={hashlib.sha256(source.encode("utf-8")).hexdigest()
                                    for source in self._pair_rule_sources}
            for root,producers in rules_by_root.items():
                dependency,channel=root
                if len(producers)!=1:
                    raise RuntimeError(
                        f"proof channel {root!r} has {len(producers)} producers"
                    )
                rule=producers[0]
                premises=tuple(rule.get("premises",()))
                predicates=[predicate for predicate,_value in premises]
                contract=rule.get("proof_factorization",{})
                if (not premises or len(predicates)!=len(set(predicates))
                        or channel==dependency
                        or channel!=rule.get("variant_id")
                        or any(not isinstance(predicate,str)
                               or not re.fullmatch(r"[a-z][a-z0-9_]*",predicate)
                               or not isinstance(value,str)
                               for predicate,value in premises)
                        or contract.get("schema")!="isolated_extensional_pair_channel_v1"
                        or contract.get("case_variable")!="$pair"
                        or contract.get("premise_tv")!=[1.0,1.0]
                        or contract.get("single_channel_producer") is not True
                        or contract.get("dependency_id")!=dependency
                        or contract.get("proof_channel_id")!=channel
                        or contract.get("premises")!=[
                            list(item) for item in premises
                        ]
                        or contract.get("rule_source_sha256")
                           not in compiled_source_hashes):
                    raise RuntimeError(
                        f"proof channel {root!r} is not safely factorable"
                    )
                topology[root]=premises
            topology_seconds=time.perf_counter()-topology_started
            activation_started=time.perf_counter()
            active_by_case={case:[] for case in missing}
            stored_attrs=getattr(self,"_pair_case_attrs",{})
            for case in missing:
                attrs=stored_attrs.get(case,supplied_attrs.get(case,{}))
                for dependency,channel in proof_roots:
                    premises=topology[(dependency,channel)]
                    if all(attrs.get(predicate)==value
                           for predicate,value in premises):
                        active_by_case[case].append((dependency,channel,premises))
            activation_join_seconds=time.perf_counter()-activation_started
            active_uses=sum(map(len,active_by_case.values()))
            active_templates=list(dict.fromkeys(
                (dependency,channel,premises)
                for case in missing
                for dependency,channel,premises in active_by_case[case]
            ))
            channel_key=lambda item:(*inference_key,item[0],item[1],item[2])
            channel_cache=getattr(self,"_pair_channel_proof_cache",None)
            if channel_cache is None:
                channel_cache={}; self._pair_channel_proof_cache=channel_cache
            query_templates=[item for item in active_templates
                             if channel_key(item) not in channel_cache]
            if len(channel_cache)+len(query_templates)>cache_limit:
                raise RuntimeError(
                    "pair proof-channel cache limit exceeded; promote a fresh "
                    "rule snapshot"
                )
            pair_stats.update(
                channel_root_requests=len(active_templates),
                channel_root_hits=len(active_templates)-len(query_templates),
                channel_root_misses=len(query_templates),
            )
            serialization_started=time.perf_counter(); facts=[]
            loaded_channels=getattr(self,"_loaded_pair_channels",None)
            if loaded_channels is None:
                loaded_channels=set(); self._loaded_pair_channels=loaded_channels
            template_audit=getattr(self,"_pair_channel_templates",None)
            if template_audit is None:
                template_audit={}; self._pair_channel_templates=template_audit
            known_template_identities={
                item["case"]:(
                    item["dependency_id"],item["proof_channel_id"],
                    tuple(tuple(fact) for fact in item["facts"]),
                )
                for item in template_audit.values()
            }
            pending_loaded=set(); pending_audit={}
            for dependency,channel,premises in query_templates:
                template=self.pair_channel_case(dependency,channel,premises)
                loaded_key=(self.version,dependency,channel,premises)
                identity=(dependency,channel,premises)
                previous_identity=known_template_identities.setdefault(template,identity)
                if previous_identity!=identity:
                    raise RuntimeError(
                        f"pair proof template hash collision at {template}"
                    )
                pending_audit[loaded_key]={
                    "case":template,"dependency_id":dependency,
                    "proof_channel_id":channel,
                    "facts":[[predicate,value] for predicate,value in premises],
                }
                if loaded_key in loaded_channels: continue
                for index,(predicate,value) in enumerate(premises,1):
                    facts.append(
                        f'(: fact_{template}_{index}_{predicate} '
                        f'({predicate.title()} {template} {json.dumps(value)}) '
                        '(STV 1.0 1.0))'
                    )
                pending_loaded.add(loaded_key)
            template_serialization_seconds=(
                time.perf_counter()-serialization_started
            )
            insertion_started=time.perf_counter()
            for offset in range(0,len(facts),1000):
                self.engine.add_atoms_no_check(
                    facts[offset:offset+1000],timeout_sec=timeout_sec
                )
            atomspace_insertion_seconds=time.perf_counter()-insertion_started
            inserted_template_atoms=len(facts)
            staged_channel_results={}
            query_started=time.perf_counter()
            for offset in range(0,len(query_templates),batch):
                items=query_templates[offset:offset+batch]
                queries=[
                    f'(: $proof (PairSignal '
                    f'{self.pair_channel_case(dependency,channel,premises)} '
                    f'{json.dumps(dependency)} {json.dumps(channel)}) $tv)'
                    for dependency,channel,premises in items
                ]
                # Each independently mined vote is a real two-hop proof:
                # predicate -> MinedPairPreference -> PairSignal.
                steps=max(10,int(self.config["pair_chain_steps"]))*len(items)
                results=self.engine.query_many(
                    queries,steps=steps,timeout_sec=timeout_sec
                ); calls+=1
                for item,proofs in zip(items,results):
                    staged_channel_results[channel_key(item)]=proofs
            query_seconds=time.perf_counter()-query_started
            candidate_channel_results={
                **channel_cache,**staged_channel_results,
            }
            unproved=[
                (dependency,channel)
                for dependency,channel,premises in active_templates
                if not candidate_channel_results.get(
                    channel_key((dependency,channel,premises))
                )
            ]
            if unproved:
                raise RuntimeError(
                    "active isolated pair channels returned no PeTTa proof: "
                    +", ".join(
                        f"{dependency}/{channel}"
                        for dependency,channel in unproved[:8]
                    )
                )
            # Publish cache and audit state only after every PeTTa mutation and
            # query has succeeded. A retired/failed worker can never leave a
            # false empty proof or a falsely "loaded" fact template behind.
            channel_cache.update(staged_channel_results)
            loaded_channels.update(pending_loaded)
            template_audit.update(pending_audit)
            self._pair_query_roots=getattr(self,"_pair_query_roots",0)+len(query_templates)
            self._pair_pruned_query_roots=(
                getattr(self,"_pair_pruned_query_roots",0)
                +len(missing)*len(proof_roots)-active_uses
            )
            self._pair_channel_activations=(
                getattr(self,"_pair_channel_activations",0)+active_uses
            )
            self._pair_reused_channel_activations=(
                getattr(self,"_pair_reused_channel_activations",0)
                +max(0,active_uses-len(query_templates))
            )
            origins=getattr(self,"_pair_proof_origins",None)
            if origins is None:
                origins={}; self._pair_proof_origins=origins
            for case,active in active_by_case.items():
                candidate_proofs=[]
                for dependency,channel,premises in active:
                    proofs=channel_cache[
                        channel_key((dependency,channel,premises))
                    ]
                    candidate_proofs.extend(proofs)
                    for proof in proofs:
                        origins[(self.version,case,proof)]=dependency
                self._pair_proof_cache[(*inference_key,case)]=candidate_proofs

            self._pair_query_calls+=calls
            pair_stats["hits"]=(pair_stats["case_root_hits"]
                                +pair_stats["channel_root_hits"])
            pair_stats["misses"]=(pair_stats["case_root_misses"]
                                  +pair_stats["channel_root_misses"])
            pair_stats["requests"]=pair_stats["hits"]+pair_stats["misses"]
            self._last_pair_cache_stats=pair_stats
            self._last_pair_reasoning_profile={
                "proof_topology_validation_seconds":topology_seconds,
                "proof_activation_join_seconds":activation_join_seconds,
                "proof_template_serialization_seconds":template_serialization_seconds,
                "proof_atomspace_insertion_seconds":atomspace_insertion_seconds,
                "proof_query_seconds":query_seconds,
                "proof_template_atoms_inserted":inserted_template_atoms,
                "total_seconds":time.perf_counter()-runtime_started,
            }
            return {case:self._pair_proof_cache[(*inference_key,case)] for case in unique},calls

        def _proof_dependency_margins(self,case,proofs,*,cache=None):
            """Return PeTTa confidence-adjusted margins keyed by dependency.

            A PairSignal STV is a probability around the balanced pair prior 0.5.
            Its base linear decision margin is therefore ``c * (2s - 1)``.  The
            log-odds transform preserves strong calibrated evidence. Reading the
            STV from the returned root proof preserves
            PeTTa's CTV propagation and revision semantics instead of reducing the
            reasoner to a Boolean gate.
            """
            if not proofs: return {}
            margin_cache=(self._pair_margin_cache if cache is None else cache)
            cache_key=("dependencies",self.version,case,tuple(proofs))
            if cache_key in margin_cache: return margin_cache[cache_key]
            cache_limit=int(self.config.get(
                "max_proof_cache_entries",DEFAULT_MAX_PROOF_CACHE_ENTRIES
            ))
            if len(margin_cache)>=cache_limit:
                raise RuntimeError(
                    "pair margin cache limit exceeded; promote a fresh rule snapshot"
                )
            margins={}
            origins=getattr(self,"_pair_proof_origins",{})
            for proof in proofs:
                origin=origins.get((self.version,case,proof))
                direct_origin=None
                if origin is None and proof.startswith("(direct-reconstruction "):
                    direct_match=re.match(
                        r'^\(direct-reconstruction \(PairSignal [A-Za-z_][A-Za-z0-9_]* '
                        r'("(?:[^"\\]|\\.)*") ',proof
                    )
                    if direct_match is None:
                        raise RuntimeError(
                            "malformed direct pair reconstruction record"
                        )
                    direct_origin=json.loads(direct_match.group(1))
                dependencies=({origin} if origin is not None else
                              {direct_origin} if direct_origin is not None else
                              set(re.findall(r"\bpair_mined_cluster_\d+\b",proof)))
                strength,confidence=proof_tv(proof)
                posterior=0.5+confidence*(strength-0.5)
                # The champion uses confidence-shrunk PeTTa posterior log odds.
                bounded=max(1e-9,min(1.0-1e-9,posterior))
                proof_margin=math.log(bounded/(1.0-bounded))
                for dependency_id in dependencies:
                    # A dependency cluster can expose several correlated proof
                    # variants. PeTTa supplies their inferred STVs; retain only the
                    # dominant canonical alternative for that evidence source.
                    previous=margins.get(dependency_id)
                    if previous is None or abs(proof_margin)>abs(previous):
                        margins[dependency_id]=proof_margin
            margin_cache[cache_key]=margins
            return margins

        def _proof_vote_margin(self,case,proofs):
            """Sum PeTTaChainer margins once per independent dependency."""
            return sum(self._proof_dependency_margins(case,proofs).values())

        @staticmethod
        def _pair_rule_family(rule):
            """Return a dataset-independent evidence-family label for one rule."""
            # A conjunction containing text-semantic evidence still belongs to the
            # text family.  Keeping this binary is important: a mixed conjunction
            # must not accidentally create a third family and receive another
            # equal-weight vote in ``balanced_rank``.
            if (rule.get("categorical_fact_family")=="llm_format"
                    or rule.get("dependency_owner")=="pair_text_semantic_top3_mean"):
                return "text_semantic"
            return (
                "text_semantic"
                if any(
                    str(predicate).startswith(("pair_text_semantic_","pair_llm_"))
                    or predicate=="pair_rel_concept_continuity_scope"
                       for predicate,_value in rule.get("premises",()))
                else "structured_symbolic"
            )

        def _relational_evaluation_activation_audit(
                self,point_specs,pair_specs):
            """Count relation evidence and rule uses in the held-out workload.

            This runs only after the normal proof calls have completed. The
            selected weighted/proof-margin paths fail closed whenever an active
            isolated channel has no PeTTa proof, so their activation counts are
            proof-backed uses rather than a mined-rule inventory.
            """
            relation_fields=(
                REL_ENTITY_CONTINUITY_SCOPE,
                REL_CONCEPT_CONTINUITY_SCOPE,
            )
            relation_proof_fields={
                REL_ENTITY_CONTINUITY_SCOPE:REL_ENTITY_CONTINUITY_PROOF_IDS,
                REL_CONCEPT_CONTINUITY_SCOPE:REL_CONCEPT_CONTINUITY_PROOF_IDS,
            }
            scope_counts={field:Counter() for field in relation_fields}
            proof_positive_candidates=0
            proof_refs_by_case=getattr(
                self,"_candidate_relational_proof_refs",{}
            )
            for _aid,case,_attrs,raw_attrs in point_specs:
                positive=False
                for field in relation_fields:
                    value=raw_attrs.get(field)
                    if value is not None:
                        scope_counts[field][str(value)]+=1
                    if value in {"recent","older"}:
                        references=(proof_refs_by_case.get(case,{}) or {}).get(
                            relation_proof_fields[field]
                        )
                        if not isinstance(references,(list,tuple)) or not references:
                            raise RuntimeError(
                                "proof-positive relation lacks provenance IDs in "
                                "the held-out candidate context"
                            )
                        positive=True
                proof_positive_candidates+=positive

            def summarize(rules,specs,*,pair):
                relation_predicates={
                    (f"pair_{field}" if pair else field):field
                    for field in relation_fields
                }
                activations=0; relational=0; proof_positive=0
                by_field_value=Counter(); rule_ids=set(); positive_ids=set()
                activated_instances=0; positive_instances=0
                for _case,attrs in specs:
                    instance_relational=False; instance_positive=False
                    for rule in rules:
                        premises=tuple(tuple(item)
                                       for item in rule.get("premises",()))
                        if not all(attrs.get(predicate)==value
                                   for predicate,value in premises):
                            continue
                        activations+=1
                        selected=[
                            (relation_predicates[predicate],str(value))
                            for predicate,value in premises
                            if predicate in relation_predicates
                        ]
                        if not selected:
                            continue
                        relational+=1; instance_relational=True
                        identifier=str(rule.get(
                            "proof_channel_id" if pair else "id",
                            rule.get("id","unknown"),
                        ))
                        rule_ids.add(identifier)
                        positive=any(
                            value in ({"left","right"} if pair
                                      else {"recent","older"})
                            for _field,value in selected
                        )
                        for field,value in selected:
                            by_field_value[(field,value)]+=1
                        if positive:
                            proof_positive+=1; instance_positive=True
                            positive_ids.add(identifier)
                    activated_instances+=instance_relational
                    positive_instances+=instance_positive
                return {
                    "instances":len(specs),
                    "all_rule_activations":activations,
                    "relational_rule_activations":relational,
                    "proof_positive_relational_rule_activations":proof_positive,
                    "instances_with_relational_rule_activation":activated_instances,
                    "instances_with_proof_positive_relational_rule_activation":(
                        positive_instances
                    ),
                    "relational_rule_ids":sorted(rule_ids),
                    "proof_positive_relational_rule_ids":sorted(positive_ids),
                    "by_field_and_value":{
                        f"{field}={value}":count
                        for (field,value),count in sorted(by_field_value.items())
                    },
                }

            point_inputs=[(case,attrs)
                          for _aid,case,attrs,_raw in point_specs]
            result={
                "candidate_contexts":len(point_specs),
                "proof_positive_candidate_contexts":proof_positive_candidates,
                "candidate_scope_values":{
                    field:dict(sorted(values.items()))
                    for field,values in scope_counts.items()
                },
                "point":summarize(
                    self.mined_rules,point_inputs,pair=False
                ),
                "pair":summarize(
                    self.pair_rules,list(pair_specs),pair=True
                ),
                "proof_positive_definition":{
                    "point":"recent or older derived scope",
                    "pair":"left or right comparison of two known scopes",
                },
                "counting_units":{
                    "point":"candidate-rule activation",
                    "pair":"oriented-comparison-rule activation",
                },
                "proof_gate_complete":bool(
                    self.config.get("aggregation")=="weighted"
                    and self._last_point_completeness.get("complete")
                    and (
                        not pair_specs
                        or self.config.get("pair_aggregation")=="proof_margin"
                    )
                ),
            }
            return result

        @staticmethod
        def _symbolic_rule_families(rule):
            """Fixed portable families; resolve ownership per dependency below."""
            families=set()
            for predicate,_value in rule.get("premises",()):
                if "lexical" in predicate or "title" in predicate:
                    families.add("lexical")
                elif "transition" in predicate:
                    families.add("transition")
                else:
                    families.add("interest")
            return families or {"interest"}

        @staticmethod
        def _midrank_scores(values):
            """Map descending values to stable [0,1] midranks."""
            # Algebraically equal vote sums can differ at floating-point epsilon.
            # Do not turn that addition-order noise into a whole rank position.
            # This precision is finer than the public eight-decimal score contract.
            values=[round(float(value),12) for value in values]
            scores=[0.0]*len(values)
            denominator=max(1,len(values)-1)
            ordered=sorted(range(len(values)),key=lambda index:(-values[index],index))
            position=0
            while position<len(ordered):
                end=position+1
                value=values[ordered[position]]
                while end<len(ordered) and values[ordered[end]]==value:
                    end+=1
                rank_score=1.0-((position+end-1)/2)/denominator
                for offset in range(position,end):
                    scores[ordered[offset]]=rank_score
                position=end
            return scores

        @staticmethod
        def _logistic(value):
            """Map an arbitrary signed proof margin to a bounded probability."""
            # 0.5 + 0.5*tanh(x/2) is algebraically sigmoid(x), but remains stable
            # when a near-certain PeTTa posterior produces a large log-odds margin.
            return 0.5+0.5*math.tanh(float(value)/2.0)

        def _pairwise_plan(self,rows,specs):
            planning_started=time.perf_counter()
            self._compile_pair_rule_matcher()
            raw_by_article={aid:(self.article(aid),raw_attrs)
                            for aid,_case,_attrs,raw_attrs in specs}
            comparisons=[]; pair_specs=[]
            comparison_limit=int(self.config["max_pair_comparisons"])
            graph_started=time.perf_counter()
            exhaustive_count=len(rows)*(len(rows)-1)//2
            if exhaustive_count>comparison_limit:
                raise ValueError(
                    "pairwise comparison budget exceeded: "
                    f"{exhaustive_count} unordered pairs for {len(rows)} "
                    f"candidates is greater than max_pair_comparisons="
                    f"{comparison_limit}"
                )
            index_pairs=(
                (left,right) for left in range(len(rows))
                for right in range(left+1,len(rows))
            )
            graph_seconds=time.perf_counter()-graph_started
            feature_seconds=0.0; reverse_seconds=0.0; bounding_seconds=0.0
            activation_seconds=0.0; cache_seconds=0.0; serialization_seconds=0.0
            activation_case_cache={}
            for left_index,right_index in index_pairs:
                left_id=rows[left_index]["article"]["id"]
                right_id=rows[right_index]["article"]["id"]
                # Preserve instance-level test/instrumentation overrides. Normal
                # execution uses the exact one-pass antisymmetric constructor.
                if "_pair_spec" in self.__dict__:
                    forward=self._pair_spec(raw_by_article[left_id],raw_by_article[right_id])
                    reverse=self._pair_spec(raw_by_article[right_id],raw_by_article[left_id])
                else:
                    forward,reverse,pair_profile=self._oriented_pair_specs(
                        raw_by_article[left_id],raw_by_article[right_id],
                        activation_case_cache,
                    )
                    feature_seconds+=pair_profile["pair_feature_derivation_seconds"]
                    reverse_seconds+=pair_profile["reverse_orientation_derivation_seconds"]
                    bounding_seconds+=pair_profile["pair_feature_bounding_seconds"]
                    activation_seconds+=pair_profile["pair_activation_join_seconds"]
                    cache_seconds+=pair_profile["pair_activation_case_cache_seconds"]
                    serialization_seconds+=pair_profile["pair_case_serialization_seconds"]
                comparisons.append((left_index,right_index,forward[0],reverse[0]))
                pair_specs.extend((forward,reverse))
            total_seconds=time.perf_counter()-planning_started
            self._last_pair_plan_profile={
                "candidate_feature_reuse":True,
                "orientation_policy":"derive_reverse_from_exact_antisymmetry",
                "unordered_comparisons":len(comparisons),
                "oriented_pair_cases":len(pair_specs),
                "unique_activation_cases_in_slate":len(activation_case_cache),
                "active_pair_predicates":len(self._pair_active_predicates),
                "mining_pair_predicates":len(getattr(
                    self,"_pair_feature_vocabulary",{}
                )),
                "comparison_graph_seconds":graph_seconds,
                "pair_feature_derivation_seconds":feature_seconds,
                "reverse_orientation_derivation_seconds":reverse_seconds,
                "pair_feature_bounding_seconds":bounding_seconds,
                "pair_activation_join_seconds":activation_seconds,
                "pair_activation_case_cache_seconds":cache_seconds,
                "pair_case_serialization_seconds":serialization_seconds,
                "pair_plan_assembly_seconds":max(0.0,total_seconds-graph_seconds
                    -feature_seconds-reverse_seconds-bounding_seconds
                    -activation_seconds-cache_seconds-serialization_seconds),
                "total_seconds":total_seconds,
            }
            return comparisons,pair_specs

        def _pairwise_rank(self,rows,specs,plan=None,proof_map=None,
                           reasoner_timeout_sec=None):
            """Rank a slate with the champion balanced-family proof tournament."""
            if not self.pair_rules or len(rows)<2:
                self._last_pair_replay=None
                for row in rows:
                    row.update(
                        ranking_score=row["score"],pairwise_score=None,
                        pairwise_margin_score=None,
                        pairwise_proof_coverage=0.0,
                        pairwise_directional_coverage=0.0,
                    )
                return rows

            comparisons,pair_specs=plan or self._pairwise_plan(rows,specs)
            if proof_map is None:
                self._ensure_pair_specs(
                    pair_specs,timeout_sec=reasoner_timeout_sec
                )
                proof_map,_calls=self._proofs_for_pair_specs(
                    pair_specs,timeout_sec=reasoner_timeout_sec
                )
            self._last_pair_replay={
                "version":self.version,
                "edges":tuple(
                    (str(rows[left]["article"]["id"]),
                     str(rows[right]["article"]["id"]),forward,reverse)
                    for left,right,forward,reverse in comparisons
                ),
                "proof_map":proof_map,
            }

            count=len(rows)
            pair_totals=[0.0]*count
            comparison_counts=[0]*count
            covered=[0]*count
            directional_covered=[0]*count

            dependency_families={}
            cluster_weights={}
            for rule in self.pair_rules:
                dependency=rule.get("dependency_id",rule["id"])
                family=self._pair_rule_family(rule)
                if (family=="text_semantic"
                        or dependency not in dependency_families):
                    dependency_families[dependency]=family
                posterior=0.5+float(rule.get(
                    "proof_confidence",rule.get("confidence",0.0)
                ))*(float(rule.get(
                    "proof_strength",rule.get("strength",0.5)
                ))-0.5)
                bounded=max(1e-9,min(1.0-1e-9,posterior))
                margin=abs(math.log(bounded/(1.0-bounded)))
                cluster_weights[dependency]=max(
                    cluster_weights.get(dependency,0.0),margin
                )

            active_families=tuple(sorted(set(dependency_families.values())))
            family_totals={
                family:[0.0]*count for family in active_families
            }
            for left,right,forward_case,reverse_case in comparisons:
                forward_proofs=proof_map[forward_case]
                reverse_proofs=proof_map[reverse_case]
                forward_margins=self._proof_dependency_margins(
                    forward_case,forward_proofs
                )
                reverse_margins=self._proof_dependency_margins(
                    reverse_case,reverse_proofs
                )
                total=0.0
                for dependency in sorted(
                        set(forward_margins)|set(reverse_margins)):
                    margin=(forward_margins.get(dependency,0.0)
                            -reverse_margins.get(dependency,0.0))
                    total+=margin
                    family=dependency_families.get(
                        dependency,"structured_symbolic"
                    )
                    family_totals.setdefault(family,[0.0]*count)
                    family_totals[family][left]+=margin
                    family_totals[family][right]-=margin
                pair_totals[left]+=total
                pair_totals[right]-=total
                comparison_counts[left]+=1
                comparison_counts[right]+=1
                if forward_proofs or reverse_proofs:
                    covered[left]+=1
                    covered[right]+=1
                directional_covered[left]+=bool(forward_proofs)
                directional_covered[right]+=bool(reverse_proofs)

            raw_pair_values=[
                pair_totals[index]/max(1,comparison_counts[index])
                for index in range(count)
            ]
            family_rank_scores={}
            family_pair_values={}
            for family,totals in family_totals.items():
                raw=[
                    totals[index]/max(1,comparison_counts[index])
                    for index in range(count)
                ]
                family_max=max(1e-12,sum(
                    weight for dependency,weight in cluster_weights.items()
                    if dependency_families.get(
                        dependency,"structured_symbolic"
                    )==family
                ))
                family_pair_values[family]=[
                    0.5+0.5*value/family_max for value in raw
                ]
                family_rank_scores[family]=self._midrank_scores(raw)

            family_count=max(1,len(family_rank_scores))
            pair_values=[
                sum(values[index] for values in family_pair_values.values())
                /family_count
                for index in range(count)
            ]
            pair_rank_scores=[
                sum(values[index] for values in family_rank_scores.values())
                /family_count
                for index in range(count)
            ]

            # Point proofs are a tie-aware live-feedback channel.  The frozen
            # offline champion gives the pair tournament full weight; a proved
            # recent skip temporarily reserves one quarter for point evidence.
            rank_denominator=max(1,count-1)
            point_ranks=[0.0]*count
            position=0
            while position<count:
                end=position+1
                signature=(
                    rows[position]["score"],
                    rows[position]["stv"]["strength"],
                    rows[position]["stv"]["confidence"],
                )
                while end<count and (
                    rows[end]["score"],
                    rows[end]["stv"]["strength"],
                    rows[end]["stv"]["confidence"],
                )==signature:
                    end+=1
                rank=1.0-(position+end-1)/2/rank_denominator
                for index in range(position,end):
                    point_ranks[index]=rank
                position=end

            feedback_active=any(
                row.get("feedback_evidence",{}).get("rule_ids")
                for row in rows
            )
            pair_weight=0.75 if feedback_active else 1.0
            for index,row in enumerate(rows):
                denominator=max(1,comparison_counts[index])
                row.update(
                    pairwise_score=round(pair_values[index],8),
                    pairwise_margin_score=round(raw_pair_values[index],8),
                    pairwise_rank_score=round(pair_rank_scores[index],8),
                    pointwise_rank_score=round(point_ranks[index],8),
                    ranking_score=round(
                        pair_weight*pair_rank_scores[index]
                        +(1.0-pair_weight)*point_ranks[index],8
                    ),
                    pairwise_proof_coverage=round(
                        covered[index]/denominator,8
                    ),
                    pairwise_directional_coverage=round(
                        directional_covered[index]/denominator,8
                    ),
                    pairwise_weight_applied=pair_weight,
                    live_feedback_rank_fusion=feedback_active,
                    pairwise_family_rank_scores={
                        family:round(values[index],8)
                        for family,values in family_rank_scores.items()
                    },
                )
            rows.sort(key=lambda row:(
                -row["ranking_score"],-row["pairwise_score"],
                -row["score"],-row["stv"]["strength"],
                -row["stv"]["confidence"],
                -row["tie_break"]["topic_prior"],
                -row["tie_break"]["format_prior"],
                -row["tie_break"]["subcategory_prior"],
                row["article"]["id"],
            ))
            return rows

        def _pairwise_replay_plan(self,rows,replay):
            """Build a sub-slate tournament from already proved pair edges."""
            if not replay or replay.get("version")!=self.version:
                return None
            positions={str(row["article"]["id"]):index
                       for index,row in enumerate(rows)}
            comparisons=[]
            for left_id,right_id,forward,reverse in replay.get("edges",()):
                if left_id not in positions or right_id not in positions:
                    continue
                comparisons.append((positions[left_id],positions[right_id],
                                    forward,reverse))
            expected=len(rows)*(len(rows)-1)//2
            if len(comparisons)!=expected:
                return None
            proof_map=replay.get("proof_map",{})
            required={case for _left,_right,forward,reverse in comparisons
                      for case in (forward,reverse)}
            if not required.issubset(proof_map):
                return None
            return (comparisons,()),proof_map

        def _rank(self,specs,groups,limit,apply_pairwise=True,
                  reasoner_timeout_sec=None,include_context=False):
            popularity=self._popularity; rows=[]; prior=float(self._click_base_rate)
            for (aid,_case,_attrs,_raw_attrs),proofs in zip(specs,groups):
                scored=[(proof_tv(proof),proof) for proof in proofs]
                inference_tv,_inference_proof=max(
                    scored,
                    key=lambda item:prior+item[0][1]*(item[0][0]-prior),
                    default=((0.0,0.0),""),
                )
                feedback_scored=[
                    (proof_tv(proof),proof) for proof in proofs
                    if LIVE_NEGATIVE_RULE_IDS.intersection(
                        re.findall(r"\bfeedback_skip_[a-z]+\b",proof)
                    )
                ]
                feedback_rule_ids=sorted(set().union(*(
                    LIVE_NEGATIVE_RULE_IDS.intersection(
                        re.findall(r"\bfeedback_skip_[a-z]+\b",proof)
                    ) for _tv,proof in feedback_scored
                ))) if feedback_scored else []

                proof_rule_ids=set().union(*(
                    set(re.findall(r"\bmined_\d+\b",proof))
                    for _tv,proof in scored
                )) if scored else set()
                fired=[
                    rule for rule in self.mined_rules
                    if rule["id"] in proof_rule_ids
                ]
                if fired:
                    weights=[
                        max(1,int(rule.get(
                            "specificity",len(rule["premises"])
                        ))) for rule in fired
                    ]
                    evidence_confidence=sum(
                        rule["confidence"]*weight
                        for rule,weight in zip(fired,weights)
                    )/sum(weights)
                    score=sum(
                        (prior+rule["confidence"]*(
                            rule["strength"]-prior
                        ))*weight
                        for rule,weight in zip(fired,weights)
                    )/sum(weights)
                elif scored:
                    values=[value for value,_proof in scored]
                    evidence_confidence=sum(
                        value[1] for value in values
                    )/len(values)
                    score=sum(
                        prior+value[1]*(value[0]-prior)
                        for value in values
                    )/len(values)
                else:
                    evidence_confidence=0.0
                    score=prior
                strength=(prior+(score-prior)/evidence_confidence
                          if evidence_confidence else prior)
                tv=(max(0.0,min(1.0,strength)),evidence_confidence)
                score_method="proof_gated_base_rate_posterior"

                if feedback_scored:
                    tv,_feedback_proof=max(
                        feedback_scored,
                        key=lambda item:(item[0][1],abs(item[0][0]-prior)),
                    )
                    score=prior+tv[1]*(tv[0]-prior)
                    score_method="pettachainer_live_feedback_revision"

                row={
                    "article":self.public_article(aid),
                    "score":round(score,8),
                    "stv":{"strength":tv[0],"confidence":tv[1]},
                    "inference_stv":{
                        "strength":inference_tv[0],
                        "confidence":inference_tv[1],
                    },
                    "baseline":popularity[aid],
                    "rules":fired,
                    "proofs":proofs,
                    "engine":"PeTTaChainer",
                    "aggregation":"weighted",
                    "score_method":score_method,
                    "tie_break":self._tie_break(self.article(aid)),
                    "feedback_evidence":{
                        "match":_raw_attrs.get(
                            LIVE_NEGATIVE_FEATURE,"none"
                        ),
                        "rule_ids":feedback_rule_ids,
                        "proof_stv":{
                            "strength":tv[0],"confidence":tv[1]
                        } if feedback_rule_ids else None,
                    },
                }
                if include_context:
                    row["_prepared_context"]=_raw_attrs
                rows.append(row)
            # Popularity is diagnostic metadata only. Ranking is proof-gated, with
            # the training-only editorial priors used only when proof values tie.
            # Article id remains the final deterministic tie-break.
            rows.sort(key=lambda r:(-r["score"],-r["stv"]["strength"],-r["stv"]["confidence"],
                                    -r["tie_break"]["topic_prior"],
                                    -r["tie_break"]["format_prior"],
                                    -r["tie_break"]["subcategory_prior"],r["article"]["id"]))
            if apply_pairwise:
                rows=self._pairwise_rank(
                    rows,specs,reasoner_timeout_sec=reasoner_timeout_sec
                )
            return rows if limit==0 else rows[:limit]

        def score(self,user,candidates=None,contexts=None,limit=None,
                  include_context=False,apply_pairwise=True,cache_result=True):
            score_started=time.perf_counter()
            if user not in self.data["users"]: raise ValueError(f"unknown user: {user}")
            if candidates is None:
                candidates=self.default_candidates(user)
                if contexts is None: contexts=self.default_candidate_context(user)
            candidates=list(dict.fromkeys(str(aid) for aid in candidates))
            for aid in candidates: self.article(aid)
            contexts=contexts or {}; limit=self.config["top_k"] if limit is None else int(limit)
            # Proof references do not become miner predicates, but they are part
            # of a candidate's provenance identity. Without them, equal categorical
            # scopes from different history origins could reuse a cached row that
            # cites the wrong proof ledger records.
            proof_fields=tuple(RELATIONAL_PROOF_FIELDS.values())
            context_key_started=time.perf_counter(); context_key=[]
            for aid in candidates:
                candidate_context=contexts.get(aid) or {}
                references=self._relational_proof_references(candidate_context)
                context_key.append((
                    aid,
                    tuple((feature,candidate_context.get(feature))
                          for feature in CONTEXT_FEATURES),
                    tuple((proof_field,references.get(proof_field,()))
                          for proof_field in proof_fields),
                ))
            context_key=tuple(context_key)
            context_key_seconds=time.perf_counter()-context_key_started
            key=(self.version,user,tuple(candidates),context_key,limit,
                 bool(include_context),bool(apply_pairwise))
            if cache_result and key in self.feed_cache:
                self._feed_rank_cache_hits=(
                    getattr(self,"_feed_rank_cache_hits",0)+1
                )
                cached=self.feed_cache.pop(key); self.feed_cache[key]=cached; return cached
            self._feed_rank_cache_misses=(
                getattr(self,"_feed_rank_cache_misses",0)+1
            )
            timeout_sec=self._serving_reasoner_timeout()
            candidate_started=time.perf_counter()
            specs=self._candidate_specs(user,candidates,contexts)
            candidate_seconds=time.perf_counter()-candidate_started
            point_started=time.perf_counter()
            self._ensure_candidate_specs(specs,timeout_sec=timeout_sec)
            groups,_calls=self._proofs_for_specs(specs,timeout_sec=timeout_sec)
            point_seconds=time.perf_counter()-point_started
            rank_started=time.perf_counter()
            ranked=self._rank(
                specs,groups,limit,reasoner_timeout_sec=timeout_sec,
                include_context=include_context,apply_pairwise=apply_pairwise,
            )
            rank_seconds=time.perf_counter()-rank_started
            if cache_result:
                self.feed_cache[key]=ranked
                while len(self.feed_cache)>256:
                    self.feed_cache.popitem(last=False)
            self._last_live_score_profile={
                "candidates":len(candidates),
                "apply_pairwise":bool(apply_pairwise),
                "context_key_seconds":round(context_key_seconds,6),
                "candidate_preparation_seconds":round(candidate_seconds,6),
                "point_reasoning_seconds":round(point_seconds,6),
                "pair_ranking_seconds":round(rank_seconds,6),
                "total_seconds":round(time.perf_counter()-score_started,6),
                "point_reasoner_query_calls":_calls,
                "point_materialization":{
                    key:(round(value,6) if isinstance(value,float) else value)
                    for key,value in getattr(
                        self,"_last_point_materialization_profile",{}
                    ).items()
                },
                "candidate_plan":{
                    key:(round(value,6) if isinstance(value,float) else value)
                    for key,value in getattr(
                        self,"_last_candidate_preparation_profile",{}
                    ).items()
                },
                "pair_plan":({
                    key:(round(value,6) if isinstance(value,float) else value)
                    for key,value in getattr(
                        self,"_last_pair_plan_profile",{}
                    ).items()
                } if apply_pairwise else {}),
            }
            return ranked


    return PairwiseRankingMixin
