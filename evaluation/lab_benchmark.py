"""Offline evaluation methods shared by the live recommendation lab."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
import uuid
from collections import Counter

from ..mining.rule_parser import POSITIVE, proof_tv


PAIR_PROOF_PREPARATION_TIMERS = (
    "proof_topology_validation_seconds",
    "proof_activation_join_seconds",
    "proof_template_serialization_seconds",
    "proof_atomspace_insertion_seconds",
)


def _proof_channel_preparation_profile(reasoning_profile):
    """Project reasoner profiling to non-overlapping pre-query preparation."""
    preparation_seconds=math.fsum(
        float(reasoning_profile.get(key,0.0))
        for key in PAIR_PROOF_PREPARATION_TIMERS
    )
    return {
        **{key:(round(value,6) if isinstance(value,float) else value)
           for key,value in reasoning_profile.items()
           if key not in {"proof_query_seconds","total_seconds"}},
        "pre_query_component_sum_seconds":round(preparation_seconds,6),
        "proof_query_seconds_excluded":round(float(
            reasoning_profile.get("proof_query_seconds",0.0)
        ),6),
        "definition":(
            "Validates proof topology, joins active rules, serializes "
            "alpha-normalized templates and inserts only new template "
            "atoms. The separately reported proof-query time is not "
            "pair preparation."
        ),
    }


def make_benchmark_mixin(
    *,
    CHALLENGER_CONFIG_KEYS,
    CONTEXT_FEATURES,
    MINING_CONFIG_KEYS,
    PAIR_FEATURE_PROFILES,
    PAIR_MARGIN_POWER_MAX,
    PAIR_MARGIN_POWER_MIN,
):
    class BenchmarkMixin:
        @staticmethod
        def _bounded_candidates(case,cap,seed):
            candidates=list(case["candidates"])
            if not cap or len(candidates)<=cap: return candidates
            stable=f"{seed}:{case.get('id')}".encode(); local=random.Random(int.from_bytes(hashlib.sha256(stable).digest()[:8],"big"))
            selected=set(local.sample(candidates,min(len(candidates),cap)))
            return [aid for aid in candidates if aid in selected]

        @staticmethod
        def _bootstrap_interval_raw(values,seed,repetitions=1000):
            if not values: return None
            local=random.Random(seed); count=len(values); estimates=[]
            for _ in range(repetitions):
                estimates.append(sum(values[local.randrange(count)] for _ in range(count))/count)
            estimates.sort()
            return (estimates[int(0.025*(repetitions-1))],
                    estimates[int(0.975*(repetitions-1))])

        @staticmethod
        def _bootstrap_interval(values,seed,repetitions=1000):
            interval=BenchmarkMixin._bootstrap_interval_raw(values,seed,repetitions)
            return ([round(interval[0],4),round(interval[1],4)]
                    if interval is not None else None)

        def _balanced_margin_probability_evaluator(self):
            """Build the exact pre-tournament balanced-margin pair evaluator."""
            if (self.config.get("pair_aggregation")!="proof_margin"
                    or self.config.get("pair_family_fusion")!="balanced_margin"):
                return None
            dependency_families={}
            for rule in self.pair_rules:
                dependency_id=rule.get("dependency_id",rule["id"])
                family=self._pair_rule_family(rule)
                if (family=="text_semantic"
                        or dependency_id not in dependency_families):
                    dependency_families[dependency_id]=family
            # Reliability runs before the timed production tournament. It must not
            # pre-populate the live margin cache and turn a cold ranking into a
            # cache-hit measurement.
            reliability_margin_cache={}

            def evaluate(forward_case,reverse_case,forward_proofs,reverse_proofs):
                forward_margins=self._proof_dependency_margins(
                    forward_case,forward_proofs,cache=reliability_margin_cache
                )
                reverse_margins=self._proof_dependency_margins(
                    reverse_case,reverse_proofs,cache=reliability_margin_cache
                )
                family_totals={}; family_counts={}
                for dependency_id in sorted(
                        set(forward_margins)|set(reverse_margins)):
                    dependency_margin=(
                        forward_margins.get(dependency_id,0.0)
                        -reverse_margins.get(dependency_id,0.0)
                    )
                    family=dependency_families.get(
                        dependency_id,"structured_symbolic"
                    )
                    family_totals[family]=(
                        family_totals.get(family,0.0)+dependency_margin
                    )
                    family_counts[family]=family_counts.get(family,0)+1
                if not family_totals:
                    return None
                active_family_margins=[
                    total/family_counts[family]
                    for family,total in family_totals.items()
                ]
                fused_margin=(
                    sum(active_family_margins)/len(active_family_margins)
                )
                return self._logistic(fused_margin)

            return evaluate

        @staticmethod
        def _pair_proof_reliability(
                prepared,planned,proof_map,bins=10,seed=7,
                comparison_probability=None):
            """Held-out reliability of individual active ``PairSignal`` proofs.

            The final tournament is an ordinal ranker and is not presented as a
            calibrated probability.  This diagnostic instead evaluates the only
            probability-bearing objects in that path: returned pair-proof STVs,
            converted to ``0.5 + confidence * (strength - 0.5)``.  Every source
            impression has total weight one, so mirrored orientations and large
            slates cannot masquerade as independent samples.
            """
            bins=int(bins)
            if bins<2 or bins>100:
                raise ValueError("pair proof reliability bins must be in [2, 100]")
            by_impression={}; aggregated_by_impression={}
            eligible_roots=proved_roots=0
            eligible_comparisons=active_aggregated_comparisons=0
            for (case,_specs,_start,_end),(rows,plan) in zip(prepared,planned):
                relevant=set(case["relevant"])
                identity=str(case.get("source_impression_id") or case.get("id"))
                observations=[]; aggregated_observations=[]
                comparisons,_pair_specs=plan
                for left_index,right_index,forward_case,reverse_case in comparisons:
                    left_relevant=(rows[left_index]["article"]["id"] in relevant)
                    right_relevant=(rows[right_index]["article"]["id"] in relevant)
                    # The mined target is defined only for clicked-versus-
                    # nonclicked pairs from one closed impression.
                    if left_relevant==right_relevant:
                        continue
                    eligible_comparisons+=1
                    forward_proofs=proof_map.get(forward_case,())
                    reverse_proofs=proof_map.get(reverse_case,())
                    for pair_case,target,proofs in (
                        (forward_case,left_relevant,forward_proofs),
                        (reverse_case,right_relevant,reverse_proofs),
                    ):
                        eligible_roots+=1
                        proved_roots+=bool(proofs)
                        for proof in proofs:
                            strength,confidence=proof_tv(proof)
                            posterior=0.5+confidence*(strength-0.5)
                            observations.append((
                                max(0.0,min(1.0,posterior)),
                                max(0.0,min(1.0,strength)),
                                1.0 if target else 0.0,
                            ))
                    if comparison_probability is not None:
                        probability=comparison_probability(
                            forward_case,reverse_case,
                            forward_proofs,reverse_proofs,
                        )
                        if probability is not None:
                            active_aggregated_comparisons+=1
                            probability=max(0.0,min(1.0,float(probability)))
                            aggregated_observations.append((
                                probability,probability,
                                1.0 if left_relevant else 0.0,
                            ))
                if observations:
                    by_impression.setdefault(identity,[]).extend(observations)
                if aggregated_observations:
                    aggregated_by_impression.setdefault(identity,[]).extend(
                        aggregated_observations
                    )
            evaluated_probability={
                "name":"active_pair_signal_left_clicked_probability",
                "definition":"q = 0.5 + confidence * (strength - 0.5)",
                "event":"the left candidate is clicked",
                "conditioning":(
                    "a held-out same-impression pair contains exactly one clicked "
                    "and one exposed nonclicked candidate, and this individual "
                    "PairSignal rule is active"
                ),
                "unit":"one returned active PairSignal proof",
                "ordinal_ranking_used":False,
            }
            if not by_impression:
                return {
                    "status":"no_active_pair_proofs",
                    "evaluated_probability":evaluated_probability,
                    "target_population":(
                        "held-out same-impression clicked-versus-nonclicked "
                        "orientations only"
                    ),
                    "eligible_direction_roots":eligible_roots,
                    "proved_direction_roots":proved_roots,
                    "aggregated_balanced_margin_pair_preference":{
                        "status":(
                            "no_active_balanced_margin_comparisons"
                            if comparison_probability is not None
                            else "not_applicable"
                        ),
                        "eligible_comparisons":eligible_comparisons,
                        "active_comparisons":active_aggregated_comparisons,
                    },
                }

            def reliability(groups,column=None,constant=None):
                if (column is None)==(constant is None):
                    raise ValueError(
                        "reliability requires exactly one prediction column or constant"
                    )
                brier=[]; log_loss=[]; buckets=[[] for _ in range(bins)]
                for observations in groups.values():
                    local_brier=[]; local_log=[]
                    weight=1.0/len(observations)
                    for row in observations:
                        prediction=(float(constant) if constant is not None
                                    else row[column])
                        target=row[2]
                        local_brier.append((prediction-target)**2)
                        bounded=max(1e-12,min(1.0-1e-12,prediction))
                        local_log.append(-(
                            target*math.log(bounded)
                            +(1.0-target)*math.log(1.0-bounded)
                        ))
                        bucket=min(bins-1,int(prediction*bins))
                        buckets[bucket].append((prediction,target,weight))
                    brier.append(math.fsum(local_brier)/len(local_brier))
                    log_loss.append(math.fsum(local_log)/len(local_log))
                total_weight=float(len(groups))
                table=[]; ece=0.0
                for index,values in enumerate(buckets):
                    mass=math.fsum(value[2] for value in values)
                    if not mass:
                        continue
                    mean_prediction=math.fsum(
                        value[0]*value[2] for value in values
                    )/mass
                    observed_rate=math.fsum(
                        value[1]*value[2] for value in values
                    )/mass
                    ece+=(mass/total_weight)*abs(
                        mean_prediction-observed_rate
                    )
                    table.append({
                        "lower":index/bins,"upper":(index+1)/bins,
                        "impression_weight_mass":mass,
                        "mean_prediction":mean_prediction,
                        "observed_target_rate":observed_rate,
                        "absolute_gap":abs(mean_prediction-observed_rate),
                    })
                return {
                    "macro_impression_brier":math.fsum(brier)/len(brier),
                    "macro_impression_log_loss":math.fsum(log_loss)/len(log_loss),
                    "equal_impression_weighted_ece":ece,
                    "reliability_bins":table,
                },{"brier":brier,"log_loss":log_loss}

            empirical_active_proof_rate=math.fsum(
                math.fsum(row[2] for row in observations)/len(observations)
                for observations in by_impression.values()
            )/len(by_impression)
            shrunk,shrunk_components=reliability(by_impression,column=0)
            raw_strength,raw_strength_components=reliability(by_impression,column=1)
            balanced_constant,balanced_components=reliability(
                by_impression,constant=0.5
            )
            empirical_constant,empirical_components=reliability(
                by_impression,
                constant=empirical_active_proof_rate
            )

            def paired_deltas(left,right,seed_offset):
                brier=[left_value-right_value
                       for left_value,right_value in zip(
                           left["brier"],right["brier"]
                       )]
                log_loss=[left_value-right_value
                          for left_value,right_value in zip(
                              left["log_loss"],right["log_loss"]
                          )]
                return {
                    "macro_impression_brier":math.fsum(brier)/len(brier),
                    "macro_impression_brier_95_ci":BenchmarkMixin._bootstrap_interval(
                        brier,int(seed)^seed_offset
                    ),
                    "macro_impression_log_loss":math.fsum(log_loss)/len(log_loss),
                    "macro_impression_log_loss_95_ci":BenchmarkMixin._bootstrap_interval(
                        log_loss,int(seed)^(seed_offset<<1)
                    ),
                    "interpretation":"negative favors the first named predictor",
                }

            def descriptive_deltas(left,right):
                brier=[left_value-right_value
                       for left_value,right_value in zip(
                           left["brier"],right["brier"]
                       )]
                log_loss=[left_value-right_value
                          for left_value,right_value in zip(
                              left["log_loss"],right["log_loss"]
                          )]
                return {
                    "macro_impression_brier":math.fsum(brier)/len(brier),
                    "macro_impression_log_loss":math.fsum(log_loss)/len(log_loss),
                    "inference":(
                        "point estimate only; no confidence interval because the "
                        "empirical constant was fitted on this same cohort"
                    ),
                    "interpretation":"negative favors the first named predictor",
                }

            shrunk_minus_raw=paired_deltas(
                shrunk_components,raw_strength_components,0xB13E2
            )
            shrunk_minus_balanced=paired_deltas(
                shrunk_components,balanced_components,0xB4500
            )
            raw_minus_balanced=paired_deltas(
                raw_strength_components,balanced_components,0xBA500
            )
            shrunk_minus_empirical=descriptive_deltas(
                shrunk_components,empirical_components
            )
            aggregated={
                "status":"not_applicable",
                "reason":(
                    "requires pair_aggregation=proof_margin and "
                    "pair_family_fusion=balanced_margin"
                ),
                "eligible_comparisons":eligible_comparisons,
            }
            if comparison_probability is not None:
                aggregated={
                    "status":"no_active_balanced_margin_comparisons",
                    "eligible_comparisons":eligible_comparisons,
                    "active_comparisons":active_aggregated_comparisons,
                }
                if aggregated_by_impression:
                    empirical_aggregate_rate=math.fsum(
                        math.fsum(row[2] for row in observations)/len(observations)
                        for observations in aggregated_by_impression.values()
                    )/len(aggregated_by_impression)
                    aggregate_probability,aggregate_components=reliability(
                        aggregated_by_impression,column=0
                    )
                    aggregate_balanced,aggregate_balanced_components=reliability(
                        aggregated_by_impression,constant=0.5
                    )
                    aggregate_empirical,aggregate_empirical_components=reliability(
                        aggregated_by_impression,constant=empirical_aggregate_rate
                    )
                    aggregated={
                        "status":"computed",
                        "evaluated_probability":{
                            "name":"balanced_margin_left_clicked_probability",
                            "definition":(
                                "q_pair = sigmoid(mean over active families of "
                                "the mean forward-minus-reverse dependency margin)"
                            ),
                            "event":"the left candidate is clicked",
                            "conditioning":(
                                "a held-out same-impression pair contains exactly "
                                "one clicked and one exposed nonclicked candidate, "
                                "and at least one dependency proof is active"
                            ),
                            "unit":"one unordered candidate comparison",
                            "ordinal_ranking_used":False,
                            "calibration_claim":(
                                "probability-shaped pre-tournament diagnostic; "
                                "reliability is measured, not assumed"
                            ),
                        },
                        "metric_weighting":(
                            "macro mean of within-impression active-comparison means"
                        ),
                        "impressions":len(aggregated_by_impression),
                        "eligible_comparisons":eligible_comparisons,
                        "active_comparisons":active_aggregated_comparisons,
                        "active_comparison_coverage":(
                            active_aggregated_comparisons/eligible_comparisons
                            if eligible_comparisons else 0.0
                        ),
                        "confidence_aware_probability":aggregate_probability,
                        "constant_probability_baselines":{
                            "constant_0_5_neutral_pair_preference":{
                                "probability":0.5,
                                "basis":"neutral left-versus-right pair preference",
                                **aggregate_balanced,
                            },
                            "constant_empirical_left_click_rate":{
                                "probability":empirical_aggregate_rate,
                                "basis":(
                                    "equal-impression left-click target rate among "
                                    "the same active comparisons"
                                ),
                                "role":(
                                    "descriptive same-cohort fitted reference, not "
                                    "a deployable or independently estimated baseline"
                                ),
                                **aggregate_empirical,
                            },
                        },
                        "paired_deltas":{
                            "confidence_aware_minus_constant_0_5":paired_deltas(
                                aggregate_components,aggregate_balanced_components,
                                0xA6650
                            ),
                        },
                        "descriptive_same_cohort_deltas":{
                            "confidence_aware_minus_empirical_constant":descriptive_deltas(
                                aggregate_components,aggregate_empirical_components,
                            ),
                        },
                    }
            return {
                "status":"computed",
                "scope":"individual_active_pair_proof_stvs_not_final_tournament",
                "evaluated_probability":evaluated_probability,
                "target_population":(
                    "held-out same-impression clicked-versus-nonclicked "
                    "orientations only; not a universal arbitrary-pair preference"
                ),
                "sampling_unit":(
                    "one source impression total weight; all mirrored orientations "
                    "and active proof channels share that unit mass"
                ),
                "metric_weighting":(
                    "macro mean of within-impression proof-observation means"
                ),
                "impressions":len(by_impression),
                "proof_observations":sum(map(len,by_impression.values())),
                "eligible_direction_roots":eligible_roots,
                "proved_direction_roots":proved_roots,
                "direction_root_coverage":(
                    proved_roots/eligible_roots if eligible_roots else 0.0
                ),
                "confidence_shrunk_posterior":shrunk,
                "raw_strength_without_confidence":raw_strength,
                "constant_probability_baselines":{
                    "constant_0_5_balanced_pair_target":{
                        "probability":0.5,
                        "basis":(
                            "protocol prior before conditioning on active-proof "
                            "availability; both pair orientations are generated"
                        ),
                        **balanced_constant,
                    },
                    "constant_empirical_active_proof_rate":{
                        "probability":empirical_active_proof_rate,
                        "basis":(
                            "equal-impression target rate among these same held-out "
                            "active-proof observations"
                        ),
                        "role":(
                            "descriptive same-cohort fitted reference, not a deployable "
                            "or independently estimated baseline"
                        ),
                        **empirical_constant,
                    },
                },
                "paired_deltas":{
                    "confidence_shrunk_minus_raw_strength":shrunk_minus_raw,
                    "confidence_shrunk_minus_constant_0_5":shrunk_minus_balanced,
                    "raw_strength_minus_constant_0_5":raw_minus_balanced,
                },
                "descriptive_same_cohort_deltas":{
                    "confidence_shrunk_minus_empirical_constant":(
                        shrunk_minus_empirical
                    ),
                },
                "aggregated_balanced_margin_pair_preference":aggregated,
                "brier_delta_shrunk_minus_raw_strength":(
                    shrunk_minus_raw["macro_impression_brier"]
                ),
                "brier_delta_95_ci":shrunk_minus_raw[
                    "macro_impression_brier_95_ci"
                ],
                "interpretation":(
                    "Lower Brier/log loss/ECE is better. This is a held-out "
                    "reliability check, not training-population calibration and "
                    "not a probability claim for the ordinal final rank score."
                ),
            }

        def benchmark(self,config):
            if self._online_events:
                raise ValueError(
                    "reload the dataset before benchmarking: live feedback is isolated "
                    "from the immutable offline evaluation protocol"
                )
            matched_direct_requested=config.get(
                "matched_direct_reasoner_ablation",False
            )
            if not isinstance(matched_direct_requested,bool):
                raise ValueError(
                    "matched_direct_reasoner_ablation must be a boolean"
                )
            force_cold=config.get(
                "force_cold_proof_cache",config.get("force_cold",False)
            )
            if not isinstance(force_cold,bool):
                raise ValueError("force_cold_proof_cache must be a boolean")
            started=time.perf_counter()
            previous_config=self.config.copy()
            mining=None
            try:
                self.configure(config)
                remine=config.get("remine",True) is not False
                stale_keys=sorted(
                    key for key in MINING_CONFIG_KEYS
                    if key in config and previous_config.get(key)!=self.config.get(key)
                )
                if stale_keys and not remine:
                    raise ValueError(
                        "remine=false cannot change mining configuration: "
                        +", ".join(stale_keys)
                    )
                if remine:
                    mining=self.mine()
            except Exception:
                # ``mine`` atomically retains the previous worker/rule metadata on
                # failure. Restore its matching configuration as well; otherwise a
                # failed benchmark could leave serving facts projected through a
                # feature profile that the still-live scorer was never compiled for.
                self.configure(previous_config)
                raise
            proof_cache_entries_before_clear={
                "point":len(self._proof_cache),
                "point_channel":len(self._point_channel_proof_cache),
                "pair_case":len(self._pair_proof_cache),
                "pair_channel":len(self._pair_channel_proof_cache),
            }
            if force_cold:
                # This is a proof-result cold run, not a model-compilation or fact-
                # materialization cold start. Keeping the identical worker/facts
                # guarantees accuracy is unchanged while reasoner cache latency is
                # measured honestly.
                self._proof_cache.clear()
                self._point_channel_proof_cache.clear()
                self._pair_proof_cache.clear()
                self._pair_channel_proof_cache.clear()
                self._pair_margin_cache.clear()
                self._pair_proof_origins.clear()
            proof_cache_entries_at_evaluation_start={
                "point":len(self._proof_cache),
                "point_channel":len(self._point_channel_proof_cache),
                "pair_case":len(self._pair_proof_cache),
                "pair_channel":len(self._pair_channel_proof_cache),
            }
            model_preparation_seconds=time.perf_counter()-started
            evaluation_started=time.perf_counter()
            benchmark_timeout_sec=self._benchmark_reasoner_timeout()
            tests=self.evaluation_cases()
            if not tests: raise ValueError("dataset has no labeled evaluation impressions")
            prepared=[]; all_specs=[]
            cap=max(0,int(config.get("max_candidates",self.config["max_candidates"])))
            seed=int(config.get("random_seed",self.config["random_seed"]))
            eval_case_limit=max(0,int(config.get("eval_case_limit",0)))
            if eval_case_limit and len(tests)>eval_case_limit:
                indexed=list(enumerate(tests))
                selected=sorted(indexed,key=lambda item:int.from_bytes(hashlib.sha256(
                    f'{seed}\0eval\0{item[1].get("source_impression_id") or item[1].get("id")}'.encode()
                ).digest(),"big"))[:eval_case_limit]
                tests=[case for _index,case in sorted(selected)]
            supplied_candidate_count=0; admitted_candidate_count=0
            supplied_relevant_count=0; admitted_relevant_count=0
            complete_relevant_slates=0; relevant_slates=0
            candidate_preparation_started=time.perf_counter()
            candidate_preparation_profile=Counter()
            for case in tests:
                supplied_candidates=list(case["candidates"])
                candidates=self._bounded_candidates(case,cap,seed)
                relevant_ids=set(map(str,case.get("relevant",())))
                supplied_relevant=relevant_ids.intersection(
                    map(str,supplied_candidates)
                )
                admitted_relevant=supplied_relevant.intersection(
                    map(str,candidates)
                )
                supplied_candidate_count+=len(supplied_candidates)
                admitted_candidate_count+=len(candidates)
                supplied_relevant_count+=len(supplied_relevant)
                admitted_relevant_count+=len(admitted_relevant)
                if supplied_relevant:
                    relevant_slates+=1
                    complete_relevant_slates+=(
                        admitted_relevant==supplied_relevant
                    )
                contexts=case.get("candidate_context",{})
                specs=self._candidate_specs(case["user"],candidates,contexts)
                for key,value in getattr(
                        self,"_last_candidate_preparation_profile",{}).items():
                    if key!="candidates": candidate_preparation_profile[key]+=value
                start=len(all_specs); all_specs.extend(specs)
                prepared.append((case,specs,start,len(all_specs)))
            candidate_preparation_seconds=(
                time.perf_counter()-candidate_preparation_started
            )
            point_reasoning_started=time.perf_counter()
            self._point_query_calls=0; self._point_query_roots=0
            self._point_pruned_query_roots=0
            self._point_channel_activations=0
            self._point_reused_channel_activations=0
            self._last_point_completeness={}
            self._ensure_candidate_specs(
                all_specs,timeout_sec=benchmark_timeout_sec
            )
            groups,reasoner_calls=self._proofs_for_specs(
                all_specs,timeout_sec=benchmark_timeout_sec
            )
            point_cache_stats=dict(self._last_point_cache_stats)
            point_reasoning_seconds=time.perf_counter()-point_reasoning_started
            self._pair_query_calls=0; self._pair_query_roots=0
            self._pair_pruned_query_roots=0; self._pair_channel_activations=0
            self._pair_reused_channel_activations=0; ranked=[]
            unordered_pair_comparisons=0; oriented_pair_cases=0
            maximum_slate_candidates=max(
                (len(specs) for _case,specs,_start,_end in prepared),default=0
            )
            maximum_slate_comparisons=0
            pair_planning_seconds=0.0; pair_reasoning_seconds=0.0
            parity_audit_seconds=0.0; rank_aggregation_seconds=0.0
            pair_reliability_seconds=0.0
            direct_ablation_seconds=0.0; direct_ranked=None; direct_planned=[]
            matched_direct_reasoner_ablation={
                "requested":matched_direct_requested,
                "status":"not_requested",
                "normal_ranking_path":"PeTTaChainer proofs",
                "direct_path_used_for_serving":False,
                "live_proof_margin_cache_mutated":False,
                "interpretation":(
                    "Opt-in matched scoring-output diagnostic only; normal benchmark "
                    "ranking remains PeTTaChainer and has no direct fallback."
                ),
            }
            reasoner_semantic_parity={
                "status":"not_applicable",
                "interpretation":(
                    "Matched direct-versus-PeTTa semantic parity only; this is not "
                    "an accuracy causal ablation or a unique-AUC estimate."
                ),
            }
            heldout_pair_proof_reliability={"status":"not_applicable"}
            pair_plan_profile=Counter()
            pair_materialization_profile={}
            pair_reasoning_profile={}
            point_row_preparation_seconds=0.0
            pair_budget_accounting_seconds=0.0
            all_pair_specs=[]
            if self.config["ranking_mode"]=="pairwise" and self.pair_rules:
                pair_planning_started=time.perf_counter()
                planned=[]
                for case,specs,start,end in prepared:
                    point_rows_started=time.perf_counter()
                    rows=self._rank(specs,groups[start:end],0,apply_pairwise=False)
                    point_row_preparation_seconds+=(
                        time.perf_counter()-point_rows_started
                    )
                    plan=self._pairwise_plan(rows,specs)
                    for key,value in getattr(self,"_last_pair_plan_profile",{}).items():
                        if isinstance(value,(int,float)) and not isinstance(value,bool):
                            aggregate_key=(
                                "sum_unique_activation_cases_per_slate"
                                if key=="unique_activation_cases_in_slate" else key
                            )
                            pair_plan_profile[aggregate_key]+=value
                    budget_started=time.perf_counter()
                    next_total=unordered_pair_comparisons+len(plan[0])
                    total_limit=int(self.config["max_total_pair_comparisons"])
                    if next_total>total_limit:
                        raise ValueError(
                            "benchmark pairwise comparison budget exceeded: "
                            f"more than {total_limit} unordered pairs across the "
                            "evaluation cohort; reduce eval_case_limit, candidate "
                            "slates, or pairwise_opponents"
                        )
                    planned.append((rows,plan)); all_pair_specs.extend(plan[1])
                    unordered_pair_comparisons=next_total
                    maximum_slate_comparisons=max(
                        maximum_slate_comparisons,len(plan[0])
                    )
                    pair_budget_accounting_seconds+=(
                        time.perf_counter()-budget_started
                    )
                oriented_pair_cases=len(all_pair_specs)
                pair_planning_seconds=time.perf_counter()-pair_planning_started
                pair_reasoning_started=time.perf_counter()
                self._ensure_pair_specs(
                    all_pair_specs,timeout_sec=benchmark_timeout_sec
                )
                pair_materialization_profile=dict(getattr(
                    self,"_last_pair_materialization_profile",{}
                ))
                pair_proof_map,_pair_calls=self._proofs_for_pair_specs(
                    all_pair_specs,timeout_sec=benchmark_timeout_sec
                )
                pair_reasoning_profile=dict(getattr(
                    self,"_last_pair_reasoning_profile",{}
                ))
                pair_cache_stats=dict(self._last_pair_cache_stats)
                pair_reasoning_seconds=time.perf_counter()-pair_reasoning_started
                pair_reliability_started=time.perf_counter()
                heldout_pair_proof_reliability=self._pair_proof_reliability(
                    prepared,planned,pair_proof_map,seed=seed,
                    comparison_probability=(
                        self._balanced_margin_probability_evaluator()
                    ),
                )
                pair_reliability_seconds=(
                    time.perf_counter()-pair_reliability_started
                )
                if matched_direct_requested:
                    # Preserve the exact pre-tournament point rows and pair plans.
                    # The normal PeTTa ranking below sorts/mutates its row objects.
                    direct_planned=[(copy.deepcopy(rows),plan)
                                    for rows,plan in planned]
                parity_audit_started=time.perf_counter()
                # A bounded matched audit answers the criticism that this shallow
                # topology may be reproducible by a direct rule table.  A separate
                # disposable worker receives fresh categorical facts and is asked
                # for every case/channel PairSignal root; it does not reuse the
                # optimized host activation path or its proof caches.  A match
                # demonstrates semantic fidelity only and deliberately does not
                # claim that PeTTa alone caused an AUC lift.
                from ..evaluation.reasoner_parity import (
                    UnsupportedParityTopology,
                    evaluate_pair_reasoner_parity,
                )
                unique_pair_specs={}
                for case,attrs in all_pair_specs:
                    unique_pair_specs.setdefault(case,(case,attrs))
                active_candidates=[]; inactive_parity_specs=[]
                for spec in unique_pair_specs.values():
                    _case,attrs=spec
                    activated={
                        rule["proof_channel_id"] for rule in self.pair_rules
                        if all(attrs.get(predicate)==value
                               for predicate,value in rule["premises"])
                    }
                    if activated:
                        active_candidates.append((spec,activated))
                    else:
                        inactive_parity_specs.append(spec)
                # Prefer cases that add channel coverage.  The audit remains
                # bounded and reports sampled_match when the evaluation slate
                # cannot exercise every compiled channel.
                active_parity_specs=[]; covered_channels=set()
                for spec,activated in active_candidates:
                    if activated.difference(covered_channels):
                        active_parity_specs.append(spec)
                        covered_channels.update(activated)
                        if len(active_parity_specs)>=96:
                            break
                selected_active={case for case,_attrs in active_parity_specs}
                active_parity_specs.extend(
                    spec for spec,_activated in active_candidates
                    if spec[0] not in selected_active
                )
                active_parity_specs=active_parity_specs[:96]
                parity_case_cap=min(
                    128,max(1,32768//max(1,len(self.pair_rules)))
                )
                # Reserve one negative/non-activation case when available. This
                # keeps the independent audit inside its hard case×channel query
                # bound even for the largest accepted rule snapshot.
                if inactive_parity_specs and parity_case_cap>1:
                    parity_specs=[
                        *active_parity_specs[:parity_case_cap-1],
                        inactive_parity_specs[0],
                    ]
                else:
                    parity_specs=active_parity_specs[:parity_case_cap]
                if len(parity_specs)<parity_case_cap:
                    selected={case for case,_attrs in parity_specs}
                    parity_specs.extend(
                        spec for spec in unique_pair_specs.values()
                        if spec[0] not in selected
                    )
                    parity_specs=parity_specs[:parity_case_cap]
                try:
                    if not parity_specs or not active_parity_specs:
                        raise UnsupportedParityTopology(
                            "no active evaluation pair channel is available"
                        )
                    parity=evaluate_pair_reasoner_parity(
                        self,parity_specs,max_cases=parity_case_cap,
                        max_rules=min(1024,max(256,len(self.pair_rules))),
                    )
                    reasoner_semantic_parity={
                        "status":parity["status"],
                        "parity_scope":parity["parity_scope"],
                        "full_model_semantic_parity":parity[
                            "full_model_semantic_parity"
                        ],
                        "sampled_semantic_parity":parity[
                            "sampled_semantic_parity"
                        ],
                        "report_kind":parity["report_kind"],
                        "interpretation":parity["interpretation"],
                        "supported_topology":parity["supported_topology"],
                        "model_sha256":parity["model_sha256"],
                        "summary":parity["summary"],
                        "errors":parity["errors"],
                    }
                except UnsupportedParityTopology as exc:
                    reasoner_semantic_parity={
                        "status":"unsupported",
                        "interpretation":(
                            "No direct semantic-parity claim was made; this is not "
                            "an accuracy causal ablation or unique-AUC estimate."
                        ),
                        "reason":str(exc),
                    }
                parity_audit_seconds=time.perf_counter()-parity_audit_started
                rank_aggregation_started=time.perf_counter()
                for (case,specs,_start,_end),(rows,plan) in zip(prepared,planned):
                    ranked.append(self._pairwise_rank(rows,specs,plan,pair_proof_map))
                rank_aggregation_seconds=time.perf_counter()-rank_aggregation_started
                if matched_direct_requested:
                    direct_ablation_started=time.perf_counter()
                    if (self.config["pair_aggregation"]!="proof_margin"
                            or self.config["aggregation"]!="weighted"):
                        matched_direct_reasoner_ablation.update(
                            status="unsupported",
                            reason=(
                                "exact direct scoring-output reconstruction supports "
                                "weighted point aggregation and compiler-issued "
                                "isolated proof_margin pair channels only"
                            ),
                        )
                    else:
                        try:
                            from ..evaluation.reasoner_parity import (
                                DirectPairReconstructor,
                                DirectPointReconstructor,
                                UnsupportedParityTopology,
                            )
                            direct_point=DirectPointReconstructor(
                                self,max_rules=min(
                                    1024,max(256,len(self.mined_rules))
                                ),
                            )
                            direct_pair=DirectPairReconstructor(
                                self,max_rules=min(
                                    1024,max(256,len(self.pair_rules))
                                ),
                            )
                            direct_point_groups=[]; direct_point_by_case={}
                            for _aid,candidate_case,attrs,_raw_attrs in all_specs:
                                reconstructed=direct_point.proof_rows(
                                    candidate_case,attrs
                                )
                                previous=direct_point_by_case.setdefault(
                                    candidate_case,reconstructed
                                )
                                if previous!=reconstructed:
                                    raise RuntimeError(
                                        "one normalized candidate case produced "
                                        "conflicting direct point activations"
                                    )
                                direct_point_groups.append(reconstructed)
                            direct_proof_map={}
                            for pair_case,attrs in all_pair_specs:
                                reconstructed=direct_pair.proof_rows(
                                    pair_case,attrs
                                )
                                previous=direct_proof_map.setdefault(
                                    pair_case,reconstructed
                                )
                                if previous!=reconstructed:
                                    raise RuntimeError(
                                        "one normalized pair case produced "
                                        "conflicting direct activations"
                                    )
                            direct_ranked=[]
                            point_order_matches=0
                            point_signature_matches=0
                            point_signatures_compared=0
                            margin_cache_keys_before=set(self._pair_margin_cache)
                            try:
                                for ((case,specs,start,end),
                                     (petta_point_rows,plan)) in zip(
                                         prepared,direct_planned):
                                    rows=self._rank(
                                        specs,direct_point_groups[start:end],0,
                                        apply_pairwise=False,
                                    )
                                    for row in rows:
                                        row["engine"]="DirectPointReconstructor"
                                    petta_index_to_id=[
                                        row["article"]["id"]
                                        for row in petta_point_rows
                                    ]
                                    direct_index_by_id={
                                        row["article"]["id"]:index
                                        for index,row in enumerate(rows)
                                    }
                                    if (len(direct_index_by_id)!=len(rows)
                                            or set(petta_index_to_id)
                                            !=set(direct_index_by_id)):
                                        raise RuntimeError(
                                            "direct point reconstruction changed "
                                            "the prepared candidate slate"
                                        )
                                    direct_point_order=[
                                        row["article"]["id"] for row in rows
                                    ]
                                    point_order_matches+=(
                                        petta_index_to_id==direct_point_order
                                    )
                                    petta_point_by_id={
                                        row["article"]["id"]:row
                                        for row in petta_point_rows
                                    }
                                    direct_point_by_id={
                                        row["article"]["id"]:row for row in rows
                                    }
                                    for article_id in petta_index_to_id:
                                        point_signature_matches+=(
                                            self._pointwise_signature(
                                                petta_point_by_id[article_id]
                                            )==self._pointwise_signature(
                                                direct_point_by_id[article_id]
                                            )
                                        )
                                        point_signatures_compared+=1
                                    comparisons,pair_specs=plan
                                    remapped_comparisons=[
                                        (
                                            direct_index_by_id[
                                                petta_index_to_id[left_index]
                                            ],
                                            direct_index_by_id[
                                                petta_index_to_id[right_index]
                                            ],
                                            forward_case,reverse_case,
                                        )
                                        for (left_index,right_index,forward_case,
                                             reverse_case) in comparisons
                                    ]
                                    direct_ranked.append(
                                        self._pairwise_rank(
                                            rows,specs,
                                            (remapped_comparisons,pair_specs),
                                            direct_proof_map,
                                        )
                                    )
                            finally:
                                # Diagnostic records must not accumulate in the
                                # live scorer's PeTTa proof-margin cache.
                                for cache_key in (
                                    set(self._pair_margin_cache)
                                    .difference(margin_cache_keys_before)
                                ):
                                    self._pair_margin_cache.pop(cache_key,None)
                            matched_direct_reasoner_ablation.update(
                                status="computed",
                                scope="full_point_and_pair_scoring_output_path",
                                supported_topologies={
                                    "point":direct_point.supported_topology,
                                    "pair":direct_pair.supported_topology,
                                },
                                model_sha256=hashlib.sha256(
                                    (direct_point.model_sha256+":"
                                     +direct_pair.model_sha256).encode("utf-8")
                                ).hexdigest(),
                                point_model_sha256=direct_point.model_sha256,
                                pair_model_sha256=direct_pair.model_sha256,
                                reconstructed_point_cases=len(
                                    direct_point_by_case
                                ),
                                reconstructed_pair_cases=len(direct_proof_map),
                                compiled_point_rules=len(direct_point.channels),
                                compiled_pair_channels=len(direct_pair.channels),
                                exact_point_slate_order_matches=(
                                    point_order_matches
                                ),
                                exact_point_slate_order_match_rate=(
                                    point_order_matches/len(prepared)
                                    if prepared else None
                                ),
                                exact_point_signature_matches=(
                                    point_signature_matches
                                ),
                                point_signatures_compared=(
                                    point_signatures_compared
                                ),
                                exact_point_signature_match_rate=(
                                    point_signature_matches
                                    /point_signatures_compared
                                    if point_signatures_compared else None
                                ),
                                interpretation=(
                                    "Matched full-scoring-output ablation for "
                                    "the supported shallow point and pair topologies: "
                                    "candidate slates, comparison edges/orientations, "
                                    "host aggregation, proof-margin transforms, "
                                    "family fusion and tie policy are identical. "
                                    "PeTTa-returned point and pair activations/STVs "
                                    "are replaced by lazy direct reconstruction. The "
                                    "normal result remains PeTTaChainer; the direct "
                                    "path is diagnostic and never a fallback."
                                ),
                            )
                        except UnsupportedParityTopology as exc:
                            matched_direct_reasoner_ablation.update(
                                status="unsupported",reason=str(exc)
                            )
                    direct_ablation_seconds=(
                        time.perf_counter()-direct_ablation_started
                    )
            else:
                pair_cache_stats={
                    "case_root_requests":0,"case_root_hits":0,
                    "case_root_misses":0,"channel_root_requests":0,
                    "channel_root_hits":0,"channel_root_misses":0,
                    "requests":0,"hits":0,"misses":0,
                }
                rank_aggregation_started=time.perf_counter()
                ranked=[self._rank(specs,groups[start:end],0)
                        for _case,specs,start,end in prepared]
                rank_aggregation_seconds=time.perf_counter()-rank_aggregation_started
                if matched_direct_requested:
                    matched_direct_reasoner_ablation.update(
                        status="unsupported",
                        reason=(
                            "matched direct point-and-pair ablation requires an active "
                            "pairwise model"
                        ),
                    )

            evaluation_relational_activation_audit=(
                self._relational_evaluation_activation_audit(
                    all_specs,all_pair_specs
                )
            )
            metric_started=time.perf_counter()
            hits=mrr=ndcg5=ndcg10=auc_total=auc_primary_total=0.0
            auc_values=[]; auc_proof_values=[]; auc_primary_values=[]; auc_pointwise_values=[]
            auc_popularity_values=[]
            auc_per_impression=[]
            auc_cases=proof_candidates=0; missing_positive=0
            pair_coverage_total=directional_coverage_total=0.0
            score_counts=Counter(); ranking_counts=Counter(); proof_ranking_counts=Counter()
            for case_index,((case,specs,_start,_end),rows) in enumerate(zip(prepared,ranked)):
                score_counts.update(row["score"] for row in rows)
                ranking_counts.update(self._ranking_signature(row) for row in rows)
                proof_ranking_counts.update(self._proof_ranking_signature(row) for row in rows)
                proof_candidates+=sum(bool(row["proofs"]) for row in rows)
                pair_coverage_total+=sum(row.get("pairwise_proof_coverage",0.0) for row in rows)
                directional_coverage_total+=sum(
                    row.get("pairwise_directional_coverage",0.0) for row in rows
                )
                relevant=set(case["relevant"]).intersection(
                    aid for aid,_candidate,_attrs,_raw_attrs in specs
                )
                ranks=[index+1 for index,row in enumerate(rows) if row["article"]["id"] in relevant]
                if not ranks:
                    missing_positive+=1
                    continue
                hits+=min(ranks)<=self.config["top_k"]
                mrr+=sum(1/rank for rank in ranks)/len(ranks)
                for cutoff,target in ((5,"ndcg5"),(10,"ndcg10")):
                    dcg=sum(1/math.log2(rank+1) for rank in ranks if rank<=cutoff)
                    ideal=sum(1/math.log2(rank+1) for rank in range(1,min(len(relevant),cutoff)+1))
                    if target=="ndcg5": ndcg5+=dcg/ideal if ideal else 0
                    else: ndcg10+=dcg/ideal if ideal else 0
                positives=[row for row in rows if row["article"]["id"] in relevant]
                negatives=[row for row in rows if row["article"]["id"] not in relevant]
                if positives and negatives:
                    case_auc=sum(
                        (self._ranking_signature(p)>self._ranking_signature(n))
                        +0.5*(self._ranking_signature(p)==self._ranking_signature(n))
                        for p in positives for n in negatives
                    )/(len(positives)*len(negatives))
                    case_primary_auc=sum(
                        (p["score"]>n["score"])+0.5*(p["score"]==n["score"])
                        for p in positives for n in negatives
                    )/(len(positives)*len(negatives))
                    case_pointwise_auc=sum(
                        (self._pointwise_signature(p)>self._pointwise_signature(n))
                        +0.5*(self._pointwise_signature(p)==self._pointwise_signature(n))
                        for p in positives for n in negatives
                    )/(len(positives)*len(negatives))
                    case_popularity_auc=sum(
                        (p["baseline"]>n["baseline"])
                        +0.5*(p["baseline"]==n["baseline"])
                        for p in positives for n in negatives
                    )/(len(positives)*len(negatives))
                    auc_total+=case_auc; auc_primary_total+=case_primary_auc
                    auc_values.append(case_auc); auc_primary_values.append(case_primary_auc)
                    auc_popularity_values.append(case_popularity_auc)
                    case_proof_auc=sum(
                        (self._proof_ranking_signature(p)>self._proof_ranking_signature(n))
                        +0.5*(self._proof_ranking_signature(p)==self._proof_ranking_signature(n))
                        for p in positives for n in negatives
                    )/(len(positives)*len(negatives))
                    auc_proof_values.append(case_proof_auc)
                    auc_pointwise_values.append(case_pointwise_auc)
                    case_identity=(case.get("source_impression_id") or case.get("id")
                                   or f"evaluation_{case_index}")
                    auc_per_impression.append({
                        "index":case_index,"id":str(case_identity),
                        "user":case["user"],
                        "candidates":len(rows),"positives":len(positives),
                        "negatives":len(negatives),"auc":case_auc,
                        "auc_proof_only":case_proof_auc,
                        "slate_digest":case.get("training_confirmation_slate_digest"),
                    })
                    auc_cases+=1
            if (matched_direct_reasoner_ablation.get("status")=="computed"
                    and direct_ranked is not None):
                petta_ablation_auc=[]; direct_ablation_auc=[]
                exact_order=0; exact_order_and_signature=0
                signature_matches=0; signature_candidates=0
                mismatched_impressions=[]
                for case_index,((case,specs,_start,_end),petta_rows,direct_rows) in enumerate(
                        zip(prepared,ranked,direct_ranked)):
                    petta_order=[row["article"]["id"] for row in petta_rows]
                    direct_order=[row["article"]["id"] for row in direct_rows]
                    if (len(set(petta_order))!=len(petta_order)
                            or set(petta_order)!=set(direct_order)):
                        raise RuntimeError(
                            "matched direct ablation changed the candidate slate"
                        )
                    order_match=petta_order==direct_order
                    exact_order+=order_match
                    petta_by_id={row["article"]["id"]:row for row in petta_rows}
                    direct_by_id={row["article"]["id"]:row for row in direct_rows}
                    slate_signature_match=True
                    for article_id in petta_order:
                        matched=(
                            article_id in direct_by_id
                            and self._ranking_signature(petta_by_id[article_id])
                            ==self._ranking_signature(direct_by_id[article_id])
                        )
                        signature_matches+=matched
                        signature_candidates+=1
                        slate_signature_match=slate_signature_match and matched
                    exact_order_and_signature+=(
                        order_match and slate_signature_match
                    )
                    if (not order_match or not slate_signature_match) and len(
                            mismatched_impressions)<32:
                        mismatched_impressions.append(str(
                            case.get("source_impression_id") or case.get("id")
                            or f"evaluation_{case_index}"
                        ))
                    relevant=set(case["relevant"]).intersection(
                        aid for aid,_candidate,_attrs,_raw_attrs in specs
                    )
                    petta_positive=[row for row in petta_rows
                                    if row["article"]["id"] in relevant]
                    petta_negative=[row for row in petta_rows
                                    if row["article"]["id"] not in relevant]
                    direct_positive=[row for row in direct_rows
                                     if row["article"]["id"] in relevant]
                    direct_negative=[row for row in direct_rows
                                     if row["article"]["id"] not in relevant]
                    if not petta_positive or not petta_negative:
                        continue
                    denominator=len(petta_positive)*len(petta_negative)
                    petta_ablation_auc.append(sum(
                        (self._ranking_signature(positive)
                         >self._ranking_signature(negative))
                        +0.5*(self._ranking_signature(positive)
                              ==self._ranking_signature(negative))
                        for positive in petta_positive
                        for negative in petta_negative
                    )/denominator)
                    direct_ablation_auc.append(sum(
                        (self._ranking_signature(positive)
                         >self._ranking_signature(negative))
                        +0.5*(self._ranking_signature(positive)
                              ==self._ranking_signature(negative))
                        for positive in direct_positive
                        for negative in direct_negative
                    )/denominator)
                if len(petta_ablation_auc)!=len(direct_ablation_auc):
                    raise RuntimeError(
                        "matched direct ablation produced an unmatched AUC cohort"
                    )
                petta_auc=(sum(petta_ablation_auc)/len(petta_ablation_auc)
                           if petta_ablation_auc else None)
                direct_auc=(sum(direct_ablation_auc)/len(direct_ablation_auc)
                            if direct_ablation_auc else None)
                auc_delta=(direct_auc-petta_auc
                           if direct_auc is not None and petta_auc is not None
                           else None)
                paired_deltas=[direct-petta for direct,petta in zip(
                    direct_ablation_auc,petta_ablation_auc
                )]
                complete_match=(
                    exact_order_and_signature==len(prepared)
                    and signature_matches==signature_candidates
                    and matched_direct_reasoner_ablation.get(
                        "exact_point_slate_order_matches"
                    )==len(prepared)
                    and matched_direct_reasoner_ablation.get(
                        "exact_point_signature_matches"
                    )==matched_direct_reasoner_ablation.get(
                        "point_signatures_compared"
                    )
                )
                matched_direct_reasoner_ablation.update(
                    status=("completed_match" if complete_match
                            else "completed_difference"),
                    auc_definition=(
                        "same served-order macro impression AUC used by the main "
                        "benchmark; ties receive half credit"
                    ),
                    eligible_auc_impressions=len(petta_ablation_auc),
                    pettachainer_auc=petta_auc,
                    direct_auc=direct_auc,
                    auc_delta_direct_minus_pettachainer=auc_delta,
                    auc_delta_95_ci=self._bootstrap_interval(
                        paired_deltas,seed^0xD1CE37
                    ),
                    exact_slate_order_matches=exact_order,
                    exact_slate_order_match_rate=(
                        exact_order/len(prepared) if prepared else None
                    ),
                    exact_candidate_signature_matches=signature_matches,
                    candidate_signatures_compared=signature_candidates,
                    exact_candidate_signature_match_rate=(
                        signature_matches/signature_candidates
                        if signature_candidates else None
                    ),
                    exact_slate_order_and_signature_matches=(
                        exact_order_and_signature
                    ),
                    exact_slate_order_and_signature_match_rate=(
                        exact_order_and_signature/len(prepared)
                        if prepared else None
                    ),
                    mismatch_impression_ids=mismatched_impressions,
                    finding=(
                        "The supported shallow point-and-pair topology has "
                        "equivalent scoring/ranking output for this matched cohort. "
                        "PeTTaChainer supplies executable "
                        "semantics and proof provenance here, but this comparison "
                        "does not show a ranking/AUC lift over its exact direct "
                        "reconstruction."
                        if complete_match else
                        "The direct reconstruction and PeTTaChainer scoring output differ; "
                        "inspect the reported mismatch impressions before making "
                        "a scoring-output-equivalence claim."
                    ),
                )
            cases=len(prepared)
            if not cases: raise ValueError("candidate configuration left no evaluable positive impressions")
            pair_delta_values=[
                pair_auc-point_auc
                for pair_auc,point_auc in zip(auc_values,auc_pointwise_values)
            ]
            pair_delta_ci=(
                self._bootstrap_interval(pair_delta_values,seed^0xC0FFEE)
                if self.config["ranking_mode"]=="pairwise" else [0.0,0.0]
            )
            if self.config["ranking_mode"]!="pairwise":
                auc_signal="lexicographic pointwise proof score, STV, and exact-tie prior"
            elif self.config["pair_aggregation"]=="proof_margin":
                family_fusion=self.config.get("pair_family_fusion","flat_margin")
                auc_signal=(f"PeTTaChainer pairwise proof_margin {family_fusion} "
                            f"{self.config['pair_margin_transform']} signed-power "
                            f"{self.config['pair_margin_power']:g} ProofRank tournament "
                            "over pointwise proofs")
            else:
                auc_signal=(f"PeTTaChainer pairwise {self.config['pair_aggregation']} "
                            "ProofRank tournament over pointwise proofs")
            metric_seconds=time.perf_counter()-metric_started
            evaluation_seconds=time.perf_counter()-evaluation_started
            total_seconds=time.perf_counter()-started
            ranking_pipeline_seconds=sum((
                candidate_preparation_seconds,
                point_reasoning_seconds,
                pair_planning_seconds,
                pair_reasoning_seconds,
                rank_aggregation_seconds,
            ))
            candidate_subtotal=math.fsum(candidate_preparation_profile.values())
            inner_pair_plan_seconds=float(pair_plan_profile.get("total_seconds",0.0))
            materialization_seconds=float(
                pair_materialization_profile.get("total_seconds",0.0)
            )
            proof_preparation_seconds=math.fsum(
                float(pair_reasoning_profile.get(key,0.0))
                for key in PAIR_PROOF_PREPARATION_TIMERS
            )
            pair_preparation_profile={
                "candidate_features":{
                    **{key:round(value,6)
                       for key,value in candidate_preparation_profile.items()},
                    "residual_admission_and_collection_seconds":round(max(
                        0.0,candidate_preparation_seconds-candidate_subtotal
                    ),6),
                    "total_seconds":round(candidate_preparation_seconds,6),
                    "definition":(
                        "Candidate/history context is calculated once per candidate, "
                        "then projected and serialized once; the resulting raw_attrs "
                        "objects are reused by every pair containing that candidate."
                    ),
                },
                "pair_construction":{
                    **{key:round(value,6) for key,value in pair_plan_profile.items()},
                    "point_row_preparation_seconds":round(
                        point_row_preparation_seconds,6
                    ),
                    "pair_budget_accounting_seconds":round(
                        pair_budget_accounting_seconds,6
                    ),
                    "outer_residual_seconds":round(max(
                        0.0,pair_planning_seconds-inner_pair_plan_seconds
                        -point_row_preparation_seconds-pair_budget_accounting_seconds
                    ),6),
                    "outer_total_seconds":round(pair_planning_seconds,6),
                    "definition":(
                        "Each unordered pair derives forward categorical relations "
                        "once; the reverse is an exact antisymmetric transform. An "
                        "inverted premise join identifies active rules without "
                        "rescanning the rule table. Pair case serialization hashes "
                        "only active-rule vectors."
                    ),
                },
                "candidate_pair_materialization":{
                    **{key:(round(value,6) if isinstance(value,float) else value)
                       for key,value in pair_materialization_profile.items()},
                    "definition":(
                        "Indexes unique pair cases and, only for non-factorized "
                        "aggregation, serializes and inserts candidate-pair facts."
                    ),
                },
                "proof_channel_preparation":(
                    _proof_channel_preparation_profile(pair_reasoning_profile)
                ),
                "profiled_pre_query_total_seconds":round(
                    candidate_preparation_seconds+pair_planning_seconds
                    +materialization_seconds+proof_preparation_seconds,6
                ),
                "timing_sum_definition":(
                    "candidate_features.total + pair_construction.outer_total + "
                    "candidate_pair_materialization.total + proof-channel topology, "
                    "activation join, template serialization and AtomSpace insertion"
                ),
            }
            cache_hits=point_cache_stats.get("hits",0)+pair_cache_stats.get("hits",0)
            cache_misses=(point_cache_stats.get("misses",0)
                          +pair_cache_stats.get("misses",0))
            cache_observed_mode=(
                "cold" if cache_hits==0 else
                "warm" if cache_misses==0 else
                "mixed"
            )
            result={"id":uuid.uuid4().hex[:8],"dataset":self.dataset_info(),"cases":cases,
                    "candidates":len(all_specs),"hit_rate":round(hits/cases,4),"mrr":round(mrr/cases,4),
                    "ndcg_at_5":round(ndcg5/cases,4),"ndcg_at_10":round(ndcg10/cases,4),
                    "ndcg":round(ndcg10/cases,4),
                    "metric_definitions":{
                        "auc":(
                            "Macro mean, over eligible impressions, of the probability "
                            "that a clicked candidate ranks above a non-clicked candidate; "
                            "ties receive half credit."
                        ),
                        "mrr":(
                            "Official MIND convention: within each impression average "
                            "1/rank over every clicked candidate, then macro-average "
                            "those impression values."
                        ),
                        "ndcg":(
                            "Normalized discounted cumulative gain from all clicked "
                            "candidates, macro-averaged over impressions."
                        ),
                    },
                    "auc":round(auc_total/auc_cases,4) if auc_cases else None,
                    "auc_proof_only":round(sum(auc_proof_values)/auc_cases,4)
                                     if auc_cases else None,
                    "auc_primary_score":round(auc_primary_total/auc_cases,4) if auc_cases else None,
                    "auc_95_ci":self._bootstrap_interval(auc_values,seed),
                    "auc_proof_only_95_ci":self._bootstrap_interval(
                        auc_proof_values,seed^0x50524F4F
                    ),
                    "auc_primary_95_ci":self._bootstrap_interval(
                        auc_primary_values,seed^0xA5A5A5A5
                    ),
                    "auc_pointwise_ranking":round(sum(auc_pointwise_values)/auc_cases,4)
                                            if auc_cases else None,
                    "same_cohort_baselines":{
                        "candidate_population":(
                            "identical held-out impressions and candidate slates"
                        ),
                        "random_expected_auc":0.5,
                        "training_click_count_popularity_auc":round(
                            sum(auc_popularity_values)/auc_cases,4
                        ) if auc_cases else None,
                        "training_click_count_popularity_auc_95_ci":(
                            self._bootstrap_interval(
                                auc_popularity_values,seed^0xB45311E
                            ) if auc_cases else [None,None]
                        ),
                        "pointwise_symbolic_auc":round(
                            sum(auc_pointwise_values)/auc_cases,4
                        ) if auc_cases else None,
                        "pointwise_symbolic_auc_95_ci":self._bootstrap_interval(
                            auc_pointwise_values,seed^0x50117
                        ) if auc_cases else [None,None],
                        "statement":(
                            "Both empirical baselines use the same frozen training "
                            "snapshot and the exact candidate population evaluated "
                            "by the pairwise ranker. Popularity is diagnostic only "
                            "and is not a serving fallback."
                        ),
                    },
                    "auc_pairwise_delta":round(
                        sum(pair_delta_values)/auc_cases,4
                    ) if auc_cases and self.config["ranking_mode"]=="pairwise" else 0.0,
                    "auc_pairwise_delta_95_ci":pair_delta_ci,
                    "pairwise_pointwise_diagnostic":{
                        "kind":"diagnostic_not_architecture_gate",
                        "status":(
                            "pass" if pair_delta_ci and pair_delta_ci[0]>0.0
                            else "hold"
                        ) if self.config["ranking_mode"]=="pairwise" else "not_applicable",
                        "criterion":"paired held-out AUC delta 95% lower bound > 0",
                        "comparator":"same-run pointwise symbolic ranker",
                        "delta_95_ci":pair_delta_ci,
                        "native_complete_impressions":not bool(cap),
                        "temporal_rule_stability_required":True,
                    },
                    "auc_signal":auc_signal,
                    "proof_coverage":round(proof_candidates/len(all_specs),4) if all_specs else 0,
                    "pairwise_proof_coverage":round(pair_coverage_total/len(all_specs),4) if all_specs else 0,
                    "pairwise_directional_coverage":round(
                        directional_coverage_total/len(all_specs),4
                    ) if all_specs else 0,
                    "seconds":round(total_seconds,3),"reasoner_batches":reasoner_calls,
                    "timings":{
                        "model_preparation_seconds":round(model_preparation_seconds,6),
                        "candidate_preparation_seconds":round(candidate_preparation_seconds,6),
                        "point_reasoning_seconds":round(point_reasoning_seconds,6),
                        "pair_planning_seconds":round(pair_planning_seconds,6),
                        "pair_reasoning_seconds":round(pair_reasoning_seconds,6),
                        "pair_proof_reliability_seconds":round(
                            pair_reliability_seconds,6
                        ),
                        "parity_audit_seconds":round(parity_audit_seconds,6),
                        "matched_direct_ablation_seconds":round(
                            direct_ablation_seconds,6
                        ),
                        "rank_aggregation_seconds":round(rank_aggregation_seconds,6),
                        "ranking_pipeline_seconds":round(
                            ranking_pipeline_seconds,6
                        ),
                        "metric_seconds":round(metric_seconds,6),
                        "evaluation_seconds":round(evaluation_seconds,6),
                        "total_seconds":round(total_seconds,6),
                    },
                    "timing_definitions":{
                        "ranking_pipeline_seconds":(
                            "candidate preparation + point reasoning + pair planning "
                            "+ pair reasoning + rank aggregation; excludes mining, "
                            "pair-proof reliability, parity audit, matched direct "
                            "ablation, metric calculation, and candidate retrieval"
                        ),
                        "pair_proof_reliability_seconds":(
                            "held-out individual PairSignal Brier/log-loss/ECE "
                            "diagnostic; excluded from ranking latency"
                        ),
                        "matched_direct_ablation_seconds":(
                            "opt-in direct channel reconstruction and identical "
                            "tournament/fusion replay; excludes its metric comparison"
                        ),
                    },
                    "pair_preparation_profile":pair_preparation_profile,
                    "reasoner_deadlines":{
                        "serving_seconds":self._serving_reasoner_timeout(),
                        "benchmark_seconds":benchmark_timeout_sec,
                        "scope":"finite parent-process deadline per scorer RPC",
                        "expiry_policy":(
                            "terminate expired worker, reconstruct the last "
                            "acknowledged rule/fact snapshot, and fail the request"
                        ),
                    },
                    "proof_cache":{
                        "requested_mode":(
                            "force_cold" if force_cold else "reuse_existing"
                        ),
                        "observed_mode":cache_observed_mode,
                        "scope":(
                            "proof-result caches only; compiled rules and grounded "
                            "facts remain identical"
                        ),
                        "accounting_unit":(
                            "unique logical point/pair case roots plus independently "
                            "cached factorized point/pair-channel roots"
                        ),
                        "entries_before_clear":proof_cache_entries_before_clear,
                        "entries_at_evaluation_start":(
                            proof_cache_entries_at_evaluation_start
                        ),
                        "point":point_cache_stats,
                        "pair":pair_cache_stats,
                        "hits":cache_hits,"misses":cache_misses,
                    },
                    "candidate_retrieval":{
                        "included":False,
                        "concurrency_measured":False,
                        "latency_sla_claimed":False,
                        "production_retrieval_recall_measured":False,
                        "source":"logged evaluation impression slate",
                        "logged_slate_admission":{
                            "candidate_cap":cap,
                            "supplied_candidates":supplied_candidate_count,
                            "admitted_candidates":admitted_candidate_count,
                            "supplied_relevant_items":supplied_relevant_count,
                            "admitted_relevant_items":admitted_relevant_count,
                            "relevant_item_inclusion_recall":(
                                admitted_relevant_count/supplied_relevant_count
                                if supplied_relevant_count else None
                            ),
                            "complete_relevant_slates":complete_relevant_slates,
                            "slates_with_supplied_relevant_items":relevant_slates,
                            "complete_relevant_slate_rate":(
                                complete_relevant_slates/relevant_slates
                                if relevant_slates else None
                            ),
                            "definition":(
                                "Admission after the optional seeded candidate cap, "
                                "relative only to relevant items already present in "
                                "the logged impression slate."
                            ),
                        },
                        "statement":(
                            "This replay evaluates ranking after candidates are supplied; "
                            "logged-slate cap admission is reported separately, but "
                            "production retrieval latency and retrieval recall are not measured."
                        ),
                    },
                    "ranking_workload":{
                        "candidates":len(all_specs),
                        "maximum_candidates_in_one_slate":maximum_slate_candidates,
                        "unordered_pair_comparisons":unordered_pair_comparisons,
                        "maximum_unordered_pair_comparisons_in_one_slate":(
                            maximum_slate_comparisons
                        ),
                        "oriented_pair_cases":oriented_pair_cases,
                        "opponent_limit":int(self.config["pairwise_opponents"]),
                        "configured_max_unordered_pair_comparisons_per_slate":int(
                            self.config["max_pair_comparisons"]
                        ),
                        "configured_max_total_unordered_pair_comparisons":int(
                            self.config["max_total_pair_comparisons"]
                        ),
                        "proof_cache_entry_limit":int(
                            self.config["max_proof_cache_entries"]
                        ),
                        "reasoner_mutation_journal":self.engine.journal_audit,
                        "comparison_policy":(
                            "complete_all_pairs"
                            if int(self.config["pairwise_opponents"])<=0
                            else "bounded_cyclic_graph"
                        ),
                        "complete_all_pairs_complexity":"sum_i n_i*(n_i-1)/2",
                    },
                    "reasoner_semantic_parity":reasoner_semantic_parity,
                    "matched_direct_reasoner_ablation":(
                        matched_direct_reasoner_ablation
                    ),
                    "heldout_pair_proof_reliability":(
                        heldout_pair_proof_reliability
                    ),
                    "pointwise_reasoner_batches":self._point_query_calls,
                    "pointwise_reasoner_roots":self._point_query_roots,
                    "pointwise_pruned_empty_roots":(
                        self._point_pruned_query_roots
                    ),
                    "pointwise_unique_case_channel_activations":(
                        self._point_channel_activations
                    ),
                    "pointwise_reused_channel_activations":(
                        self._point_reused_channel_activations
                    ),
                    "pointwise_proof_factorization":(
                        "alpha_normalized_isolated_channels"
                        if self.config["aggregation"]=="weighted"
                        else "disabled_shared_engagement_revision"
                    ),
                    "pointwise_proof_completeness":dict(
                        self._last_point_completeness
                    ),
                    "pointwise_proof_templates":len(
                        self._point_channel_templates
                    ),
                    "pointwise_proof_template_audit":sorted(
                        self._point_channel_templates.values(),
                        key=lambda item:(item["rule_id"],item["proof_channel_id"]),
                    ),
                    "pairwise_reasoner_batches":self._pair_query_calls,
                    "pairwise_reasoner_roots":self._pair_query_roots,
                    "pairwise_pruned_empty_roots":self._pair_pruned_query_roots,
                    "pairwise_unique_case_channel_activations":self._pair_channel_activations,
                    "pairwise_reused_channel_activations":self._pair_reused_channel_activations,
                    "pairwise_proof_factorization":(
                        "alpha_normalized_isolated_channels"
                        if self.config["pair_aggregation"]=="proof_margin"
                        else "disabled_shared_posterior"
                    ),
                    "pairwise_proof_templates":len(self._pair_channel_templates),
                    "pairwise_proof_template_audit":sorted(
                        self._pair_channel_templates.values(),
                        key=lambda item:(item["dependency_id"],item["proof_channel_id"]),
                    ),
                    "evaluation_relational_activation_audit":(
                        evaluation_relational_activation_audit
                    ),
                    "rules":len(self.mined_rules),"pair_rules":len(self.pair_rules),
                    "mining":mining,"config":self.config.copy(),
                    "requested_cases":len(tests),"sampled_without_positive":missing_positive,
                    "auc_cases":auc_cases,"auc_per_impression":auc_per_impression,
                    "score_unique":len(score_counts),
                    "score_tie_groups":sum(count>1 for count in score_counts.values()),
                    "score_tied_candidates":sum(count for count in score_counts.values() if count>1),
                    "ranking_unique":len(ranking_counts),
                    "proof_ranking_unique":len(proof_ranking_counts),
                    "ranking_tie_groups":sum(count>1 for count in ranking_counts.values()),
                    "evaluation_protocol":"native_impressions" if not cap else "seeded_uniform_candidate_sample",
                    "eval_case_limit":eval_case_limit}
            # Compatibility alias for older clients.  This is intentionally marked
            # as a diagnostic: only ``compare_profiles`` can promote an architecture.
            result["promotion_gate"]={
                **result["pairwise_pointwise_diagnostic"],
                "deprecated_alias":True,
            }
            self.runs.insert(0,result); return result

        def compare_profiles(self,config):
            """Run a proof-backed champion/challenger architecture promotion gate.

            The currently active architecture is the frozen champion.  A narrowly
            scoped ``challenger_config`` may change the mining strategy/depth, pair
            feature profile, and/or proof-margin power; evaluation controls remain
            frozen.  Both sides are independently remined and evaluated through a
            freshly compiled PeTTaChainer worker on the identical cohort. A challenger is
            retained only when its paired impression-level AUC interval clears
            ``min_delta`` on the complete, uncapped cohort.
            """
            if self._online_events:
                raise ValueError(
                    "reload the dataset before architecture comparison: live feedback "
                    "is isolated from the immutable offline evaluation protocol"
                )
            if not isinstance(config,dict):
                raise ValueError("architecture comparison config must be an object")
            champion_config=self.config.copy()
            champion_profile=str(champion_config["pair_feature_profile"])
            requested_champion=config.get("champion_pair_feature_profile",champion_profile)
            if str(requested_champion)!=champion_profile:
                raise ValueError(
                    "champion_pair_feature_profile must equal the currently active "
                    f"frozen champion ({champion_profile})"
                )
            raw_challenger_config=config.get("challenger_config",{})
            if raw_challenger_config is None: raw_challenger_config={}
            if not isinstance(raw_challenger_config,dict):
                raise ValueError("challenger_config must be an object")
            unknown_challenger_keys=sorted(
                set(raw_challenger_config)-CHALLENGER_CONFIG_KEYS
            )
            if unknown_challenger_keys:
                raise ValueError(
                    "challenger_config contains unsupported keys: "
                    +", ".join(unknown_challenger_keys)
                )
            challenger_overrides=dict(raw_challenger_config)
            legacy_challenger_value=config.get(
                "challenger_pair_feature_profile",config.get("pair_feature_profile")
            )
            nested_challenger_value=challenger_overrides.get("pair_feature_profile")
            if (legacy_challenger_value is not None
                    and nested_challenger_value is not None
                    and str(legacy_challenger_value)!=str(nested_challenger_value)):
                raise ValueError(
                    "challenger pair feature profile conflicts with challenger_config"
                )
            challenger_value=(nested_challenger_value
                              if nested_challenger_value is not None
                              else legacy_challenger_value)
            if challenger_value is None and challenger_overrides:
                challenger_value=champion_profile
            if challenger_value is None:
                raise ValueError("challenger_pair_feature_profile is required")
            challenger_profile=str(challenger_value)
            if challenger_profile not in PAIR_FEATURE_PROFILES:
                raise ValueError(
                    "challenger_pair_feature_profile must be one of: "
                    + ", ".join(sorted(PAIR_FEATURE_PROFILES))
                )
            challenger_overrides["pair_feature_profile"]=challenger_profile
            if "pair_margin_power" in challenger_overrides:
                try:
                    challenger_power=float(challenger_overrides["pair_margin_power"])
                except (TypeError,ValueError) as exc:
                    raise ValueError(
                        "challenger pair_margin_power must be a finite number between "
                        f"{PAIR_MARGIN_POWER_MIN} and {PAIR_MARGIN_POWER_MAX}"
                    ) from exc
                if (not math.isfinite(challenger_power)
                        or not PAIR_MARGIN_POWER_MIN<=challenger_power<=PAIR_MARGIN_POWER_MAX):
                    raise ValueError(
                        "challenger pair_margin_power must be a finite number between "
                        f"{PAIR_MARGIN_POWER_MIN} and {PAIR_MARGIN_POWER_MAX}"
                    )
                challenger_overrides["pair_margin_power"]=challenger_power
            challenger_effective_config={**champion_config,**challenger_overrides}
            if champion_config.get("ranking_mode")!="pairwise":
                raise ValueError("architecture comparison requires ranking_mode=pairwise")
            try:
                min_delta=float(config.get("min_delta",0.0))
            except (TypeError,ValueError) as exc:
                raise ValueError("min_delta must be a finite number between 0 and 1") from exc
            if not math.isfinite(min_delta) or not 0.0<=min_delta<=1.0:
                raise ValueError("min_delta must be a finite number between 0 and 1")
            try:
                repetitions=int(config.get("bootstrap_repetitions",1000))
                cap=max(0,int(config.get(
                    "max_candidates",champion_config.get("max_candidates",0)
                )))
                eval_case_limit=max(0,int(config.get("eval_case_limit",0)))
                bootstrap_seed=int(config.get(
                    "bootstrap_seed",champion_config.get("random_seed",7)
                ))
            except (TypeError,ValueError) as exc:
                raise ValueError(
                    "bootstrap_repetitions, max_candidates, eval_case_limit, and "
                    "bootstrap_seed must be integers"
                ) from exc
            if repetitions<1 or repetitions>100000:
                raise ValueError("bootstrap_repetitions must be between 1 and 100000")

            full_case_count=len(self.evaluation_cases())
            common={"max_candidates":cap,"eval_case_limit":eval_case_limit,
                    "remine":True}
            champion_run=challenger_run=None
            try:
                champion_run=self.benchmark({
                    **champion_config,**common,
                })
                challenger_run=self.benchmark({
                    **challenger_effective_config,**common,
                })
                champion_records=champion_run["auc_per_impression"]
                challenger_records=challenger_run["auc_per_impression"]
                identity_fields=("index","id","candidates","positives","negatives")
                champion_identities=[tuple(row[key] for key in identity_fields)
                                     for row in champion_records]
                challenger_identities=[tuple(row[key] for key in identity_fields)
                                       for row in challenger_records]
                if champion_identities!=challenger_identities:
                    raise RuntimeError(
                        "champion and challenger did not score an identical "
                        "held-out impression cohort"
                    )
                if not champion_records:
                    raise ValueError("evaluation cohort has no impressions with pairwise AUC")
                paired=[]; deltas=[]; proof_deltas=[]
                for champion_row,challenger_row in zip(
                    champion_records,challenger_records
                ):
                    delta=challenger_row["auc"]-champion_row["auc"]
                    proof_delta=(challenger_row["auc_proof_only"]
                                 -champion_row["auc_proof_only"])
                    deltas.append(delta); proof_deltas.append(proof_delta)
                    paired.append({
                        "index":champion_row["index"],"id":champion_row["id"],
                        "champion_auc":round(champion_row["auc"],8),
                        "challenger_auc":round(challenger_row["auc"],8),
                        "delta":round(delta,8),
                    })
                raw_interval=self._bootstrap_interval_raw(
                    deltas,bootstrap_seed^0x4348414D,repetitions
                )
                interval=[round(raw_interval[0],6),round(raw_interval[1],6)]
                raw_proof_interval=self._bootstrap_interval_raw(
                    proof_deltas,bootstrap_seed^0x50524F4D,repetitions
                )
                proof_interval=[round(raw_proof_interval[0],6),
                                round(raw_proof_interval[1],6)]
                full_cohort=(
                    cap==0 and eval_case_limit==0
                    and champion_run["cases"]==challenger_run["cases"]==full_case_count
                    and champion_run["auc_cases"]==champion_run["cases"]
                    and challenger_run["auc_cases"]==challenger_run["cases"]
                    and champion_run["sampled_without_positive"]==0
                    and challenger_run["sampled_without_positive"]==0
                )
                passed=bool(full_cohort and raw_interval[0]>min_delta)
                restoration=None
                if not passed:
                    self.configure(champion_config)
                    restoration=self.mine()

                def architecture_summary(run,profile):
                    run_config=run["config"]
                    engine={
                        "fixed_combinations":"fpMiner -> PeTTaChainer",
                        "target_aware":
                            "fpMiner unary + target-aware expansion -> PeTTaChainer",
                        "conditional_llm":
                            "fpMiner LLM unary + stable conditional expansion -> PeTTaChainer",
                        "conditional_llm_seed_only":
                            "fpMiner LLM discovery seeds + stable conditional children -> PeTTaChainer",
                        "petta_conditional_seed_only":
                            "PeTTa target-aware fpMiner + PeTTaChainer",
                        "petta_mdl_seed_only":
                            "PeTTa MDL-selected target rules + PeTTaChainer",
                        "petta_hierarchical_seed_only":
                            "PeTTa hierarchical target rules + PeTTaChainer",
                    }[run_config["miner_strategy"]]
                    return {
                        "profile":profile,"run_id":run["id"],"auc":run["auc"],
                        "pair_margin_power":run_config["pair_margin_power"],
                        "config":run_config.copy(),
                        "architecture_config":{
                            key:run_config[key] for key in sorted(CHALLENGER_CONFIG_KEYS)
                        },
                        "auc_proof_only":run["auc_proof_only"],"cases":run["cases"],
                        "auc_cases":run["auc_cases"],"candidates":run["candidates"],
                        "proof_coverage":run["proof_coverage"],
                        "pairwise_proof_coverage":run["pairwise_proof_coverage"],
                        "rules":run["rules"],"pair_rules":run["pair_rules"],
                        "reasoner_batches":run["reasoner_batches"],
                        "pairwise_reasoner_batches":run["pairwise_reasoner_batches"],
                        "mining":run["mining"],
                        "engine":engine,
                    }

                reasons=[]
                if cap: reasons.append("max_candidates must be 0")
                if eval_case_limit: reasons.append("eval_case_limit must be 0")
                if champion_run["auc_cases"]!=full_case_count or challenger_run["auc_cases"]!=full_case_count:
                    reasons.append("every held-out impression must have a complete AUC case")
                cohort_payload=json.dumps(
                    champion_identities,ensure_ascii=False,separators=(",",":")
                ).encode("utf-8")
                return {
                    "kind":"champion_challenger_architecture_gate",
                    "status":"pass" if passed else "hold",
                    "decision":("promote_challenger" if passed else "retain_champion"),
                    "criterion":"paired held-out AUC delta 95% lower bound > min_delta",
                    "metric":"serving_ranking_auc",
                    "min_delta":min_delta,
                    "full_cohort_eligible":full_cohort,
                    "ineligibility_reasons":reasons,
                    "cohort_fingerprint":hashlib.sha256(cohort_payload).hexdigest(),
                    "paired_impressions":len(deltas),
                    "mean_auc_delta":round(sum(deltas)/len(deltas),6),
                    "delta_95_ci":interval,
                    "mean_proof_only_auc_delta":round(
                        sum(proof_deltas)/len(proof_deltas),6
                    ),
                    "proof_only_delta_95_ci":proof_interval,
                    "bootstrap":{"seed":bootstrap_seed,"repetitions":repetitions,
                                 "unit":"held-out impression"},
                    "champion":architecture_summary(champion_run,champion_profile),
                    "challenger":architecture_summary(challenger_run,challenger_profile),
                    "per_impression":paired,
                    "active_pair_feature_profile":self.config["pair_feature_profile"],
                    "active_pair_margin_power":self.config["pair_margin_power"],
                    "active_miner_strategy":self.config["miner_strategy"],
                    "active_conjunctions":self.config["conjunctions"],
                    "active_pair_conjunctions":self.config["pair_conjunctions"],
                    "champion_restored":not passed,
                    "restoration_mining":restoration,
                    "distinct_from":"pairwise_pointwise_diagnostic",
                }
            except BaseException as exc:
                try:
                    self.configure(champion_config)
                    self.mine()
                except BaseException as restore_exc:
                    raise RuntimeError(
                        "architecture comparison failed and champion restoration also failed: "
                        f"{restore_exc}"
                    ) from exc
                raise

        def training_confirmation(self,config):
            """Confirm one challenger on a chronological tail of training only."""
            from ..evaluation.training_gate import run_training_confirmation
            return run_training_confirmation(
                self,config,context_features=CONTEXT_FEATURES,
                positive_actions=POSITIVE,
                challenger_config_keys=CHALLENGER_CONFIG_KEYS,
            )

        def tune(self,config):
            """Select mining/scoring settings on one slice and report once on another."""
            objective=str(config.get("objective", "auc"))
            objective_fields={"auc":"auc","mrr":"mrr","ndcg_at_10":"ndcg_at_10"}
            if objective not in objective_fields:
                raise ValueError("objective must be auc, mrr, or ndcg_at_10")
            objective_field=objective_fields[objective]
            source_key=next((key for key in ("eval_impressions","evaluation","tests","impressions")
                             if isinstance(self.data.get(key),list)),None)
            if source_key is None or len(self.data[source_key])<20:
                raise ValueError("auto-tuning needs at least 20 labeled evaluation impressions")
            original_source=self.data[source_key]
            split=max(10,min(len(original_source)-10,int(len(original_source)*0.7)))
            tuning_source=original_source[:split]; heldout_source=original_source[split:]
            base_support=max(1,int(config.get("min_support",self.config["min_support"])))
            base_ratio=max(0,int(config.get("negative_ratio",self.config["negative_ratio"])))
            base_rules=max(1,int(config.get("max_rules",self.config["max_rules"])))
            supports=config.get("support_grid",[base_support,base_support*2])
            depths=config.get("depth_grid",[2,3])
            ratios=config.get("negative_ratio_grid",[0,base_ratio] if base_ratio else [0,4])
            aggregations=config.get("aggregation_grid",["max","weighted"])
            supports=list(dict.fromkeys(max(1,int(value)) for value in supports))[:4]
            depths=list(dict.fromkeys(int(value) for value in depths if 2<=int(value)<=4))[:3]
            ratios=list(dict.fromkeys(max(0,int(value)) for value in ratios))[:3]
            aggregations=list(dict.fromkeys(str(value) for value in aggregations
                                           if str(value) in {"max","weighted"}))[:2]
            if not depths or not aggregations: raise ValueError("auto-tuning grid is empty")
            common={"top_k":int(config.get("top_k",self.config["top_k"])),
                    "chain_steps":int(config.get("chain_steps",self.config["chain_steps"])),
                    "max_candidates":int(config.get("max_candidates",0)),
                    "random_seed":int(config.get("random_seed",self.config["random_seed"])),
                    "feature_profile":str(config.get("feature_profile",self.config["feature_profile"])),
                    "rule_rank":"quality"}
            tuning_results=[]; heldout_result=None; evaluation_cache={}
            original_config=self.config.copy(); original_runs=list(self.runs)
            original_pending=self.pending_events; succeeded=False

            def discard_run(result):
                if self.runs and self.runs[0].get("id")==result.get("id"): self.runs.pop(0)

            try:
                self.data[source_key]=tuning_source
                for depth in depths:
                    for support in supports:
                        for ratio in ratios:
                            mining_config={**common,"conjunctions":depth,"min_support":support,
                                           "negative_ratio":ratio,
                                           "max_rules":max(base_rules,60 if depth>2 else base_rules)}
                            self.configure(mining_config); mining=self.mine()
                            fingerprint=hashlib.sha256(json.dumps([
                                [rule["premises"],rule["target"],rule["strength"],rule["confidence"],
                                 rule["negative_strength"],rule["negative_confidence"]]
                                for rule in self.mined_rules
                            ],sort_keys=True,separators=(",",":")).encode()).hexdigest()
                            for aggregation in aggregations:
                                trial_config={**mining_config,"aggregation":aggregation,"remine":False}
                                cache_key=(fingerprint,aggregation,common["top_k"],common["max_candidates"],
                                           common["random_seed"],common["chain_steps"])
                                reused=cache_key in evaluation_cache
                                if reused:
                                    self.configure({"aggregation":aggregation})
                                    result=evaluation_cache[cache_key]
                                else:
                                    result=self.benchmark(trial_config); discard_run(result)
                                    evaluation_cache[cache_key]=result
                                tuning_results.append({"config":{key:trial_config[key] for key in
                                    ("min_support","conjunctions","max_rules","negative_ratio","aggregation")},
                                    "auc":result["auc"],"mrr":result["mrr"],
                                    "ndcg_at_5":result["ndcg_at_5"],"ndcg_at_10":result["ndcg_at_10"],
                                    "proof_coverage":result["proof_coverage"],"rules":result["rules"],
                                    "premise_atoms":sum(len(rule["premises"]) for rule in self.mined_rules),
                                    "seconds":result["seconds"],"mining_seconds":mining["seconds"],
                                    "reused_evaluation":reused})
                peak_objective=max(row[objective_field] for row in tuning_results)
                objective_tolerance=max(0.0,float(config.get(
                    "objective_tolerance",config.get("ndcg_tolerance",0.0025))))
                finalists=[row for row in tuning_results
                           if row[objective_field]>=peak_objective-objective_tolerance]
                # A small validation fluctuation should not promote a much larger
                # symbolic model.  Within the configured nDCG tolerance, apply a
                # deterministic parsimony tie-break before secondary metrics.
                best=min(finalists,key=lambda row:(row["premise_atoms"],row["rules"],
                    -(row["auc"] if row["auc"] is not None else -1),-row["mrr"],
                    row["mining_seconds"],row["seconds"]))
                best_config={**common,**best["config"]}
                self.data[source_key]=heldout_source
                self.configure(best_config); mining=self.mine()
                heldout_result=self.benchmark({**best_config,"remine":False})
                heldout_result["kind"]="auto_tune_holdout"; heldout_result["mining"]=mining
                succeeded=True
            finally:
                self.data[source_key]=original_source
                if not succeeded:
                    self.runs[:]=original_runs
                    try:
                        self.configure(original_config); self.mine()
                    except Exception:
                        pass
                    self.pending_events=original_pending
            return {"best_config":best_config,"tuning_results":tuning_results,
                    "heldout_result":heldout_result,"split":{"tuning":len(tuning_source),"heldout":len(heldout_source)},
                    "objective":f"{objective} with parsimony tolerance, then secondary metrics and mining time",
                    "objective_metric":objective_field,"objective_tolerance":objective_tolerance,
                    # Retain the old key for clients that already render it.
                    "ndcg_tolerance":objective_tolerance,"state":self.state()}

    return BenchmarkMixin
