"""Regression coverage for isolated weighted point-rule proof channels."""

from collections import Counter
import hashlib
import json
import re
import unittest

from recommendation.evaluation.reasoner_parity import DirectPointReconstructor
from recommendation.app.server import IsolatedPeTTaChainer, Lab, proof_tv


def _point_model(count=7):
    rules=[]; shared=[]; channels=[]
    for index in range(1,count+1):
        rule_id=f"mined_{index}"
        variant_id=f"point_variant_{index}"
        decision_id=f"point_decision_rule_{index}"
        channel=f"point_{rule_id}"
        predicate=f"feature_{index}"
        premise=((predicate,"active"),)
        strength=0.55+index/100.0
        confidence=0.60+index/100.0
        negative_strength=0.25
        negative_confidence=0.50
        ctv=(f'(CTV (STV {strength} {confidence}) '
             f'(STV {negative_strength} {negative_confidence}))')
        shared.append(
            f'(: {rule_id} (Implication ({predicate.title()} $case "active") '
            f'(Engagement $case "click")) {ctv})'
        )
        mined=(f'(MinedPointPreference $case {json.dumps(rule_id)} '
               f'{json.dumps(channel)})')
        signal=(f'(PointSignal $case {json.dumps(rule_id)} '
                f'{json.dumps(channel)})')
        variant=(
            f'(: {variant_id} (Implication '
            f'({predicate.title()} $case "active") {mined}) {ctv})'
        )
        decision=(
            f'(: {decision_id} (Implication {mined} {signal}) '
            '(CTV (STV 1.0 1.0) (STV 0.0 1.0)))'
        )
        channels.extend((variant,decision))
        rules.append({
            "id":rule_id,
            "point_variant_id":variant_id,
            "point_decision_id":decision_id,
            "point_proof_channel_id":channel,
            "premises":premise,
            "target":"click",
            "strength":strength,
            "confidence":confidence,
            "negative_strength":negative_strength,
            "negative_confidence":negative_confidence,
            "specificity":1,
            "point_proof_factorization":{
                "schema":"isolated_extensional_point_channel_v1",
                "case_variable":"$case",
                "premise_tv":[1.0,1.0],
                "single_channel_producer":True,
                "rule_id":rule_id,
                "variant_id":variant_id,
                "decision_id":decision_id,
                "proof_channel_id":channel,
                "premises":[list(item) for item in premise],
                "rule_source_sha256":hashlib.sha256(
                    variant.encode("utf-8")
                ).hexdigest(),
                "decision_source_sha256":hashlib.sha256(
                    decision.encode("utf-8")
                ).hexdigest(),
            },
        })
    return rules,shared,channels


def _bare_lab(*,engine,aggregation="weighted",cache_limit=100):
    rules,shared,channels=_point_model()
    lab=object.__new__(Lab)
    lab.config={
        "aggregation":aggregation,
        "query_batch_size":32,
        "chain_steps":1,
        "max_proof_cache_entries":cache_limit,
    }
    lab.version=5; lab.engine=engine
    lab.mined_rules=rules
    lab._point_rule_sources=shared
    lab._point_channel_sources=channels
    lab._proof_cache={}; lab._point_channel_proof_cache={}
    lab._loaded_point_channels=set(); lab._point_channel_templates={}
    lab._point_query_calls=0; lab._point_query_roots=0
    lab._point_pruned_query_roots=0
    lab._point_channel_activations=0
    lab._point_reused_channel_activations=0
    lab._last_point_completeness={}; lab._last_point_cache_stats={}
    lab._articles={
        "article_1":{
            "id":"article_1","title":"Seven active rules",
            "topic":"test","format":"test",
        },
    }
    lab._popularity=Counter(); lab._click_base_rate=0.2
    lab._tie_break_stats={}
    return lab,shared,channels


class _RecordingEngine:
    def __init__(self):
        self.added=[]; self.queries=[]

    def add_atoms_no_check(self,atoms,timeout_sec=None):
        self.added.extend(atoms)

    def query_many(self,queries,*,steps,timeout_sec):
        self.queries.append((tuple(queries),steps,timeout_sec))
        return [[f'{query} (STV 0.7 0.8)'] for query in queries]


class WeightedPointChannelTest(unittest.TestCase):
    def test_seven_simultaneously_active_rules_are_all_petta_proven(self):
        engine=IsolatedPeTTaChainer()
        self.addCleanup(engine.close)
        lab,shared,channels=_bare_lab(engine=engine)
        engine.replace([*shared,*channels])
        attrs={f"feature_{index}":"active" for index in range(1,8)}
        specs=[("article_1","candidate_1",attrs,{})]

        groups,calls=lab._proofs_for_specs(specs,timeout_sec=30)

        self.assertEqual(calls,1)
        self.assertEqual(len(groups),1)
        self.assertEqual(len(groups[0]),7)
        proven=set().union(*(
            set(re.findall(r"\bmined_\d+\b",proof)) for proof in groups[0]
        ))
        self.assertEqual(proven,{f"mined_{index}" for index in range(1,8)})
        self.assertEqual(lab._point_query_roots,7)
        self.assertEqual(lab._point_channel_activations,7)
        self.assertTrue(lab._last_point_completeness["complete"])
        self.assertEqual(
            lab._last_point_completeness["proven_active_channel_uses"],7
        )

        direct=DirectPointReconstructor(lab)
        direct_proofs=direct.proof_rows("candidate_1",attrs)
        petta_rows=lab._rank(specs,groups,0,apply_pairwise=False)
        direct_rows=lab._rank(
            specs,[direct_proofs],0,apply_pairwise=False
        )
        self.assertEqual(len(direct_proofs),7)
        self.assertEqual(
            Lab._pointwise_signature(petta_rows[0]),
            Lab._pointwise_signature(direct_rows[0]),
        )
        for actual,expected in zip(
            sorted(proof_tv(proof) for proof in groups[0]),
            sorted(proof_tv(proof) for proof in direct_proofs),
        ):
            self.assertAlmostEqual(actual[0],expected[0],places=14)
            self.assertAlmostEqual(actual[1],expected[1],places=14)

    def test_channel_cache_cap_fails_before_reasoner_mutation(self):
        engine=_RecordingEngine()
        lab,_shared,_channels=_bare_lab(
            engine=engine,cache_limit=6
        )
        attrs={f"feature_{index}":"active" for index in range(1,8)}

        with self.assertRaisesRegex(
                RuntimeError,"point proof-channel cache limit"):
            lab._proofs_for_specs([
                ("article_1","candidate_1",attrs,{})
            ],timeout_sec=30)

        self.assertEqual(engine.added,[])
        self.assertEqual(engine.queries,[])
        self.assertEqual(lab._proof_cache,{})
        self.assertEqual(lab._point_channel_proof_cache,{})
        self.assertEqual(lab._point_channel_templates,{})

    def test_max_and_hybrid_keep_the_shared_engagement_query(self):
        for aggregation in ("max","hybrid"):
            with self.subTest(aggregation=aggregation):
                engine=_RecordingEngine()
                lab,_shared,_channels=_bare_lab(
                    engine=engine,aggregation=aggregation
                )
                lab._proofs_for_specs([
                    ("article_1","candidate_1",{"feature_1":"active"},{})
                ],timeout_sec=30)

                self.assertEqual(len(engine.queries),1)
                roots,_steps,_timeout=engine.queries[0]
                self.assertEqual(
                    roots,('(: $proof (Engagement candidate_1 "click") $tv)',)
                )
                self.assertFalse(any("PointSignal" in root for root in roots))
                self.assertEqual(lab._point_channel_proof_cache,{})


if __name__=="__main__":
    unittest.main()
