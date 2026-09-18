"""Frozen content evidence must pass through actual mining and PeTTa proofs."""

import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from recommendation.app.server import Handler, Lab, load_semantic_snapshot
from recommendation.pipelines.semantic_data import build_semantic_projection
from recommendation.features.semantic_workspace import build_semantic_workspace_facts
from recommendation.features.text_embeddings import build_text_embedding_sidecar
from recommendation.tests.fixtures import FakeEncoder, lexical_fixture


class SemanticLabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory()
        root=Path(cls.temporary.name)
        source=lexical_fixture()
        for event in source['events']:
            event['history']=['past']
        corpus=root/'source.json'
        corpus.write_text(json.dumps(source))
        sidecar=root/'vectors.npz'
        # Unit fixture only: sorted IDs matched,past,unrelated. Real benchmark
        # artifacts use the content-verified, frozen MiniLM sidecar instead.
        build_text_embedding_sidecar(corpus,sidecar,
                                     encoder=FakeEncoder([[1,0],[1,0],[0,1]]),
                                     model_name='unit-fixture',model_revision='v1')
        cls.data=build_semantic_projection(source,sidecar,provenance_corpus=corpus)
        cls.lab=Lab(cls.data,config={
            'pair_feature_profile':'workspace_centered',
            'pair_family_fusion':'balanced_rank',
            'min_support':2,'pair_min_support':2,'max_rules':8,'pair_max_rules':8,
        })

    @classmethod
    def tearDownClass(cls):
        cls.lab.engine.close()
        cls.temporary.cleanup()

    def test_centered_evidence_is_actually_mined_and_proved(self):
        rules=[r for r in self.lab.pair_rules
               if ('pair_text_semantic_centered_attention_t8_similarity','left') in r['premises']]
        self.assertTrue(rules)
        self.assertTrue(all(r['source'].endswith('fpMiner.metta') for r in rules))
        result=self.lab.benchmark({'remine':False})
        self.assertEqual(result['auc_proof_only'],1.0)
        self.assertTrue(self.lab._pair_proof_cache)

    def test_embeddings_cannot_rank_without_preference_proofs(self):
        def no_proofs(specs, **_kwargs):
            return {spec[0]:[] for spec in specs},0
        with patch.object(self.lab,'_proofs_for_pair_specs',side_effect=no_proofs):
            result=self.lab.benchmark({'remine':False})
        self.assertEqual(result['auc_proof_only'],0.5)

    def test_live_and_historical_projection_are_identical(self):
        actual=self.lab.features('reader',self.lab.article('matched'))
        expected=build_semantic_workspace_facts('matched',['past'],self.lab._article_text_vectors,
                                               self.lab._semantic_workspace_model)
        for key,value in expected.items():
            self.assertEqual(actual[key],value,key)
            self.assertEqual(self.data['events'][0][key],value,key)

    def test_missing_evidence_does_not_become_zero_similarity(self):
        pair=self.lab._pair_features(
            {'text_semantic_centered_attention_t8_similarity':None},
            {'text_semantic_centered_attention_t8_similarity':0.2},{},{})
        self.assertEqual(pair['pair_text_semantic_centered_attention_t8_similarity'],'right_known')

    def test_wrong_sidecar_fingerprint_fails_before_mining(self):
        bad=copy.deepcopy(self.data)
        bad['metadata']['semantic_workspace']['embedding_file_sha256']='0'*64
        with patch('recommendation.app.server.PeTTa',side_effect=AssertionError('mining started')):
            with self.assertRaisesRegex(ValueError,'frozen provenance'):
                Lab(bad)

    def test_changed_background_model_fails_before_mining(self):
        bad=copy.deepcopy(self.data)
        bad['semantic_workspace_model']['mean_vector'][0]+=0.1
        with patch('recommendation.app.server.PeTTa',side_effect=AssertionError('mining started')):
            with self.assertRaisesRegex(ValueError,'background model'):
                Lab(bad)

    def test_inline_vectors_cannot_override_missing_sidecar_evidence(self):
        for field in ('article_text_vectors','article_entity_vectors'):
            with self.subTest(field=field):
                bad=copy.deepcopy(self.data)
                bad[field]={'unverified_article':[1.0,0.0]}
                with patch('recommendation.app.server.PeTTa',side_effect=AssertionError('mining started')):
                    with self.assertRaisesRegex(ValueError,'inline vector'):
                        Lab(bad)

    def test_residual_mode_cannot_double_count_encoder_variants(self):
        before=self.lab.config.copy()
        with self.assertRaisesRegex(ValueError,'clustered dependencies'):
            self.lab.configure({'pair_dependency_mode':'residual_hypergraph'})
        self.assertEqual(self.lab.config,before)

    def test_load_semantic_snapshot_rejects_plain_fixture(self):
        path=Path(self.temporary.name)/'plain.json'
        path.write_text(json.dumps(lexical_fixture()))
        with self.assertRaisesRegex(ValueError,'prepared semantic_workspace'):
            load_semantic_snapshot(path)

    def test_dataset_reload_preserves_semantic_architecture(self):
        active=MagicMock()
        active.symbolic_only=False
        active._semantic_workspace_model={'schema':'test'}
        active.config={'pair_feature_profile':'workspace_centered'}
        active.lock=threading.RLock()
        replacement=MagicMock()
        body=json.dumps({'path':'new_media.json.gz'}).encode()
        handler=object.__new__(Handler)
        handler.path='/api/dataset/load'
        handler.headers={'Content-Length':str(len(body))}
        handler.rfile=io.BytesIO(body)
        handler.send_json=MagicMock()
        with patch('recommendation.app.server.LAB',active), \
                patch('recommendation.app.server.load_semantic_snapshot',return_value=self.data) as loader, \
                patch('recommendation.app.server.Lab',return_value=replacement) as constructor:
            handler.do_POST()
        loader.assert_called_once_with('new_media.json.gz')
        constructor.assert_called_once_with(data=self.data,symbolic_only=False,config=active.config)
        active.close.assert_called_once_with()
        self.assertIn('state',handler.send_json.call_args.args[0])


if __name__=='__main__':
    unittest.main()
