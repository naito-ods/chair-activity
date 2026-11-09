import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hierarchy import Taxonomy,select_resolution,validate_assertion

def toy():
    nodes=[{'id':'root','parent':None,'depth':0}]
    for key,parent,depth in [('A','root',1),('B','root',1),('A1','A',2),('A2','A',2),
                             ('B1','B',2),('a','A1',3),('b','A1',3),('c','A2',3),('d','B1',3)]:
        nodes.append({'id':key,'parent':parent,'depth':depth})
    return Taxonomy({'version':'synthetic','root':'root','nodes':nodes})

class HierarchyTests(unittest.TestCase):
    def setUp(self):self.t=toy()
    def test_exact_and_sibling_and_cross_branch(self):
        for pred,expected in [('a',1),('b',2/3),('c',1/3),('d',0)]:
            self.assertAlmostEqual(self.t.semantic_credit('a',pred),expected)
    def test_parent_prediction_is_partial(self):
        self.assertEqual(self.t.at('A1',3),None)
        self.assertAlmostEqual(self.t.semantic_credit('a','A1'),2/3)
        self.assertAlmostEqual(self.t.path_f1('a','A1'),.8)
    def test_root_and_unknown_never_earn_credit(self):
        for pred in ['root',None,'unmapped']:
            self.assertEqual(self.t.semantic_credit('a',pred),0)
    def test_partial_truth_not_fabricated_as_leaf(self):
        self.assertIsNone(self.t.semantic_credit('A1','a'))
        self.assertIsNone(self.t.at('A',2))
    def test_depth_adjusted_distance_equivalence(self):
        for pred in self.t.nodes:
            expected=1-(self.t.distance('a',pred)+3-self.t.depth(pred))/6
            self.assertAlmostEqual(self.t.semantic_credit('a',pred),expected)
    def test_cycles_rejected(self):
        doc={'version':'bad','root':'r','nodes':[{'id':'r','parent':None,'depth':0},
             {'id':'x','parent':'y','depth':1},{'id':'y','parent':'x','depth':2}]}
        with self.assertRaises(ValueError):Taxonomy(doc)
    def test_parent_mass_returns_parent_when_leaf_uncertain(self):
        a=select_resolution(self.t,{'a':.45,'b':.4,'d':.15},{1:.8,2:.8,3:.8},allow_uncalibrated_demo=True)
        self.assertEqual(a['concept_id'],'A1');self.assertTrue(a['demonstration_only'])
    def test_unknown_mass_not_renormalized_away(self):
        a=select_resolution(self.t,{'a':.4,'not_mapped':.6},{1:.8,2:.8,3:.8},allow_uncalibrated_demo=True)
        self.assertIsNone(a['concept_id'])
    def test_parent_mass_not_invented_as_descendant(self):
        a=select_resolution(self.t,{'A1':.9,'d':.1},{1:.8,2:.8,3:.8},allow_uncalibrated_demo=True)
        self.assertEqual(a['concept_id'],'A1')
    def test_unvalidated_real_scores_rejected(self):
        with self.assertRaises(ValueError):select_resolution(self.t,{'a':1},{1:.8,2:.8,3:.8})
    def test_log_does_not_establish_clinical_action(self):
        with self.assertRaises(ValueError):
            validate_assertion({'domain':'clinical_action','evidence_basis':'Observed',
                                'assertion_source':'raw_event','source_refs':['pickup_event']})
    def test_legacy_human_evidence_stays_unspecified(self):
        self.assertTrue(validate_assertion({'domain':'clinical_action','evidence_basis':'Unspecified',
                                            'assertion_source':'human_annotation'}))
        with self.assertRaises(ValueError):
            validate_assertion({'domain':'clinical_action','evidence_basis':'Inferred',
                                'assertion_source':'human_annotation'})
    def test_observed_needs_raw_reference(self):
        with self.assertRaises(ValueError):
            validate_assertion({'domain':'device_observation','evidence_basis':'Observed','assertion_source':'raw_event'})
    def test_model_must_be_inferred(self):
        with self.assertRaises(ValueError):
            validate_assertion({'domain':'clinical_action','evidence_basis':'Unspecified','assertion_source':'model_inference'})
    def test_model_cannot_assert_human_confirmation(self):
        assertion={'domain':'clinical_action','evidence_basis':'Inferred',
                   'assertion_source':'model_inference','human_confirmed':False}
        self.assertTrue(validate_assertion(assertion))
        for value in (True,None):
            with self.assertRaises(ValueError):
                validate_assertion(dict(assertion,human_confirmed=value))
        del assertion['human_confirmed']
        with self.assertRaises(ValueError):validate_assertion(assertion)

if __name__=='__main__':unittest.main()
