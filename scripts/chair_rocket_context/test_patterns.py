"""Synthetic tests for case-isolated and parent-excluded context features."""
import unittest,json
import numpy as np
import pandas as pd
from patterns import context_features,anchor_features

class PatternTests(unittest.TestCase):
    def test_case_boundaries(self):
        meta=pd.DataFrame({'seq':[1,1,2,2]});score=np.arange(8.).reshape(4,2);available=np.ones((4,2),bool);raw=np.ones((4,2))
        f,_=context_features(meta,score,available,raw,'past');score[2:]=-999
        g,_=context_features(meta,score,available,raw,'past');np.testing.assert_array_equal(f[:2],g[:2])
    def test_unavailable_score_not_evidence(self):
        meta=pd.DataFrame({'seq':[1,1]});score=np.array([[1.,-1000.],[1.,9999.]]);available=np.array([[True,False],[True,False]])
        f,_=context_features(meta,score,available,np.zeros((2,1)),'current');np.testing.assert_array_equal(f[:,1],[0,0])
    def test_whole_self_parent_excluded(self):
        visible=pd.DataFrame({'seq':[1,1,1,2],'id':['a','b','c','other'],
          'start_s':[0.,30.,90.,0.],'end_s':[25.,85.,120.,120.],
          'confirmed':[True]*4,'treatment_normalized':['A','B','C','B']})
        meta=pd.DataFrame({'id':['w1','w2'],'seq':[1,1],'start_s':[35.,50.],'end_s':[45.,60.],
          'known_teacher':[True,True],'annotation_id':['b','b']})
        features,proof=anchor_features(visible,meta,['A','B','C'])
        self.assertTrue(all(set(json.loads(s))=={'a','c'} for s in proof.anchor_ids))
        altered=visible.copy();altered.loc[altered.id.eq('b'),'treatment_normalized']='CORRUPTED'
        again,_=anchor_features(altered,meta,['A','B','C']);np.testing.assert_array_equal(features,again)
    def test_hidden_rows_removed_before_features(self):
        visible=pd.DataFrame({'seq':[1],'id':['a'],'start_s':[0.],'end_s':[20.],
          'confirmed':[True],'treatment_normalized':['A']})
        query=pd.DataFrame({'id':['q'],'seq':[1],'start_s':[30.],'end_s':[60.],
          'known_teacher':[False],'annotation_id':['unknown-gap']})
        features,proof=anchor_features(visible,query,['A','B']);self.assertEqual(json.loads(proof.anchor_ids.iloc[0]),['a'])
        self.assertEqual(features.shape,(1,4*(2+3)))

if __name__=='__main__':unittest.main()
