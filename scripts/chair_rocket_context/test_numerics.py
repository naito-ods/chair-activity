"""Synthetic checks; no clinical input required."""
import unittest
import numpy as np
from numerics import variance_mask,bounded_late_fusion,decode_sequence

class NumericsTests(unittest.TestCase):
    def test_quantization_and_constant(self):
        np.testing.assert_array_equal(variance_mask([1,1,1],[0,1e-16,1e-4]),[False,False,True])
    def test_relative_precision(self):
        self.assertFalse(variance_mask([1e8],[100])[0])
    def test_query_does_not_fit_mask(self):
        mean=np.array([1.,1.]);var=np.array([1e-16,1.]);before=variance_mask(mean,var)
        _query=np.array([[1e20,-1e20]])
        np.testing.assert_array_equal(variance_mask(mean,var),before)
    def test_available_comes_from_classes(self):
        scores=bounded_late_fusion([[-2000,1,0]],[[-3000,-100,1]],[[True,True,False]])
        self.assertTrue(np.isfinite(scores[0,0]));self.assertAlmostEqual(scores[0,1],.6)
        self.assertTrue(np.isneginf(scores[0,2]))
    def test_no_transition_weight_is_local_argmax(self):
        s=np.array([[2,0],[0,3],[1,0]],float)
        np.testing.assert_array_equal(decode_sequence(s,np.full((2,2),.5),0),s.argmax(axis=1))
    def test_invalid_transition_is_rejected(self):
        with self.assertRaises(ValueError):decode_sequence([[1,0]],[[1,1],[1,1]])

if __name__=='__main__':unittest.main()
