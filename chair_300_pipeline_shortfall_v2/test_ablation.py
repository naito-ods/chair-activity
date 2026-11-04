"""Logic smoke tests: run with python test_ablation.py"""
import numpy as np
import chair_ablation as ab
import chair_dtw_v3_1 as base


def run():
    rng=np.random.default_rng(21)
    for n,m in [(20,20),(18,29),(64,42)]:
        a=rng.normal(size=(n,9))
        b=rng.normal(size=(m,9))
        a[:3,3:6]=np.nan
        b[4:8,3:6]=np.nan
        old=base.dtw_cost(a,b,3,3,.25)
        new=ab._shape(a,b,3,3,.25,'A_full')
        assert np.isclose(old,new,atol=1e-12), (old,new)
    a=np.zeros((10,9),dtype=float)
    b=a.copy();b[:,3:6]=3.2
    assert np.isclose(ab._shape(a,b,3,3,.25,'B_state_events'),0.), 'B must ignore numeric values'
    assert ab._shape(a,b,3,3,.25,'C_numeric_only')>0., 'C must react to numeric values'
    assert np.isclose(ab._shape(a,b,3,3,.25,'E_mask_only_control'),0.), 'E ignores values'
    b[:5,3:6]=np.nan
    assert ab._shape(a,b,3,3,.25,'E_mask_only_control')>0., 'E detects missingness'
    a[:,:3]=0.;b[:,:3]=0.;a[:,6:]=0.;b[:,6:]=0.
    a[:,3:6]=np.nan;b[:,3:6]=2.5
    assert np.isclose(ab._shape(a,b,3,3,.25,'D_missing_neutral'),0.), 'D should not penalize one-sided missing numeric values'
    assert ab._shape(a,b,3,3,.25,'A_full')>0., 'A baseline penalizes one-sided missing'
    print('All ablation unit tests passed')

if __name__=='__main__':run()
