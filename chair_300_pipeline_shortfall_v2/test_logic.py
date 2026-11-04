"""Small unit tests. Run: python test_logic.py"""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from chair_dtw_v3_1 import (clip_refine_boundaries,boundary_matching,dtw_cost,local_cost_matrix)

# Interior boundaries: one near edge -> snap; one deep inside -> drop.
eps=pd.DataFrame({'operation_type':['HP1'],'start_s':[20.],'end_s':[45.]})
b,aud=clip_refine_boundaries([20.5,30.,68.],eps,100.,1.,10.)
assert np.allclose(b,[20.,68.]), b
assert [x['action'] for x in aud]==['snapped_to_hp_episode_edge','removed_internal_hp_episode','unchanged']

# One-to-one matching: two predicted edges near one reference edge count only once.
s=boundary_matching([10.,10.5,30.],[10.3,30.],1.)
assert s['n_match']==2 and s['match_precision']<1

A=np.array([[1,2.,.2],[1,3.,.1],[2,np.nan,.0],[0,2.,.0]],float)
B=A.copy()
assert abs(dtw_cost(A,B,1,1,.4))<1e-9
assert abs(dtw_cost(A,B[::-1],1,1,.4)-dtw_cost(B[::-1],A,1,1,.4))<1e-9
print('All unit tests passed')

# Memory-safe exact-event reader: only relevant columns/interval and bounded output.
import gzip
import tempfile
from pathlib import Path
from chair_dtw_v3_1 import read_exact_activity_events, OnDemandCaseData
with tempfile.TemporaryDirectory() as td:
    fname = Path(td) / 'events_original.csv.gz'
    with gzip.open(fname, 'wt', newline='', encoding='utf-8') as fh:
        fh.write('occur_date,occur_time,log_device_id,event_code,event_value,camera_device_id,detection_code,detection_value,time_s\n')
        for i in range(20000):
            code = '20742' if i % 2 == 0 else '90012'
            fh.write(f'2024/03/29,00:00:00,A,{code},{i},,,,{i/100.}\n')
    rows, seen = read_exact_activity_events(fname, 50.0, 150.0, ['20742','20743'], max_points=50)
    assert seen['20742']==5000, seen
    assert seen['20743']==0, seen
    assert len(rows)<=50 and (rows.event_code=='20742').all(), len(rows)
    assert rows.time_s.between(50.,150.,inclusive='left').all()
print('Streaming reader tests passed')
