"""Self-contained tests of case-first selection (no real chair logs needed)."""
import pandas as pd
from case_first_sampling import choose_case_first

# 100 different cases, multiple provisional Activity types per case.
rows=[]
for i in range(100):
    for j,t in enumerate(['HP1','HP2','syringe_sensor','idle'],1):
        rows.append({'case_id':f'case_{i:03d}','activity_id':j,
                     'start_s':j*10.,'end_s':j*10.+8.,'duration_s':8.,
                     'activity_type':t})
table=pd.DataFrame(rows)
def sig(row):
    return {'case_id':row.case_id,'activity_id':row.activity_id}
selected,feat,types,rejected,coverage=choose_case_first(table,150,3,42,sig,('HP1','HP2','syringe_sensor','idle'))
assert len(selected)==150,(len(selected),coverage.n_selected.sum())
assert selected.case_id.nunique()==100,selected.case_id.nunique()
assert set(selected.iloc[:100].selection_stage)=={'coverage'}
assert coverage.n_selected.max()<=3
assert len(feat)==len(selected)
assert not len(rejected)
# Same seed -> same selected manifest and order.
s2,*_=choose_case_first(table,150,3,42,sig,('HP1','HP2','syringe_sensor','idle'))
assert selected[['case_id','activity_id']].equals(s2[['case_id','activity_id']])
# Failing the first candidate must be handled without losing the case.
def sig_fail(row):
    if row.case_id=='case_000' and row.activity_id!=4: return None
    return sig(row)
s3,f3,_,rej,cov=choose_case_first(table,150,3,42,sig_fail,('HP1','HP2','syringe_sensor','idle'))
assert s3.case_id.nunique()==100 and len(s3)==150
assert len(rej)>=1
assert cov.loc[cov.case_id=='case_000','n_selected'].iat[0]>=1
# A truly impossible case is reported, not given an invented Activity.
def sig_impossible(row):
    if row.case_id=='case_000':return None
    return sig(row)
s4,_,_,rej4,cov4=choose_case_first(table,150,3,42,sig_impossible,('HP1','HP2','syringe_sensor','idle'))
assert s4.case_id.nunique()==99
assert cov4.loc[cov4.case_id=='case_000','status'].iat[0]=='no_usable_activity'
try:
    choose_case_first(table,99,3,42,sig)
except ValueError as e:
    assert '--activities' in str(e)
else:
    raise AssertionError('should reject insufficient DTW capacity')
print('Case-first selection tests PASS: 100-case coverage, fallback, determinism, shortfall audit')
