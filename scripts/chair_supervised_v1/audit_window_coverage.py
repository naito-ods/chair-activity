"""Account for short unannotated gaps/tails that fixed windows cannot cover."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from window_data import complement,starts,union_intervals

def audit(source):
    source=source.resolve();out=source/'windows30'
    annotations=pd.read_csv(source/'annotations_aligned.csv');cases=pd.read_csv(source/'cases_manifest.csv')
    candidates=pd.read_csv(out/'unannotated_candidates.csv')
    rows=[]
    for case in cases.itertuples():
        intervals=annotations.loc[annotations.seq.eq(case.seq),['start_s','end_s']].to_numpy(float)
        gaps=complement(intervals,0.,case.raw_duration_s)
        for index,(a,b) in enumerate(gaps):
            q=candidates[candidates.seq.eq(case.seq)&candidates.source_gap_start_s.eq(a)&candidates.source_gap_end_s.eq(b)]
            expected=len(starts(a,b,30,15))
            # CSV roundtrip changes a few floating last bits; fallback tolerant.
            if len(q)!=expected:
                q=candidates[candidates.seq.eq(case.seq)&np.isclose(candidates.source_gap_start_s,a,atol=1e-8)&np.isclose(candidates.source_gap_end_s,b,atol=1e-8)]
            assert len(q)==expected
            covered=sum(y-x for x,y in union_intervals(q[['start_s','end_s']].to_numpy(float)))
            uncovered=max(0.,b-a-covered)
            rows.append(dict(seq=int(case.seq),case_id=case.case_id,split=case.split,gap_index=index,
                gap_start_s=a,gap_end_s=b,gap_duration_s=b-a,candidate_windows=len(q),
                covered_duration_s=covered,uncovered_duration_s=uncovered,
                reason='GAP_SHORTER_THAN_30S' if not len(q) else 'FIXED_STEP_TAIL' if uncovered>1e-6 else 'FULLY_COVERED'))
    result=pd.DataFrame(rows);result.to_csv(out/'unannotated_gap_audit.csv',index=False)
    summary=dict(unannotated_gaps=len(result),gaps_without_30s_window=int(result.candidate_windows.eq(0).sum()),
        unannotated_duration_s=float(result.gap_duration_s.sum()),window_covered_duration_s=float(result.covered_duration_s.sum()),
        uncovered_short_gap_and_tail_duration_s=float(result.uncovered_duration_s.sum()),
        candidate_windows=int(result.candidate_windows.sum()),
        interpretation='Uncovered short gaps/tails remain unknown, never background negatives or asserted classified duration. Overlapping candidate-window durations are unioned before reporting coverage.')
    assert summary['candidate_windows']==len(candidates)
    (out/'process/window_coverage_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);audit(ap.parse_args().source)
