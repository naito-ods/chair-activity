"""Separate macro-scale and tail-candidate audit; original flags stay fixed."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from analyze_patterns import BASE,PATTERNS,make_blocks,map_observations,make_points,classify_blocks,log


def tail_candidates(frame,scale):
    result=frame[frame.tail_low_n_bins>0].copy()
    result['scale']=scale
    result['reasons_not_strict']=[';'.join(reason for failed,reason in [
        (not r.supported,'INSUFFICIENT_NUMERIC_SUPPORT'),(r.active_s<30,'ACTIVE_LT30S'),
        (r.high_fraction<.6,'HIGH_FRACTION_LT60PCT'),(r.tail_prefix_high_fraction<.8,'PREFIX_HIGH_LT80PCT'),
        (r.tail_fraction_active>.25,'LOW_TAIL_NOT_SHORT'),(r.tail_low_n_bins<3,'LOW_TAIL_LT3_BINS'),
        (r.tail_low_equivalent_s<2,'LOW_TAIL_LT2_BIN_EQUIVALENT_SECONDS'),
        (r.tail_last_observation_gap_active_s>max(2,.05*r.active_s),'END_OBSERVATION_GAP')] if failed)
        for r in result.itertuples()]
    result['screening_score']=result.high_fraction*result.tail_prefix_high_fraction/(1+result.tail_fraction_active)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output.resolve()
    if (out/'tail_transition_candidates.csv').exists():
        raise FileExistsError('Scale review already exists; use a fresh version to preserve outputs.')
    hp2=pd.read_csv(out/'hp2_blocks.csv');runs=pd.read_csv(out/'hp_runs.csv');intervals=pd.read_csv(out/'hp_intervals.csv')
    obs=pd.read_csv(out/'observations.csv',dtype={'code':str})
    baseline_sets={pattern:set(hp2.loc[hp2[field],'case_id']) for pattern,field in PATTERNS.items()}
    candidates=[tail_candidates(hp2,'macro30')];sensitivity=[]
    # This is a semantic-scale comparison, not selection for a higher score.
    for name,gap in [('macro300',300.),('same_hp_stage',86400.)]:
        config={**BASE,'merge_gap_s':gap}
        blocks,mapping=make_blocks(runs,intervals,gap)
        points=make_points(map_observations(obs,mapping,blocks))
        frame,points=classify_blocks(blocks,points,config)
        candidates.append(tail_candidates(frame,name))
        for pattern,field in PATTERNS.items():
            found=set(frame.loc[frame[field],'case_id']);old=baseline_sets[pattern]
            sensitivity.append(dict(scenario=name,merge_gap_s=gap,pattern=pattern,n_cases=len(found),
                n_blocks=int(frame[field].sum()),n_evaluable_cases=frame.loc[frame.supported,'case_id'].nunique(),
                baseline_cases=len(old),case_jaccard=len(found&old)/len(found|old) if found|old else np.nan,
                added_case_ids=';'.join(sorted(found-old)),lost_case_ids=';'.join(sorted(old-found))))
        if name=='same_hp_stage':
            blocks.to_csv(out/'stage_blocks.csv',index=False)
            frame.to_csv(out/'stage_hp2_blocks.csv',index=False)
            points.to_csv(out/'stage_hp2_points.csv',index=False)
        log(out,'extended_scale_review',scenario=name,merge_gap_s=gap,
            n_hp2_blocks=len(frame),strict_tail_cases=frame.loc[frame.high_then_short_low,'case_id'].nunique())
    candidate=pd.concat(candidates,ignore_index=True)
    candidate.to_csv(out/'tail_transition_candidates.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(out/'scale_sensitivity.csv',index=False)
    # Show unsupported candidates as candidates, not relabeled strict positives.
    representatives=[]
    for scale,g in candidate.groupby('scale'):
        g=g.sort_values('screening_score',ascending=False).drop_duplicates('case_id').head(6)
        for rank,row in enumerate(g.itertuples()):
            representatives.append(dict(pattern='TAIL_SCREEN_CANDIDATE',scale=scale,rank=rank,
                case_id=row.case_id,block_id=row.block_id,evidence_score=row.screening_score,
                strict_pattern=row.high_then_short_low,why=row.reasons_not_strict,
                terminal_low_near_stop=row.terminal_low_near_stop))
    pd.DataFrame(representatives).to_csv(out/'candidate_representatives.csv',index=False)
    print(pd.DataFrame(sensitivity).to_string(index=False))


if __name__=='__main__':main()
