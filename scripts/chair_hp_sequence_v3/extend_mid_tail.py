"""Apply user's clarified HIGH -> short MID hypothesis, retaining LOW audit."""
from pathlib import Path
import argparse
import shutil
import json

import numpy as np
import pandas as pd

from analyze_patterns import BASE,make_blocks,map_observations,make_points,classify_blocks,log


def rename_mid(frame):
    return frame.rename(columns={col:col.replace('tail_low','tail_mid').replace('short_low','short_mid')
        .replace('terminal_low','terminal_mid') for col in frame})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output.resolve()
    if (out/'mid_tail_blocks.csv').exists():
        raise FileExistsError('MID extension already exists; use a fresh output version to preserve flags.')
    blocks=pd.read_csv(out/'blocks.csv');points=pd.read_csv(out/'hp2_points.csv')
    primary=pd.read_csv(out/'hp2_blocks.csv');cases=pd.read_csv(out/'case_patterns.csv')
    config={**BASE,'tail_zone':'M'}
    (out/'mid_tail_plan.json').write_text(json.dumps(dict(config=config,
        reason='User clarified that with LOW<=5000 and HIGH around40000, the final lower-speed interval is MID, not LOW.',
        legacy_low_tail='Preserved as audit; not substituted for the corrected hypothesis',
        near_stop='All tail observations close to their own run end, including multiple short runs, are flagged',
        primary_patterns=['POST_HP3_LOW_HP2','HIGH_DOMINANT_OBSERVED','HIGH_THEN_SHORT_MID']),indent=2))
    frame,_=classify_blocks(blocks,points,config);frame=rename_mid(frame)
    frame.to_csv(out/'mid_tail_blocks.csv',index=False)
    for name in ['hp2_blocks.csv','case_patterns.csv','pattern_prevalence.csv','representatives.csv','evaluation_summary.json']:
        shutil.copy2(out/name,out/'process'/('before_mid_clarification_'+name))
    # Add fields; preserve all original LOW-based flags and counts.
    columns=['case_id','block_id','high_then_short_mid','final_hp2_high_then_short_mid',
        'terminal_mid_near_stop','tail_mid_n_bins','tail_mid_equivalent_s']
    primary=primary.merge(frame[columns],on=['case_id','block_id'],validate='one_to_one')
    primary.to_csv(out/'hp2_blocks.csv',index=False)
    counts=frame.groupby('case_id').agg(n_high_then_short_mid=('high_then_short_mid','sum'),
        n_final_hp2_high_then_short_mid=('final_hp2_high_then_short_mid','sum'))
    cases=cases.merge(counts,on='case_id',how='left',validate='one_to_one')
    evaluable=cases.status=='HP2_PATTERN_EVALUABLE'
    for field in ['high_then_short_mid','final_hp2_high_then_short_mid']:
        cases['n_'+field]=cases['n_'+field].fillna(0).astype(int)
        cases['has_'+field]=np.where(evaluable,cases['n_'+field]>0,np.nan)
    cases.to_csv(out/'case_patterns.csv',index=False)
    prevalence=pd.read_csv(out/'pattern_prevalence.csv')
    additions=[]
    for name,field in [('HIGH_THEN_SHORT_MID','high_then_short_mid'),
                        ('FINAL_HP2_HIGH_THEN_SHORT_MID','final_hp2_high_then_short_mid')]:
        pos=frame[frame[field]]
        additions.append(dict(pattern=name,field=field,n_cases=pos.case_id.nunique(),n_blocks=len(pos),
            n_evaluable_cases=frame.loc[frame.supported,'case_id'].nunique(),n_evaluable_blocks=int(frame.supported.sum()),
            near_stop_blocks=int(pos.terminal_mid_near_stop.sum())))
    prevalence=pd.concat([prevalence,pd.DataFrame(additions)],ignore_index=True)
    prevalence.to_csv(out/'pattern_prevalence.csv',index=False)
    reps=[]
    for name,field in [('HIGH_THEN_SHORT_MID','high_then_short_mid'),
                        ('FINAL_HP2_HIGH_THEN_SHORT_MID','final_hp2_high_then_short_mid')]:
        g=frame[frame[field]].copy()
        g['evidence_score']=g.rpm_observed_fraction*np.log1p(g.active_s)*g.tail_mid_equivalent_s*np.where(g.terminal_mid_near_stop,.25,1)
        g=g.sort_values('evidence_score',ascending=False).drop_duplicates('case_id').head(6)
        for rank,row in enumerate(g.itertuples()):
            reps.append(dict(pattern=name,rank=rank,case_id=row.case_id,block_id=row.block_id,
                evidence_score=row.evidence_score,why='User-corrected HIGH to MID; observed supported tail',
                terminal_mid_near_stop=row.terminal_mid_near_stop))
    rep=pd.DataFrame(reps,columns=['pattern','rank','case_id','block_id','evidence_score','why','terminal_mid_near_stop'])
    rep.to_csv(out/'representatives_mid.csv',index=False)
    oldrep=pd.read_csv(out/'representatives.csv')
    pd.concat([oldrep,rep],ignore_index=True).to_csv(out/'representatives.csv',index=False)
    # Multi-signal tail evidence on exactly matching HP2 run and second-bin.
    paired=[]
    for row in frame[frame.high_then_short_mid].itertuples():
        g=points[(points.case_id==row.case_id)&(points.block_id==row.block_id)]
        last_high=g.loc[g.zone=='H','active_clock_s'].max()
        for phase,sample in [('high',g[g.zone=='H']),('tail_mid',g[(g.zone=='M')&(g.active_clock_s>last_high)])]:
            for code in ['rpm','torque','bulk','pedal1','pedal2']:
                v=sample[code].dropna()
                paired.append(dict(case_id=row.case_id,block_id=row.block_id,phase=phase,code=code,
                    n_pairs=len(v),raw_median=v.median() if len(v) else np.nan))
    pd.DataFrame(paired,columns=['case_id','block_id','phase','code','n_pairs','raw_median']).to_csv(out/'mid_tail_signal_support.csv',index=False)
    # Preserve candidate reasons so absent strict matches cannot be mistaken for absence of a sequence.
    candidate=frame[frame.tail_mid_n_bins>0].copy()
    candidate['screening_score']=candidate.high_fraction*candidate.tail_prefix_high_fraction/(1+candidate.tail_fraction_active)
    candidate.to_csv(out/'mid_tail_candidates.csv',index=False)
    # Repeat the original one-factor sensitivity with MID as the endpoint.
    runs=pd.read_csv(out/'hp_runs.csv');intervals=pd.read_csv(out/'hp_intervals.csv')
    obs=pd.read_csv(out/'observations.csv',dtype={'code':str})
    variants=[('high_band5',{'high_tolerance':.05}),('high_band15',{'high_tolerance':.15}),
        ('merge15',{'merge_gap_s':15.}),('merge60',{'merge_gap_s':60.}),('merge120',{'merge_gap_s':120.}),
        ('merge300',{'merge_gap_s':300.}),('same_hp_stage',{'merge_gap_s':86400.}),
        ('support5_25pct',{'min_positive_bins':5,'min_observed_fraction':.25}),
        ('support15_60pct',{'min_positive_bins':15,'min_observed_fraction':.6}),
        ('tail_min5s',{'tail_min_equivalent_s':5.}),('tail_max15pct',{'tail_max_fraction':.15})]
    sensitivity=[]
    for scenario,changes in variants:
        cfg={**config,**changes}
        if cfg['merge_gap_s']!=BASE['merge_gap_s']:
            vb,vm=make_blocks(runs,intervals,cfg['merge_gap_s']);vp=make_points(map_observations(obs,vm,vb))
        else:vb,vp=blocks,points
        vf,_=classify_blocks(vb,vp,cfg);vf=rename_mid(vf)
        for name,field in [('HIGH_THEN_SHORT_MID','high_then_short_mid'),('FINAL_HP2_HIGH_THEN_SHORT_MID','final_hp2_high_then_short_mid')]:
            base=set(frame.loc[frame[field],'case_id']);found=set(vf.loc[vf[field],'case_id']);union=base|found
            sensitivity.append(dict(scenario=scenario,pattern=name,n_cases=len(found),n_blocks=int(vf[field].sum()),
                n_evaluable_cases=vf.loc[vf.supported,'case_id'].nunique(),baseline_cases=len(base),
                case_jaccard=len(base&found)/len(union) if union else np.nan,
                added_case_ids=';'.join(sorted(found-base)),lost_case_ids=';'.join(sorted(base-found)),parameters=json.dumps(changes)))
        log(out,'mid_sensitivity_complete',scenario=scenario)
    pd.DataFrame(sensitivity).to_csv(out/'mid_tail_sensitivity.csv',index=False)
    summary=json.loads((out/'evaluation_summary.json').read_text())
    summary['corrected_primary_patterns']=['POST_HP3_LOW_HP2','HIGH_DOMINANT_OBSERVED','HIGH_THEN_SHORT_MID']
    summary['pattern_prevalence']=prevalence.to_dict('records')
    summary['user_mid_tail_clarification']=True
    (out/'evaluation_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    log(out,'mid_extension_complete',prevalence=additions)


if __name__=='__main__':main()
