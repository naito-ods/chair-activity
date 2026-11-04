"""HP-first sequence evidence; overlapping observed patterns, not treatment labels."""
from pathlib import Path
import argparse
import json
import time
import hashlib

import numpy as np
import pandas as pd

SEED=42
CODE_NAMES={'20742':'rpm','20743':'torque','20108':'bulk','10014':'pedal1','10015':'pedal2'}
PATTERNS={'POST_HP3_LOW_HP2':'post_hp3_low','HIGH_DOMINANT_OBSERVED':'high_dominant',
          'HIGH_THEN_SHORT_LOW':'high_then_short_low',
          'FINAL_HP2_HIGH_THEN_SHORT_LOW':'final_hp2_high_then_short_low'}
BASE=dict(merge_gap_s=30.,low_max=5000.,high_center=40000.,high_tolerance=.10,
          min_positive_bins=8,min_observed_fraction=.4,min_active_s=10.,
          low_dominant_fraction=.7,high_dominant_fraction=.8,
          tail_max_fraction=.25,tail_min_equivalent_s=2.,tail_min_bins=3,
          long_high_min_s=15.,tail_prefix_high_fraction=.8,near_stop_s=3.)


def log(out,event,**data):
    row=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),event=event,**data)
    with (out/'process/pattern_analysis_log.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)


def wquant(values,weights,q):
    values=np.asarray(values,float);weights=np.asarray(weights,float)
    good=np.isfinite(values)&np.isfinite(weights)&(weights>0)
    if not good.any():return np.nan
    x=values[good];w=weights[good];order=np.argsort(x,kind='stable');x=x[order];w=w[order]
    return float(x[min(np.searchsorted(np.cumsum(w),q*w.sum(),side='left'),len(x)-1)])


def unknown_overlap(unknown,start,end):
    if end<=start or not len(unknown):return 0.
    return float(np.maximum(0,np.minimum(unknown[:,1],end)-np.maximum(unknown[:,0],start)).sum())


def make_blocks(runs,intervals,gap_s):
    blocks=[];mapping=[]
    intervals_by_case={case:g for case,g in intervals.groupby('case_id',sort=False)}
    for case,g in runs.groupby('case_id',sort=False):
        unknown=intervals_by_case[case].loc[lambda t:~t.known,['start_s','end_s']].to_numpy(float)
        active_block=None;prior=None;block_id=0
        for row in g.sort_values('start_s').itertuples():
            gap=row.start_s-prior.end_s if prior is not None else np.nan
            missing=unknown_overlap(unknown,prior.end_s,row.start_s) if prior is not None else np.nan
            combine=(active_block is not None and row.hp==active_block['hp'] and gap<=gap_s and missing<=1e-8)
            if not combine:
                block_id+=1
                active_block=dict(case_id=case,block_id=block_id,hp=int(row.hp),start_s=row.start_s,
                    end_s=row.end_s,active_s=0.,n_runs=0,previous_hp=int(prior.hp) if prior else np.nan,
                    previous_gap_s=gap,unknown_gap_s=missing)
                blocks.append(active_block)
            mapping.append(dict(case_id=case,run_id=int(row.run_id),block_id=block_id,
                run_start_s=row.start_s,run_end_s=row.end_s,run_active_s=row.active_s,
                cumulative_active_before=active_block['active_s']))
            active_block['active_s']+=row.active_s
            active_block['n_runs']+=1
            active_block['end_s']=row.end_s
            prior=row
    frame=pd.DataFrame(blocks)
    frame['elapsed_s']=frame.end_s-frame.start_s
    frame['post_hp3']=(frame.hp==2)&(frame.previous_hp==3)&(frame.unknown_gap_s<=1e-8)
    frame['context']=np.select([frame.post_hp3,frame.previous_hp.isna(),frame.unknown_gap_s>1e-8],
        ['POST_HP3','FIRST_OBSERVED_BLOCK','UNKNOWN_GAP'],default='OTHER_KNOWN_PREDECESSOR')
    return frame,pd.DataFrame(mapping)


def map_observations(obs,mapping,blocks):
    mapped=obs.loc[obs.run_id.notna()].merge(mapping,on=['case_id','run_id'],validate='many_to_one')
    mapped['weight_s']=(np.minimum(mapped.second_bin+1,mapped.run_end_s)-
                         np.maximum(mapped.second_bin,mapped.run_start_s)).clip(lower=0)
    mapped['bin_active_start']=mapped.cumulative_active_before+(
        np.maximum(mapped.second_bin,mapped.run_start_s)-mapped.run_start_s)
    mapped['bin_active_end']=mapped.cumulative_active_before+(
        np.minimum(mapped.second_bin+1,mapped.run_end_s)-mapped.run_start_s)
    mapped['active_clock_s']=mapped.cumulative_active_before+(mapped.time_s-mapped.run_start_s)
    mapped=mapped.merge(blocks[['case_id','block_id','active_s']],on=['case_id','block_id'],validate='many_to_one')
    mapped['active_progress']=mapped.active_clock_s/mapped.active_s
    assert (mapped.weight_s>0).all()
    assert mapped.groupby(['case_id','block_id','code']).weight_s.sum().le(
        mapped.groupby(['case_id','block_id','code']).active_s.first()+1e-7).all()
    return mapped


def make_points(mapped):
    hp2=mapped[mapped.hp==2].copy()
    keys=['case_id','run_id','second_bin']
    assert not hp2.duplicated(keys+['code']).any()
    pivot=hp2.pivot(index=keys,columns='code',values='value').rename(columns=CODE_NAMES)
    rpm=hp2[hp2.code=='20742'].drop(columns=['value','code']).set_index(keys)
    points=rpm.join(pivot,how='left',validate='one_to_one').reset_index()
    for key in CODE_NAMES.values():
        if key not in points:points[key]=np.nan
    return points.sort_values(['case_id','block_id','active_clock_s'])


def zoned(points,config):
    p=points.copy();value=p.rpm.to_numpy(float)
    low=config['low_max'];lo=config['high_center']*(1-config['high_tolerance'])
    hi=config['high_center']*(1+config['high_tolerance'])
    p['zone']=np.select([value==0,(value>0)&(value<=low),(value>low)&(value<lo),
                        (value>=lo)&(value<=hi),value>hi],['Z','L','M','H','X'],default='INVALID')
    return p


def classify_blocks(blocks,points,config):
    p=zoned(points,config)
    tail_zone=config.get('tail_zone','L')
    lookup={(case,int(block)):g for (case,block),g in p.groupby(['case_id','block_id'],sort=False)}
    rows=[]
    for b in blocks[blocks.hp==2].itertuples():
        g=lookup.get((b.case_id,int(b.block_id)),pd.DataFrame(columns=p.columns))
        pos=g[g.rpm>0]
        observed=float(g.weight_s.sum());positive=float(pos.weight_s.sum())
        row=dict(case_id=b.case_id,block_id=b.block_id,hp=2,start_s=b.start_s,end_s=b.end_s,
            active_s=b.active_s,elapsed_s=b.elapsed_s,n_runs=b.n_runs,previous_hp=b.previous_hp,
            previous_gap_s=b.previous_gap_s,unknown_gap_s=b.unknown_gap_s,post_hp3=bool(b.post_hp3),context=b.context,
            n_rpm_bins=len(g),n_positive_bins=len(pos),observed_equivalent_s=observed,
            positive_observed_equivalent_s=positive,rpm_observed_fraction=observed/b.active_s,
            positive_observed_fraction=positive/b.active_s,
            rpm_q10=wquant(g.rpm,g.weight_s,.1),rpm_q50=wquant(g.rpm,g.weight_s,.5),
            rpm_q90=wquant(g.rpm,g.weight_s,.9))
        for letter,name in [('L','low'),('M','mid'),('H','high'),('X','above_high')]:
            row[name+'_fraction']=float(pos.loc[pos.zone==letter,'weight_s'].sum()/positive) if positive else np.nan
        row['zero_fraction']=float(g.loc[g.zone=='Z','weight_s'].sum()/observed) if observed else np.nan
        supported=(len(pos)>=config['min_positive_bins'] and positive/b.active_s>=config['min_observed_fraction']
                   and b.active_s>=config['min_active_s'])
        row['supported']=bool(supported)
        row['pattern_status']='EVALUABLE_OBSERVED_HP2' if supported else 'INSUFFICIENT_HP2_NUMERIC_SUPPORT'
        row['low_dominant']=bool(supported and row['low_fraction']>=config['low_dominant_fraction'])
        row['high_dominant']=bool(supported and row['high_fraction']>=config['high_dominant_fraction'] and row['zero_fraction']<=.1)
        early=pos[pos.active_progress<=.3]
        row['initial_low_fraction']=float(early.loc[early.zone=='L','weight_s'].sum()/early.weight_s.sum()) if len(early) else np.nan
        row['post_hp3_low']=bool(supported and b.post_hp3 and row['low_dominant'] and
                                  row['initial_low_fraction']>=config['low_dominant_fraction'])
        row.update(high_then_short_low=False,terminal_low_near_stop=False,
            tail_low_n_bins=0,tail_low_equivalent_s=0.,tail_span_active_s=np.nan,
            tail_fraction_active=np.nan,tail_prefix_high_fraction=np.nan,tail_last_observation_gap_active_s=np.nan)
        if len(pos) and pos.zone.iloc[-1]==tail_zone and (pos.zone=='H').any():
            last_high=float(pos.loc[pos.zone=='H','active_clock_s'].max())
            tail=pos[(pos.active_clock_s>last_high)&(pos.zone==tail_zone)]
            if len(tail):
                start=float(tail.bin_active_start.min());end=float(tail.bin_active_end.max())
                prefix=pos[pos.active_clock_s<float(tail.active_clock_s.min())]
                high_s=float(prefix.loc[prefix.zone=='H','weight_s'].sum())
                prefix_fraction=high_s/prefix.weight_s.sum() if len(prefix) else np.nan
                tail_s=float(tail.weight_s.sum());span=end-start
                row.update(tail_low_n_bins=len(tail),tail_low_equivalent_s=tail_s,
                    tail_span_active_s=span,tail_fraction_active=span/b.active_s,
                    tail_prefix_high_fraction=prefix_fraction,
                    tail_last_observation_gap_active_s=b.active_s-end)
                near_stop=bool((tail.run_end_s-tail.time_s).max()<=config['near_stop_s'])
                row['terminal_low_near_stop']=near_stop
                row['high_then_short_low']=bool(supported and b.active_s>=30 and
                    row['high_fraction']>=.6 and high_s>=config['long_high_min_s'] and
                    prefix_fraction>=config['tail_prefix_high_fraction'] and start/b.active_s>=.7 and
                    span/b.active_s<=config['tail_max_fraction'] and len(tail)>=config['tail_min_bins'] and
                    tail_s>=config['tail_min_equivalent_s'] and
                    b.active_s-end<=max(2,.05*b.active_s))
        rows.append(row)
    frame=pd.DataFrame(rows)
    last=frame.groupby('case_id').block_id.transform('max')
    frame['is_final_hp2_block']=frame.block_id==last
    frame['final_hp2_high_then_short_low']=frame.high_then_short_low&frame.is_final_hp2_block
    return frame,p


def case_table(manifest,audit,hp2):
    rows=[];bycase={case:g for case,g in hp2.groupby('case_id')}
    stat=audit.set_index('case_id')
    for m in manifest.itertuples():
        row=dict(case_id=m.case_id,date=m.date,duration_s=m.duration_s)
        a=stat.loc[m.case_id]
        types=[f'HP{i}' for i in [1,2,3] if a[f'hp{i}_s']>0]
        row['hp_combination']='+'.join(types) if types else 'NO_HP_WORK_OBSERVED'
        row['dominant_hp']=int(np.argmax([a[f'hp{i}_s'] for i in [1,2,3]])+1) if types else np.nan
        g=bycase.get(m.case_id,pd.DataFrame(columns=hp2.columns))
        row['n_hp2_blocks']=len(g);row['n_supported_hp2_blocks']=int(g.supported.sum())
        row['status']=('NO_TARGET_SIGNAL' if m.signal_status=='NO_TARGET_SIGNAL' else
                       'NO_HP_WORK_OBSERVATION' if not types else 'NO_HP2_OBSERVED' if not len(g) else
                       'HP2_PATTERN_EVALUABLE' if g.supported.any() else 'HP2_NUMERIC_SUPPORT_INSUFFICIENT')
        for name,field in PATTERNS.items():
            row['has_'+field]=bool(g[field].any()) if g.supported.any() else np.nan
            row['n_'+field]=int(g[field].sum())
        row['has_high_then_short_low_away_from_stop']=bool((g.high_then_short_low&~g.terminal_low_near_stop).any()) if g.supported.any() else np.nan
        row['all_supported_high_dominant']=bool(g.loc[g.supported,'high_dominant'].all()) if g.supported.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean(values,repeats=2000):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    if not len(values):return np.nan,np.nan,np.nan
    rng=np.random.default_rng(SEED)
    samples=rng.choice(values,size=(repeats,len(values)),replace=True).mean(axis=1)
    return float(values.mean()),float(np.quantile(samples,.025)),float(np.quantile(samples,.975))


def transition_comparisons(hp2):
    block=hp2[hp2.supported].copy();effects=[];case_rows=[]
    for cap in [30,120,300,np.inf]:
        post=block.post_hp3&(block.previous_gap_s<=cap)
        other=(~block.post_hp3)&block.previous_hp.isin([1,2])&(block.unknown_gap_s<=1e-8)&(block.previous_gap_s<=cap)
        eligible=block[post|other].copy();eligible['context']=np.where(post.loc[eligible.index],'POST_HP3','OTHER_KNOWN_PREDECESSOR')
        records=[]
        for (case,context),g in eligible.groupby(['case_id','context']):
            low=float(np.average(g.low_fraction,weights=g.positive_observed_equivalent_s))
            rpm=float(np.average(g.rpm_q50,weights=g.positive_observed_equivalent_s))
            rec=dict(case_id=case,context=context,gap_cap_s=cap,n_blocks=len(g),low_fraction=low,rpm_q50=rpm)
            records.append(rec);case_rows.append(rec)
        table=pd.DataFrame(records)
        if table.empty:continue
        paired=table.pivot(index='case_id',columns='context',values='low_fraction')
        if all(c in paired for c in ['POST_HP3','OTHER_KNOWN_PREDECESSOR']):
            paired=paired.dropna();delta=paired.POST_HP3-paired.OTHER_KNOWN_PREDECESSOR
        else:delta=pd.Series(dtype=float)
        mean,lo,hi=bootstrap_mean(delta)
        for context,g in table.groupby('context'):
            _,cl,ch=bootstrap_mean(g.low_fraction)
            effects.append(dict(context=context,gap_cap_s=cap,n_cases=g.case_id.nunique(),n_blocks=int(g.n_blocks.sum()),
                median_case_low_fraction=g.low_fraction.median(),mean_case_low_fraction=g.low_fraction.mean(),
                ci_low=cl,ci_high=ch,n_paired_cases=len(delta),within_case_delta=mean,
                paired_ci_low=lo,paired_ci_high=hi))
    return pd.DataFrame(effects),pd.DataFrame(case_rows)


def analyze(out):
    if (out/'hp2_blocks.csv').exists():raise FileExistsError('Use a new output folder; preserve existing sequence analysis.')
    manifest=pd.read_csv(out/'case_manifest.csv');audit=pd.read_csv(out/'hp_status_audit.csv')
    intervals=pd.read_csv(out/'hp_intervals.csv');runs=pd.read_csv(out/'hp_runs.csv')
    observations=pd.read_csv(out/'observations.csv',dtype={'code':str})
    assert len(manifest)==300 and not manifest.case_id.duplicated().any()
    plan=dict(version='HP_SEQUENCE_V3',seed=SEED,**BASE,
        low_definition_source='user: <=5000 rpm; zero kept separate',
        high_definition_source='user: around40000 rpm; +/-10% is analyst tolerance, +/-5/15% sensitivity',
        macro_policy='same HP across known OFF gaps <=merge_gap_s; never cross unknown gap or another HP',
        hp_mapping={'1':'HP1','2':'HP2','3':'HP3 scaler'},
        positive_observation_weight='intersection of actual HP run and observed integer-second bin; proxy, not continuous measured duration',
        pattern_policy='overlapping evidence flags; no forced three-class partition',
        joint_signal_policy='same case, HP2 run, integer-second bin; no temporal forward fill or interpolation',
        hypotheses=['next HP2 after HP3 low-dominant','observed high-dominant HP2','long high then short observed low tail'],
        tail_limitation='low tail can reflect shutdown deceleration; near-stop flag retained; not proof of intentional work',
        external_validation=False)
    (out/'analysis_plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False))
    log(out,'started',requested_cases=300,cases_hp2=int((audit.hp2_s>0).sum()),config=BASE)
    blocks,mapping=make_blocks(runs,intervals,BASE['merge_gap_s'])
    mapped=map_observations(observations,mapping,blocks)
    points=make_points(mapped)
    hp2,points=classify_blocks(blocks,points,BASE)
    blocks.to_csv(out/'blocks.csv',index=False);mapping.to_csv(out/'run_to_block.csv',index=False)
    hp2.to_csv(out/'hp2_blocks.csv',index=False);points.to_csv(out/'hp2_points.csv',index=False)
    numeric=[]
    for (case,block,code),g in mapped.groupby(['case_id','block_id','code'],sort=False):
        numeric.append(dict(case_id=case,block_id=int(block),code=code,n_bins=len(g),
            observed_equivalent_s=g.weight_s.sum(),observed_fraction=g.weight_s.sum()/g.active_s.iloc[0],
            raw_q10=wquant(g.value,g.weight_s,.1),raw_q50=wquant(g.value,g.weight_s,.5),
            raw_q90=wquant(g.value,g.weight_s,.9)))
    pd.DataFrame(numeric).to_csv(out/'block_numeric_summary.csv',index=False)
    cases=case_table(manifest,audit,hp2);cases.to_csv(out/'case_patterns.csv',index=False)
    prevalence=[]
    for name,field in PATTERNS.items():
        eligible=hp2.supported if field!='post_hp3_low' else hp2.supported&hp2.post_hp3
        positive=hp2[hp2[field]]
        prevalence.append(dict(pattern=name,field=field,n_cases=positive.case_id.nunique(),n_blocks=len(positive),
            n_evaluable_cases=hp2[eligible].case_id.nunique(),n_evaluable_blocks=int(eligible.sum()),
            near_stop_blocks=int(positive.terminal_low_near_stop.sum()) if 'tail' in field or 'short_low' in field else 0))
    pd.DataFrame(prevalence).to_csv(out/'pattern_prevalence.csv',index=False)
    effects,context_cases=transition_comparisons(hp2)
    effects.to_csv(out/'transition_effects.csv',index=False);context_cases.to_csv(out/'context_case_comparisons.csv',index=False)
    conditional=[]
    for (case,zone),g in points.groupby(['case_id','zone']):
        for code in ['torque','bulk','pedal1','pedal2']:
            valid=g[code].notna()
            conditional.append(dict(case_id=case,zone=zone,code=code,n_pairs=int(valid.sum()),
                raw_median=wquant(g.loc[valid,code],g.loc[valid,'weight_s'],.5),
                raw_q10=wquant(g.loc[valid,code],g.loc[valid,'weight_s'],.1),
                raw_q90=wquant(g.loc[valid,code],g.loc[valid,'weight_s'],.9)))
    pd.DataFrame(conditional).to_csv(out/'conditional_numeric.csv',index=False)
    # Multivariate tail support: contemporaneous signal values, never invented holds.
    tail_support=[]
    for row in hp2[hp2.high_then_short_low].itertuples():
        g=points[(points.case_id==row.case_id)&(points.block_id==row.block_id)]
        last_high=g.loc[g.zone=='H','active_clock_s'].max()
        for phase,sample in [('high',g[g.zone=='H']),('tail_low',g[(g.zone=='L')&(g.active_clock_s>last_high)])]:
            for code in ['rpm','torque','bulk','pedal1','pedal2']:
                ok=sample[code].notna()
                tail_support.append(dict(case_id=row.case_id,block_id=row.block_id,phase=phase,code=code,
                    n_pairs=int(ok.sum()),raw_median=wquant(sample.loc[ok,code],sample.loc[ok,'weight_s'],.5)))
    pd.DataFrame(tail_support,columns=['case_id','block_id','phase','code','n_pairs','raw_median']).to_csv(out/'tail_signal_support.csv',index=False)
    representatives=[]
    for name,field in PATTERNS.items():
        g=hp2[hp2[field]].copy()
        g['evidence_score']=g.rpm_observed_fraction*np.log1p(g.active_s)
        if 'short_low' in field:
            g['evidence_score']*=np.where(g.terminal_low_near_stop,.25,1)*g.tail_low_equivalent_s
        g=g.sort_values('evidence_score',ascending=False).drop_duplicates('case_id').head(6)
        for rank,row in enumerate(g.itertuples()):
            representatives.append(dict(pattern=name,rank=rank,case_id=row.case_id,block_id=row.block_id,
                evidence_score=row.evidence_score,why='highest supported observed evidence; distinct case; not clinical exemplar',
                terminal_low_near_stop=row.terminal_low_near_stop))
    pd.DataFrame(representatives).to_csv(out/'representatives.csv',index=False)
    # One-factor-at-a-time sensitivity; no parameter is chosen for best clustering.
    variants=[('high_band5',dict(high_tolerance=.05)),('high_band15',dict(high_tolerance=.15)),
        ('merge15',dict(merge_gap_s=15.)),('merge60',dict(merge_gap_s=60.)),('merge120',dict(merge_gap_s=120.)),
        ('support5_25pct',dict(min_positive_bins=5,min_observed_fraction=.25)),
        ('support15_60pct',dict(min_positive_bins=15,min_observed_fraction=.6)),
        ('tail_min5s',dict(tail_min_equivalent_s=5.)),('tail_max15pct',dict(tail_max_fraction=.15)),
        ('high_dominant90pct',dict(high_dominant_fraction=.9))]
    sens=[]
    for name,changes in variants:
        config={**BASE,**changes}
        if config['merge_gap_s']!=BASE['merge_gap_s']:
            vb,vm=make_blocks(runs,intervals,config['merge_gap_s'])
            vp=make_points(map_observations(observations,vm,vb))
        else:vb,vp=blocks,points
        vh,_=classify_blocks(vb,vp,config)
        for pattern,field in PATTERNS.items():
            before=set(hp2.loc[hp2[field],'case_id']);after=set(vh.loc[vh[field],'case_id'])
            union=before|after
            sens.append(dict(scenario=name,pattern=pattern,n_cases=len(after),n_blocks=int(vh[field].sum()),
                n_evaluable_cases=vh.loc[vh.supported,'case_id'].nunique(),
                baseline_cases=len(before),case_jaccard=len(before&after)/len(union) if union else np.nan,
                added_case_ids=';'.join(sorted(after-before)),lost_case_ids=';'.join(sorted(before-after)),
                parameters=json.dumps(changes)))
        log(out,'sensitivity_complete',scenario=name)
    pd.DataFrame(sens).to_csv(out/'sensitivity.csv',index=False)
    # Raw signal availability is an independent audit, not a usage classifier.
    signal=mapped.groupby(['hp','code']).agg(n_cases=('case_id','nunique'),n_bins=('value','size')).reset_index()
    signal.to_csv(out/'hp_signal_availability.csv',index=False)
    transitions=blocks[blocks.previous_hp.notna()].groupby(['previous_hp','hp']).agg(
        n_blocks=('block_id','size'),n_cases=('case_id','nunique'),median_gap_s=('previous_gap_s','median')).reset_index()
    transitions.to_csv(out/'macro_transition_summary.csv',index=False)
    result=dict(requested_cases=300,cases_hp_work=int((audit.run_count>0).sum()),cases_hp2=int((audit.hp2_s>0).sum()),
        n_hp2_blocks=len(hp2),n_supported_hp2_blocks=int(hp2.supported.sum()),
        n_evaluable_hp2_cases=hp2.loc[hp2.supported,'case_id'].nunique(),
        case_status_counts=cases.status.value_counts().to_dict(),
        pattern_prevalence=prevalence,config=BASE,forced_partition=False,external_validation=False,
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'evaluation_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    log(out,'complete',**result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    analyze(p.parse_args().output.resolve())
