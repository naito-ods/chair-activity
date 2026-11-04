"""Compact, raw-unit features for fixed high-dominant HP2 units and their stages.

Only prior compact CSVs are read. Original events, assignments and prior outputs
remain unchanged. Missing numeric observations remain NaN.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

SIGNALS={'torque':'20743','pedal2':'10015','bulk':'20108','rpm':'20742'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def wq(value,weight,q):
    value=np.asarray(value,float);weight=np.asarray(weight,float)
    good=np.isfinite(value)&np.isfinite(weight)&(weight>0)
    if not good.any():return np.nan
    value=value[good];weight=weight[good];order=np.argsort(value,kind='stable')
    value=value[order];weight=weight[order]
    return float(value[min(np.searchsorted(np.cumsum(weight),q*weight.sum(),side='left'),len(value)-1)])


def unit_name(case,block,kind):
    return f'{case}__{kind}{int(block):03d}'


def build_level(units,runs,point_table,manifest,case_patterns,kind):
    """Return raw features, definitions and numeric support for one analysis unit."""
    definitions={};features=[];supports=[]
    run_lookup={(case,int(block)):g.sort_values('start_s') for (case,block),g in runs.groupby(['case_id','block_id'],sort=False)}
    point_lookup={(case,int(block)):g for (case,block),g in point_table.groupby(['case_id','block_id'],sort=False)}
    meta=manifest.set_index('case_id');context=case_patterns.set_index('case_id')
    for unit in units.itertuples():
        key=(unit.case_id,int(unit.block_id));r=run_lookup[key]
        p=point_lookup.get(key,pd.DataFrame(columns=point_table.columns))
        uid=unit_name(unit.case_id,unit.block_id,kind)
        row={'unit_id':uid}
        def add(feature,value,view,block,label,transform='identity',scale_hint='robust_observed',**extra):
            row[feature]=value
            definition=dict(feature=feature,view=view,block=block,label=label,
                            transform=transform,scale_hint=scale_hint,**extra)
            if feature in definitions:assert definitions[feature]==definition
            definitions[feature]=definition
        duration=r.active_s.to_numpy(float)
        gaps=np.maximum(0,r.start_s.to_numpy(float)[1:]-r.end_s.to_numpy(float)[:-1])
        # Internal gap=0 for a single run is a known structural absence, not an
        # imputation of an unobserved sensor value.
        gap_q50=float(np.quantile(gaps,.5)) if len(gaps) else 0.
        gap_q90=float(np.quantile(gaps,.9)) if len(gaps) else 0.
        add('active_s',unit.active_s,'structure','duration','実作動時間（秒）','log1p')
        add('active_duty',unit.active_s/unit.elapsed_s,'structure','continuity','経過時間中のHP2実作動割合')
        add('n_runs',len(r),'structure','repetition','原子的な作動run数','log1p')
        add('run_q50_s',float(np.quantile(duration,.5)),'structure','run_duration','個々の作動run長の中央値（秒）','log1p')
        add('run_q90_s',float(np.quantile(duration,.9)),'structure','run_duration','個々の作動run長のq90（秒）','log1p')
        add('off_gap_q50_s',gap_q50,'structure','off_gap','内部の既知OFF間隔中央値（秒、単一runは0）','log1p')
        add('off_gap_q90_s',gap_q90,'structure','off_gap','内部の既知OFF間隔q90（秒、単一runは0）','log1p')
        add('run_rate_per_active_min',len(r)/unit.active_s*60,'quality','derived_structure','run数/実作動分（構造特徴との重複につき主距離には入れない）')
        add('single_run',float(len(r)==1),'quality','structure_support','単一runで内部gapが無いことの指標')
        add('internal_gap_count',len(gaps),'quality','structure_support','観測された内部OFF間隔数')
        assert np.isclose(duration.sum(),unit.active_s,atol=1e-7),(uid,duration.sum(),unit.active_s)
        for signal,code in SIGNALS.items():
            if len(p):
                good=p[signal].notna()&np.isfinite(p[signal])
                values=p.loc[good,signal];weights=p.loc[good,'weight_s']
            else:values=pd.Series(dtype=float);weights=pd.Series(dtype=float)
            observed=float(weights.sum());fraction=observed/unit.active_s
            assert fraction<=1+1e-7,(uid,signal,fraction)
            q25=wq(values,weights,.25);q50=wq(values,weights,.5);q75=wq(values,weights,.75);q90=wq(values,weights,.9)
            iqr=q75-q25 if len(values) else np.nan
            view='rpm_audit' if signal=='rpm' else 'numeric'
            hint='fixed_4000_rpm_audit_only' if signal=='rpm' else 'robust_observed'
            for suffix,value in [('q50',q50),('q90',q90),('iqr',iqr)]:
                add(f'{signal}_{suffix}',value,view,signal,f'{signal}：RPM対応観測binの{suffix}',scale_hint=hint,code=code)
            add(f'{signal}_n_bins',len(values),'quality',signal+'_support',f'{signal}：RPMと対応する観測bin数',code=code)
            add(f'{signal}_observed_equivalent_s',observed,'quality',signal+'_support',f'{signal}：観測bin相当秒',code=code)
            add(f'{signal}_observed_fraction',fraction,'quality',signal+'_support',f'{signal}：作動時間中の観測bin相当割合',code=code)
            supports.append(dict(unit_level='block' if kind=='B' else 'same_hp_stage',unit_id=uid,
                case_id=unit.case_id,block_id=int(unit.block_id),signal=signal,code=code,
                n_bins=len(values),observed_equivalent_s=observed,observed_fraction=fraction,
                raw_q25=q25,raw_q50=q50,raw_q75=q75,raw_q90=q90,raw_iqr=iqr,
                support_basis='same HP2 run and integer-second bin as observed RPM; no forward fill'))
        known_predecessor=np.isfinite(unit.previous_hp) and np.isfinite(unit.unknown_gap_s) and unit.unknown_gap_s<=1e-8
        for hp in [1,2,3]:
            add(f'previous_hp{hp}',float(unit.previous_hp==hp) if known_predecessor else np.nan,
                'context','predecessor',f'直前作動HPがHP{hp}（未知前駆はNaN）')
        add('previous_gap_s',float(unit.previous_gap_s) if known_predecessor else np.nan,
            'context','predecessor','直前の作動HPからの既知間隔（秒）','log1p')
        add('first_observed_block',float(not np.isfinite(unit.previous_hp)),
            'context','position','最初に観測されたHP作動ブロック（真の最初を保証しない）')
        add('case_progress_start',unit.start_s/float(meta.loc[unit.case_id,'duration_s']),
            'context','position','症例経過時間に対する区間開始位置')
        add('post_hp3',float(unit.post_hp3) if known_predecessor else np.nan,
            'context','predecessor','既知HP3の次に現れるHP2')
        add('case_has_post_hp3_low',context.loc[unit.case_id,'has_post_hp3_low'],
            'context','case_sequence','同症例にHP3後低回転HP2が観測される')
        add('case_has_high_then_short_mid',context.loc[unit.case_id,'has_high_then_short_mid'],
            'context','case_sequence','同症例に高回転後短い中回転が観測される')
        add('high_fraction',getattr(unit,'high_fraction',np.nan),'rpm_audit','rpm_regime','正の観測回転数中の高回転帯割合')
        features.append(row)
    return pd.DataFrame(features),pd.DataFrame(definitions.values()),pd.DataFrame(supports)


def case_level(features,catalog,units,all_hp2):
    joined=units[['unit_id','case_id']].merge(features,on='unit_id',validate='one_to_one')
    definitions={};rows=[]
    all_active=all_hp2.groupby('case_id').active_s.sum()
    for case,g in joined.groupby('case_id',sort=False):
        unit_group=units[units.case_id==case]
        row={'case_id':case}
        for entry in catalog.itertuples():
            if entry.view=='numeric':
                supported=g[f'{entry.block}_n_bins']>=3
                values=pd.to_numeric(g.loc[supported,entry.feature],errors='coerce').dropna()
            else:
                values=pd.to_numeric(g[entry.feature],errors='coerce').dropna()
            if entry.view in ['structure','numeric','rpm_audit']:
                for suffix,value in [('median',values.median() if len(values) else np.nan),
                    ('iqr',values.quantile(.75)-values.quantile(.25) if len(values)>=2 else np.nan)]:
                    feature=entry.feature+'__'+suffix
                    row[feature]=value
                    # Raw feature aggregation is equal per primary block. Case
                    # IQR remains unknown with only one observed block.
                    definitions[feature]=dict(feature=feature,view=entry.view,block=entry.block,
                        label=entry.label+('：症例内ブロック中央値' if suffix=='median' else '：症例内ブロックIQR（2以上必要）'),
                        transform=entry.transform,scale_hint=entry.scale_hint)
                feature=entry.feature+'__n_observed_blocks';row[feature]=len(values)
                definitions[feature]=dict(feature=feature,view='quality',block='case_support',
                    label=entry.label+'：観測ブロック数',transform='identity',scale_hint='robust_observed')
            elif entry.view=='context':
                feature=entry.feature+'__fraction'
                row[feature]=values.mean() if len(values) else np.nan
                definitions[feature]=dict(feature=feature,view='context',block=entry.block,
                    label=entry.label+'：症例内平均/割合',transform=entry.transform,scale_hint=entry.scale_hint)
            elif entry.view=='quality':
                feature=entry.feature+'__median';row[feature]=values.median() if len(values) else np.nan
                definitions[feature]=dict(feature=feature,view='quality',block=entry.block,
                    label=entry.label+'：症例内中央値',transform=entry.transform,scale_hint=entry.scale_hint)
        additions=[('n_high_blocks',len(g),'高回転主体ブロック数','log1p'),
            ('high_active_total_s',unit_group.active_s.sum(),'高回転主体ブロックの実作動時間合計','log1p'),
            ('high_active_fraction_all_hp2',unit_group.active_s.sum()/all_active.loc[case],
             '全HP2実作動時間に対する高回転主体ブロックの割合','identity')]
        for feature,value,label,transform in additions:
            row[feature]=value;definitions[feature]=dict(feature=feature,view='structure',block='case_structure',
                label=label,transform=transform,scale_hint='robust_observed')
        for signal in SIGNALS:
            for suffix,value,label in [
                ('total_paired_bins',int(g[f'{signal}_n_bins'].sum()),'全sourceblockのRPM対応観測bin合計'),
                ('eligible_unit_count',int((g[f'{signal}_n_bins']>=3).sum()),'3bin以上観測されたblock数'),
                ('sourceall_unit_count',len(g),'source高回転主体block数')]:
                feature=f'{signal}__{suffix}';row[feature]=value
                definitions[feature]=dict(feature=feature,view='quality',block='case_support',
                    label=f'{signal}：{label}',transform='identity',scale_hint='robust_observed')
        rows.append(row)
    return pd.DataFrame(rows),pd.DataFrame(definitions.values())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();root=out.parent.parent
    source=root/'Output/chair_hp_sequence_300_v3_20261104'
    if (out/'features.csv').exists():raise FileExistsError('Use a new output version; existing feature tables are preserved.')
    (out/'process').mkdir(parents=True,exist_ok=True)
    def log(event,**info):
        record=dict(time_utc=datetime.now(timezone.utc).isoformat(),event=event,**info)
        with (out/'process/feature_log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(json.dumps(record,ensure_ascii=False),flush=True)
    started=time.time()
    names=['hp2_blocks.csv','run_to_block.csv','hp_runs.csv','hp2_points.csv','block_numeric_summary.csv',
           'blocks.csv','case_patterns.csv','case_manifest.csv','stage_blocks.csv','stage_hp2_points.csv','stage_hp2_blocks.csv']
    hashes={name:sha(source/name) for name in names}
    hp2=pd.read_csv(source/'hp2_blocks.csv')
    units=hp2[hp2.high_dominant].copy().reset_index(drop=True)
    assert len(units)==203 and units.case_id.nunique()==67
    units.insert(0,'unit_id',[unit_name(c,b,'B') for c,b in zip(units.case_id,units.block_id)])
    units['status']='FIXED_HIGH_DOMINANT_HP2'
    runs=pd.read_csv(source/'hp_runs.csv');runmap=pd.read_csv(source/'run_to_block.csv')
    mapped_runs=runs.drop(columns=['interval_id'],errors='ignore').merge(
        runmap[['case_id','run_id','block_id']],on=['case_id','run_id'],validate='one_to_one')
    selected_runs=mapped_runs.merge(units[['case_id','block_id']],on=['case_id','block_id'],validate='many_to_one')
    points=pd.read_csv(source/'hp2_points.csv',low_memory=False)
    manifest=pd.read_csv(source/'case_manifest.csv');cases=pd.read_csv(source/'case_patterns.csv')
    log('preflight',fixed_blocks=len(units),fixed_cases=units.case_id.nunique(),source_scope=len(manifest))
    feature,catalog,support=build_level(units,selected_runs,points,manifest,cases,'B')
    case_feature,case_catalog=case_level(feature,catalog,units,hp2)
    # Build exact run-to-stage mapping using retained atomic HP2 run intervals.
    stage_all=pd.read_csv(source/'stage_blocks.csv');stage_all=stage_all[stage_all.hp==2].copy()
    stage_full=pd.read_csv(source/'stage_hp2_blocks.csv')
    stage_runs=[]
    for case,g in mapped_runs[mapped_runs.hp==2].groupby('case_id',sort=False):
        stages=stage_all[stage_all.case_id==case].sort_values('start_s')
        start=stages.start_s.to_numpy(float);end=stages.end_s.to_numpy(float)
        ids=stages.block_id.to_numpy(int)
        for r in g.itertuples():
            ix=int(np.searchsorted(start,r.start_s,side='right')-1)
            assert ix>=0 and r.end_s<=end[ix]+1e-7,(case,r.run_id)
            rec=r._asdict();rec.pop('Index',None);rec['primary_block_id']=rec.pop('block_id');rec['block_id']=int(ids[ix]);stage_runs.append(rec)
    stage_runs=pd.DataFrame(stage_runs)
    linked=selected_runs[['case_id','run_id','block_id']].rename(columns={'block_id':'primary_block_id'}).merge(
        stage_runs[['case_id','run_id','block_id']],on=['case_id','run_id'],validate='one_to_one')
    containment=linked.groupby(['case_id','block_id']).primary_block_id.agg(lambda x:';'.join(str(int(v)) for v in sorted(set(x)))).reset_index()
    stage_units=stage_full.merge(containment,on=['case_id','block_id'],validate='one_to_one')
    stage_units=stage_units.rename(columns={'primary_block_id':'contains_primary_high_blocks'})
    stage_units.insert(0,'unit_id',[unit_name(c,b,'S') for c,b in zip(stage_units.case_id,stage_units.block_id)])
    stage_units['status']='STAGE_CONTAINS_FIXED_PRIMARY_HIGH_BLOCK'
    assert set(stage_units.case_id)==set(units.case_id)
    # Every primary high block belongs to precisely one stage; no selection on
    # whether the broader stage still qualifies as high-dominant.
    unique_links=linked[['case_id','primary_block_id','block_id']].drop_duplicates()
    assert len(unique_links)==203
    stage_points=pd.read_csv(source/'stage_hp2_points.csv',low_memory=False)
    sr=stage_runs.merge(stage_units[['case_id','block_id']],on=['case_id','block_id'],validate='many_to_one')
    stage_feature,stage_catalog,stage_support=build_level(stage_units,sr,stage_points,manifest,cases,'S')
    support=pd.concat([support,stage_support],ignore_index=True)
    # Preserve all300 source cases and explicit out-of-scope status.
    scope=manifest[['case_id','date','duration_s','signal_status']].merge(cases[['case_id','status']],on='case_id',validate='one_to_one')
    scope=scope.rename(columns={'status':'source_hp2_status'})
    scope['similarity_scope_status']=np.where(scope.case_id.isin(units.case_id),'ELIGIBLE_HIGH_HP2',
        np.where(scope.signal_status=='NO_TARGET_SIGNAL','NO_TARGET_SIGNAL',
        np.where(scope.source_hp2_status=='NO_HP_WORK_OBSERVATION','NO_HP_WORK_OBSERVATION','OUTSIDE_FIXED_HIGH_HP2_SUBSET')))
    scope=scope.merge(units.groupby('case_id').size().rename('n_fixed_high_blocks'),on='case_id',how='left')
    scope.n_fixed_high_blocks=scope.n_fixed_high_blocks.fillna(0).astype(int)
    outputs={'units.csv':units,'features.csv':feature,'feature_catalog.csv':catalog,
        'case_features.csv':case_feature,'case_feature_catalog.csv':case_catalog,
        'scope_audit.csv':scope,'signal_support.csv':support,'stage_units.csv':stage_units,
        'stage_features.csv':stage_feature,'stage_feature_catalog.csv':stage_catalog}
    for name,frame in outputs.items():frame.to_csv(out/name,index=False)
    assert not feature.unit_id.duplicated().any()
    assert not stage_feature.unit_id.duplicated().any()
    assert not case_feature.case_id.duplicated().any() and len(case_feature)==67
    for frame,key in [(feature,'unit_id'),(stage_feature,'unit_id'),(case_feature,'case_id')]:
        arr=frame.drop(columns=key).to_numpy(float)
        assert not np.isinf(arr).any()
    assert hashes=={name:sha(source/name) for name in names},'Source compact tables changed'
    metadata=dict(version='HP2_SIMILARITY_V4_FEATURES',fixed_primary_blocks=203,fixed_primary_cases=67,
        source_case_scope=300,stage_units=len(stage_units),stage_cases=stage_units.case_id.nunique(),
        n_block_features=len(catalog),n_case_features=len(case_catalog),n_stage_features=len(stage_catalog),
        feature_views=catalog.groupby('view').size().to_dict(),source= str(source),source_sha256=hashes,
        original_inputs_unchanged=True,script_sha256=sha(Path(__file__)),python=sys.version,
        numeric_policy='Raw weighted quantiles of jointly observed same-run/second RPM-anchored bins; missing stays NaN',
        structure_policy='Exact atomic HP2 run durations and known internal OFF gaps; single-run internal gap0 is observed structural absence',
        case_policy='Equal block weighting; numeric aggregates use only same-signal blocks with >=3 RPM-paired bins; IQR unknown with fewer than2 supported blocks',
        stage_policy='Stages selected by containment of original203 high blocks, not by stage high-dominant qualification',
        support_mask='Not applied in raw feature extraction; numeric minimum bins/coverage to be applied explicitly by analysis',
        rpm_policy='Audit only; tiny within-high RPM IQR must not dominate primary distance',
        elapsed_seconds=time.time()-started)
    (out/'feature_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    log('complete',**metadata)


if __name__=='__main__':main()
