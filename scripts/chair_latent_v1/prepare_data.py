"""Read-only, case-level macro-sequence export for annotation-free latent study.

This is a separately versioned branch. No Activity boundaries or old labels are
changed. Numeric bin summaries are observed-only statistics, never held states.
"""
from pathlib import Path
import argparse, hashlib, json, datetime
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'Output/chair_hp_sequence_300_v3_20261104'
CHANNELS=['OFF_share','HP1_share','HP2_share','HP3_share','RPM','torque','pedal2']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--bins',type=int,default=48);a=ap.parse_args();out=a.output
    out.mkdir(parents=True,exist_ok=True);(out/'process').mkdir(exist_ok=True)
    if (out/'data.npz').exists():raise FileExistsError('Preserve previous export; use new output')
    names=['case_manifest.csv','hp_intervals.csv','hp_runs.csv','hp2_points.csv','case_patterns.csv','observations.csv']
    before={n:sha(SOURCE/n) for n in names}
    m=pd.read_csv(SOURCE/'case_manifest.csv');patterns=pd.read_csv(SOURCE/'case_patterns.csv')
    intervals=pd.read_csv(SOURCE/'hp_intervals.csv');runs=pd.read_csv(SOURCE/'hp_runs.csv')
    assert len(m)==300 and m.case_id.is_unique and set(m.case_id)==set(patterns.case_id)==set(intervals.case_id)
    assert not intervals.duplicated(['case_id','interval_id']).any()
    assert not runs.duplicated(['case_id','run_id']).any()
    print(json.dumps(dict(event='preflight',requested_cases=300,unit='whole original case',bins=a.bins,
        requested_hp2_evaluable=int(patterns.status.eq('HP2_PATTERN_EVALUABLE').sum()))),flush=True)
    integrity=[]
    for row in m.itertuples():
        source=Path(row.source_path);meta=source.parent/'meta.json'
        sh=sha(source);mh=sha(meta)
        integrity.append(dict(case_id=row.case_id,source_path=str(source),source_sha256=sh,
            metadata_sha256=mh,source_matches=sh==row.source_sha256,metadata_matches=mh==row.meta_sha256))
    integ=pd.DataFrame(integrity);integ.to_csv(out/'source_integrity_before.csv',index=False)
    assert integ.source_matches.all() and integ.metadata_matches.all()
    obs=pd.read_csv(SOURCE/'observations.csv',usecols=['case_id','code','time_s','value','hp','hp_known','run_id','interval_id','second_bin'],dtype={'code':str})
    assert set(obs.case_id)<=set(m.case_id)
    assert not obs.duplicated(['case_id','code','interval_id','second_bin']).any()
    obs=obs[obs.hp.eq(2)&obs.code.isin(['20742','20743','10015'])].copy()
    duration=m.set_index('case_id').duration_s
    obs['progress_bin']=np.floor(obs.time_s/obs.case_id.map(duration)*a.bins).astype(int).clip(0,a.bins-1)
    raw=np.full((len(m),a.bins,7),np.nan);support=np.zeros_like(raw)
    groups={k:v for k,v in intervals.groupby('case_id')};og={k:v for k,v in obs.groupby('case_id')}
    for i,row in enumerate(m.itertuples()):
        edges=np.linspace(0,row.duration_s,a.bins+1);width=row.duration_s/a.bins
        ints=groups[row.case_id]
        known=np.zeros(a.bins);state=np.zeros((a.bins,4))
        assert np.isclose(ints.duration_s.sum(),row.duration_s)
        for r in ints.itertuples():
            overlap=np.maximum(0,np.minimum(edges[1:],r.end_s)-np.maximum(edges[:-1],r.start_s))
            if np.isfinite(r.hp):known+=overlap;state[:,int(r.hp)]+=overlap
        good=known>=.5*width
        raw[i,good,:4]=state[good]/known[good,None]
        support[i,:,:4]=known[:,None]
        q=og.get(row.case_id)
        if q is None:continue
        for (code,b),g in q.groupby(['code','progress_bin']):
            j={'20742':4,'20743':5,'10015':6}[code]
            support[i,b,j]=len(g)
            if len(g)>=3:raw[i,b,j]=float(g.value.median())
    eligible=patterns.set_index('case_id').loc[m.case_id].status.eq('HP2_PATTERN_EVALUABLE').to_numpy()
    ledger=m[['case_id','date','chair','duration_s','signal_status']].copy()
    ledger['hp2_source_status']=patterns.set_index('case_id').loc[m.case_id].status.to_numpy()
    ledger['eligible']=eligible
    for j,c in enumerate(CHANNELS):ledger[c+'_observed_bins']=np.isfinite(raw[:,:,j]).sum(1)
    assert eligible.sum()==249
    assert (ledger.loc[eligible,'RPM_observed_bins']>0).all()
    # Fixed random split before observing any model scores. Same case never crosses splits.
    rng=np.random.default_rng(20261104);ix=rng.permutation(np.flatnonzero(eligible))
    ledger['split']='OUTSIDE_HP2_EVALUATION'
    for label,part in [('train',ix[:149]),('validation',ix[149:199]),('test',ix[199:])]:ledger.loc[part,'split']=label
    masks=np.ones((len(m),4,a.bins),bool)
    for i in range(len(m)):
        # Four complementary hidden layouts: each target is evaluated exactly once.
        order=np.random.default_rng(20261104+100+i).permutation(a.bins//3)
        for rep in range(4):
            chunks=order[rep*(len(order)//4):(rep+1)*(len(order)//4)]
            hidden=np.concatenate([np.arange(3*k,3*k+3) for k in chunks])
            masks[i,rep,hidden]=False
        assert (~masks[i]).sum(0).min()==1 and (~masks[i]).sum(0).max()==1
    np.savez_compressed(out/'data.npz',raw=raw,support=support,numeric_visible=masks,case_ids=m.case_id.to_numpy(str),channels=np.array(CHANNELS))
    ledger.to_csv(out/'case_ledger.csv',index=False)
    m.to_csv(out/'case_manifest.csv',index=False)
    plan=dict(version='chair-latent-v1',seed=20261104,requested_cases=300,hp2_evaluable_cases=249,
        bins=a.bins,channels=CHANNELS,numeric_aggregation='HP2 observed-only per-case progress-bin median; >=3 observed run/second samples',
        hp_state='exact interval overlaps, occupancy among known duration; >=50% bin state known',
        state_conditioning='Full observed HP state allowed to all models. Only numeric values hidden.',
        missing='NaN + explicit support/mask. No numeric ffill or interpolation. Internal model estimates never become observations.',
        evaluation='Four complementary layouts hide contiguous 3-bin chunks, 25% bins each, all numeric channels jointly. Numeric targets only.',
        splits=ledger.split.value_counts().to_dict(),case_ids_by_split={s:ledger.loc[ledger.split.eq(s),'case_id'].tolist() for s in ['train','validation','test']},
        primary='paired case-mean of channel-mean standardized numeric squared error; same hidden observed targets for every model',
        acceptance=dict(practical_relative_gain=.05,paired_bootstrap_ci_lower_gt_zero=True,
            beats_semantic_and_linear=True,mask_control_not_comparable=True,raw_recurrence_beyond_known_flags_required=True),
        interpretation='No clinical treatment classes. No fixed k. 48-bin macro representation cannot validate short MID tails.',
        original_data_readonly=True,source_sha256=before,
        reference='https://pmc.ncbi.nlm.nih.gov/articles/PMC8183613/',
        scope='cryoDRGN-inspired masked continuous latent representation, not cryoDRGN cryo-EM reconstruction software')
    (out/'analysis_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    after={n:sha(SOURCE/n) for n in names};assert before==after
    (out/'process/preflight.json').write_text(json.dumps(dict(sources_unchanged=True,source_hashes=before,
        unique_cases=len(m),unique_hp_intervals=len(intervals),unique_hp_runs=len(runs),
        duplicate_case_keys=0,duplicate_interval_keys=0,duplicate_run_keys=0,all_300_original_file_hashes_verified=True,
        prepared_at=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
    print(json.dumps(dict(event='prepared',split_counts=ledger.split.value_counts().to_dict(),raw_shape=raw.shape,
        numeric_observed_bins=np.isfinite(raw[:,:,4:]).sum((0,1)).tolist(),sources_unchanged=True)),flush=True)

if __name__=='__main__':main()
