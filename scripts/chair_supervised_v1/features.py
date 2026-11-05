"""Observed-only interval features for separately versioned supervised study.

No labels, annotation IDs, dates, seq numbers, or absolute timestamps become
features. State events persist until the next same-code event. Initial and
invalid states stay unknown. Numeric signals are NEVER forward-filled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATE_CODES={'20039':(0.,1.,2.,3.),'20011':(0.,1.,2.,3.),
             '20101':(0.,1.),'20774':(0.,1.),'20776':None}
NUMERIC_CODES=['20742','20743','20108','10014','10015']
READ_CODES=list(STATE_CODES)+NUMERIC_CODES+['90012']

def state_segments(events,code,start,end):
    """Exact clipped state partition. Every unknown segment remains NaN."""
    q=events[events.code.eq(str(code))].sort_values('time_s',kind='stable')
    q=q.drop_duplicates('time_s',keep='last')
    t=q.time_s.to_numpy(float);v=q.value.to_numpy(float)
    good=np.isfinite(v)
    allowed=STATE_CODES[str(code)]
    if allowed is not None:good &= np.isin(v,allowed)
    else:good &= v>=0
    v=np.where(good,v,np.nan)
    cuts=np.unique(np.r_[float(start),t[(t>start)&(t<end)],float(end)])
    indices=np.searchsorted(t,cuts[:-1],side='right')-1
    values=np.full(len(indices),np.nan);seen=indices>=0
    values[seen]=v[indices[seen]]
    rows=[]
    for a,b,c in zip(cuts[:-1],cuts[1:],values):
        if b<=a:continue
        same=bool(rows and ((np.isnan(c) and np.isnan(rows[-1][2])) or c==rows[-1][2]))
        if same:rows[-1][1]=float(b)
        else:rows.append([float(a),float(b),float(c)])
    ans=np.asarray(rows,float).reshape(-1,3)
    assert np.isclose(np.sum(ans[:,1]-ans[:,0]),end-start)
    return ans

def state_lookup(segments,times):
    t=np.asarray(times,float);out=np.full(len(t),np.nan)
    ix=np.searchsorted(segments[:,0],t,side='right')-1
    good=(ix>=0)&(t<segments[-1,1])
    out[good]=segments[ix[good],2]
    return out

def weighted_quantile(x,w,q):
    x=np.asarray(x,float);w=np.asarray(w,float)
    good=np.isfinite(x)&np.isfinite(w)&(w>0)
    if not good.any():return np.nan
    x=x[good];w=w[good];order=np.argsort(x,kind='stable');x=x[order];w=w[order]
    i=np.searchsorted(np.cumsum(w),q*w.sum(),side='left')
    return float(x[min(i,len(x)-1)])

def numeric_summary(values,prefix,result):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    result[prefix+'_n_obs']=float(len(x))
    for name,q in [('q10',.1),('q25',.25),('q50',.5),('q75',.75),('q90',.9)]:
        result[prefix+'_'+name]=float(np.quantile(x,q)) if len(x) else np.nan
    result[prefix+'_mean']=float(x.mean()) if len(x) else np.nan
    result[prefix+'_std']=float(x.std()) if len(x) else np.nan
    result[prefix+'_range']=float(np.ptp(x)) if len(x) else np.nan

def extract_features(events,start,end,include_context=False,profile_bins=0):
    """Return numeric features for [start,end), in full-raw relative seconds.

    events: pandas DataFrame with time_s(float), code(str), value(float/NaN).
    include_context permits offline +/-60sec context, never annotation labels.
    profile_bins adds within-interval HP/RPM temporal summaries. The caller must
    keep all windows from a case together in every train/evaluation split.
    """
    start=float(start);end=float(end)
    if not np.isfinite(start+end) or end<=start:raise ValueError('Invalid interval')
    if not {'time_s','code','value'}<=set(events):raise ValueError('Missing event columns')
    duration=end-start
    win=events[events.time_s.ge(start)&events.time_s.lt(end)]
    result={'duration_s':duration,'quality_target_event_count':float(len(win))}
    result['quality_target_events_per_s']=len(win)/duration
    hp=state_segments(events,'20039',start,end)
    for code in READ_CODES:
        q=win[win.code.eq(code)]
        result[f'count_{code}']=float(len(q));result[f'rate_{code}']=len(q)/duration
    for code,allowed in STATE_CODES.items():
        state=hp if code=='20039' else state_segments(events,code,start,end)
        widths=state[:,1]-state[:,0];known=np.isfinite(state[:,2]);known_s=float(widths[known].sum())
        result[f'state_{code}_known_fraction']=known_s/duration
        result[f'state_{code}_unknown_fraction']=1-known_s/duration
        result[f'state_{code}_n_changes']=float(sum(np.isfinite(a) and np.isfinite(b) and a!=b for a,b in zip(state[:-1,2],state[1:,2])))
        if allowed is None:
            result[f'state_{code}_weighted_mean']=float(np.sum(state[known,2]*widths[known])/known_s) if known_s else np.nan
            for name,q in [('q10',.1),('q50',.5),('q90',.9)]:result[f'state_{code}_{name}']=weighted_quantile(state[:,2],widths,q)
            continue
        for value in allowed:
            own=widths[state[:,2]==value]
            result[f'state_{code}_{int(value)}_share_known']=float(own.sum()/known_s) if known_s else np.nan
            # Proportion of known duration, never an assertion about unknown time.
            if code=='20039' and value>0:
                label=f'hp{int(value)}';ownrows=state[state[:,2]==value]
                result[label+'_n_runs']=float(len(own)) if known_s else np.nan
                result[label+'_active_s']=float(own.sum()) if known_s else np.nan
                result[label+'_run_q50_s']=float(np.median(own)) if len(own) else np.nan
                result[label+'_run_q90_s']=float(np.quantile(own,.9)) if len(own) else np.nan
                result[label+'_first_progress']=float((ownrows[0,0]-start)/duration) if len(ownrows) else np.nan
                result[label+'_last_progress']=float((ownrows[-1,1]-start)/duration) if len(ownrows) else np.nan
    # Strict adjacent transitions do not bridge OFF or UNKNOWN intervals.
    for a in [0,1,2,3]:
        for b in [0,1,2,3]:
            if a==b:continue
            result[f'hp_transition_{a}_{b}']=float(np.sum((hp[:-1,2]==a)&(hp[1:,2]==b)))
    off=hp[hp[:,2]==0]
    result['off_run_q50_s']=float(np.median(off[:,1]-off[:,0])) if len(off) else np.nan
    result['off_run_q90_s']=float(np.quantile(off[:,1]-off[:,0],.9)) if len(off) else np.nan
    for code in NUMERIC_CODES:
        q=win[win.code.eq(code)&np.isfinite(win.value)]
        numeric_summary(q.value,f'num_{code}',result)
        result[f'num_{code}_observed_second_count']=float(np.unique(np.floor(q.time_s)).size)
        if code in ['20742','20743']:
            states=state_lookup(hp,q.time_s)
            for h in [1,2,3]:
                vals=q.value.to_numpy()[states==h]
                result[f'num_{code}_hp{h}_n']=float(len(vals))
                result[f'num_{code}_hp{h}_q50']=float(np.median(vals)) if len(vals) else np.nan
    rpm=win[win.code.eq('20742')&np.isfinite(win.value)].copy()
    hv=state_lookup(hp,rpm.time_s);rpm=rpm.loc[hv==2]
    positive=rpm[rpm.value>0]
    n=len(positive)
    zones={'low':positive.value.le(5000),'mid':positive.value.gt(5000)&positive.value.lt(36000),
           'high':positive.value.ge(36000)&positive.value.le(44000),'above_high':positive.value.gt(44000)}
    for zone,mask in zones.items():
        result[f'rpm_hp2_{zone}_observed_fraction']=float(mask.sum()/n) if n else np.nan
    result['rpm_hp2_zero_observed_fraction']=float(rpm.value.eq(0).mean()) if len(rpm) else np.nan
    high=rpm[rpm.value.ge(36000)&rpm.value.le(44000)]
    result['rpm_hp2_first_high_progress']=float((high.time_s.min()-start)/duration) if len(high) else np.nan
    result['rpm_hp2_last_high_progress']=float((high.time_s.max()-start)/duration) if len(high) else np.nan
    h3=hp[hp[:,2]==3]
    def unknown_between(a,b):
        w=np.maximum(0,np.minimum(hp[:,1],b)-np.maximum(hp[:,0],a))
        return bool(w[np.isnan(hp[:,2])].sum()>1e-9)
    result['hp3_before_first_high']=float(any(r[1]<=high.time_s.min() and not unknown_between(r[1],high.time_s.min()) for r in h3)) if len(high) else np.nan
    result['hp3_after_last_high']=float(any(r[0]>=high.time_s.max() and not unknown_between(high.time_s.max(),r[0]) for r in h3)) if len(high) else np.nan
    if profile_bins:
        for i in range(int(profile_bins)):
            a=start+duration*i/profile_bins;b=start+duration*(i+1)/profile_bins
            sub=extract_features(events,a,b,False,0)
            selected=[k for k in sub if k.startswith('state_20039_') and k.endswith('share_known')]
            selected+=['state_20039_known_fraction','num_20742_q50','num_20743_q50','num_20108_q50',
                       'num_10015_q50','rpm_hp2_low_observed_fraction','rpm_hp2_mid_observed_fraction','rpm_hp2_high_observed_fraction']
            for k in selected:result[f'profile_{i}_{k}']=sub[k]
        # Temporal trend is defined only when both actual halves have evidence.
        for code in NUMERIC_CODES:
            q=win[win.code.eq(code)&np.isfinite(win.value)]
            early=q.loc[q.time_s<start+duration/2,'value'];late=q.loc[q.time_s>=start+duration/2,'value']
            result[f'num_{code}_late_minus_early']=float(late.median()-early.median()) if len(early)>=3 and len(late)>=3 else np.nan
    if include_context:
        full_start=float(events.time_s.min());full_end=float(events.time_s.max())+1e-6
        for prefix,a,b in [('context_before',max(full_start,start-60),start),('context_after',end,min(full_end,end+60))]:
            available=max(0,b-a);result[prefix+'_available_s']=available
            if available<=0:
                # Schema uses a valid dummy temporal query, then erases values.
                ctx=extract_features(events,start,end,False,0);ctx={k:np.nan for k in ctx}
            else:ctx=extract_features(events,a,b,False,0)
            for k in ['state_20039_known_fraction','state_20039_0_share_known','state_20039_1_share_known',
                      'state_20039_2_share_known','state_20039_3_share_known','num_20742_q50','num_20743_q50',
                      'num_10015_q50','rpm_hp2_low_observed_fraction','rpm_hp2_mid_observed_fraction','rpm_hp2_high_observed_fraction',
                      'hp1_n_runs','hp2_n_runs','hp3_n_runs','state_20101_0_share_known','state_20774_1_share_known']:
                result[prefix+'_'+k]=ctx[k]
    assert all(np.isscalar(v) for v in result.values())
    if any(np.isinf(v) for v in result.values()):raise ValueError('Infinite feature')
    return result

def synthetic_checks():
    e=pd.DataFrame({'time_s':[2.,4.,5.,6.,7.,8.,9.],
        'code':['20039','20742','20039','20742','20039','20742','20039'],
        'value':[2.,0.,0.,100.,99.,40000.,3.]})
    h=state_segments(e,'20039',0,10)
    assert np.allclose(h[:,0],[0,2,5,7,9])
    assert np.isnan(h[0,2]) and np.isnan(h[3,2])
    x=extract_features(e,0,10)
    assert x['state_20039_known_fraction']==.6
    assert np.isnan(x['num_20743_q50']) and x['num_20743_n_obs']==0
    assert x['num_20742_q50']==100 and x['num_20742_hp2_q50']==0
    assert x['rpm_hp2_zero_observed_fraction']==1
    # State prior to an annotation boundary is retained; future state is not.
    assert state_segments(e,'20039',3,4)[0,2]==2
    assert np.isclose(sum(state_segments(e,'20039',3,8)[:,1]-state_segments(e,'20039',3,8)[:,0]),5)
    y=extract_features(e,0,10,True,3)
    assert set(x)<=set(y) and not any(np.isinf(v) for v in y.values())
    return {'initial_unknown_preserved':True,'invalid_state_unknown':True,'zero_distinct_from_missing':True,
            'exact_state_duration_partition':True,'asof_state_boundary':True,'observed_numeric_no_ffill':True,
            'profile_and_context_schema_finite_or_nan':True}

if __name__=='__main__':
    import json
    print(json.dumps(synthetic_checks()))
