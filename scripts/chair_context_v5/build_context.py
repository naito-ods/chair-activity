#!/usr/bin/env python3
"""v5 whole-case context extraction; read-only compact v3 inputs, no DTW."""
from pathlib import Path
import argparse, hashlib, json, datetime
import numpy as np
import pandas as pd

VERSION='context-v5.2'
FILES=['case_manifest','hp_status_audit','hp_intervals','hp_runs','blocks','hp2_points','hp2_blocks','case_patterns']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def zones(x):
 return np.select([x.eq(0),x.gt(0)&x.le(5000),x.gt(5000)&x.lt(36000),x.ge(36000)&x.le(44000),x.gt(44000)],['Z','L','M','H','X'],default='UNKNOWN')
def has_motif(seq, pattern):
 """Ordered observed subsequence, reset by unknown; OFF is not an event."""
 n=0
 for s in seq:
  if s=='UNKNOWN': n=0
  elif s==pattern[n]:
   n+=1
   if n==len(pattern): return True
 return False

def synthetic_checks():
 assert list(zones(pd.Series([0,1,5000,5001,35999,36000,44000,44001,np.nan])))==['Z','L','L','M','M','H','H','X','UNKNOWN']
 assert has_motif(['HP3','L','H'],['HP3','L','H'])
 assert not has_motif(['HP3','UNKNOWN','L','H'],['HP3','L','H'])
 assert has_motif(['H','M','L','HP3'],['H','L','HP3'])
 return ['RPM boundary inclusive/exclusive checks','unknown barrier prevents motifs','ordered subsequence allows intervening known operations']

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--source',type=Path,default=Path('Output/chair_hp_sequence_300_v3_20261104')); ap.add_argument('--output',type=Path,default=Path('Output/chair_context_v5_20261104')); a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=True); (a.output/'process').mkdir(exist_ok=True)
 if (a.output/'case_context_features.csv').exists(): raise SystemExit('Refusing to overwrite existing extraction; use a new output directory')
 logpath=a.output/'process/feature_log.jsonl'
 def log(event,**kw):
  with logpath.open('a') as f: f.write(json.dumps(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event,**kw),ensure_ascii=False)+'\n')
 tests=synthetic_checks(); log('synthetic_checks_passed',checks=tests)
 hashes={n:sha(a.source/(n+'.csv')) for n in FILES}
 d={n:pd.read_csv(a.source/(n+'.csv'),low_memory=False) for n in FILES}
 manifest=d['case_manifest']; assert len(manifest)==300 and manifest.case_id.is_unique
 pts=d['hp2_points']; pts['zone']=zones(pts.rpm); pts=pts.sort_values(['case_id','run_id','time_s'])
 assert not pts.duplicated(['case_id','run_id','second_bin']).any()
 for name,t in d.items(): assert set(t.case_id)<=set(manifest.case_id),name
 log('inputs_validated',requested_cases=300,point_rows=len(pts),hashes=hashes)
 groups={n:{k:g for k,g in t.groupby('case_id',sort=False)} for n,t in d.items() if n!='case_manifest'}
 features=[]; scopes=[]; events=[]; transitions=[]; sequences=[]
 for c in manifest.itertuples(index=False):
  cid=c.case_id; dur=c.duration_s
  def tab(n): return groups[n].get(cid,d[n].iloc[:0])
  au=tab('hp_status_audit').iloc[0]; cp=tab('case_patterns').iloc[0]
  ints=tab('hp_intervals'); runs=tab('hp_runs').sort_values('start_s'); blocks=tab('blocks').sort_values('start_s'); b2=tab('hp2_blocks'); p=pts[pts.case_id.eq(cid)]
  unknown=ints[~ints.known.astype(bool)]
  known=au.hp_known_s>0; hp2seen=bool((runs.hp==2).any())
  f={'case_id':cid,'duration_s':dur,'hp_known_fraction':au.hp_known_s/dur if dur>0 else np.nan,'unknown_hp_s':au.unknown_s,'hp1_observed':int((runs.hp==1).any()) if known else np.nan,'hp3_observed':int((runs.hp==3).any()) if known else np.nan,'hp1_active_s':au.hp1_s if known else np.nan,'hp2_active_s':au.hp2_s if known else np.nan,'hp3_active_s':au.hp3_s if known else np.nan,'n_hp2_runs':int((runs.hp==2).sum()) if known else np.nan,'old_high_dominant':cp.has_high_dominant,'old_post_hp3_low':cp.has_post_hp3_low,'old_high_then_short_mid':cp.has_high_then_short_mid,'old_final_high_then_short_mid':cp.has_final_hp2_high_then_short_mid}
  for hp in [1,3]:
   hr=runs[runs.hp.eq(hp)]
   f[f'n_hp{hp}_runs']=len(hr) if known else np.nan
   f[f'hp{hp}_longest_run_s']=hr.active_s.max() if len(hr) else (0 if known else np.nan)
   for threshold in [2,5]: f[f'hp{hp}_supported_{threshold}s']=int(hr.active_s.sum()>=threshold) if known else np.nan
  pos=p[p.rpm>0]; positive_s=pos.weight_s.sum()
  f['hp2_observed_equivalent_s']=p.weight_s.sum() if hp2seen else np.nan
  f['hp2_positive_observed_equivalent_s']=positive_s if hp2seen else np.nan
  f['hp2_observed_coverage']=p.weight_s.sum()/au.hp2_s if au.hp2_s>0 else np.nan
  for z,label in [('L','low'),('M','mid'),('H','high')]:
   q=p[p.zone.eq(z)]; f[f'{label}_n_bins']=len(q) if hp2seen else np.nan; f[f'{label}_equivalent_s']=q.weight_s.sum() if hp2seen else np.nan; f[f'{label}_fraction']=q.weight_s.sum()/positive_s if positive_s>0 else np.nan; f[f'any_{label}_observed']=int(len(q)>0) if hp2seen else np.nan; f[f'{label}_supported']=int(len(q)>=3 and q.weight_s.sum()>=2) if hp2seen else np.nan
  high=p[p.zone.eq('H')]; first=high.time_s.min(); last=high.time_s.max()
  f.update(first_high_s=first,last_high_s=last,first_high_progress=first/dur if dur>0 else np.nan,last_high_progress=last/dur if dur>0 else np.nan)
  for band in ['mid','low']:
   bb=b2[b2.supported.astype(bool)&b2[f'{band}_fraction'].ge(.6)]
   f[f'n_{band}_dominant_blocks']=len(bb) if hp2seen else np.nan
   for label,mask in [('before_first_high',bb.end_s.le(first)),('after_last_high',bb.start_s.ge(last)),('between_first_last_high',bb.start_s.gt(first)&bb.end_s.lt(last))]:
    f[f'{band}_block_{label}']=int(mask.any()) if len(high) else np.nan
  def clear(s,e): return not ((unknown.start_s<e)&(unknown.end_s>s)).any()
  h3=runs[runs.hp.eq(3)]
  for key,mask in [('hp3_before_first_high',h3.end_s<=first),('hp3_after_last_high',h3.start_s>=last),('high_hp3_high',h3.start_s.gt(first)&h3.end_s.lt(last))]:
   selected=h3[mask]; candidates=[]
   for r in selected.itertuples():
    lo=min(first,r.start_s) if key!='hp3_after_last_high' else last
    hi=max(last,r.end_s) if key=='high_hp3_high' else max(first if key=='hp3_before_first_high' else last,r.end_s)
    candidates.append(clear(lo,hi))
   f[key]=int(any(candidates)) if len(high) and known else np.nan
  mids=[]; runstats={}
  for r in runs[runs.hp.eq(2)].itertuples():
   q=p[p.run_id.eq(r.run_id)]; qp=q[q.rpm>0]; m=q[q.zone.eq('M')]; total=qp.weight_s.sum(); coverage=total/r.active_s if r.active_s>0 else 0
   fractions={z:q.loc[q.zone.eq(z),'weight_s'].sum()/total if total>0 else np.nan for z in ['L','M','H']}
   supported=len(qp)>=3 and total>=2 and coverage>=.4 and r.active_s>=2
   dominant=next((z for z,v in fractions.items() if v>=.6),'MIXED') if supported else 'AMBIGUOUS'
   runstats[r.run_id]=(q,dominant,supported,fractions,coverage)
   if len(m):
    nearstart=(m.time_s-r.start_s<=3).all(); nearstop=(r.end_s-m.time_s<=3).all(); near_edges=((m.time_s-r.start_s<=3)|(r.end_s-m.time_s<=3)).all()
    mids.append((bool(supported and len(qp)>=3 and m.weight_s.sum()>=2 and fractions['M']>=.6),nearstart,nearstop,near_edges,bool(((m.time_s-r.start_s>3)&(r.end_s-m.time_s>3)).any())))
  f.update(n_mid_dominant_runs=sum(x[0] for x in mids) if hp2seen else np.nan,n_mid_only_near_start_runs=sum(x[1] for x in mids) if hp2seen else np.nan,n_mid_only_near_stop_runs=sum(x[2] for x in mids) if hp2seen else np.nan,n_mid_only_edges_runs=sum(x[3] for x in mids) if hp2seen else np.nan,n_mid_interior_runs=sum(x[4] for x in mids) if hp2seen else np.nan,n_mid_dominant_blocks=int((b2.supported.astype(bool)&b2.mid_fraction.ge(.6)).sum()) if hp2seen else np.nan)
  local=[]
  def add(level,kind,start,end,active,runids='',support=np.nan,extra=None):
   item=dict(case_id=cid,unit_level=level,kind=kind,start_s=start,end_s=end,active_s=active,run_ids=runids,supported=support)
   if extra:item.update(extra)
   local.append(item)
  for r in runs.itertuples():
   if r.hp!=2:
    for lev in ['run','observed_regime']: add(lev,'HP'+str(int(r.hp)),r.start_s,r.end_s,r.active_s,str(r.run_id),True)
   else:
    q,dom,sup,fr,cov=runstats[r.run_id]
    add('run','H2_'+dom,r.start_s,r.end_s,r.active_s,str(r.run_id),sup,dict(low_fraction=fr['L'],mid_fraction=fr['M'],high_fraction=fr['H'],observed_coverage=cov))
    # Observed bands retain all H support; gaps >2 seconds are explicit observation gaps, not filled.
    prev=None; episode=[]
    def finish(ep):
     if not ep:return
     z=ep[0].zone; add('observed_regime',z,max(r.start_s,ep[0].second_bin),min(r.end_s,ep[-1].second_bin+1),sum(x.weight_s for x in ep),str(r.run_id),len(ep)>=3 and sum(x.weight_s for x in ep)>=2,dict(n_bins=len(ep)))
    for x in q.itertuples():
     if prev is not None and (x.zone!=prev.zone or x.time_s-prev.time_s>2):
      finish(episode); episode=[]
      if x.time_s-prev.time_s>2: add('observed_regime','RPM_UNOBSERVED',min(r.end_s,prev.second_bin+1),max(r.start_s,x.second_bin),0,str(r.run_id),False)
     episode.append(x); prev=x
    finish(episode)
  for b in blocks.itertuples():
   extra={}; kind='HP'+str(int(b.hp)); sup=True
   if b.hp==2:
    bb=b2[b2.block_id.eq(b.block_id)].iloc[0]; sup=bool(bb.supported); fr={'L':bb.low_fraction,'M':bb.mid_fraction,'H':bb.high_fraction}; dom=next((z for z,v in fr.items() if v>=.6),'MIXED') if sup else 'AMBIGUOUS'; kind='H2_'+dom; extra=dict(low_fraction=fr['L'],mid_fraction=fr['M'],high_fraction=fr['H'],observed_coverage=bb.rpm_observed_fraction)
   ids=runs[(runs.start_s>=b.start_s)&(runs.end_s<=b.end_s)&(runs.hp==b.hp)].run_id.astype(str)
   add('block',kind,b.start_s,b.end_s,b.active_s,'|'.join(ids),sup,extra)
  for u in unknown.itertuples():
   for lev in ['run','block','observed_regime']: add(lev,'UNKNOWN',u.start_s,u.end_s,0,'',False)
  seqrow={'case_id':cid,'raw_hp_sequence':' > '.join('HP'+str(int(x)) for x in runs.hp)}
  for lev in ['block','run','observed_regime']:
   ev=sorted([x for x in local if x['unit_level']==lev],key=lambda x:(x['start_s'],x['end_s']))
   compressed=[]
   for i,e in enumerate(ev):
    e['token_id']=f'{lev}_{i:05d}'; events.append(e)
    if not compressed or compressed[-1]!=e['kind']: compressed.append(e['kind'])
    if i:
     prev=ev[i-1]; gap=max(0,e['start_s']-prev['end_s']); ug=sum(max(0,min(e['start_s'],u.end_s)-max(prev['end_s'],u.start_s)) for u in unknown.itertuples())
     transitions.append(dict(case_id=cid,unit_level=lev,from_token=prev['token_id'],to_token=e['token_id'],from_kind=prev['kind'],to_kind=e['kind'],gap_s=gap,unknown_gap_s=ug,confident_order=ug==0 and 'UNKNOWN' not in [e['kind'],prev['kind']]))
   seqrow[lev+'_sequence']=' > '.join(compressed)
   if lev=='block':
    f['block_high_mid_high']=int(has_motif(compressed,['H2_H','H2_M','H2_H'])) if hp2seen else np.nan
    f['block_high_low_high']=int(has_motif(compressed,['H2_H','H2_L','H2_H'])) if hp2seen else np.nan
   if lev=='observed_regime':
    # Missing RPM intervals break RPM motifs; UNKNOWN state also breaks them.
    obs=[x['kind'] if x['kind']!='RPM_UNOBSERVED' else 'UNKNOWN' for x in ev]
    for name,pat in [('hp3_low_high',['HP3','L','H']),('high_low_hp3',['H','L','HP3']),('low_high',['L','H']),('low_high_low',['L','H','L']),('high_mid_high',['H','M','H']),('high_mid',['H','M']),('high_hp3_high_observed',['H','HP3','H'])]: f['motif_'+name]=int(has_motif(obs,pat)) if hp2seen else np.nan
  rawhp=[]
  for x in sorted([dict(start_s=r.start_s,kind='HP'+str(int(r.hp))) for r in runs.itertuples()]+[dict(start_s=u.start_s,kind='UNKNOWN') for u in unknown.itertuples()],key=lambda x:x['start_s']):
   if not rawhp or rawhp[-1]!=x['kind']:rawhp.append(x['kind'])
  seqrow['compressed_hp_sequence']=' > '.join(rawhp); sequences.append(seqrow); features.append(f)
  included=bool(f['high_supported']==1); anyhigh=len(high)>0
  reason='INCLUDED_SUPPORTED_HP2_HIGH' if included else ('HIGH_SPARSE' if anyhigh else str(cp.status) if not hp2seen else 'NO_HP2_HIGH_OBSERVED')
  scopes.append(dict(case_id=cid,selected_source_cases=300,included_context_high=included,any_hp2_high=anyhigh,old_high_dominant=bool(cp.has_high_dominant==True),scope_reason=reason,high_n_bins=len(high),high_equivalent_s=high.weight_s.sum(),source_status=cp.status))
  if len(features)%50==0:log('cases_processed',count=len(features))
 F=pd.DataFrame(features); S=pd.DataFrame(scopes)
 assert S.included_context_high.sum()==97, S.included_context_high.sum()
 assert S.any_hp2_high.sum()==99
 assert S.old_high_dominant.sum()==67
 assert F.case_id.is_unique and len(F)==300
 output={'case_context_features':F,'cohort_scope':S,'context_events':pd.DataFrame(events),'context_transitions':pd.DataFrame(transitions),'case_sequences':pd.DataFrame(sequences)}
 catalog=[]
 for col in F.columns:
  if col=='case_id':continue
  block='support' if any(t in col for t in ['known','coverage','observed_equivalent']) else 'middle' if 'mid' in col else 'low' if 'low' in col else 'order' if 'motif' in col or 'before' in col or 'after' in col or 'hp3_high' in col else 'location' if 'first_' in col or 'last_' in col else 'hp_presence'
  kind='binary' if (col.startswith(('any_','old_','motif_','block_high_')) or col.endswith(('_supported','_observed','_2s','_5s')) or '_block_before_' in col or '_block_after_' in col or '_block_between_' in col or col in ['hp3_before_first_high','hp3_after_last_high','high_hp3_high']) else 'count' if col.startswith('n_') or col.endswith('_n_bins') else 'continuous'
  missing='NaN means no applicable HP2/HP observation or no H timestamp; zero means not observed in recorded portion, never confirmed clinical absence.'
  label=col.replace('_',' ')
  if 'fraction' in col and col.startswith(('low_','mid_','high_')):label+='; positive RPM overlap-weight proxy denominator, not continuously measured duration'
  if col.startswith('motif_'):label+='; ordered observed subsequence; UNKNOWN and RPM observation gaps >2s reset; intermediate known states allowed; not sustained phase proof'
  if 'only_near' in col:label+='; all MID observations within 3s of respective HP2 run edge(s), ramp candidate only'
  if '_block_before_' in col or '_block_after_' in col or '_block_between_' in col: label+='; entire supported >=60% band-dominant block lies before/after/strictly between raw first/last HP2 HIGH; no clinical phase implication'
  if col.startswith('block_high_'):label+='; ordered supported >=60% band-dominant macroblock subsequence; UNKNOWN resets'
  if col=='high_hp3_high':label+='; conservative UNKNOWN barrier check spans entire first-to-last HIGH interval'
  catalog.append(dict(feature=col,block=block,label=label,kind=kind,missing_policy=missing))
 output['feature_catalog']=pd.DataFrame(catalog)
 source=pd.DataFrame([dict(table=n,path=str((a.source/(n+'.csv')).resolve()),sha256=hashes[n],rows=len(d[n])) for n in FILES]); output['source_manifest']=source
 for n,t in output.items():t.to_csv(a.output/(n+'.csv'),index=False)
 after={n:sha(a.source/(n+'.csv')) for n in FILES}; assert after==hashes
 meta=dict(version=VERSION,requested_cases=300,actual_case_rows=len(F),supported_high_cases=int(S.included_context_high.sum()),any_high_cases=int(S.any_hp2_high.sum()),old_high_dominant_cases=int(S.old_high_dominant.sum()),parameters=dict(high_range=[36000,44000],low_max=5000,high_support_bins=3,high_support_equivalent_s=2,run_dominance=.6,run_support_bins=3,run_support_equivalent_s=2,run_coverage_min=.4,run_active_min_s=2,ramp_edge_s=3,observation_gap_reset_s=2),synthetic_checks=tests,source_hashes_unchanged=after==hashes,source_hashes=hashes,notes=['All 300 cases retained; four all-target missing cases remain explicitly outside primary high cohort.','Operation order is whole first-light-ON to final-light-OFF interval inherited unchanged from v3.','No DTW, no reassignment of Stage2 classes, no raw/wide source loaded.','No zero filling; numeric duration is run/bin overlap-weight observation proxy.','Block tokens may be mixed; observed_regime preserves short H inside mixed/M blocks.','RPM_UNOBSERVED and UNKNOWN break observed motif matching. OFF gaps are permitted and visible via transition gap_s.','Negative observed flags describe recorded portion; hp_known_fraction records state coverage.'])
 (a.output/'extraction_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
 log('extraction_complete',**{k:meta[k] for k in ['actual_case_rows','supported_high_cases','any_high_cases','old_high_dominant_cases']},event_rows=len(events),sources_unchanged=True)
 print(json.dumps({k:meta[k] for k in ['actual_case_rows','supported_high_cases','any_high_cases','old_high_dominant_cases']}))
if __name__=='__main__':main()
