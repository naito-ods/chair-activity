"""Leakage-aware neighboring-score and visible-annotation features.

Inputs are generic private tables; no dataset, clinical results or credentials
are embedded. Missing numeric scores use a neutral value AND availability mask.
"""

import json

import numpy as np

import pandas as pd

def context_features(meta,scores,available,raw,mode):
    lags={'current':[0],'past':[0,-1,-2,-4],'bidir':[0,-1,-2,-4,1,2,4],
          'neighbors':[-1,-2,-4,1,2,4]}[mode]
    n=len(meta);base=np.concatenate([np.clip(scores,-1,1)*available,available.astype(float)],axis=1)
    fields=[];names=[]
    for lag in lags:
        block=np.zeros_like(base);present=np.zeros((n,1))
        for _,g in meta.groupby('seq',sort=False):
            ix=g.index.to_numpy();src=np.arange(len(ix))+lag;good=(src>=0)&(src<len(ix))
            block[ix[good]]=base[ix[src[good]]];present[ix[good]]=1
        fields.extend([block,present]);names.extend([f'lag{lag}_score_or_available_{j}' for j in range(base.shape[1])]+[f'lag{lag}_present'])
    if mode!='neighbors':fields.append(raw);names.extend([f'current_raw_{j}' for j in range(raw.shape[1])])
    return np.concatenate(fields,axis=1),names

def anchor_features(visible,meta,classes):
    rows=[];provenance=[]
    for r in meta.itertuples():
        g=visible[visible.seq.eq(r.seq)]
        # Every training target removes its complete parent, not only this window.
        self_id=r.annotation_id if r.known_teacher else None
        if self_id is not None:g=g[g.id.ne(self_id)]
        before=g[(g.end_s<=r.start_s)&g.confirmed].sort_values('end_s',ascending=False).head(2)
        after=g[(g.start_s>=r.end_s)&g.confirmed].sort_values('start_s').head(2)
        features=[];used=[];mid=(r.start_s+r.end_s)/2
        for direction,q in [('previous',before),('next',after)]:
            records=list(q.itertuples())
            for j in range(2):
                onehot=np.zeros(len(classes));present=0.;distance=0.;duration=0.
                if j<len(records):
                    p=records[j];present=1.;distance=np.log1p(abs(mid-(p.end_s if direction=='previous' else p.start_s))/30)
                    duration=np.log1p((p.end_s-p.start_s)/30)
                    if p.treatment_normalized in classes:onehot[classes.index(p.treatment_normalized)]=1
                    used.append(p.id)
                features.extend(onehot.tolist()+[present,distance,duration])
        assert self_id not in used
        rows.append(features);provenance.append(dict(id=r.id,seq=int(r.seq),self_parent_excluded=self_id or '',anchor_ids=json.dumps(used)))
    return np.asarray(rows,float),pd.DataFrame(provenance)
