"""Post-freeze inspection of a continuous linear latent representation.

Run ONLY after final_evaluation.py evaluate. All full-observation analyses here
are exploratory posthoc descriptions; they must not be used for further tuning
or described as independent validation. No clusters/classes are imposed. The
linear latent core is a conditional low-rank model, not a neural VAE/cryoDRGN.
"""
from pathlib import Path
import argparse
import datetime
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import spearmanr

from experiment import MaskedFactors
from final_evaluation import load_linear
from prepare_data import SOURCE, CHANNELS, ROOT, sha
from semantic_baseline import semantic_features


def robust_scale(x, train):
    """Training-only coordinate median/IQR; explicit NaNs remain missing."""
    center = np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0 for v in x[train].T])
    scale = np.array([max(.1, float(np.subtract(*np.percentile(v[np.isfinite(v)], [75,25]))))
                      if np.isfinite(v).any() else 1.0 for v in x[train].T])
    supported = np.isfinite(x[train]).sum(0) >= 2
    return (x-center)/scale, center, scale, supported


def pairwise_observed_l1(x):
    """Mean normalized absolute difference over shared finite coordinates."""
    n = len(x)
    distance = np.full((n,n), np.nan)
    shared = np.zeros((n,n), dtype=int)
    for i in range(n):
        valid = np.isfinite(x[i]) & np.isfinite(x)
        count = valid.sum(1)
        diff = np.where(valid, np.abs(x-x[i]), 0.0).sum(1)
        distance[i] = np.divide(diff,count,out=np.full(n,np.nan),where=count>0)
        shared[i] = count
    return distance, shared


def whitening(z, train):
    center = z[train].mean(0)
    covariance = np.atleast_2d(np.cov(z[train], rowvar=False))
    vals, vectors = np.linalg.eigh(covariance)
    floor = max(1e-8, float(vals.max())*1e-6)
    matrix = (vectors/np.sqrt(np.maximum(vals,floor))[None,:]) @ vectors.T
    return (z-center)@matrix, center, matrix, vals


def rho(x,y, minimum=3):
    mask = np.isfinite(x)&np.isfinite(y)
    n = int(mask.sum())
    if n < minimum or np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return np.nan, n
    return float(spearmanr(x[mask],y[mask]).statistic), n


def full_semantic(out):
    """Recompute the exact frozen comparator using each signal's actual time."""
    manifest = pd.read_csv(out/'case_manifest.csv')
    intervals = pd.read_csv(SOURCE/'hp_intervals.csv')
    runs = pd.read_csv(SOURCE/'hp_runs.csv')
    points = pd.read_csv(SOURCE/'hp2_points.csv', low_memory=False)
    stamps = pd.read_csv(SOURCE/'observations.csv',
                         usecols=['case_id','code','time_s','run_id','second_bin','hp'],dtype={'code':str})
    stamps = stamps[stamps.hp.eq(2)&stamps.code.isin(['20742','20743','10015'])]
    keys = ['case_id','run_id','second_bin']
    for code,signal in [('20742','rpm'),('20743','torque'),('10015','pedal2')]:
        times = stamps[stamps.code.eq(code)][keys+['time_s']].rename(columns={'time_s':signal+'_time_s'})
        assert not times.duplicated(keys).any()
        points = points.merge(times,on=keys,how='left',validate='one_to_one')
    features, names = semantic_features(manifest,intervals,runs,points,np.ones((len(manifest),48),bool),bins=48)
    return features,names


def inspect(out, repeats=10):
    # This guard precedes all test data reads. The inspection cannot accidentally
    # become a development-stage visualization or a covert model selector.
    if not (out/'test_summary.json').is_file() or not (out/'selection_freeze.json').is_file():
        raise RuntimeError('Final selection and test evaluation must finish before exploratory inspection')
    if (out/'representation_summary.json').exists():
        raise FileExistsError('Representation inspection already exists; preserve it')
    frozen = json.loads((out/'selection_freeze.json').read_text())
    test_summary = json.loads((out/'test_summary.json').read_text())
    data = np.load(out/'data.npz')
    ledger = pd.read_csv(out/'case_ledger.csv')
    rows = pd.read_csv(out/'development_comparison.csv').set_index('model')
    candidate = frozen['choices']['candidate']
    name = str(rows.loc[candidate,'latent_model']) if rows.loc[candidate,'family'] in ['linear_local','hybrid'] else frozen['choices']['linear']
    if rows.loc[name,'family'] != 'linear':
        name = frozen['choices']['linear']
    path = out/'models'/f'{name}.npz'
    before = {str(p):sha(p) for p in [out/'data.npz',out/'scaling.npz',path,out/'case_ledger.csv',out/'selection_freeze.json',out/'test_summary.json']}
    for p in [SOURCE/'observations.csv',SOURCE/'hp2_points.csv',SOURCE/'hp_intervals.csv',SOURCE/'hp_runs.csv',SOURCE/'case_patterns.csv']:
        before[str(p)] = sha(p)
    ids = np.flatnonzero(ledger.split.isin(['train','validation','test']))
    l = ledger.iloc[ids].reset_index(drop=True)
    train = np.flatnonzero(l.split.eq('train'))
    test = np.flatnonzero(l.split.eq('test'))
    assert len(ids)==249 and len(train)==149 and len(test)==50 and l.case_id.is_unique
    assert l.case_id.tolist() == np.asarray(data['case_ids'])[ids].tolist()
    raw = data['raw'][ids]
    scaling = np.load(out/'scaling.npz')
    x = raw.copy(); x[:,:,4] /= 40000
    for c in [5,6]:x[:,:,c] = np.log1p(x[:,:,c])
    x = (x-scaling['center'])/scaling['scale']
    a = x.reshape(len(x),-1)
    weights = np.sqrt(np.tile([.25,.25,.25,.25,1,1,1],48))
    a = a*weights
    visible = np.isfinite(a)
    model = load_linear(path)
    z = model.transform(a,visible)
    zw,zc,whiten,eigenvalues = whitening(z,train)
    latent_distance = squareform(pdist(zw))
    coordinates = l[['case_id','split']].copy()
    for j in range(model.k):
        coordinates[f'z{j+1}'] = z[:,j]
        coordinates[f'whitened_z{j+1}'] = zw[:,j]
    patterns = pd.read_csv(SOURCE/'case_patterns.csv')
    pattern_cols = ['has_post_hp3_low','has_high_dominant','has_high_then_short_mid','has_final_hp2_high_then_short_mid']
    coordinates = coordinates.merge(patterns[['case_id']+pattern_cols],on='case_id',how='left',validate='one_to_one')
    # Full comparator features are recomputed, never copied from masked views or
    # old report labels. Labels above are descriptive only, after all training.
    full_features,names = full_semantic(out)
    features = full_features[ids]
    fs,fc,fscale,used = robust_scale(features,train)
    semantic_distance,shared = pairwise_observed_l1(fs[:,used])
    np.savez_compressed(out/'semantic_fullvisible.npz',features=features,feature_names=np.array(names),
                        training_center=fc,training_scale=fscale,training_supported=used,case_ids=l.case_id.to_numpy(str))
    numeric_support = np.isfinite(raw[:,:,4:]).mean(1)
    context = pd.read_csv(ROOT/'Output/chair_context_v5_20261104/case_context_features.csv').set_index('case_id')
    quality = np.column_stack([context.loc[l.case_id,'hp_known_fraction'].to_numpy(float),
                               numeric_support,np.log1p(data['support'][ids,:,4:].sum(1))])
    qs,qc,qscale,qused = robust_scale(quality,train)
    quality_distance,_ = pairwise_observed_l1(qs[:,qused])
    pair_index = np.triu_indices(len(l),1)
    quality_rho,n_pairs = rho(latent_distance[pair_index],quality_distance[pair_index])
    semantic_rho,semantic_pairs = rho(latent_distance[pair_index],semantic_distance[pair_index])
    pd.DataFrame([dict(comparison='latent_vs_quality_distance',spearman=quality_rho,n_pairs=n_pairs),
                  dict(comparison='latent_vs_semantic_distance',spearman=semantic_rho,n_pairs=semantic_pairs)]).to_csv(out/'representation_distance_audit.csv',index=False)
    neighbors = []
    agreement = []
    for i in test:
        latent_near = train[np.argsort(latent_distance[i,train],kind='stable')[:5]]
        allowed = train[np.isfinite(semantic_distance[i,train])]
        semantic_near = allowed[np.argsort(semantic_distance[i,allowed],kind='stable')[:5]]
        inter = len(set(latent_near)&set(semantic_near)); union = len(set(latent_near)|set(semantic_near))
        agreement.append(dict(case_id=l.case_id.iloc[i],latent_semantic_top5_jaccard=inter/union if union else np.nan))
        for method,indices,distance in [('latent',latent_near,latent_distance),('semantic',semantic_near,semantic_distance)]:
            for rank,j in enumerate(indices,1):
                row = dict(case_id=l.case_id.iloc[i],query_split='test',method=method,rank=rank,
                           neighbor_case_id=l.case_id.iloc[j],neighbor_split='train',distance=float(distance[i,j]),
                           shared_semantic_features=int(shared[i,j]),quality_distance=float(quality_distance[i,j]))
                for flag in pattern_cols:
                    row['query_'+flag] = coordinates.loc[i,flag]
                    row['neighbor_'+flag] = coordinates.loc[j,flag]
                neighbors.append(row)
    pd.DataFrame(neighbors).to_csv(out/'representation_neighbors.csv',index=False)
    pd.DataFrame(agreement).to_csv(out/'representation_neighbor_agreement.csv',index=False)
    stability = []
    for rep in range(repeats):
        retained = np.sort(np.random.default_rng(20261104+30000+rep).choice(train,round(.8*len(train)),replace=False))
        refit = MaskedFactors(model.k,iterations=model.iterations,regularization=model.reg).fit(a[retained])
        refit_z = refit.transform(a,visible)
        refit_zw,_,_,_ = whitening(refit_z,retained)
        refit_d = cdist(refit_zw[test],refit_zw[retained])
        for qi,i in enumerate(test):
            base = retained[np.argsort(latent_distance[i,retained],kind='stable')[:5]]
            revised = retained[np.argsort(refit_d[qi],kind='stable')[:5]]
            common = len(set(base)&set(revised)); union = len(set(base)|set(revised))
            stability.append(dict(repeat=rep,case_id=l.case_id.iloc[i],n_train_retained=len(retained),
                                  top5_retention=common/5,top5_jaccard=common/union,
                                  original_neighbors='|'.join(l.case_id.iloc[base]),refit_neighbors='|'.join(l.case_id.iloc[revised])))
        print(json.dumps(dict(event='representation_refit',repeat=rep,n_train=len(retained))),flush=True)
    stability = pd.DataFrame(stability)
    stability.to_csv(out/'representation_stability.csv',index=False)
    # These are direct summaries of observed raw macro bins, not generated
    # waveforms, not continuously measured phases and not clinical labels.
    trends = l[['case_id','split']].copy()
    for c,signal in enumerate(CHANNELS[4:],4):
        early = raw[:,:24,c];late = raw[:,24:,c]
        values = np.full(len(raw),np.nan)
        for i in range(len(raw)):
            if np.isfinite(early[i]).sum()>=3 and np.isfinite(late[i]).sum()>=3:
                values[i] = np.nanmedian(late[i])-np.nanmedian(early[i])
        trends[signal+'_late_minus_early'] = values
        trends[signal+'_early_bins'] = np.isfinite(early).sum(1)
        trends[signal+'_late_bins'] = np.isfinite(late).sum(1)
    trends.to_csv(out/'raw_temporal_trends.csv',index=False)
    correlations = []
    trend_correlations = []
    groups = [('train',train),('validation',np.flatnonzero(l.split.eq('validation'))),('test',test)]
    # Raw unwhitened latent axes preserve the loading model's coordinate basis.
    # Axis sign/basis is arbitrary; correlations describe this fitted model only.
    for j in range(model.k):
        for feature in pattern_cols+[c+'_late_minus_early' for c in CHANNELS[4:]]:
            source = trends if feature in trends else coordinates
            y = pd.to_numeric(source[feature],errors='coerce').to_numpy(float)
            for split,index in groups:
                rr,nn = rho(z[index,j],y[index])
                row = dict(axis=f'z{j+1}',feature=feature,split=split,spearman=rr,n_supported=nn,
                           status='POSTHOC_EXPLORATORY_NOT_INDEPENDENT_CONFIRMATION')
                correlations.append(row)
                if split=='train' and feature.endswith('_late_minus_early') and nn>=20 and np.isfinite(rr):
                    trend_correlations.append(row)
    selected = max(trend_correlations,key=lambda row:abs(row['spearman'])) if trend_correlations else None
    selected_rows = [r for r in correlations if selected and r['axis']==selected['axis'] and r['feature']==selected['feature']]
    pd.DataFrame(correlations).to_csv(out/'representation_correlations.csv',index=False)
    coordinates.to_csv(out/'latent_coordinates.csv',index=False)
    np.savez_compressed(out/'representation_geometry.npz',latent_center=zc,whitening_matrix=whiten,
                        training_covariance_eigenvalues=eigenvalues,feature_weights=weights,
                        quality_center=qc,quality_scale=qscale,quality_supported=qused)
    after = {p:sha(p) for p in before}
    assert before==after,'Read-only inspection modified an input'
    summary = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   linear_core=name,predictive_candidate=candidate,n_cases=len(l),n_train=len(train),n_test=len(test),
                   model_kind='linear masked low-rank conditional latent representation; not neural VAE',
                   latent_dim=model.k,no_clusters_fitted=True,all_input_hashes_unchanged=True,input_sha256=before,
                   full_observation_encoder=True,raw_observations_unchanged=True,
                   scaling='Saved original TRAIN channel transforms/scales; covariance whitening fit on original TRAIN only',
                   neighbor_candidates='TRAIN cases only for TEST queries; one row per case',
                   semantic_distance='Exact frozen semantic features recomputed with full visibility/per-signal timestamps; TRAIN median/IQR max(.1) scale; shared-finite-coordinate L1 mean',
                   quality_distance_spearman=quality_rho,distance_pair_count=n_pairs,
                   quality_caveat='Pairs are dependent. Low association does not exclude recording/missingness confounding. Support itself can depend on operation.',
                   latent_semantic_distance_spearman=semantic_rho,
                   latent_semantic_top5_jaccard_mean=float(pd.DataFrame(agreement).latent_semantic_top5_jaccard.mean()),
                   stability=dict(repeats=repeats,retained_train_fraction=.8,
                                  top5_retention_median=float(stability.top5_retention.median()),
                                  top5_retention_q25=float(stability.top5_retention.quantile(.25)),
                                  top5_retention_q75=float(stability.top5_retention.quantile(.75)),
                                  top5_jaccard_median=float(stability.top5_jaccard.median()),
                                  scope='Conditional fixed k/regularization and original channel scales. Mean/loadings/whitening refit on TRAIN subset. Original/refit neighbors compared within same retained TRAIN candidates; not full pipeline selection stability or external validation.'),
                   strongest_train_axis_trend=selected_rows,
                   posthoc_warning='Full-observation analysis performed after frozen predictive test. Axis/trend selection uses TRAIN but all associations remain exploratory; no further tuning permitted.',
                   new_operation_pattern_validated=False,
                   predictive_gate_passed=test_summary['predictive_gate_passed'],
                   interpretation='Stable new coordinates or correlations alone do not establish added meaningful raw operation patterns beyond known HP/RPM order features.')
    (out/'representation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=True)+'\n')
    print(json.dumps({k:summary[k] for k in ['linear_core','n_cases','quality_distance_spearman','latent_semantic_distance_spearman','stability','strongest_train_axis_trend']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=10)
    args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    inspect(args.output,args.repeats)
