"""Compare fixed-cohort non-DTW representations; preserve all original results."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, silhouette_samples)
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

SEED = 42
VIEWS = {
    'levels': ['levels'],
    'operations': ['operations'],
    'combined': ['levels', 'operations'],
    'conditional': ['levels', 'operations', 'conditional'],
    'coarse': ['levels', 'operations', 'coarse'],
}
METHODS = ['kmeans', 'ward', 'gmm']


def log(out, event, **kwargs):
    record = dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), event=event, **kwargs)
    with (out / 'process/modeling_log.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)


def transform(raw, catalog):
    """Fit preprocessing on exactly the supplied cases, including each subsample.

    NaN remains unknown in exported raw features. Algorithm-only median imputation
    centers missing values at the observed cohort median; it does not mean OFF/0.
    Each semantic block receives equal Euclidean weight after robust scaling.
    """
    array = np.asarray(raw, float)
    kept = np.flatnonzero(np.isfinite(array).sum(axis=0) >= max(5, int(.10*len(array))))
    array = array[:, kept]
    if not array.shape[1]:
        raise ValueError('No sufficiently observed features')
    median = np.nanmedian(array, axis=0)
    q10, q25, q75, q90 = np.nanquantile(array, [.1, .25, .75, .9], axis=0)
    scale = np.maximum(q75-q25, .25*(q90-q10))
    standard = np.nanstd(array, axis=0)
    scale = np.where(scale > 1e-9, scale, standard)
    variable = np.flatnonzero(scale > 1e-9)
    kept = kept[variable]
    array = array[:, variable]
    median, scale = median[variable], scale[variable]
    matrix = np.clip((np.where(np.isfinite(array), array, median)-median)/scale, -5, 5)
    meta = catalog.iloc[kept].copy()
    # A signal block groups its related quantiles, preventing feature count alone
    # from giving one modality a larger nominal weight.
    blocks = meta['block'].astype(str).to_numpy()
    for block in np.unique(blocks):
        matrix[:, blocks == block] /= np.sqrt(sum(blocks == block))
    assert np.isfinite(matrix).all() and matrix.shape[1] > 0
    return matrix, meta, dict(median=median.tolist(), scale=scale.tolist(),
                             clip_z=5, selected_columns=meta.feature.tolist())


def fit_labels(matrix, method, k, seed):
    with threadpool_limits(limits=1):
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter('always', ConvergenceWarning)
            if method == 'kmeans':
                model = KMeans(n_clusters=k, n_init=30, random_state=seed)
                labels = model.fit_predict(matrix)
                extra = dict(converged=True, inertia=float(model.inertia_))
            elif method == 'ward':
                model = AgglomerativeClustering(n_clusters=k, linkage='ward')
                labels = model.fit_predict(matrix)
                extra = dict(converged=True)
            else:
                # Diagonal covariance is estimable at n=296 with many correlated
                # features; it is one comparison model, not clinical validation.
                model = GaussianMixture(n_components=k, covariance_type='diag',
                                        n_init=5, max_iter=500, reg_covar=1e-3,
                                        random_state=seed)
                labels = model.fit_predict(matrix)
                extra = dict(converged=bool(model.converged_), bic=float(model.bic(matrix)),
                             aic=float(model.aic(matrix)))
            extra['warnings'] = '; '.join(str(w.message) for w in warning_list)
    return np.asarray(labels, int), extra


def eta_squared(values, labels):
    values = np.asarray(values, float)
    den = np.sum((values-np.mean(values))**2)
    return float(sum(np.sum(labels == cl) * (np.mean(values[labels == cl])-np.mean(values))**2
                     for cl in np.unique(labels)) / den) if den > 0 else 0.


def canonicalize(labels, matrix):
    # Names are descriptive IDs only. Consistent centroid sorting improves display.
    centers = [(cl, np.median(matrix[labels == cl], axis=0)) for cl in np.unique(labels)]
    order = sorted(centers, key=lambda pair: tuple(np.round(pair[1], 8)))
    mapping = {old: new+1 for new, (old, _) in enumerate(order)}
    return np.asarray([mapping[x] for x in labels])


def evaluate_candidate(view, method, k, raw, catalog, quality, dates, duration, repeats):
    matrix, _, _ = transform(raw, catalog)
    labels, extra = fit_labels(matrix, method, k, SEED)
    labels = canonicalize(labels, matrix)
    unique, sizes = np.unique(labels, return_counts=True)
    silhouette = float(silhouette_score(matrix, labels)) if len(unique) > 1 else np.nan
    stability = []
    rng = np.random.default_rng(SEED)
    for iteration in range(repeats):
        ix = np.sort(rng.choice(len(raw), size=int(round(.8*len(raw))), replace=False))
        submatrix, _, _ = transform(raw[ix], catalog)
        sublabels, subextra = fit_labels(submatrix, method, k, SEED+iteration+1)
        if subextra['converged'] and len(set(sublabels)) == k:
            stability.append(float(adjusted_rand_score(labels[ix], sublabels)))
    if not stability:
        stability = [np.nan]
    quality_labels, _ = fit_labels(quality, 'kmeans', k, SEED)
    missing_labels, _ = fit_labels(quality, 'ward', k, SEED)
    quality_ari = max(adjusted_rand_score(labels, quality_labels),
                      adjusted_rand_score(labels, missing_labels))
    structurally_valid = (len(unique) == k and sizes.min() >= 10 and sizes.max()/len(raw) <= .9
                          and np.isfinite(silhouette) and silhouette > 0 and extra['converged'])
    row = dict(model_id=f'{view}__{method}__k{k}', view=view, method=method, k=k,
        n_cases=len(raw), n_features=matrix.shape[1], silhouette=silhouette,
        min_cluster_n=int(sizes.min()), largest_fraction=float(sizes.max()/len(raw)),
        stability_median=float(np.median(stability)), stability_q25=float(np.quantile(stability,.25)),
        stability_q75=float(np.quantile(stability,.75)), stability_min=float(np.min(stability)),
        stability_n_valid=int(np.isfinite(stability).sum()),
        stability_n_failed=int(repeats-np.isfinite(stability).sum()),
        quality_ari=float(quality_ari), duration_eta2=eta_squared(duration, labels),
        date_nmi=float(normalized_mutual_info_score(dates, labels)), valid=bool(structurally_valid),
        selection_score=float(silhouette + .35*np.median(stability)-.10*max(0,quality_ari)),
        **extra)
    return row, labels, stability


def refine_stability(method, k, raw, catalog, labels, n_repeats):
    rows = []
    rng = np.random.default_rng(SEED+1000)
    for iteration in range(n_repeats):
        ix = np.sort(rng.choice(len(raw), int(round(.8*len(raw))), replace=False))
        matrix, _, _ = transform(raw[ix], catalog)
        sublabels, extra = fit_labels(matrix, method, k, SEED+iteration+1001)
        valid = extra['converged'] and len(set(sublabels)) == k
        rows.append(dict(iteration=iteration, n_cases=len(ix), ari=float(adjusted_rand_score(labels[ix],sublabels)),
                         converged=extra['converged'], k_actual=len(set(sublabels)), valid_repeat=bool(valid)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--screen-repeats', type=int, default=12)
    parser.add_argument('--stability-repeats', type=int, default=60)
    args = parser.parse_args()
    out = args.output.resolve()
    if (out / 'selected_models.csv').exists():
        parser.error('Model outputs already exist. Use a new output directory to preserve this analysis version.')
    (out / 'process').mkdir(exist_ok=True)
    started = time.time()
    manifest = pd.read_csv(out / 'case_manifest.csv')
    features = pd.read_csv(out / 'features.csv').set_index('case_id')
    catalog = pd.read_csv(out / 'feature_catalog.csv', dtype={'code': str})
    assert len(manifest) == 300 and not manifest.case_id.duplicated().any()
    assert set(manifest.case_id) == set(features.index)
    features = features.reindex(manifest.case_id)
    behavioral = catalog[catalog.view.isin(['levels','operations','conditional','coarse'])].feature.tolist()
    usable = features[behavioral].notna().any(axis=1).to_numpy()
    selected_cases = manifest.loc[usable].reset_index(drop=True)
    assert usable.sum() == 296, f'Unexpected usable case count: {usable.sum()}'
    skipped = manifest.loc[~usable].copy()
    skipped['reason'] = 'NO_TARGET_SIGNAL; all specified events unrecorded; user requested skip'
    skipped.to_csv(out / 'skipped_cases.csv', index=False)
    all_raw = features.loc[usable].copy()
    year = pd.to_datetime(selected_cases['date']).dt.year.astype(str).to_numpy()
    duration = selected_cases.duration_s.to_numpy(float)
    qualcat = catalog[catalog.view == 'quality'].reset_index(drop=True)
    qualraw = all_raw[qualcat.feature].to_numpy(float)
    quality_matrix, _, _ = transform(qualraw, qualcat)
    log(out, 'preflight', requested_cases=300, usable_cases=int(usable.sum()), skipped_cases=len(skipped),
        fixed_seed=SEED, decision='User explicitly requested skipping the four cases with no target signals')
    predeclared = dict(version='numeric_multiview_v1', seed=SEED, views=VIEWS, methods=METHODS,
        k_range=list(range(2,9)), min_cluster_n=10, max_largest_fraction=.90,
        selection='Within each view/method, prefer structurally valid candidates; maximize silhouette + '
                  '0.35*screen_median_ARI - 0.10*max(0,quality_ARI); ties prefer smaller k. '
                  'If none pass structural rules, retain best exploratory candidate and flag invalid.',
        screen_repeats=args.screen_repeats, final_repeats=args.stability_repeats,
        subsample_fraction=.8, preprocessing='refitted independently in every subsample',
        imputation='algorithm-only per-feature observed median; raw missingness retained as NaN',
        scaling='median/IQR with q10-q90 and SD fallback; clipping +/-5; equal semantic-block weights',
        quality_control='max ARI against KMeans and Ward partitions of quality/missingness features at same k',
        temporal_change='coarse progress summaries only; no DTW',
        external_validation=False, software={n:importlib.metadata.version(n) for n in
                                             ['numpy','pandas','scikit-learn','scipy','matplotlib']})
    (out / 'model_plan.json').write_text(json.dumps(predeclared, indent=2, ensure_ascii=False))
    view_data = {}
    tasks = []
    for view, parts in VIEWS.items():
        cat = catalog[catalog.view.isin(parts)].reset_index(drop=True)
        if cat.empty:
            continue
        raw_all = all_raw[cat.feature].to_numpy(float)
        view_ix = np.flatnonzero(np.isfinite(raw_all).any(axis=1))
        raw = raw_all[view_ix]
        view_data[view] = raw, cat, view_ix
        for method in METHODS:
            for k in range(2,9):
                tasks.append((view,method,k,raw,cat,view_ix))
    log(out, 'per_view_scope', views={v:len(data[2]) for v,data in view_data.items()},
        correction='No fully missing view is assigned using imputed values; levels has seven additional NO_VIEW_SIGNAL cases')
    log(out, 'candidate_search_started', n_candidates=len(tasks), workers=args.workers)
    results = Parallel(n_jobs=args.workers, backend='loky')(
        delayed(evaluate_candidate)(view,method,k,raw,cat,quality_matrix[view_ix],year[view_ix],
                                    duration[view_ix],args.screen_repeats)
        for view,method,k,raw,cat,view_ix in tasks)
    candidates = pd.DataFrame([r[0] for r in results])
    candidates.to_csv(out / 'model_candidates.csv', index=False)
    label_lookup = {r[0]['model_id']:r[1] for r in results}
    winners = []
    for (_, _), group in candidates.groupby(['view','method']):
        valid = group[group.valid]
        pool = valid if len(valid) else group
        winners.append(pool.sort_values(['selection_score','k'],ascending=[False,True]).iloc[0].to_dict())
    log(out, 'candidate_search_complete', selected_models=len(winners),
        structural_valid=int(sum(bool(r['valid']) for r in winners)))
    refined = Parallel(n_jobs=args.workers, backend='loky')(
        delayed(refine_stability)(r['method'],int(r['k']),*view_data[r['view']][:2],
                                  label_lookup[r['model_id']],args.stability_repeats) for r in winners)
    stability_rows = []
    for winner, rows in zip(winners,refined):
        values = np.asarray([r['ari'] for r in rows if r['valid_repeat']])
        assert len(values), 'No valid stability repetitions'
        for row in rows:
            stability_rows.append(dict(model_id=winner['model_id'],**row))
        winner['screen_stability_median'] = winner['stability_median']
        winner['stability_median'] = float(np.median(values))
        winner['stability_q25'] = float(np.quantile(values,.25))
        winner['stability_q75'] = float(np.quantile(values,.75))
        winner['stability_min'] = float(values.min())
        winner['stability_repeats'] = len(rows)
        winner['stability_valid_repeats'] = len(values)
        winner['stability_failed_repeats'] = len(rows)-len(values)
        winner['evidence_status'] = ('EXPLORATORY_STRUCTURAL_FAILURE' if not winner['valid'] else
            'RECORDING_LINKED_WARNING' if winner['quality_ari'] >= .5 else
            'INTERNALLY_STABLE_CANDIDATE' if winner['stability_median'] >= .75 else
            'EXPLORATORY_UNSTABLE')
    selected = pd.DataFrame(winners)
    selected.to_csv(out / 'selected_models.csv', index=False)
    pd.DataFrame(stability_rows).to_csv(out / 'stability_repeats.csv',index=False)
    assignments, sizes, representatives, profiles, embeddings, preprocessing = [],[],[],[],[],{}
    # Embeddings are fitted once per view, then reused for all methods in that view.
    coords = {}
    for view,(raw,cat,view_ix) in view_data.items():
        matrix, usedcat, prep = transform(raw,cat)
        preprocessing[view] = prep
        with threadpool_limits(limits=1):
            pc = PCA(n_components=2,random_state=SEED).fit_transform(matrix)
            ts = TSNE(n_components=2,perplexity=30,init='pca',max_iter=1500,
                      learning_rate='auto',random_state=SEED,n_jobs=1).fit_transform(matrix)
        coords[view] = matrix,pc,ts
    for winner in winners:
        mid,view = winner['model_id'],winner['view']
        labels = label_lookup[mid]
        view_ix = view_data[view][2]
        view_cases = selected_cases.iloc[view_ix].reset_index(drop=True)
        view_raw = all_raw.iloc[view_ix].reset_index(drop=True)
        case_labels = dict(zip(view_cases.case_id, labels))
        matrix,pc,ts = coords[view]
        individual_silhouette = silhouette_samples(matrix,labels)
        for index,row in manifest.iterrows():
            assignments.append(dict(case_id=row.case_id,model_id=mid,view=view,method=winner['method'],
                k=winner['k'],cluster=int(case_labels[row.case_id]) if row.case_id in case_labels else np.nan,
                status=('CLASSIFIED' if row.case_id in case_labels else
                        'NO_VIEW_SIGNAL' if usable[index] else 'NO_TARGET_SIGNAL')))
        for ix,row in view_cases.iterrows():
            embeddings.append(dict(model_id=mid,case_id=row.case_id,pc1=pc[ix,0],pc2=pc[ix,1],
                                   tsne1=ts[ix,0],tsne2=ts[ix,1],silhouette=individual_silhouette[ix]))
        for cl in sorted(set(labels)):
            ix = np.flatnonzero(labels==cl)
            sizes.append(dict(model_id=mid,cluster=int(cl),n_cases=len(ix)))
            distances = cdist(matrix[ix],matrix[ix])
            med_local = np.argmin(distances.sum(axis=1))
            others = [j for j in np.argsort(distances[med_local],kind='stable') if j != med_local]
            order = [med_local]+others[:min(4,len(ix)-1)]
            for rank,local in enumerate(order):
                original = ix[local]
                representatives.append(dict(model_id=mid,cluster=int(cl),rank=rank,
                    case_id=view_cases.case_id.iloc[original],is_medoid=bool(local==med_local),
                    distance_to_center=float(distances[med_local,local])))
            group = view_raw.iloc[ix]
            for feature in catalog.feature:
                values = group[feature].dropna()
                profiles.append(dict(model_id=mid,cluster=int(cl),n_cases=len(ix),feature=feature,
                    raw_median=values.median() if len(values) else np.nan,
                    raw_q25=values.quantile(.25) if len(values) else np.nan,
                    raw_q75=values.quantile(.75) if len(values) else np.nan,observed_cases=len(values)))
    pd.DataFrame(assignments).to_csv(out / 'assignments_long.csv',index=False)
    pd.DataFrame(sizes).to_csv(out / 'cluster_sizes.csv',index=False)
    pd.DataFrame(representatives).to_csv(out / 'representatives.csv',index=False)
    pd.DataFrame(profiles).to_csv(out / 'class_profiles.csv',index=False)
    pd.DataFrame(embeddings).to_csv(out / 'embedding.csv',index=False)
    (out / 'preprocessing.json').write_text(json.dumps(preprocessing,indent=2))
    mids = selected.model_id.tolist()
    global_labels = {}
    for mid in mids:
        view = selected.loc[selected.model_id==mid,'view'].iloc[0]
        full = np.full(len(selected_cases),-1,dtype=int)
        full[view_data[view][2]] = label_lookup[mid]
        global_labels[mid] = full
    agreement = pd.DataFrame(index=mids,columns=mids,dtype=float)
    comparison_n = pd.DataFrame(index=mids,columns=mids,dtype=int)
    for a in mids:
        for b in mids:
            common = (global_labels[a]>0)&(global_labels[b]>0)
            agreement.loc[a,b] = adjusted_rand_score(global_labels[a][common],global_labels[b][common])
            comparison_n.loc[a,b] = int(common.sum())
    agreement.to_csv(out / 'model_comparison_ari.csv',index_label='model_id')
    comparison_n.to_csv(out / 'pairwise_model_comparison_n.csv',index_label='model_id')
    # Also provide a strictly fixed 289-case intersection for comparisons across all views.
    common_all = np.logical_and.reduce([global_labels[mid]>0 for mid in mids])
    fixed_agreement = pd.DataFrame([[adjusted_rand_score(global_labels[a][common_all],global_labels[b][common_all])
                                    for b in mids] for a in mids],index=mids,columns=mids)
    fixed_agreement.to_csv(out / 'model_comparison_ari_common_cohort.csv',index_label='model_id')
    selected_cases.loc[common_all].to_csv(out / 'common_comparison_cohort.csv',index=False)
    # Compare one-dimensional nuisance groupings independently of the primary views.
    nuisance = []
    for mid in mids:
        labels = label_lookup[mid]
        view = selected.loc[selected.model_id==mid,'view'].iloc[0]
        vi = view_data[view][2]
        nuisance.append(dict(model_id=mid,year_nmi=normalized_mutual_info_score(year[vi],labels),
            duration_eta2=eta_squared(duration[vi],labels),single_device=True,
            note='Single chair in this cohort; device confounding and external validity cannot be tested'))
    pd.DataFrame(nuisance).to_csv(out / 'confound_diagnostics.csv',index=False)
    metadata = dict(elapsed_seconds=time.time()-started,python=sys.version,executable=sys.executable,
        platform=platform.platform(),requested_cases=300,classified_cases=int(usable.sum()),
        skipped_cases=len(skipped),models=len(selected),candidate_models=len(candidates),
        workers=args.workers,gpu_used=False,gpu_reason='nvidia-smi cannot communicate with driver; no environment changes',
        peak_self_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        assignment_status='new version; original DTW/Two-stage assignments untouched',
        inputs={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in
                ['features.csv','feature_catalog.csv','case_manifest.csv']})
    (out / 'modeling_metadata.json').write_text(json.dumps(metadata,indent=2))
    log(out,'modeling_complete',**metadata)


if __name__=='__main__':
    main()
