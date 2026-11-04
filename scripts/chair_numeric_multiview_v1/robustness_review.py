"""Support-threshold sensitivity; never replace the primary model assignments."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score

from cluster_compare import VIEWS, SEED, transform, fit_labels, log


SCENARIOS = [('support10_known20',10,.20), ('support20_known20',20,.20),
             ('support30_known50',30,.50)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    plan=dict(scenarios=SCENARIOS, purpose='Sensitivity only; primary labels retained',
        rules='Numeric levels/conditional features withheld when source numeric signal has too few observed bins. '
              'State-derived features withheld when last-value-held known duration is below specified fraction. '
              'Joint HP match feature withheld if either state code has low known time. '
              'Coarse features withheld using whole-case support (not bin-local support). '
              'Conditional support threshold uses source code total bins, not within-state count; limitation retained.',
        seed=SEED, fixed_k=True)
    (out/'sensitivity_plan.json').write_text(json.dumps(plan,indent=2))
    features=pd.read_csv(out/'features.csv').set_index('case_id')
    catalog=pd.read_csv(out/'feature_catalog.csv',dtype={'code':str})
    manifest=pd.read_csv(out/'case_manifest.csv')
    audit=pd.read_csv(out/'signal_audit.csv',dtype={'code':str})
    selected=pd.read_csv(out/'selected_models.csv')
    assignments=pd.read_csv(out/'assignments_long.csv')
    repeats=pd.read_csv(out/'stability_repeats.csv')
    log(out,'support_sensitivity_started',scenarios=SCENARIOS,
        rationale='Independent reviewer found low-known-time syringe subgroup; test observation-support dependence without altering primary labels')
    support = audit.pivot(index='case_id',columns='code',values='n_observed_bins')
    results, sensitivity_labels = [],[]
    for scenario,min_bins,min_known in SCENARIOS:
        adjusted=features.copy()
        for field in catalog.itertuples():
            if field.view not in ['levels','conditional','operations','coarse']:
                continue
            code=str(field.code)
            if field.view in ['levels','conditional'] or (field.view=='coarse' and 'q50' in field.feature):
                if code not in support:
                    continue
                mask=support[code].reindex(adjusted.index)<min_bins
            else:
                key=f'E{code}__known_fraction'
                if key not in adjusted:
                    continue
                mask=adjusted[key]<min_known
                if 'selected_match_given_running' in field.feature:
                    mask=mask | (adjusted['E20011__known_fraction']<min_known)
            adjusted.loc[mask,field.feature]=np.nan
        for model in selected.itertuples():
            cat=catalog[catalog.view.isin(VIEWS[model.view])].reset_index(drop=True)
            raw=adjusted[cat.feature]
            ids=raw.index[raw.notna().any(axis=1)]
            baseline=assignments[(assignments.model_id==model.model_id)&assignments.cluster.notna()].set_index('case_id')
            ids=ids.intersection(baseline.index,sort=False)
            matrix,_,_=transform(raw.loc[ids].to_numpy(float),cat)
            labels,extra=fit_labels(matrix,model.method,int(model.k),SEED)
            ari=adjusted_rand_score(baseline.loc[ids,'cluster'],labels)
            # Cluster overlap is a descriptive retention diagnostic; each original
            # class chooses its maximal Jaccard match independently (not a new label).
            truth=baseline.loc[ids,'cluster'].to_numpy(int)
            for cl in sorted(set(truth)):
                mask=truth==cl
                best=max((np.sum(mask&(labels==other))/np.sum(mask|(labels==other))
                          for other in np.unique(labels)),default=np.nan)
                results.append(dict(model_id=model.model_id,scenario=scenario,min_numeric_bins=min_bins,
                    min_state_known_fraction=min_known,n_common_cases=len(ids),
                    n_baseline_cases=len(baseline),n_no_support=len(baseline)-len(ids),
                    ari=float(ari),baseline_cluster=int(cl),cluster_best_jaccard=float(best),
                    baseline_cluster_n=int(mask.sum()),converged=extra['converged'],k_actual=len(set(labels))))
            sensitivity_labels.extend(dict(model_id=model.model_id,scenario=scenario,case_id=case,
                sensitivity_cluster=int(label)+1) for case,label in zip(ids,labels))
    pd.DataFrame(results).to_csv(out/'support_sensitivity.csv',index=False)
    pd.DataFrame(sensitivity_labels).to_csv(out/'support_sensitivity_assignments.csv',index=False)
    # Full class-specific recording support, so an overall low quality-only ARI
    # cannot conceal one poorly observed subgroup.
    quality_rows=[]
    for model in selected.itertuples():
        subset=assignments[(assignments.model_id==model.model_id)&assignments.cluster.notna()]
        for cl,g in subset.groupby('cluster'):
            ids=g.case_id
            for code in audit.code.unique():
                values=audit[(audit.case_id.isin(ids))&(audit.code==code)]
                knownkey=f'E{code}__known_fraction'
                known=features.loc[ids,knownkey] if knownkey in features else pd.Series(dtype=float)
                quality_rows.append(dict(model_id=model.model_id,cluster=int(cl),code=code,
                    n_cases=len(ids),cases_with_events=int((values.n_events>0).sum()),
                    median_observed_bins=float(values.n_observed_bins.median()),
                    cases_under20_observed_bins=int((values.n_observed_bins<20).sum()),
                    median_observed_fraction=float(values.observed_fraction.median()),
                    mean_state_known_fraction=float(known.mean()) if len(known) else np.nan,
                    median_state_known_fraction=float(known.median()) if len(known) else np.nan,
                    cases_under20pct_known=int((known<.2).sum()) if len(known) else np.nan))
    pd.DataFrame(quality_rows).to_csv(out/'class_quality_audit.csv',index=False)
    # Two independent descriptive coordinates, not a claim of validated 12 classes.
    operation=assignments[assignments.model_id=='operations__kmeans__k6'].set_index('case_id')
    level=assignments[assignments.model_id=='levels__kmeans__k2'].set_index('case_id')
    cross=manifest[['case_id','date','duration_s','signal_status']].copy().set_index('case_id')
    cross['operation_cluster']=operation.cluster
    cross['numeric_cluster']=level.cluster
    cross['operation_status']=operation.status
    cross['numeric_status']=level.status
    cross['cross_group']=[f'O{int(op)}-N{int(num)}' if np.isfinite(op) and np.isfinite(num)
                         else 'NO_NUMERIC_VIEW' if np.isfinite(op) else 'NO_TARGET_SIGNAL'
                         for op,num in zip(cross.operation_cluster,cross.numeric_cluster)]
    cross.reset_index().to_csv(out/'cross_view_groups.csv',index=False)
    counts=pd.crosstab(cross.operation_cluster,cross.numeric_cluster.fillna('NO_NUMERIC'))
    counts.to_csv(out/'cross_view_counts.csv',index_label='operation_cluster')
    # Inspect per-feature recording counts, never automatically treat absent records as non-use.
    log(out,'support_sensitivity_complete',models=len(selected),scenarios=len(SCENARIOS),
        primary_assignments_sha256=hashlib.sha256((out/'assignments_long.csv').read_bytes()).hexdigest(),
        output='support_sensitivity.csv; class_quality_audit.csv; cross_view_counts.csv')


if __name__=='__main__':
    main()
