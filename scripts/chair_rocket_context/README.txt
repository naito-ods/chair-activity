ROCKET numerical guard and temporal inference helpers

This module provides the reusable numerical rules for a separately versioned,
exploratory case-grouped ROCKET/context experiment. It contains no clinical data,
fitted models, case IDs, local reports or notification addresses.

Run synthetic checks:
  python -m unittest discover -s scripts/chair_rocket_context -p 'test_*.py' -v

Integration order for each CASE-GROUPED fold:
1. Fit missing-value encoding on training windows only. Unknown is distinct from
   observed zero. Preserve raw NaN tensors for audit; do not interpolate numerics.
2. Fit official sktime ROCKET on training windows, then convert its output to
   float64 before StandardScaler and Ridge. Use normalise=False when raw numeric
   levels are intentionally retained.
3. Fit StandardScaler(with_mean=False) on training ROCKET features only.
   keep = variance_mask(scaler.mean_, scaler.var_)
   Apply the same keep mask after scaler.transform() to training and query rows.
   Fit RidgeClassifier(alpha=1) with equal-case/parent/window sample weighting.
   Fit one binary head per known instrument and a multiclass treatment head.
   Unknown instrument labels must not remove valid treatment labels; use copied
   boolean masks to avoid mutating shared pandas/numpy metadata.
4. For contextual ablation, derive context from RAW observations only. Keep
   past-only and future-using offline variants distinct. Refit context imputation
   and scaling inside each training fold; never insert human instrument labels
   or query-neighbor clinical truth into predictors.
5. bounded_late_fusion(local, context, available, .2) combines80% local margin
   with20% clipped raw-context margin. available comes from the TRAINING head's
   class list, never from a numeric sentinel threshold. Scores are not probabilities.
6. Optional decode_sequence uses a uniform initial distribution and fixed0.2
   transition weight. A reference experiment used0.5 pseudo-count per transition.
   Learn transitions only in training cases: self-transitions from truly adjacent
   regular-grid windows of the same known parent; cross-parent transitions from
   consecutive confirmed intervals with gap0..5seconds and represented classes.
   Save observed counts separately from pseudo-count-smoothed probabilities.

Freeze all rules before evaluating a new cohort. If a rule follows inspection of
OOF results, label its scores adaptive exploration, not independent validation.
Save all variants, excluded cases/parents, coverage, class/case support, common-
parent comparisons for different window lengths, per-case metrics, raw replay,
source hashes, train/query IDs, parameter seeds, versions and evaluator findings.
Use paired CASE bootstrap for internal uncertainty, not window resampling.

A numerical guard prevents float32 quantization-level variations from being
magnified; numerical correctness alone does not establish clinical model quality.
Unannotated time is unknown, not background-negative. Full-domain sequence
inference can propagate a candidate through unknown time and cannot establish
no-procedure or clinical boundaries without corresponding reference annotations.
