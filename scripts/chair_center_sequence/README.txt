Center-time local logs + procedure sequence, version 0.1.4

Generic code and synthetic tests only. Private input data, fitted models,
per-case predictions, study results and notification addresses are excluded.
Keep real-data reanalysis in isolated private output directories and verify
source hashes, coverage and case-held-out metrics. Do not overwrite source
annotations or treat model candidates as confirmed clinical observations.

Target and input
---------------
The target is the confirmed label at the center TIME, not the label of an
entire input window. A raw-bound, annotation-independent grid defaults to 1 s
cells: centers 0.5, 1.5, ... . The final partial cell is retained. Membership
uses [start,end). There is no five-second boundary trim and no majority label
across an input window. Annotation boundaries never determine feature windows.

Every center is kept, including annotation gaps. Exactly one confirmed,
nonempty, non-unknown parent is required for a clinical teacher. Overlap,
pending and unannotated centers retain target=unknown and are excluded from
clinical supervision. Unknown means annotation status, not no-procedure.
Predictions for unknown times are separate, unconfirmed candidate columns.

An interval shorter than the grid step can still have no center. The complete
annotation_coverage.csv records every parent and the reason if it contributes
no teacher. A finer step is configurable; do not manufacture finer raw signal
resolution by interpolation. The grid must be selected before model scoring.

Local model
-----------
1. Raw summaries in 2, 5, 10 s windows around each center: state occupancy,
   raw record counts, state changes and observed numeric statistics.
2. ROCKET on a 30 s, 1 s/bin tensor around the same center: state occupancies,
   unknown-state channels, observed-only numeric medians and masks. A separate
   raw-bound coverage channel distinguishes input padding from in-case time.
3. Train-only numeric median/IQR encoding, explicit missing sentinel AND mask;
   train-only imputation/scaling. Numeric values are never carried forward.
   Same-code state changes persist, with initial/invalid values unknown.
4. Train-only ROCKET feature scaling, float64 classifier design, and a guard
   against amplification of constant/float32-quantization-sized variance.
5. Case/parent-balanced teachers plus explicit inverse-class-mass balancing.
   SGD logistic classification, alpha=.0001, 10 epochs, seed=20261106. The two
   standardized feature blocks are divided by sqrt(their feature counts).

ROCKET has 10000 kernels by default. Transforms and SGD updates use batches of
256, avoiding a whole-cohort center x 20000 feature matrix. Base raw summaries
and tensors are still held in memory for one CV run. A summary-only ablation
is available with --num-kernels 0. This is a newly versioned SGD local head,
not a claim of parity with an older Ridge-based model.

The 20101 pickup state is not evidence of actual air/water discharge. Whether
blowing is identifiable from these logs is an empirical question. The design
can include short blowing labels but does not guarantee discrimination.

Procedure model
---------------
A weak first-order HMM prior combines the local classifier scores with:
  - observed transitions between adjacent, unambiguously confirmed procedures;
  - a separate mean duration per class, balanced across training cases.
Only training-case annotations are accepted into the learned prior. Pending,
overlapping and unknown gaps censor the sequence; they do not teach clinical
transitions. Contiguous equal labels are merged only internally for duration
estimation; source parents and training IDs remain separate.
Visible annotation lengths are not censoring-corrected estimates of complete
clinical action durations.
These are lengths of visible annotation intervals, not censoring-corrected
estimates of complete clinical action durations.

Mean durations define exponential survival / geometric dwell times. This is
NOT an explicit-duration HSMM or a model of higher-order dependencies. All
transitions have positive smoothed probabilities; one-center procedures are
allowed and no minimum 30 s duration is imposed. Missing class duration support
uses a training-only pooled median, or a recorded fixed 30 s fallback if no
eligible complete interval exists. Strength=.2, transition smoothing=.5 and a
uniform initial distribution are fixed defaults, not tuned on test cases.

Viterbi operates on each complete case independently. It is OFFLINE and can
use later local evidence. --mode past restricts LOCAL feature windows only;
it does not make the complete Viterbi pipeline an online detector.
Class-balanced local scores are not calibrated clinical confidence values.

Validation and preservation
---------------------------
Use identical pre-existing case folds for local-only and local+sequence.
No query-case labels, annotation boundaries, instruments or evidence fields
are model inputs. The sequence prior fits directly on training annotations;
there is no learned stacker using in-sample base predictions and no need for
nested cross-fitting for this two-stage generative prior.

Report case/parent-weighted accuracy and macro-F1, raw-center per-label
precision/recall/F1, parent any-center/majority-center detection, missed-grid
parents and unseen training classes. Parent detection is not event-IoU
matching. Boundary timing and event-IoU evaluation remain additional planned
checks before claiming better procedure detection. The old 30 s interior-only
accuracy is not directly comparable to this enlarged center-time test set.

All source inputs and prepared artifacts are hashed; changed or incompatible
inputs fail closed. Outputs require new directories. Fold audit records full
training/query case sets, class support and all sequence-parent exclusions.
Do not publish clinical inputs, prepared arrays, model files or local reports.

Execution (only when an analysis run is requested)
-------------------------------------------------
Python 3.10+, numpy, pandas, scikit-learn; sktime+numba for ROCKET.
Use the existing project environment; this milestone installs no packages.
From this directory, put src on PYTHONPATH:

  PYTHONPATH=src python -m unittest discover -s tests -v
  PYTHONPATH=src python -m chair_center_sequence.workflow --help

Prepare from previously verified, narrow cached events:

  PYTHONPATH=src python -m chair_center_sequence.workflow prepare \
    --aligned /private/previous/annotations_aligned.csv \
    --manifest /private/previous/cases_manifest.csv \
    --folds /private/previous/case_folds.csv \
    --events-dir /private/previous/cases \
    --annotation-source /private/source/annotations.csv \
    --output /private/new_preparation

Required cache/source fields:
  manifest: seq,raw_duration_s,raw_start,source_path,source_sha256
  folds: seq,fold (integer fold IDs, at least two folds)
  aligned: id,seq,start_s,end_s,treatment_normalized,confirmed
  source: id,seq,start_datetime,end_datetime,treatment,status,source_sha256
          status=確認済み is confirmed; all other statuses are pending.
  events: /events-dir/<seq>.csv.gz with time_s,code,value. code is a string;
          values may be NaN. Case bounds come from the raw manifest, never
          from min/max timestamps of the selected feature codes.

Then explicitly evaluate into another new directory:

  PYTHONPATH=src python -m chair_center_sequence.workflow evaluate \
    --prepared /private/new_preparation --output /private/new_evaluation

Settings are frozen in config.json and code fingerprints. Changed settings
require separate preparation/output versions. The code imports without
opening project data or starting fitting. No background job is installed.

Numerical/performance corrections in 0.1.3
----------------------------------------
- Center-relative half-open bin endpoints; no endpoint reassociation.
- Exact local state overlaps, explicit unknown integration and bound padding;
  no 1-known complement or subtraction of large cumulative durations.
- Coalesce equal-state heartbeats only for occupancy, retain raw record counts.
- Exactly constant numeric observations have mean=value and std=0.
- Train-only summary scales below 1024*eps(float64)*max(1,abs(mean)) are set to1
  to disable numerical-noise amplification without discarding input fields.
- One-vs-rest sigmoid normalization in log space prevents all-zero underflow.
- Exact float64 standardized designs are cached transiently across unchanged
  SGD epochs, then removed. --cache-dir chooses an explicit writable cache root.
- Vectorized extraction matches the scalar reference, including fractional
  boundaries, initial unknown, padding and missing numeric observations.

Additional helpers, when present, use explicit paths and preserve input data.
fullfit.py saves an all-case reusable model separately from held-out scoring.
run_fold.py schedules one fixed fold with class-independent SGD threads.
Thread-count changes require numerical parity verification; they are not
changes to seed, epoch order, weights, regularization or train/query cases.
Do not silently mix independent fold runs: validate exact one-time query
coverage, training-class provenance, model hashes and source case folds.

Centering for the SGD head in 0.1.4
---------------------------------
ROCKET StandardScaler now centers with TRAINING means by default. An
uncentered transform can have large mean/standard-deviation ratios even after
variance guarding, which can give SGD poorly conditioned input. Earlier Ridge
heads handled an intercept differently, so a scaler suitable for that pipeline
is not automatically suitable for SGD. The constructor option center_rocket=False
is retained solely to reproduce the previous representation explicitly.

Training/query cases must still be separate. Never center using query/cohort
means or present an inspected-cohort correction as independent validation.
Stored models preserve their fitted scaler.with_mean value; loading an older
model does not change its predictions. Grouped comparisons and private source
model/checkpoint provenance must be reported independently of this generic code.
