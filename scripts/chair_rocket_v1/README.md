# ROCKET time-series classification and raw replay

This separately versioned experiment retains temporal inputs instead of reducing
each window to finite-observation indicators. Clinical datasets, fitted models,
case figures, local reports and notification addresses must remain local.

The pipeline is `120s window -> 26 channels x 120 one-second bins -> official
sktime ROCKET -> StandardScaler(with_mean=False) -> case-weighted RidgeClassifier`.
There are 10,000 random kernels, two features per kernel, fixed alpha=1, and no
hyperparameter selection. The classifier follows the documented ROCKET + Ridge
construction, rather than calling RocketClassifier's internal RidgeClassifierCV:
the latter does not accept case weights and does not enforce case-grouped CV.
See https://www.sktime.net/docs/api-reference/sktimeclassificationkernel-basedrocketclassifier/.

Four discrete channels are represented by per-second durations in each state
plus an explicit unknown category. Five numeric channels retain actual observed
per-second medians and observed masks. Numeric gaps stay NaN in saved tensors.
The finite model input uses an explicit standardized missing token (-3) plus
its observed mask. It does not assert that unknown is OFF or an observed zero.
No numeric interpolation or forward filling is used. Raw event data remain
read-only. Sub-second numeric detail is aggregated; this is not a lossless raw
event representation. Random convolution pooling does not preserve every
absolute timestamp or every aspect of global event ordering.

Numerical centering/scaling uses training-only observed medians and IQRs, with
observed-range and unit fallbacks. ROCKET's per-instance normalization is off.
A temporal observation-mask-only control is fitted separately. It is an
observation-support control, not proof of a pure recording-system effect.

Instruments use independent binary Ridge classifiers with decision threshold0.
Treatment uses multiclass Ridge argmax. Margins are not calibrated probabilities.
Training weights equalize cases, parents within cases, and windows within parents.
Five GroupKFold folds keep each case together, with preprocessing refitted inside
each training fold. No query case may be in a saved model's training-case list.

## Local inputs

Requires the sibling `chair_supervised_v1` code and its locally extracted metadata.
Window CSVs need unique id, seq, annotation_id, start_s, end_s, confirmed,
instrument_labels (JSON), treatment_normalized, and split=development for training.
Every window must be exactly120s; no short-window fallback is applied. The caller
must preserve the extraction ledger for annotations/cases with no eligible windows.
Case manifests must provide date, chair, raw_duration_s, source_sha256 and seq.

Use an isolated Python environment. Install `requirements.txt`. CPU execution
is sufficient; GPU is not required. Do not change system drivers for this code.

```bash
python build_tensors.py --c31-root "$C31_ROOT" \
  --metadata "$PRIVATE_TRAIN_WINDOWS" --case-manifest "$PRIVATE_CASE_MANIFEST" \
  --output "$NEW_OUTPUT/train_tensors.npz"

python train.py --metadata "$PRIVATE_TRAIN_WINDOWS" \
  --tensors "$NEW_OUTPUT/train_tensors.npz" --output "$NEW_MODEL_OUTPUT"

python replay_cases.py --c31-root "$C31_ROOT" --model-root "$NEW_MODEL_OUTPUT" \
  --annotations "$PRIVATE_ALIGNED_ANNOTATIONS" --case-manifest "$PRIVATE_CASE_MANIFEST" \
  --cases "$QUERY_SEQS" --output "$NEW_REPLAY_OUTPUT"
```

Paths are explicit: historical absolute paths embedded in old manifests are not
used to find raw files. All output destinations must be new. Pickles must be
trusted local artifacts; the compatibility loader supports original pilot module
names without changing model parameters. Replay writes exact-window predictions,
raw tensors with NaN, margins, source/model hashes, annotation tables, per-case
figures and a combined PDF. Focus figures zoom around human annotation intervals;
the full-case figures/PDF retain the entire raw domain.

## Interpretation

Window labels inherited from human intervals are weak labels. Only windows fully
inside a single confirmed parent with an available target are scored. Boundary,
mixed and unannotated windows retain their scope and are not counted as negatives.
Short parents and unscanned tails are audited. Overlapping windows are not
independent cases. Descriptive agreement on hand-selected cases is not population
performance. Choose replay cases by annotation-pattern diversity before inspecting
predictions, preserve every selected result, and do not tune frozen models using
the replay. Source, model and previous-stage ID compatibility must be verified.

This pipeline generates exploratory proposals, not validated automatic clinical
annotations. Raw support, same-instrument clinical confusion, background rejection,
and external-case validation remain separate questions.
