# Chair Log Two-stage DTW Research Plan

## Overview

This project studies dental chair operation logs using unsupervised time-series analysis. The aim is to discover recurring operation patterns from the logs first, then evaluate whether those patterns are stable and interpretable across cases.

The current analysis architecture is:

```text
Preprocessed chair logs
        |
        v
Hybrid v3 Activity extraction
        |
        v
Stage 1: State + Events clustering
        |
        v
Stage 2: Numeric DTW within eligible Stage 1 classes
        |
        v
Replay: Medoid + multi-case Member Overlay
        |
        v
Stability / missingness / external-validation review
```

The central research principle is that an unsupervised cluster is initially a **candidate log-level operation pattern**, not a confirmed dental procedure label.

---

## 1. Research objectives

### Primary objective

Determine whether dental chair logs contain recurring operation patterns that can be segmented into Activities and clustered reproducibly across independent cases.

### Secondary objectives

1. Compare broad state/event patterns before introducing numeric signals.
2. Determine whether numeric signals provide meaningful finer substructure within a broad class.
3. Evaluate whether the same Stage 2 class shows similar raw waveforms in several cases.
4. Quantify cluster stability with case-level resampling ARI.
5. Detect and control clustering driven by missingness or observation coverage.
6. Identify limitations of the current six-code representation and design a broader event model if required.

---

## 2. Current status

### Hybrid v3

Hybrid v3 has been expanded to 300 cases and provides hierarchical Activity intervals.

### Case coverage at the current DTW input layer

Current case-first DTW selection requested 300 cases but found usable DTW signatures for 296 cases.

Four cases failed every candidate Activity with `no_usable_sequence`:

```text
2024-05-21__C512KC1503__Chair_log_seq_1558
2024-06-13__C512KC1503__Chair_log_seq_1604
2024-11-21__C512KC1503__Chair_log_seq_1930
2024-12-18__C512KC1503__Chair_log_seq_1996
```

The failure occurs after Activity extraction. Each affected case has 12 refined Activity candidates, but none can be represented with the currently selected six DTW signal codes.

The four logs contain other events, including frequent `30117`, `30116`, `30123`, `30122`, and `30121` events. Therefore this is a **feature-coverage limitation**, not evidence that the source logs are empty.

### Current DTW feature groups

State/event-oriented codes:

```text
20039  20011  20101
```

Numeric codes:

```text
20742  20743  20108
```

These codes remain the baseline model until a separately versioned broader-event experiment is designed.

---

## 3. Research hypothesis

### H1 - Stage 1 recurrence

State/event behavior contains broad recurring operation patterns that are stable across cases.

Evidence expected:

- reasonable cluster balance
- multiple distinct cases per cluster
- case-subsampling ARI remains high across repeats
- clusters are not explained mainly by data availability

### H2 - Stage 2 added value

Within at least some Stage 1 classes, numeric signals provide meaningful substructure not captured by state/events alone.

Evidence expected:

- valid numeric coverage for enough Activities/cases
- interpretable raw-level or shape differences
- positive silhouette / separation without extreme imbalance
- stable Stage 2 assignments under case-level resampling

### H3 - Replay recurrence

Activities assigned to the same Stage 2 subcluster show similar raw temporal behavior across distinct cases.

Evidence expected:

- medoid is representative of multiple members
- multi-case Member Overlay shows recurring signal motifs
- similarity is visible in raw observations, not only after normalization

### H4 - Missingness is not the primary cluster generator

Useful clusters should not be reproduced mainly by observation masks or recording coverage.

Evidence expected:

- weak relationship between distance and coverage difference
- low agreement with missingness-only control clusters
- raw signal features still differ meaningfully between clusters

---

## 4. Analysis plan

### Phase A - Preserve a stable baseline

Keep the current six-code DTW as the reference model.

Do not change the signal set while evaluating the 296 usable cases. This provides a stable baseline against which later broader-event models can be compared.

#### Pilot continuation

If the explicit research decision is to continue before redesigning the feature set, run the 296-case pilot with full audit of the four excluded cases.

```bash
source ~/venvs/chair_c31/bin/activate
cd "/media/knaito/Samsung_SSD/C31/chair_300_pipeline_shortfall_v2"

python run_300_pipeline.py \
  --output-root "/media/knaito/Samsung_SSD/C31/Output" \
  --stage all \
  --resume \
  --allow-case-shortfall \
  --min-case-fraction 0.95
```

This run should always be described as a **296-case usable-signal analysis derived from a 300-case Hybrid cohort**, not as a complete 300-case DTW analysis.

---

### Phase B - Evaluate Stage 1

Stage 1 uses state + event behavior for coarse grouping.

For each Stage 1 cluster, report:

- cluster ID
- number of Activities
- number of distinct cases
- selected k / silhouette
- case-level subsampling ARI distribution
- provisional operation interpretation, if any
- observation/missingness diagnostics

#### Decision rule

Do not promote a Stage 1 cluster to further interpretation if it is dominated by very few cases, highly unstable, or strongly associated with missingness alone.

---

### Phase C - Evaluate Stage 2

Within each Stage 1 class, run numeric DTW only when numeric coverage is sufficient.

For each eligible parent cluster, report:

- number of numeric-eligible Activities
- number of distinct numeric-eligible cases
- selected Stage 2 k
- silhouette
- subcluster sizes
- case-level Stage 2 resampling ARI
- relationship between numeric distance and observation coverage

`NOT_CLUSTERED` is a valid outcome when coverage is insufficient. Do not force k >= 2.

---

### Phase D - Replay classified classes

Replay fixes the existing cluster labels and displays original observations. It is not another clustering stage.

Run:

```bash
source ~/venvs/chair_c31/bin/activate
cd "/media/knaito/Samsung_SSD/C31/chair_two_stage_replay_v1"

python visualize_two_stage_classes.py \
  --two-stage-root "/media/knaito/Samsung_SSD/C31/Output/chair_two_stage_300" \
  --data-root "/media/knaito/Samsung_SSD/C31/Output/chair_model_A_results" \
  --max-members 7 \
  --max-points 500 \
  --dpi 110
```

#### Replay outputs

```text
BXX_SYY_medoid_raw_1s.png
BXX_SYY_members_overlay.png
class_recurrence.csv
plotted_activities.csv
plot_errors.csv
```

#### Medoid review

The medoid is an observed Activity closest to other class members under the clustering distance. It is preferred over an artificial average waveform for interpretation.

Review:

- state sequence
- numeric level
- rise/fall behavior
- duration
- observation coverage

#### Member Overlay review

Overlay Activities from different cases.

The x-axis may be shown as 0-100% Activity progress to compare unequal durations. This is a visualization transform only; it should not be described as DTW warping unless actual DTW alignment is shown.

The overlay is considered promising when several independent cases reproduce similar raw signal patterns.

---

## 5. Two meanings of reproducibility

### 5.1 Pattern recurrence

Question:

> Do different cases assigned to the same class show similar original chair behavior?

Evaluate with:

- medoid raw plot
- multi-case Member Overlay
- class size
- number of distinct cases

### 5.2 Partition stability

Question:

> If some cases are removed and the clustering is re-run, is a similar partition recovered?

Evaluate with:

- `stage1_case_subsampling.csv`
- `stage2_case_subsampling.csv`
- Adjusted Rand Index (ARI)

ARI near 1 indicates high internal partition agreement. It does **not** establish prediction performance on unseen cases.

---

## 6. Class-level review checklist

For each candidate Stage 2 class, complete the following table.

| Item | Question | Desired evidence |
|---|---|---|
| Recurrence | Does the Member Overlay show a repeated pattern? | Similar raw rise/fall/level across cases |
| Representation | Are multiple independent cases included? | More than a tiny case-specific cluster |
| Stability | Is case-subsampling ARI consistently high? | Stable across repeats |
| Missingness | Is the class independent of simple observation coverage? | Separation remains in raw signals |
| Medoid | Does the medoid resemble the other members? | Representative, not an outlier |
| Numeric value | Does Stage 2 add information beyond Stage 1? | Meaningful raw-level/shape differences |
| Interpretation | Is the sensor interpretation plausible? | Consistent with known/provisional code behavior |
| External evidence | Is there a clinical/device reference? | Annotation or equipment knowledge when available |

---

## 7. Current limitations

### 7.1 Four uncovered cases

The current six-code representation cannot generate a usable DTW sequence for four of the 300 Hybrid cases.

Do not solve this by replacing absent signals with zero.

### 7.2 Sensor semantics are provisional

Working interpretations of codes such as `20039`, `20011`, `20101`, and `20742` are hypotheses derived from data behavior unless independently confirmed.

### 7.3 Internal stability is not external validation

Case-subsampling ARI measures robustness within the observed cohort. Future external validation should use new cases or independent annotations.

### 7.4 Time normalization in replay

A 0-100% Member Overlay helps visually compare different Activity durations but can hide absolute-duration differences. Always retain duration as a separate feature/statistic.

### 7.5 Preprocessed 1-second replay is not the full raw event stream

Replay plots from `raw_1s.csv.gz` show observed 1-second values. They are not a millisecond-level reconstruction of the original DAT event timing.

---

## 8. Planned broader-event experiment

The 4-case coverage failure motivates a separately versioned experiment that represents a broader event vocabulary, potentially including useful `30xxx` event families.

This experiment should answer:

1. Which event codes occur across most cases?
2. Which codes are device/firmware-specific?
3. Can a common event representation cover the four excluded cases?
4. Does the broader model discover operation structure or merely recording-system differences?
5. Does it improve stability and replay recurrence compared with the six-code baseline?

### Required controls

- code prevalence table by case
- missingness-only negative control
- observation-coverage distance correlation
- fixed case cohort for fair comparison
- same case-level resampling protocol
- raw replay of representative clusters

Do not merge the broader-event experiment into the baseline until it demonstrates added value.

---

## 9. Outcome categories

After Stage 1 + Stage 2 + Replay, classify each discovered class into one of these research statuses:

### A. Stable recurrent candidate

- multiple cases
- stable ARI
- reproducible raw pattern
- not explained mainly by missingness

Suitable for further interpretation / annotation.

### B. Exploratory candidate

- plausible pattern
- limited case count or moderate stability

Retain for larger-cohort testing.

### C. Data-availability cluster

- strongly associated with observation coverage / missing channels

Treat as a recording artifact candidate, not an operation type.

### D. Unstable / unsupported

- low stability
- very small case support
- no repeated raw pattern

Do not interpret clinically.

### E. Not clusterable

- insufficient numeric or event data

Report as such rather than forcing a result.

---

## 10. Success criteria for the current research stage

The current approximately 300-case study is successful if it can produce a small set of candidate operation patterns for which:

1. Stage 1 grouping is case-stable.
2. Stage 2, when applicable, provides additional numeric structure.
3. the class contains several independent cases.
4. raw replay shows repeated temporal behavior.
5. missingness diagnostics do not explain the clustering.
6. all exclusions and parameter choices are auditable.

Clinical naming is a later validation step, not a success requirement for this stage.

---

## 11. Recommended next actions

1. Complete the 296-case baseline pipeline with exclusion audit preserved.
2. Generate Stage 2 replay plots.
3. Review every Stage 2 class with `class_recurrence.csv`, Member Overlay, and ARI.
4. Mark each class A-E using the outcome categories above.
5. Select only stable/recurrent candidates for deeper event-level inspection.
6. Design the broader-event experiment as a separate model version.
7. After method stability is established, expand toward the full preprocessed dataset.
8. Add external annotation/device-definition validation when available.

---

## 12. Key files

### Inputs / intermediate data

```text
Output/chair_model_A_results/
Output/chair_hierarchical_v3_300/
```

### Current analysis outputs

```text
Output/chair_dtw_v3_1_300_casefirst/
Output/chair_ablation_300/
Output/chair_two_stage_300/
```

### Important audit / result files

```text
case_coverage_summary.json
case_coverage.csv
rejected_activity_windows.csv
processing_errors.csv
stage1_fixed_assignments.csv
two_stage_assignments.csv
stage2_summary.csv
stage2_medoids.csv
stage1_case_subsampling.csv
stage2_case_subsampling.csv
class_recurrence.csv
```

---

## 13. Reporting language

Preferred wording:

- "candidate operation pattern"
- "log-level cluster"
- "provisional sensor interpretation"
- "internally stable under case-level resampling"
- "recurrent raw pattern across multiple cases"

Avoid unsupported wording:

- "this cluster is treatment X"
- "the model diagnosed X"
- "ARI proves external reproducibility"
- "missing means off"

---

## 14. Repository agent instructions

CLI agents working in this project should read `AGENTS.md` before modifying analysis code. `AGENTS.md` contains the non-negotiable scientific, data-integrity, reproducibility, and memory-safety rules for this research project.
