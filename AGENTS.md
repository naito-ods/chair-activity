# AGENTS.md - Chair Log Activity / DTW Research Agent Instructions

## User-confirmed definitions for the non-DTW branches (2026-11-04)

These session-confirmed definitions supersede older provisional mappings below
for separately versioned numeric/HP/context workflows; they do not authorize
changing existing Stage 2 assignments.

- Lamp 90012 first ON starts the case, final OFF ends it. Intermediate OFF stays
  in the case. State/event changes persist until the next record; unknown before
  the first applicable observation remains unknown.
- 20039: operating HP, 0 OFF / 1 HP1 / 2 HP2 / 3 HP3 (scaler).
- 20011: selected HP. 20774: scaler operating, 20776: scaler output setting.
- 20742: RPM, 20743: torque, 20108: bulk opening, 10014/10015: pedal1/pedal2.
- 20101 syringe: 0 picked up, 1 stored. Pickup does not establish spraying.
- LOW is positive RPM <=5000, MID >5000 and below the high band. User defines
  high near40000; the current analyst tolerance is36000..44000 inclusive, with
  narrower/wider band sensitivity. Zero and missing are separate.
- Numeric plots/features retain raw observed values. Never extend a numeric
  value beyond its actual operating run to manufacture continuous waveforms.
- User corrected the short lower-speed tail after HIGH to MID, not LOW.
- Whole-case context, HP type and large operation order are the main current
  research focus. Short waveform variation alone may carry little meaning.
- Record decisions, review findings and corrections in separate process files.
- Publish code/re-execution guidance to GitHub at work milestones; keep clinical
  data, per-case figures, local reports and notification addresses out of commits.

## 1. Mission

This repository analyzes dental chair logs to discover reproducible operation patterns without assuming clinical treatment labels in advance.

The working pipeline is:

`Hybrid v3 Activity extraction -> Stage 1 State+Events clustering -> Stage 2 Numeric DTW -> Replay / reproducibility review`

The main goal is not to assign a treatment name automatically. The goal is to identify stable, interpretable **log-level operation patterns** and determine which patterns are reproducible across independent cases.

## 2. Non-negotiable scientific rules

1. **Do not treat unsupervised clusters as clinical diagnoses or treatment names.**
   - Cluster labels such as HP1, HP2, syringe, idle, etc. are provisional sensor/operation interpretations unless externally validated.
2. **Do not replace missing observations with zero.**
   - Missing/unknown and OFF/0 are different states.
3. **Preserve raw numeric values.**
   - Normalized values may be used for shape/distance calculations, but interpretation and plots must retain raw values when possible.
4. **Never silently remove cases or Activities.**
   - Record exclusions, failure reason, requested count, actual count, and affected IDs.
5. **Use case-level resampling for stability analysis.**
   - Avoid leakage caused by treating multiple Activities from the same case as independent validation units.
6. **Do not claim external reproducibility from internal ARI alone.**
   - Case-subsampling ARI measures internal stability, not performance on unseen patients/cases.
7. **Do not change segmentation boundaries, DTW weights, missing-data policy, or code mappings merely to improve silhouette.**
   - Any methodological change must be recorded as a new analysis version and compared against the previous version.
8. **Do not overwrite source/preprocessed data.**
   - Analysis outputs go to new output directories. Source files are read-only unless explicitly requested.

## 3. Current verified project state

### Hybrid v3

- Hybrid v3 has been expanded to 300 cases.
- Activity extraction itself succeeded for the four cases that later failed DTW input generation.

### DTW coverage issue

The current 300-case selection produced:

- Requested cases: 300
- Cases with at least one usable DTW Activity: 296
- Missing cases: 4
- Failure class: `SIGNATURE_FAILED_FOR_ALL_CANDIDATES`
- Failure reason: `no_usable_sequence`

Affected case IDs:

- `2024-05-21__C512KC1503__Chair_log_seq_1558`
- `2024-06-13__C512KC1503__Chair_log_seq_1604`
- `2024-11-21__C512KC1503__Chair_log_seq_1930`
- `2024-12-18__C512KC1503__Chair_log_seq_1996`

For these cases:

- Hybrid v3 produced 12 refined Activity candidates per case.
- Activity time ranges overlap the preprocessed time axis.
- The currently selected six DTW signal codes have no usable observations.
- Other events are present; the logs are not empty.
- Frequent other event codes include `30117`, `30116`, `30123`, `30122`, and `30121`.

Therefore, do **not** describe these four cases as corrupt or empty without additional evidence.

## 4. Signal groups currently used by DTW

### State / event-oriented codes

- `20039`
- `20011`
- `20101`

### Numeric codes

- `20742`
- `20743`
- `20108`

### Treatment interval marker

- `90012`
  - first transition/value `1`: analysis start candidate
  - final transition/value `0`: analysis end candidate

### Provisional semantic hypotheses

These mappings are working hypotheses derived from observed logs, not confirmed manufacturer specifications:

- `20039`: handpiece work candidate; values may distinguish HP1 / HP2 / off
- `20011`: handpiece selection candidate
- `20101`: syringe pickup/return candidate
- `20742`: RPM-related candidate

Never promote these hypotheses to confirmed clinical meanings without external validation.

## 5. Two-stage analysis model

### Stage 1: State + Events

Purpose:

- identify broad operation patterns using discrete state/event behavior
- reduce sensitivity to sparse numeric observations

Primary outputs/metrics:

- cluster assignments
- selected k
- silhouette
- cluster balance
- case-level subsampling ARI
- case count per cluster

### Stage 2: Numeric DTW within Stage 1 classes

Purpose:

- subdivide a coarse Stage 1 class using numeric signal shapes and raw numeric levels
- only run when numeric coverage is sufficient

A Stage 2 group may legitimately remain `NOT_CLUSTERED` if numeric data are insufficient. Do not force a split.

### Replay / reproducibility review

Replay does not re-cluster Activities. It fixes the existing Stage 2 labels and visualizes the original observations.

Use two complementary concepts of reproducibility:

1. **Pattern recurrence:** multiple independent cases in the same class show similar raw temporal behavior.
2. **Partition stability:** case-level resampling and re-clustering produce similar partitions, measured by ARI.

These are related but not equivalent.

## 6. Replay plots and interpretation

### Medoid Raw plot

A medoid is an actual observed Activity, not an average waveform.

Use it to inspect:

- state transitions
- raw numeric level
- duration
- whether the representative is plausible for the class

### Member Overlay

Overlay several Activities from different cases in the same Stage 2 subcluster.

Rules:

- prefer distinct cases
- x-axis may be normalized to Activity progress 0-100% for visual comparison
- do not describe this as DTW-warped time unless actual warping is plotted
- do not interpolate missing regions simply to make curves look smooth
- display raw observations when possible

Good signs:

- similar rise/fall patterns across cases
- similar operating levels
- repeated temporal motifs
- multiple distinct cases represented

Warning signs:

- medoid looks distinctive but other members scatter widely
- class depends on only one or two cases
- class separation follows observation coverage rather than signal behavior
- class identity disappears under case-level subsampling

## 7. Primary reproducibility files

Expected Two-stage outputs include:

- `stage1_fixed_assignments.csv`
- `two_stage_assignments.csv`
- `stage2_summary.csv`
- `stage2_medoids.csv`
- `stage1_case_subsampling.csv`
- `stage2_case_subsampling.csv`

Replay outputs include:

- `BXX_SYY_medoid_raw_1s.png`
- `BXX_SYY_members_overlay.png`
- `class_recurrence.csv`
- `plotted_activities.csv`
- `plot_errors.csv`

When reporting a Stage 2 class, include at minimum:

- Stage 1 cluster ID
- Stage 2 subcluster ID
- number of Activities
- number of distinct cases
- medoid case / Activity ID
- silhouette or selected-k evidence when available
- case-subsampling ARI distribution or median
- numeric observation coverage
- qualitative replay result

## 8. Environment and paths

Primary current environment:

- OS: Ubuntu
- Python: 3.10
- virtual environment: `~/venvs/chair_c31`
- external SSD root: `/media/knaito/Samsung_SSD/C31`

Do not create the Python venv on the Samsung SSD if the mounted filesystem does not support Linux symlinks. Keep the venv under the Linux home directory.

Common paths:

```text
/media/knaito/Samsung_SSD/C31/Output/chair_model_A_results
/media/knaito/Samsung_SSD/C31/Output/chair_hierarchical_v3_300
/media/knaito/Samsung_SSD/C31/Output/chair_dtw_v3_1_300_casefirst
/media/knaito/Samsung_SSD/C31/Output/chair_ablation_300
/media/knaito/Samsung_SSD/C31/Output/chair_two_stage_300
```

## 9. Memory / performance rules

The original workstation has limited RAM, so prefer memory-safe operations even when running on Ubuntu.

- do not load all wide `raw_1s.csv.gz` cases into memory simultaneously
- use `usecols` to load only required signals
- use chunked reads for wide event-count data
- close matplotlib figures after saving
- avoid high-DPI multi-member plots by default
- cache/reuse completed validated results when possible
- write intermediate audit CSVs instead of retaining large objects in memory

If a memory failure occurs, reduce plotting load before changing analytical methodology.

## 10. Current executable workflows

### Activate environment

```bash
source ~/venvs/chair_c31/bin/activate
```

### 296/300 pilot continuation

Use only when the research decision is explicitly to continue with the 296 usable cases while preserving the four exclusions in the audit trail:

```bash
cd "/media/knaito/Samsung_SSD/C31/chair_300_pipeline_shortfall_v2"
python run_300_pipeline.py \
  --output-root "/media/knaito/Samsung_SSD/C31/Output" \
  --stage all \
  --resume \
  --allow-case-shortfall \
  --min-case-fraction 0.95
```

This is a **296-case pilot continuation**, not a completed 300-case analysis.

### Replay existing Two-stage classes

```bash
cd "/media/knaito/Samsung_SSD/C31/chair_two_stage_replay_v1"
python visualize_two_stage_classes.py \
  --two-stage-root "/media/knaito/Samsung_SSD/C31/Output/chair_two_stage_300" \
  --data-root "/media/knaito/Samsung_SSD/C31/Output/chair_model_A_results" \
  --max-members 7 \
  --max-points 500 \
  --dpi 110
```

## 11. Required quality-control behavior for CLI agents

Before running a stage:

1. inspect input paths and required files
2. print requested case/activity counts
3. validate that previous-stage IDs match the next-stage manifest
4. refuse to silently reuse incompatible output generated with different settings

After running a stage:

1. report actual cases and Activities used
2. report excluded IDs and reasons
3. verify assignment row counts and unique case counts
4. verify no duplicate `(case_id, activity_id)` keys unless explicitly expected
5. record parameters / seed / version
6. preserve output files needed to reproduce the result

When a count mismatch occurs, stop and diagnose rather than silently continuing unless an explicit shortfall option was requested.

## 12. Research questions the agent should keep visible

1. Are Stage 1 clusters stable across cases, or driven by case-specific recording patterns?
2. Does Stage 2 add meaningful numeric substructure beyond Stage 1?
3. Do Stage 2 classes recur as similar raw waveforms in multiple independent cases?
4. Are clusters driven by missingness / observation masks rather than actual operation behavior?
5. Can a broader event representation recover cases not covered by the current six-code DTW without creating a recording-system cluster?
6. Which clusters remain stable when moving from the 100-case pilot to approximately 300 cases and eventually the full dataset?
7. Which clusters can be externally validated with annotations or device knowledge?

## 13. Future method changes

A likely next research branch is a broader event representation that includes useful `30xxx` event families. Treat this as a **new model version**, not a patch to the current six-code DTW.

Before adopting additional codes:

- quantify prevalence by case
- distinguish event absence from unobserved/missing data
- verify whether codes are device/firmware/site dependent
- test a missingness-only negative control
- compare clustering against the current Stage 1 baseline
- retain raw-event auditability

Do not add codes solely because they rescue the four currently excluded cases.

## 14. Definition of a promising class

A class is a research candidate when most of the following hold:

- represented by multiple distinct cases
- reasonable cluster size / balance
- stable under case-level resampling
- replay shows recurrent raw operation patterns
- numeric separation is not explained mainly by observation coverage
- medoid is representative of members, not an outlier
- interpretation is consistent with available sensor semantics

Even then, label it as a **candidate operation pattern** until externally validated.
