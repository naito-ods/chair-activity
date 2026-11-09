# Hierarchical annotation semantics

Keep device observations, clinical concepts, and evidence provenance separate.
This implementation adds a versioned single-parent taxonomy and partial-credit
evaluation without changing the original labels or fitting a classifier.

`src/hierarchy.py` uses the Python standard library. Run the synthetic checks:

```bash
python -m unittest discover -s tests -v
```

Load a private reviewed taxonomy as a JSON document with `version`, `root`, and
`nodes` (`id`, `parent`, and integer `depth`). Root depth is zero. Clinical
evaluation levels are one, two, and three. The constructor rejects duplicate
IDs, missing parents, cycles, and inconsistent depths. Domain-specific labels,
their mappings, source logs, reports, model outputs, and case identifiers are
deliberately kept outside this public package.

```python
import json
import sys
sys.path.insert(0, "src")
from hierarchy import Taxonomy

with open("private_taxonomy.json", encoding="utf-8") as stream:
    taxonomy = Taxonomy(json.load(stream))

# truth and prediction are concept IDs from that private mapping.
major = taxonomy.at(prediction, 1)
middle = taxonomy.at(prediction, 2)
detail = taxonomy.at(prediction, 3)
distance = taxonomy.distance(truth, prediction)
partial_credit = taxonomy.semantic_credit(truth, prediction)
```

Preserve the original exact-label accuracy. For each hierarchy level, freeze a
truth-resolved mask shared by all models. Unresolved or too-shallow predictions
remain in the denominator. Unresolved truth does not become an artificial
"unknown" class that can earn a match. Compute case/parent/point weights once;
do not merge original annotation parents when they share an ancestor.

On depth-three truth, the proposed equal-level partial credit is

```text
1 - (tree_distance + 3 - prediction_depth) / 6
  = common_ancestor_depth / 3
  = (major_match + middle_match + detail_match) / 3
```

It awards one for the same detail, two thirds for a correct middle concept,
one third for a correct major concept, and zero for root or unresolved output.
This is a declared evaluation utility, not a measure of clinical harm.
`path_f1` exposes ancestor-path F1 as a supplementary depth-based similarity.
Report output depth and coverage alongside every score. Compare against
train-only majority predictions and always-root/always-unknown controls.

`select_resolution` aggregates mutually exclusive outcome masses over
ancestors and follows one consistent top-down path. Unknown mass is never
redistributed. It requires a calibration contract for real-data use. The
`allow_uncalibrated_demo` switch is for synthetic demonstrations only. Choose
thresholds on training/validation data separate from the final test cases.

`schema/annotation.schema.json` illustrates the sidecar contract:

- Record presence is separate from interpretation and evidence type.
- `Observed` refers to traceable device facts.
- `Inferred` identifies model assertions, never human-confirmed references.
- Legacy human annotations with missing evidence keep `Unspecified` provenance.
- No new event in an interval does not imply no ongoing state or no activity.
- Unknown gaps remain unknown even when a separate candidate is available.

The schema can be validated with a Draft 7 JSON Schema validator. The
Python assertion validator additionally prevents device observations being
silently promoted to clinical actions.

References:

- [SNOMED CT logical model](https://docs.snomed.org/snomed-ct-practical-guides/snomed-ct-starter-guide/5-snomed-ct-logical-model): subtype and attribute relationships are distinct. This package is not a SNOMED CT release or mapping.
- [Kosmopoulos et al., Evaluation Measures for Hierarchical Classification](https://arxiv.org/abs/1306.6802): hierarchical evaluation and metric failure modes.
- [Wu and Palmer, Verb Semantics and Lexical Selection](https://aclanthology.org/P94-1019/): depth-based semantic similarity.
