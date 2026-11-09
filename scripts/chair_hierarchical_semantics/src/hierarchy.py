"""Small single-parent taxonomy, explicit partial resolution, and bounded credit.

This module knows no medical meanings. A reviewed, versioned mapping supplies
them. Sensor observations never become clinical assertions through this code.
"""
from collections import defaultdict
import math


class Taxonomy:
    def __init__(self, document):
        self.version = document['version']
        self.root = document['root']
        self.nodes = {n['id']: dict(n) for n in document['nodes']}
        if len(self.nodes) != len(document['nodes']) or self.root not in self.nodes:
            raise ValueError('Duplicate nodes or missing root')
        self.paths = {}
        def visit(key, active):
            if key in self.paths:
                return self.paths[key]
            if key in active or key not in self.nodes:
                raise ValueError('Cycle or missing parent')
            node = self.nodes[key]
            if key == self.root:
                if node['parent'] is not None or node['depth'] != 0:
                    raise ValueError('Invalid root')
                path = []
            else:
                path = visit(node['parent'], active | {key}) + [key]
                if node['depth'] != len(path) or node['depth'] not in (1,2,3):
                    raise ValueError('Depth must be consecutive, 1..3')
            self.paths[key] = path
            return path
        for key in self.nodes:
            visit(key, set())

    def depth(self, concept):
        return self.nodes[concept]['depth'] if concept in self.nodes else 0

    def at(self, concept, depth):
        if depth not in (1,2,3):
            raise ValueError('Root is not an evaluation level')
        path = self.paths.get(concept, [])
        return path[depth-1] if len(path) >= depth else None

    def lca_depth(self, truth, predicted):
        n = 0
        for a,b in zip(self.paths.get(truth, []), self.paths.get(predicted, [])):
            if a != b:
                break
            n += 1
        return n

    def distance(self, truth, predicted):
        if truth not in self.nodes or predicted not in self.nodes:
            return None
        return self.depth(truth)+self.depth(predicted)-2*self.lca_depth(truth,predicted)

    def semantic_credit(self, truth, predicted):
        """Depth-adjusted distance credit for a fully specified depth-3 truth.

        credit=1-(distance + (3-predicted_depth))/6 = shared_depth/3.
        Unresolved predictions receive zero. An unresolved truth is unscorable.
        This is a proposed equal-level utility, not a clinical severity measure.
        """
        if self.depth(truth) != 3:
            return None
        return self.lca_depth(truth,predicted)/3.0

    def path_f1(self, truth, predicted):
        """Ancestor-path F1 (root excluded), a depth-based similarity check."""
        if truth not in self.nodes:
            return None
        denom = self.depth(truth)+self.depth(predicted)
        return 2*self.lca_depth(truth,predicted)/denom if denom else 0.0


def select_resolution(taxonomy, masses, thresholds, *, calibrated=False,
                      allow_uncalibrated_demo=False):
    """Choose the deepest supported node; unknown mass is never redistributed.

    Inputs are mutually exclusive outcome masses, not arbitrary classifier
    margins. A mass at a partial node supports its ancestors, not descendants.
    A production caller must provide a separate calibration contract.
    """
    if not calibrated and not allow_uncalibrated_demo:
        raise ValueError('Calibration required before real-data selective output')
    values = list(masses.values())
    if any(not math.isfinite(v) or v < 0 for v in values) or sum(values) > 1+1e-9:
        raise ValueError('Mass must be finite, nonnegative and sum to at most one')
    if set(thresholds) != {1,2,3} or any(not 0<t<=1 for t in thresholds.values()):
        raise ValueError('Require an independently chosen threshold per level')
    totals = defaultdict(float)
    for concept,value in masses.items():
        for ancestor in taxonomy.paths.get(concept,[]):
            totals[ancestor] += value
    # A single top-down path avoids inconsistent separate level predictions.
    current = taxonomy.root
    chosen = None
    for depth in (1,2,3):
        choices = [k for k,n in taxonomy.nodes.items() if n['parent']==current]
        choices.sort(key=lambda k:(-totals[k], k))
        if not choices or totals[choices[0]] < thresholds[depth]:
            break
        chosen = choices[0]
        current = chosen
    return {'concept_id':chosen,'resolved_depth':taxonomy.depth(chosen),
            'interpretation_status':'Identified' if chosen else 'Unknown',
            'evidence_basis':'Inferred','scores_calibrated':calibrated,
            'demonstration_only':not calibrated,
            'selected_mass':totals[chosen] if chosen else None}


def validate_assertion(assertion):
    domain = assertion['domain']
    basis = assertion['evidence_basis']
    source = assertion['assertion_source']
    if basis not in {'Observed','Inferred','Unspecified'}:
        raise ValueError('Invalid evidence basis')
    if domain == 'clinical_action' and basis == 'Observed':
        raise ValueError('This schema reserves Observed for direct device facts')
    if basis == 'Observed' and (domain != 'device_observation' or not assertion.get('source_refs')):
        raise ValueError('Observed requires traceable device evidence')
    if source == 'model_inference' and basis != 'Inferred':
        raise ValueError('Model outputs remain inferred')
    if source == 'model_inference' and assertion.get('human_confirmed') is not False:
        raise ValueError('Model outputs must explicitly remain unconfirmed')
    if source == 'human_annotation' and basis != 'Unspecified':
        raise ValueError('Legacy annotation evidence is unspecified')
    return True
