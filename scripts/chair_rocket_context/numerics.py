"""Train-only ROCKET variance guards and exploratory temporal inference helpers.

Only numeric arrays enter these helpers. They contain no dataset, fitted model,
clinical labels or file paths. Margins are discriminative energies, not probabilities.
"""
import numpy as np


def variance_mask(training_mean, training_variance):
    """Drop variation below 10 float32 eps relative to the training feature mean.

    ROCKET may emit float32 features. Promoting them to float64 before scaling
    prevents later arithmetic loss but does not remove their quantization noise.
    Fit statistics on each training fold only. Apply the same mask to queries.
    Use variance, not StandardScaler.scale_: the latter is1 for constant features.
    """
    mean=np.asarray(training_mean,dtype=np.float64)
    var=np.asarray(training_variance,dtype=np.float64)
    if mean.shape!=var.shape or mean.ndim!=1:raise ValueError('Expected matching feature vectors')
    if not np.isfinite(mean).all() or not np.isfinite(var).all() or (var<0).any():raise ValueError('Invalid training statistics')
    threshold=10*np.finfo(np.float32).eps*np.maximum(1,np.abs(mean))
    return np.sqrt(var)>threshold


def bounded_late_fusion(local, context, available, context_weight=.2):
    """Bound raw-context influence while preserving train-fold class vocabulary.

    available must derive from training model classes, never from a margin cutoff:
    an actual margin can be more negative than an unavailable-class sentinel.
    No label from a query case is used. Weights are fixed experimental settings,
    not calibrated probabilities or evidence of clinical validity.
    """
    local=np.asarray(local,dtype=np.float64);context=np.asarray(context,dtype=np.float64)
    available=np.asarray(available,dtype=bool)
    if local.shape!=context.shape or local.shape!=available.shape or local.ndim!=2:raise ValueError('Mismatched score/class arrays')
    if not 0<=context_weight<=1 or not available.any(axis=1).all():raise ValueError('Invalid weight or no available class')
    if not np.isfinite(local[available]).all() or not np.isfinite(context[available]).all():raise ValueError('Nonfinite available score')
    scores=(1-context_weight)*local+context_weight*np.clip(context,-1,1)
    return np.where(available,scores,-np.inf)


def decode_sequence(energies, transition_probabilities, transition_weight=.2):
    """Viterbi over uncalibrated energies, with a uniform initial distribution.

    Transition counts/probabilities must be fitted on training cases only. Training
    must not manufacture transitions across unknown annotation gaps. A full-domain
    decoder can still propagate a candidate THROUGH unknown query time; this does
    not establish continued treatment, background absence, or boundary accuracy.
    """
    energies=np.asarray(energies,dtype=np.float64);p=np.asarray(transition_probabilities,dtype=np.float64)
    if energies.ndim!=2 or energies.shape[0]==0:raise ValueError('Expected a nonempty time/class matrix')
    n=energies.shape[1]
    if p.shape!=(n,n) or not np.isfinite(p).all() or (p<=0).any():raise ValueError('Expected positive smoothed transition rows')
    if not np.allclose(p.sum(axis=1),1) or not np.isfinite(energies).all() or transition_weight<0:raise ValueError('Invalid transition/energy array')
    score=energies[0].copy();back=[]
    for row in energies[1:]:
        candidates=score[:,None]+transition_weight*np.log(p)
        ix=candidates.argmax(axis=0);score=row+candidates[ix,np.arange(n)];back.append(ix)
    last=int(score.argmax());path=[last]
    for ix in reversed(back):last=int(ix[last]);path.append(last)
    return np.asarray(path[::-1],dtype=int)
