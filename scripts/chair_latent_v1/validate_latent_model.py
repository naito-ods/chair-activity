"""Numerical, masking and synthetic checks for the independent NumPy VAE."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from latent_model import MaskedVAE


def validate():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(5, 6))
    m = rng.random(x.shape) > .3
    x[~m] = np.nan
    model = MaskedVAE(latent_dim=2, hidden_dim=4, beta=.03, seed=19,
                      feature_weights=np.arange(1,7)/3)
    model._initialize(x, m)
    eps = rng.normal(size=(5,2))
    value, gradients, _ = model.loss_and_grad(x, m, eps)
    max_errors = {}
    delta = 1e-5
    for name, parameter in model.params.items():
        errors = []
        for index in np.ndindex(parameter.shape):
            old = parameter[index]
            parameter[index] = old + delta
            plus = model.loss_and_grad(x,m,eps)[0]
            parameter[index] = old - delta
            minus = model.loss_and_grad(x,m,eps)[0]
            parameter[index] = old
            numerical = (plus-minus)/(2*delta)
            errors.append(abs(numerical-gradients[name][index]))
        max_errors[name] = float(max(errors))
    assert max(max_errors.values()) < 1e-7, max_errors
    # Denoising has different encoder and target masks. Check its token gradient
    # as well: supervised-by-observation targets must not enter the encoder.
    encoder_mask = m.copy()
    encoder_mask[:, 2:4] = False
    _, dg, _ = model.loss_and_grad(x,m,eps,input_mask=encoder_mask)
    denoise_gradient_max = 0.0
    for name, parameter in model.params.items():
        for index in np.ndindex(parameter.shape):
            old = parameter[index]
            parameter[index] = old + delta
            plus = model.loss_and_grad(x,m,eps,input_mask=encoder_mask)[0]
            parameter[index] = old - delta
            minus = model.loss_and_grad(x,m,eps,input_mask=encoder_mask)[0]
            parameter[index] = old
            denoise_gradient_max = max(denoise_gradient_max,abs((plus-minus)/(2*delta)-dg[name][index]))
    assert denoise_gradient_max < 1e-7
    hidden_changed = x.copy()
    hidden_changed[m & ~encoder_mask] += 500
    assert np.array_equal(model.predict(x,encoder_mask),model.predict(hidden_changed,encoder_mask))
    changed = x.copy()
    changed[~m] = 1e100
    other, other_grad, _ = model.loss_and_grad(changed,m,eps)
    assert value == other
    assert all(np.array_equal(gradients[k],other_grad[k]) for k in gradients)
    # At a clipped log-variance head the analytic derivative must be zero.
    old_bv = model.params['bv'].copy()
    model.params['bv'][:] = -20
    clipped = model.loss_and_grad(x,m,eps)[1]
    assert np.all(clipped['Wv'] == 0) and np.all(clipped['bv'] == 0)
    model.params['bv'][:] = old_bv
    # A zero-valued observed target differs from a missing value at the encoder.
    observed_zero = np.array([[0., 1.,2.,3.,4.,5.]])
    unknown = observed_zero.copy(); unknown[0,0] = np.nan
    mz = np.ones_like(observed_zero,dtype=bool); mu = mz.copy(); mu[0,0] = False
    assert not np.array_equal(model.transform(observed_zero,mz),model.transform(unknown,mu))
    # Heldout cases from an explicitly low-rank generating process. Loss should
    # improve on training-only feature means despite independently missing inputs.
    factors = rng.normal(size=(100,2))
    loading = rng.normal(size=(2,10))*.5
    full = np.tanh(factors @ loading) + rng.normal(scale=.02,size=(100,10))
    mask = rng.random(full.shape) > .2
    sparse = np.where(mask,full,np.nan)
    train = sparse[:75]; mt = mask[:75]
    val = sparse[75:]; mv = mask[75:]
    synth = MaskedVAE(latent_dim=2, hidden_dim=16, beta=.001, epochs=600,
                      lr=.005, patience=100, seed=21).fit(train,mt,val,mv)
    means = np.nanmean(train,axis=0)
    baseline = np.mean([np.mean((full[i,mask[i]]-means[mask[i]])**2) for i in range(75,100)])
    errors = synth.predict(val,mv)-full[75:]
    score = np.mean([np.mean(errors[i,mv[i]]**2) for i in range(25)])
    assert score < .5*baseline, (score,baseline)
    same = MaskedVAE(latent_dim=2, hidden_dim=16, beta=.001, epochs=600,
                     lr=.005, patience=100, seed=21).fit(train,mt,val,mv)
    assert np.array_equal(synth.predict(val,mv),same.predict(val,mv))
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)/'model.npz'
        synth.save(path)
        restored = MaskedVAE.load(path)
        assert np.array_equal(synth.predict(val,mv),restored.predict(val,mv))
        try:
            synth.save(path)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing model was overwritten')
    ae = MaskedVAE(latent_dim=2,hidden_dim=4,beta=0,epochs=5).fit(train,mt)
    assert np.array_equal(ae.predict(val,mv),ae.predict(val,mv))
    # Genuine denoising validation: the evaluated target columns are never
    # visible to the encoder, and dropout leaves allowed conditioning unchanged.
    numeric = np.arange(10) >= 2
    train_visible = mt.copy()
    val_visible = mv.copy(); val_visible[:, 2:4] = False
    val_target = mv & ~val_visible
    # Every synthetic validation row needs one target; choose another observed
    # numeric coordinate if both default target entries happened to be absent.
    for i in range(len(val_target)):
        if not val_target[i].any():
            j = np.flatnonzero(mv[i] & numeric)[0]
            val_visible[i,j] = False; val_target[i,j] = True
    denoise = MaskedVAE(latent_dim=2,hidden_dim=16,beta=.001,epochs=600,lr=.005,
                        patience=100,seed=21,input_dropout=.2,
                        protected_features=~numeric).fit(train,mt,val,val_target,
                                                        input_mask=train_visible,input_mask_val=val_visible)
    prediction = denoise.predict(val,val_visible)
    hidden_error = float(np.mean([np.mean((prediction[i,val_target[i]]-val[i,val_target[i]])**2) for i in range(len(val))]))
    hidden_baseline = float(np.mean([np.mean((means[val_target[i]]-val[i,val_target[i]])**2) for i in range(len(val))]))
    assert hidden_error < .6*hidden_baseline, (hidden_error,hidden_baseline)
    # Callback alignment: caller-defined validation metric receives exactly the
    # encoder-visible posterior-mean prediction. Saved best epoch must minimize
    # that metric, and saving never serializes a closure containing observations.
    seen = []
    def custom_score(pred):
        assert pred.shape == val.shape and np.isfinite(pred).all()
        # Explicit case-equal, channel-group-equal scores for two target groups.
        groups = [np.arange(10) < 3, np.arange(10) >= 3]
        case_scores = []
        for i in range(len(val)):
            entries = []
            for group in groups:
                selected = val_target[i] & group
                if selected.any():
                    entries.append(np.mean((pred[i,selected]-val[i,selected])**2))
            case_scores.append(np.mean(entries))
        result = float(np.mean(case_scores))
        seen.append(result)
        return result
    callback_model = MaskedVAE(latent_dim=2,hidden_dim=8,beta=.001,epochs=35,seed=5).fit(
        train,mt,val,val_target,input_mask_val=val_visible,validation_score_fn=custom_score)
    assert len(seen) == len(callback_model.history)
    assert np.array_equal(seen,[r['val_reconstruction'] for r in callback_model.history])
    assert abs(min(seen)-custom_score(callback_model.predict(val,val_visible))) < 1e-12
    with tempfile.TemporaryDirectory() as td:
        callback_path = Path(td)/'callback_model.npz'
        callback_model.save(callback_path)
        callback_restored = MaskedVAE.load(callback_path)
        assert np.array_equal(callback_model.predict(val,val_visible),callback_restored.predict(val,val_visible))
    assert not any(callable(v) for v in callback_model.__dict__.values())
    try:
        MaskedVAE(epochs=1).fit(train,mt,val,val_target,input_mask_val=val_visible,
                              validation_score_fn=lambda p: float('nan'))
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite callback was accepted')
    result = dict(gradient_max_absolute_error_by_parameter=max_errors,
                  denoising_gradient_max_absolute_error=float(denoise_gradient_max),
                  hidden_targets_do_not_enter_encoder=True,
                  masked_target_placeholders_ignored=True,
                  logvariance_clipping_gradient=True, observed_zero_distinct_from_unknown=True,
                  synthetic_val_mse=float(score),synthetic_mean_baseline_mse=float(baseline),
                  deterministic_seed=True,save_load_exact=True,overwrite_refused=True,
                  deterministic_ae=True, denoising_hidden_val_mse=hidden_error,
                  denoising_hidden_mean_baseline_mse=hidden_baseline,
                  callback_metric_controls_best_epoch=True,
                  callback_save_load_predictions_exact=True,
                  callback_not_persisted=True,nonfinite_callback_rejected=True)
    return result


if __name__ == '__main__':
    print(json.dumps(validate(),indent=2))
