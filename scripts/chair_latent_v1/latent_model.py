"""Small NumPy masked AE/VAE for a separately versioned exploratory analysis.

This is inspired by continuous latent modelling, not the cryoDRGN image/volume
likelihood. Inputs must already be scaled using training cases only. A missing
input is represented by a trainable encoder token AND an explicit mask. It is
never turned into an observed zero. Only observed targets contribute to loss.

``fit(x, mask, x_val, mask_val)`` minimizes per-case weighted observed MSE plus
``beta * mean_case(sum_latent(KL)/latent_dim)``. beta=0 is a deterministic AE.
Validation reconstruction uses the posterior mean and controls early stopping;
it is NOT an independent test. The caller must hide test targets before input
construction and assess them separately. No state/clinical labels are consumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class MaskedVAE:
    def __init__(self, latent_dim=4, hidden_dim=48, beta=0.001, seed=0,
                 epochs=400, lr=0.003, patience=50, feature_weights=None,
                 batch_size=None, input_dropout=0.0, protected_features=None):
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.beta = float(beta)
        self.seed = int(seed)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.patience = int(patience)
        self.feature_weights = feature_weights
        self.batch_size = batch_size
        self.input_dropout = float(input_dropout)
        self.protected_features = protected_features
        self.params = {}
        self.history = []
        self.best_epoch = 0
        if min(self.latent_dim, self.hidden_dim, self.epochs, self.patience) < 1:
            raise ValueError('Dimensions, epochs and patience must be positive')
        if self.beta < 0 or self.lr <= 0:
            raise ValueError('beta must be nonnegative and lr positive')
        if not 0 <= self.input_dropout < 1:
            raise ValueError('input_dropout must be in [0,1)')

    @staticmethod
    def _check(x, mask):
        x = np.asarray(x, dtype=np.float64)
        m = np.asarray(mask, dtype=bool)
        if x.ndim != 2 or m.shape != x.shape or not len(x):
            raise ValueError('Expected nonempty 2D x and same-shape mask')
        if not np.isfinite(x[m]).all():
            raise ValueError('Mask marks a nonfinite value as observed')
        return x, m

    def _initialize(self, x, mask):
        self.rng = np.random.default_rng(self.seed)
        d, h, k = x.shape[1], self.hidden_dim, self.latent_dim
        def weight(a, b):
            return self.rng.normal(0, np.sqrt(2 / (a + b)), size=(a, b))
        sums = np.sum(np.where(mask, x, 0.0), axis=0)
        counts = mask.sum(axis=0)
        # These are model tokens, not filled observations. Initially the training
        # coordinate mean, or 0 in an entirely unsupported model coordinate.
        tokens = np.divide(sums, counts, out=np.zeros(d), where=counts > 0)
        self.params = dict(token=tokens, We=weight(2*d, h), be=np.zeros(h),
                           Wm=weight(h, k), bm=np.zeros(k),
                           Wv=weight(h, k)*0.1, bv=np.full(k, -2.0),
                           Wd=weight(k, h), bd=np.zeros(h),
                           Wo=weight(h, d), bo=np.zeros(d))
        self.weights_ = np.ones(d) if self.feature_weights is None else np.asarray(self.feature_weights, float)
        if self.weights_.shape != (d,) or not np.isfinite(self.weights_).all() or (self.weights_ <= 0).any():
            raise ValueError('feature_weights must be positive finite values, one per feature')
        self.protected_ = np.zeros(d, dtype=bool) if self.protected_features is None else np.asarray(self.protected_features, bool)
        if self.protected_.shape != (d,):
            raise ValueError('protected_features needs one boolean per feature')

    def _forward(self, x, mask, epsilon=None):
        p = self.params
        values = np.where(mask, x, p['token'][None, :])
        a = np.concatenate([values, mask.astype(float)], axis=1)
        e = np.tanh(a @ p['We'] + p['be'])
        mu = e @ p['Wm'] + p['bm']
        raw_lv = e @ p['Wv'] + p['bv']
        lv = np.clip(raw_lv, -8.0, 5.0)
        sd = np.exp(0.5 * lv)
        if self.beta == 0 or epsilon is None:
            z = mu
            used_eps = np.zeros_like(mu)
        else:
            used_eps = np.asarray(epsilon, float)
            if used_eps.shape != mu.shape:
                raise ValueError('epsilon shape differs from latent coordinates')
            z = mu + sd * used_eps
        dh = np.tanh(z @ p['Wd'] + p['bd'])
        y = dh @ p['Wo'] + p['bo']
        return y, (a, e, mu, raw_lv, lv, sd, used_eps, z, dh)

    def loss_and_grad(self, x, mask, epsilon=None, input_mask=None):
        """Return loss, gradient dict and diagnostics, without updating weights.

        Pass fixed epsilon for repeatable VAE gradient checks. None uses mean z.
        Missing target placeholders (NaN or any finite value) are ignored.
        """
        x, m = self._check(x, mask)
        im = m if input_mask is None else self._check(x, input_mask)[1]
        if not self.params:
            self._initialize(x, m)
        if not m.any(axis=1).all():
            raise ValueError('Training/loss rows require an observed target; audit unsupported cases separately')
        y, cache = self._forward(x, im, epsilon)
        a, e, mu, raw_lv, lv, sd, eps, z, dh = cache
        n, d = x.shape
        weights = m * self.weights_[None, :]
        weights = weights / weights.sum(axis=1, keepdims=True) / n
        diff = np.where(m, y - x, 0.0)
        reconstruction = np.sum(weights * diff * diff)
        kl = float(np.mean(np.sum(0.5*(mu*mu + np.exp(lv) - 1 - lv), axis=1) / self.latent_dim))
        loss = float(reconstruction + self.beta * kl)
        gy = 2 * weights * diff
        p = self.params
        g = dict(Wo=dh.T @ gy, bo=gy.sum(axis=0))
        gdh = (gy @ p['Wo'].T) * (1 - dh*dh)
        g['Wd'] = z.T @ gdh
        g['bd'] = gdh.sum(axis=0)
        gz = gdh @ p['Wd'].T
        gmu = gz + self.beta * mu / (n*self.latent_dim)
        glv = gz * (0.5*sd*eps) + self.beta * 0.5*(np.exp(lv)-1) / (n*self.latent_dim)
        glv *= (raw_lv > -8) & (raw_lv < 5)
        if self.beta == 0:
            glv[:] = 0
        g['Wm'], g['bm'] = e.T @ gmu, gmu.sum(axis=0)
        g['Wv'], g['bv'] = e.T @ glv, glv.sum(axis=0)
        ge = (gmu @ p['Wm'].T + glv @ p['Wv'].T) * (1 - e*e)
        g['We'], g['be'] = a.T @ ge, ge.sum(axis=0)
        ga = ge @ p['We'].T
        g['token'] = np.sum(np.where(~im, ga[:, :d], 0.0), axis=0)
        return loss, g, dict(reconstruction=float(reconstruction), kl=kl)

    def _reconstruction(self, x, mask, input_mask=None):
        im = mask if input_mask is None else self._check(x, input_mask)[1]
        y, _ = self._forward(x, im)
        w = mask * self.weights_[None, :]
        denom = w.sum(axis=1)
        if (denom <= 0).any():
            raise ValueError('Validation row lacks observations')
        error = np.where(mask, y-x, 0.0)
        return float(np.mean(np.sum(w*error*error, axis=1)/denom))

    def fit(self, x, mask, x_val=None, mask_val=None, input_mask=None, input_mask_val=None,
            validation_score_fn=None):
        """Fit observed targets, optionally from separate visible encoder inputs.

        ``mask`` and ``mask_val`` select observed loss targets. ``input_mask``
        and ``input_mask_val`` select encoder-visible values in the same arrays.
        With denoising dropout only encoder visibility is randomly reduced;
        protected_features remain unchanged. Validation input masks are fixed.
        All validation targets must be absent from input_mask_val if validation
        is intended as a heldout-target evaluation; caller controls that policy.
        ``validation_score_fn(mean_prediction)`` may supply the scalar stopping
        metric (lower is better), e.g. case/channel-equal MSE after undoing a
        residual transform. It must use validation data only. No callback or
        callback closure is retained or persisted in the model artifact.
        """
        x, mask = self._check(x, mask)
        input_mask = mask if input_mask is None else self._check(x, input_mask)[1]
        if not mask.any(axis=1).all():
            raise ValueError('Unsupported training rows must be audited, not silently dropped')
        if (x_val is None) != (mask_val is None):
            raise ValueError('Provide both x_val and mask_val')
        if validation_score_fn is not None and x_val is None:
            raise ValueError('validation_score_fn requires validation data')
        if validation_score_fn is not None and not callable(validation_score_fn):
            raise TypeError('validation_score_fn must be callable')
        if x_val is not None:
            x_val, mask_val = self._check(x_val, mask_val)
            input_mask_val = mask_val if input_mask_val is None else self._check(x_val, input_mask_val)[1]
            if x_val.shape[1] != x.shape[1] or not mask_val.any(axis=1).all():
                raise ValueError('Invalid validation shape or unsupported validation rows')
        self._initialize(x, mask)
        self.history = []
        mom = {k: np.zeros_like(v) for k, v in self.params.items()}
        vel = {k: np.zeros_like(v) for k, v in self.params.items()}
        step = 0
        best = float('inf')
        best_params = None
        n = len(x)
        batch_size = n if self.batch_size is None else min(n, int(self.batch_size))
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        for epoch in range(1, self.epochs + 1):
            order = self.rng.permutation(n)
            for start in range(0, n, batch_size):
                ix = order[start:start+batch_size]
                epsilon = self.rng.normal(size=(len(ix), self.latent_dim)) if self.beta > 0 else None
                visible = input_mask[ix].copy()
                if self.input_dropout:
                    hide = self.rng.random(visible.shape) < self.input_dropout
                    hide[:, self.protected_] = False
                    visible &= ~hide
                _, grad, _ = self.loss_and_grad(x[ix], mask[ix], epsilon, input_mask=visible)
                norm = np.sqrt(sum(np.sum(v*v) for v in grad.values()))
                factor = min(1.0, 10.0/max(norm, 1e-12))
                step += 1
                for k in self.params:
                    gv = grad[k] * factor
                    mom[k] = .9*mom[k] + .1*gv
                    vel[k] = .999*vel[k] + .001*gv*gv
                    self.params[k] -= self.lr*(mom[k]/(1-.9**step))/(np.sqrt(vel[k]/(1-.999**step))+1e-8)
            _, _, diagnostics = self.loss_and_grad(x, mask, input_mask=input_mask)
            if validation_score_fn is not None:
                ymean, _ = self._forward(x_val, input_mask_val)
                val = float(validation_score_fn(ymean))
                if not np.isfinite(val):
                    raise ValueError('Validation callback returned a nonfinite score')
            else:
                val = self._reconstruction(x_val, mask_val, input_mask_val) if x_val is not None else diagnostics['reconstruction']
            self.history.append(dict(epoch=epoch, train_reconstruction=diagnostics['reconstruction'],
                                     train_kl=diagnostics['kl'], val_reconstruction=val))
            if val < best - 1e-8:
                best = val
                best_params = {k: v.copy() for k, v in self.params.items()}
                self.best_epoch = epoch
            if x_val is not None and epoch - self.best_epoch >= self.patience:
                break
        self.params = best_params
        return self

    def transform(self, x, mask):
        x, mask = self._check(x, mask)
        if not self.params:
            raise RuntimeError('Model has not been fitted')
        return self._forward(x, mask)[1][2]

    def decode(self, z):
        z = np.asarray(z, dtype=float)
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError('Invalid latent shape')
        p = self.params
        return np.tanh(z @ p['Wd'] + p['bd']) @ p['Wo'] + p['bo']

    def predict(self, x, mask):
        return self.decode(self.transform(x, mask))

    def save(self, path):
        """Store parameters/settings/history; refuse to overwrite an artifact."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        settings = {k: getattr(self, k) for k in ['latent_dim','hidden_dim','beta','seed','epochs','lr','patience','batch_size','input_dropout']}
        payload = dict(self.params)
        payload.update(feature_weights=self.weights_, protected_features=self.protected_, settings_json=np.asarray(json.dumps(settings)),
                       history_json=np.asarray(json.dumps(self.history)), best_epoch=np.asarray(self.best_epoch))
        with path.open('xb') as handle:
            np.savez_compressed(handle, **payload)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            settings = json.loads(str(data['settings_json']))
            model = cls(**settings, feature_weights=data['feature_weights'].copy(), protected_features=data['protected_features'].copy())
            excluded = {'feature_weights','protected_features','settings_json','history_json','best_epoch'}
            model.params = {k: data[k].copy() for k in data.files if k not in excluded}
            model.weights_ = data['feature_weights'].copy()
            model.protected_ = data['protected_features'].copy()
            model.history = json.loads(str(data['history_json']))
            model.best_epoch = int(data['best_epoch'])
        return model
