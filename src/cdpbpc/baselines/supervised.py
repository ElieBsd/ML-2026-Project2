"""Supervised detectors. ORACLE references only: they are trained on real fakes.

A defender usually has no fakes. These methods show what having them would buy, and
what it costs when the fakes of the training set do not match those met later (another
attack, session, printer or phone). They are never a fake-free method.

- Logistic regression on the 8 per-class deviations binned_z (cdpbpc.scores): scaling
  then L2-penalized logistic regression, classes balanced. Its strength C is chosen by
  cross-validation grouped by template inside the training data (a template is never in
  two folds), on the balanced log-loss.
- lrt_plugin: the likelihood-ratio test of the thesis (Tutt 2025, eqs. 6.18-6.21,
  p. 125-127) with the flip probabilities of fakes Q_b learned on training fakes, with
  the same Jeffreys smoothing as the codebook of originals.

The interface is generic: features Z (one row per image), labels y (1 = fake,
0 = original) and groups (template_id of each image).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from cdpbpc.codebook import estimate
from cdpbpc.patterns import N_PATTERNS
from cdpbpc.scores import lrt

MAX_ITER = 1000


def _check_xyg(Z, y, groups) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if Z.ndim != 2:
        raise ValueError(f"expected features of shape (N, d), got {Z.shape}")
    if y.shape != (len(Z),) or groups.shape != (len(Z),):
        raise ValueError(f"expected {len(Z)} labels and groups, got {y.shape} and {groups.shape}")
    if not np.isfinite(Z).all():
        raise ValueError("features must be finite")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 (original) or 1 (fake)")
    if len(np.unique(y)) < 2:
        raise ValueError("need both originals and fakes")
    return Z, y.astype(int), groups


def _model(C: float, seed: int) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=MAX_ITER, random_state=seed),
    )


def grouped_folds(y, groups, n_splits: int = 5, seed: int = 0) -> list:
    """Cross-validation folds that never split a group (template) between two folds.

    Groups are shuffled with the seed, then balanced over folds (GroupKFold). Every
    training part must hold both classes. Returns [(train_idx, val_idx), ...].
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    if n_splits < 2:
        raise ValueError(f"need at least 2 folds, got {n_splits}")
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"{len(np.unique(groups))} templates cannot fill {n_splits} folds")
    cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(np.zeros((len(y), 1)), y, groups))
    for train, _ in folds:
        if len(np.unique(y[train])) < 2:
            raise ValueError("a training fold lacks originals or fakes")
    return folds


def fit_logistic(
    Z, y, groups, Cs, n_splits: int = 5, seed: int = 0
) -> tuple[Pipeline, float, dict]:
    """ORACLE. Logistic regression of fake (1) against original (0), C chosen by grouped CV.

    Z: (N, d) features (binned_z with 8 classes); y: (N,) 0/1; groups: (N,) template of
    each image. For each C of the grid, the balanced log-loss is averaged over the
    validation folds of grouped_folds; the smallest loss wins, ties going to the smallest
    C (strongest penalty). The model is then refit on all the data with that C.
    Returns (model, C, {C: mean validation log-loss}).
    """
    Z, y, groups = _check_xyg(Z, y, groups)
    Cs = sorted(float(c) for c in Cs)
    if not Cs or min(Cs) <= 0:
        raise ValueError(f"expected a non-empty grid of C > 0, got {Cs}")
    folds = grouped_folds(y, groups, n_splits, seed)
    cv = {}
    for C in Cs:
        losses = []
        for train, val in folds:
            m = _model(C, seed).fit(Z[train], y[train])
            w = compute_sample_weight("balanced", y[val]) if len(np.unique(y[val])) > 1 else None
            losses.append(
                log_loss(y[val], m.predict_proba(Z[val])[:, 1], sample_weight=w, labels=[0, 1])
            )
        cv[C] = float(np.mean(losses))
    best = select_C(cv)
    return _model(best, seed).fit(Z, y), best, cv


def select_C(cv: dict) -> float:
    """C with the smallest validation loss; on a tie, the smallest C (strongest penalty)."""
    if not cv:
        raise ValueError("empty grid")
    return min(cv, key=lambda c: (cv[c], c))


def logistic_score(model: Pipeline, Z) -> np.ndarray:
    """ORACLE score: log-odds of fake given by a fitted model; large = suspect. (N,)."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError(f"expected features of shape (N, d), got {Z.shape}")
    return model.decision_function(Z)


def fake_codebook(D, L) -> np.ndarray:
    """ORACLE. Flip probability of fakes per pattern, pooled over training fakes.

    Q_b(omega) = (sum D + 0.5) / (sum L + 1), the Jeffreys estimate used for the codebook
    of originals. D, L: (N, 512) counts of the training fakes. Returns (512,).
    """
    D = np.asarray(D)
    L = np.asarray(L)
    if D.shape != L.shape or D.ndim != 2 or D.shape[1] != N_PATTERNS:
        raise ValueError(f"expected two (N, {N_PATTERNS}) arrays, got {D.shape} and {L.shape}")
    if (D < 0).any() or (D > L).any():
        raise ValueError("expected 0 <= D <= L")
    return estimate(L.sum(axis=0), D.sum(axis=0))


def lrt_plugin(D, L, codebook, q) -> np.ndarray:
    """ORACLE score: log-likelihood ratio with Q_b learned on fakes (fake_codebook).

    sum over patterns of D log(Q/P) + (L - D) log((1 - Q)/(1 - P)), P = codebook of
    originals, both kept in [1e-3, 1 - 1e-3] (cdpbpc.scores.lrt). Large = suspect.
    """
    codebook = np.asarray(codebook, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if codebook.shape != (N_PATTERNS,) or q.shape != (N_PATTERNS,):
        raise ValueError(f"expected two ({N_PATTERNS},) codebooks, got {codebook.shape}, {q.shape}")
    return lrt(D, L, codebook, q)
