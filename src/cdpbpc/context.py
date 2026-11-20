"""Models of the flip probability with a context larger than 3x3.

A model gives each symbol of a template a flip probability q from the neighbourhood of
that symbol in the template. The BPC codebook is the 3x3 table (Tutt 2025, Sec. 4.3).
The larger models are:

    back-off tables  counts per neighbourhood with 13, 21 and 25 bits (the full 5x5),
                     each level shrunk towards the level below:
                     p_b(c) = (k_c + m_b p_parent(c)) / (n_c + m_b)
    ring logistic    logistic regression on logit(p_9), the 16 bits of the 5x5 ring,
                     those bits times the centre colour, and the centre colour
    boosted trees    HistGradientBoostingClassifier on the bits of the 5x5 or 7x7 window

They are judged by their log-loss on reads of templates never used for training.

Data come as flip counts: K[i, r, c] = number of reads of template i in which symbol
(r, c) flipped, out of n reads (diagnostics.flip_counts). Positions are the symbols at
distance >= margin from the border; every comparison uses the same positions.

Bits of a neighbourhood are read in the order of LEVELS[25], the first offset being the
most significant bit. The index of a level is then the 25-bit index shifted right, so a
model only ever needs the 25-bit index of each position.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from cdpbpc.scores import CLASS_EDGES_8, reliability_class

OFF9 = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
ARMS = [(-2, 0), (0, -2), (0, 2), (2, 0)]
RING8 = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
CORNERS = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
LEVELS = {9: OFF9, 13: OFF9 + ARMS, 21: OFF9 + ARMS + RING8, 25: OFF9 + ARMS + RING8 + CORNERS}
PARENT = {13: 9, 21: 13, 25: 21}
TOP_BITS = 25
CENTRE_BIT = TOP_BITS - 1 - OFF9.index((0, 0))  # bit of the symbol itself in the 25-bit index

BACKOFF_MASSES = (2, 5, 10, 20, 50, 100, 200, 500)
CEILING_A = (0.5, 1, 2, 4, 8, 16)
Q_CLIP = 1e-5  # q is kept in [1e-5, 1 - 1e-5] in the measures


def _reach(bits: int) -> int:
    return max(max(abs(a), abs(b)) for a, b in LEVELS[bits])


def crop(x: np.ndarray, margin: int) -> np.ndarray:
    """Positions at distance >= margin from the border: (..., L, L) -> (..., L-2m, L-2m)."""
    x = np.asarray(x)
    side = x.shape[-1]
    return x[..., margin : side - margin, margin : side - margin]


def context_index(t: np.ndarray, bits: int = TOP_BITS, margin: int = 2) -> np.ndarray:
    """Index of the neighbourhood LEVELS[bits] of each position.

    t: (..., L, L) binary, 1 = white. Returns int32 (..., L-2m, L-2m). With bits=9 and
    margin=1 this is patterns.pattern_index.
    """
    if bits not in LEVELS:
        raise ValueError(f"bits must be one of {tuple(LEVELS)}, got {bits}")
    if margin < _reach(bits):
        raise ValueError(f"the {bits}-bit neighbourhood needs margin >= {_reach(bits)}")
    t = np.asarray(t)
    side = t.shape[-1]
    h = side - 2 * margin
    if h < 1 or t.shape[-2] != side:
        raise ValueError(f"expected square templates larger than 2 * margin, got {t.shape}")
    idx = np.zeros(t.shape[:-2] + (h, h), dtype=np.int32)
    for k, (a, b) in enumerate(LEVELS[bits]):
        sl = t[..., margin + a : margin + a + h, margin + b : margin + b + h].astype(np.int32)
        idx |= sl << (bits - 1 - k)
    return idx


def parent_index(idx: np.ndarray, child_bits: int, parent_bits: int) -> np.ndarray:
    """Index of the smaller neighbourhood: the child index shifted right."""
    return np.asarray(idx) >> (child_bits - parent_bits)


def window_bits(t: np.ndarray, radius: int, margin: int) -> np.ndarray:
    """Bits of the (2r+1) x (2r+1) window of each position, row by row.

    t: (N, L, L). Returns (N * (L-2m)^2, (2r+1)^2) uint8; rows follow crop(t, margin).
    """
    if margin < radius:
        raise ValueError("the window does not fit: need margin >= radius")
    t = np.asarray(t)
    side = t.shape[-1]
    h = side - 2 * margin
    cols = [
        t[..., margin + a : margin + a + h, margin + b : margin + b + h].reshape(-1)
        for a in range(-radius, radius + 1)
        for b in range(-radius, radius + 1)
    ]
    return np.stack(cols, axis=1).astype(np.uint8)


# ------------------------------------------------------------------------------- tables


@dataclass(frozen=True)
class TableCounts:
    """Flips k and reads n per neighbourhood, for each level up to `top` bits.

    k[b], n[b]: (2^b,) float64. The counts of a level are the sums of its children.
    """

    k: dict
    n: dict
    top: int


def table_counts(
    t: np.ndarray, K: np.ndarray, n: int, margin: int = 2, bits: int = TOP_BITS, chunk: int = 64
) -> TableCounts:
    """Counts of the back-off tables, from flip counts K over n reads.

    t, K: (N, L, L); only positions at distance >= margin count.
    """
    t = np.asarray(t)
    K = np.asarray(K)
    if t.shape != K.shape or t.ndim != 3:
        raise ValueError(f"expected two (N, L, L) arrays, got {t.shape} and {K.shape}")
    size = 1 << bits
    k_top = np.zeros(size)
    n_top = np.zeros(size)
    for s in range(0, len(t), chunk):
        idx = context_index(t[s : s + chunk], bits, margin).ravel()
        k_top += np.bincount(idx, weights=crop(K[s : s + chunk], margin).ravel(), minlength=size)
        n_top += np.bincount(idx, minlength=size) * float(n)
    k, nn = {bits: k_top}, {bits: n_top}
    for b in LEVELS:
        if b < bits:
            k[b] = k_top.reshape(1 << b, -1).sum(1)
            nn[b] = n_top.reshape(1 << b, -1).sum(1)
    return TableCounts(k=k, n=nn, top=bits)


def backoff_predict(
    counts: TableCounts, masses: dict, idx: np.ndarray, bits: int = TOP_BITS, top: int | None = None
) -> np.ndarray:
    """Flip probability of the back-off table with `bits` bits at each position.

    idx: indices of the positions with `top` bits (default counts.top). The 9-bit level
    is (k + 0.5) / (n + 1); each larger level b is (k + m_b p_parent) / (n + m_b), and
    m_b = inf gives the parent itself. masses: {13: m13, 21: m21, 25: m25}, only the
    levels up to `bits` are read. Returns float64 of the shape of idx.
    """
    top = counts.top if top is None else top
    idx = np.asarray(idx)
    c = idx >> (top - 9)
    p = (counts.k[9][c] + 0.5) / (counts.n[9][c] + 1.0)
    for b in (13, 21, 25):
        if b > bits:
            break
        m = float(masses[b])
        if np.isinf(m):
            continue
        c = idx >> (top - b)
        p = (counts.k[b][c] + m * p) / (counts.n[b][c] + m)
    return p


def select_masses(folds: list, grid=BACKOFF_MASSES, margin: int = 3) -> tuple[dict, dict]:
    """Back-off masses chosen by log-loss on validation templates, level by level.

    folds: list of (counts, t_val, K_val, n_val): tables counted on some templates and
    validation reads of others; the loss is summed over folds. 13 bits is chosen first,
    then 21 with m13 fixed, then 25. Returns ({13: m, 21: m, 25: m}, {b: {m: loss}}).
    """
    prepared = [
        (counts, context_index(tv, TOP_BITS, margin).ravel(), crop(Kv, margin).ravel(), nv)
        for counts, tv, Kv, nv in folds
    ]
    chosen, losses = {}, {}
    for b in (13, 21, 25):
        losses[b] = {}
        for m in grid:
            trial = {**chosen, b: m}
            total, reads = 0.0, 0.0
            for counts, idx, K, n in prepared:
                q = backoff_predict(counts, trial, idx, bits=b)
                total += _loss_sum(K, n, q)
                reads += K.size * n
            losses[b][m] = total / reads
        chosen[b] = min(losses[b], key=losses[b].get)
    return chosen, losses


# ----------------------------------------------------------------------- learned models


def binomial_rows(F: np.ndarray, K: np.ndarray, n: int):
    """Two weighted rows per position: (features, 1, weight K) and (features, 0, n - K).

    Rows of weight 0 are left out. F: (P, d); K: (P,). Returns X, y, w.
    """
    K = np.asarray(K).ravel()
    flip, clean = K > 0, K < n
    X = np.concatenate([F[flip], F[clean]])
    y = np.concatenate([np.ones(flip.sum(), np.int8), np.zeros(clean.sum(), np.int8)])
    w = np.concatenate([K[flip], n - K[clean]]).astype(np.float64)
    return X, y, w


def ring_features(idx: np.ndarray, p9: np.ndarray) -> np.ndarray:
    """Variables of the ring logistic for positions with 25-bit indices idx.

    Columns: logit(p_9); the 16 ring bits (order of LEVELS[25][9:]); the same bits times
    the centre colour; the centre colour. Returns (P, 34) float32.
    """
    idx = np.asarray(idx).ravel()
    p9 = np.clip(np.asarray(p9, dtype=np.float64).ravel(), Q_CLIP, 1 - Q_CLIP)
    ring = ((idx[:, None] >> np.arange(15, -1, -1)) & 1).astype(np.float32)
    centre = ((idx >> CENTRE_BIT) & 1).astype(np.float32)[:, None]
    logit = np.log(p9 / (1 - p9)).astype(np.float32)[:, None]
    return np.concatenate([logit, ring, ring * centre, centre], axis=1)


def fit_ring_logistic(F: np.ndarray, K: np.ndarray, n: int, C: float) -> LogisticRegression:
    """L2 logistic regression of the flips on ring_features, inverse penalty C."""
    X, y, w = binomial_rows(F, K, n)
    return LogisticRegression(C=C, max_iter=1000).fit(X, y, sample_weight=w)


def fit_trees(
    F: np.ndarray,
    K: np.ndarray,
    n: int,
    max_iter: int = 300,
    learning_rate: float = 0.1,
    max_leaf_nodes: int = 63,
    min_samples_leaf: int = 200,
    seed: int = 0,
) -> HistGradientBoostingClassifier:
    """Boosted trees on window bits F (window_bits), flips K over n reads."""
    X, y, w = binomial_rows(F, K, n)
    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        early_stopping=False,
        random_state=seed,
    )
    return model.fit(X, y, sample_weight=w)


def trees_predict(
    model: HistGradientBoostingClassifier,
    F: np.ndarray,
    n_iter: int | None = None,
    chunk: int = 1 << 20,
) -> np.ndarray:
    """Flip probability after the first n_iter trees (all of them if None). (P,) float64."""
    out = []
    for s in range(0, len(F), chunk):
        Fc = F[s : s + chunk]
        if n_iter is None or n_iter >= model.n_iter_:
            out.append(model.predict_proba(Fc)[:, 1])
            continue
        for i, proba in enumerate(model.staged_predict_proba(Fc), start=1):
            if i == n_iter:
                out.append(proba[:, 1])
                break
    return np.concatenate(out)


def trees_stage_losses(
    model: HistGradientBoostingClassifier,
    F: np.ndarray,
    K: np.ndarray,
    n: int,
    chunk: int = 1 << 20,
) -> np.ndarray:
    """Log-loss on (F, K) after each number of trees 1..n_iter_. (n_iter_,) float64."""
    K = np.asarray(K).ravel()
    totals = np.zeros(model.n_iter_)
    for s in range(0, len(F), chunk):
        for i, proba in enumerate(model.staged_predict_proba(F[s : s + chunk])):
            totals[i] += _loss_sum(K[s : s + chunk], n, proba[:, 1])
    return totals / (K.size * n)


# ----------------------------------------------------------------------------- measures


def _loss_sum(K: np.ndarray, n: int, q: np.ndarray) -> float:
    q = np.clip(np.asarray(q, dtype=np.float64), Q_CLIP, 1 - Q_CLIP)
    K = np.asarray(K, dtype=np.float64)
    return float(-(K * np.log(q) + (n - K) * np.log1p(-q)).sum())


def log_loss(K: np.ndarray, n: int, q: np.ndarray) -> float:
    """Mean negative log-likelihood per symbol read, in nats.

    -mean[K log q + (n - K) log(1 - q)] / n over positions; q kept in [1e-5, 1 - 1e-5].
    """
    return _loss_sum(K, n, q) / (np.size(K) * n)


def per_template_log_loss(K: np.ndarray, n: int, q: np.ndarray) -> np.ndarray:
    """log_loss of each template. K, q: (N, ...). Returns (N,)."""
    q = np.clip(np.asarray(q, dtype=np.float64), Q_CLIP, 1 - Q_CLIP)
    K = np.asarray(K, dtype=np.float64)
    ll = -(K * np.log(q) + (n - K) * np.log1p(-q))
    return ll.reshape(len(K), -1).mean(1) / n


def predicted_disagreement(q: np.ndarray) -> float:
    """D_model = mean of 2 q (1 - q): how often two reads of a position disagree."""
    q = np.asarray(q, dtype=np.float64)
    return float((2 * q * (1 - q)).mean())


def observed_disagreement(K: np.ndarray, n: int) -> float:
    """D_obs = mean of 2 K (n - K) / (n (n - 1)) over positions read n >= 2 times."""
    if n < 2:
        raise ValueError("disagreement between reads needs n >= 2")
    K = np.asarray(K, dtype=np.float64)
    return float((2 * K * (n - K) / (n * (n - 1))).mean())


def per_template_disagreement(K: np.ndarray, n: int) -> np.ndarray:
    """observed_disagreement of each template. K: (N, ...) counts of any integer type."""
    if n < 2:
        raise ValueError("disagreement between reads needs n >= 2")
    K = np.asarray(K, dtype=np.float64)
    return (2 * K * (n - K) / (n * (n - 1))).reshape(len(K), -1).mean(1)


def per_template_predicted_disagreement(q: np.ndarray) -> np.ndarray:
    """predicted_disagreement of each template. q: (N, ...). Returns (N,)."""
    q = np.asarray(q, dtype=np.float64)
    return (2 * q * (1 - q)).reshape(len(q), -1).mean(1)


def remaining_rho(K: np.ndarray, n: int, q: np.ndarray) -> float:
    """rho_left = 1 - D_obs / D_model: the reproducible share the model still misses."""
    return 1.0 - observed_disagreement(K, n) / predicted_disagreement(q)


def ceiling_log_loss(K: np.ndarray, n: int, q: np.ndarray, a: float) -> float:
    """Log-loss of the per-position ceiling: each read predicted from the n - 1 others.

    Read r is given (K - delta_r + a q) / (n - 1 + a): a flipped read gets
    (K - 1 + a q) / (n - 1 + a) and a clean one (K + a q) / (n - 1 + a). It uses the
    other prints of the same template, so it is a ceiling, not a model.
    """
    K = np.asarray(K, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    q1 = np.clip((K - 1 + a * q) / (n - 1 + a), Q_CLIP, 1 - Q_CLIP)
    q0 = np.clip((K + a * q) / (n - 1 + a), Q_CLIP, 1 - Q_CLIP)
    return float(-(K * np.log(q1) + (n - K) * np.log1p(-q0)).mean() / n)


def calibration(K: np.ndarray, n: int, q: np.ndarray, edges=CLASS_EDGES_8) -> list[dict]:
    """Per bin of q: share of positions, mean predicted q, observed flip rate K / n."""
    q = np.asarray(q, dtype=np.float64).ravel()
    K = np.asarray(K, dtype=np.float64).ravel()
    cls = reliability_class(q, edges)
    out = []
    for b in range(len(edges) - 1):
        m = cls == b
        cnt = int(m.sum())
        out.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "share": cnt / q.size,
                "mean_q": float(q[m].mean()) if cnt else float("nan"),
                "observed": float(K[m].sum() / (cnt * n)) if cnt else float("nan"),
            }
        )
    return out
