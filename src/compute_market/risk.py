"""Discrete probability risk operators. Loss is positive when wealth decreases."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray


def distribution(values: ArrayLike, probabilities: ArrayLike) -> tuple[NDArray, NDArray]:
    x = np.asarray(values, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if x.ndim != 1 or p.shape != x.shape or x.size == 0:
        raise ValueError("values and probabilities must be equal, nonempty vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("finite values and nonnegative probabilities required")
    if not np.isclose(p.sum(), 1.0, atol=1e-10, rtol=0):
        raise ValueError("probabilities must sum to one; implicit normalization is forbidden")
    return x, p


def expected_shortfall(losses: ArrayLike, probabilities: ArrayLike, alpha: float = .95) -> float:
    """Upper-tail ES, including fractional mass at a discrete quantile atom."""
    x, p = distribution(losses, probabilities)
    if not 0 <= alpha < 1:
        raise ValueError("alpha must lie in [0, 1)")
    remaining = 1.0 - alpha
    tail_sum = 0.0
    for i in np.argsort(-x, kind="stable"):
        mass = min(float(p[i]), remaining)
        tail_sum += mass * float(x[i])
        remaining -= mass
        if remaining <= 1e-14:
            break
    return tail_sum / (1.0 - alpha)


def risk(losses: ArrayLike, probabilities: ArrayLike, alpha: float = .95,
         weight: float = .25) -> float:
    x, p = distribution(losses, probabilities)
    if not 0 <= weight <= 1 or not 0 <= alpha < 1:
        raise ValueError("weight in [0,1] and alpha in [0,1) required")
    return float((1-weight) * (p @ x) + weight * expected_shortfall(x, p, alpha))


def certainty_equivalent(values: ArrayLike, probabilities: ArrayLike,
                        alpha: float = .95, weight: float = .25) -> float:
    return -risk(-np.asarray(values, float), probabilities, alpha, weight)


def nested_value(tree, leaf_values: ArrayLike, alpha: float, weight: float,
                 discounts: ArrayLike | None = None) -> NDArray:
    """Backward value for a fixed policy; input is a full node vector, leaves used."""
    value = np.array(leaf_values, float, copy=True)
    if value.shape != (tree.size,):
        raise ValueError("leaf_values must be a full node vector")
    d = np.ones(tree.size) if discounts is None else np.asarray(discounts, float)
    if d.shape != value.shape or np.any(d <= 0) or not np.isfinite(d).all():
        raise ValueError("positive finite edge discounts required")
    for n in reversed(range(tree.size)):
        ch = tree.children[n]
        if ch:
            value[n] = certainty_equivalent(
                [d[m] * value[m] for m in ch],
                [tree.probability[m] for m in ch], alpha, weight)
    return value
