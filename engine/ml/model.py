"""
Model interface and a simple first implementation.

A model-agnostic boundary (`SignalModel`) so the learning/validation harness
never depends on a specific algorithm -- a logistic model today, gradient-boosted
trees later, swapped without touching validate.py or the scanner. The first
concrete model is deliberately SIMPLE: standardized features + L2-regularized
logistic regression. With the data volume a realistic lookback produces (dozens
to low-hundreds of sessions x ~17 events), a simple, regularized, interpretable
model is the right call -- a complex model overfits and the walk-forward
correctly rejects it. Start with something the validation can actually trust;
upgrade only if the honest out-of-sample numbers justify it.

No third-party ML dependency: the logistic model is implemented directly in
numpy (gradient descent with L2). This keeps the install light and the behavior
fully inspectable, which matters more here than squeezing the last bit of fit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class SignalModel(Protocol):
    """The boundary the harness depends on. Any model implements this."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> SignalModel: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    @property
    def feature_names(self) -> list[str]: ...


@dataclass
class StandardScaler:
    """Causal-safe standardizer: fit on TRAIN only, apply to test."""

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> StandardScaler:
        # NaN-aware. Event feature vectors are HETEROGENEOUS: a vwap event has no
        # leg_* features, a leg event has no level_* features, so the stacked
        # frame is full of NaN by construction. Standardize on the present values
        # (nanmean/nanstd) and impute missing -> the column mean (neutral). An
        # all-NaN or constant column collapses to mean 0 / std 1. Without this the
        # logistic trains on NaN and predict_proba returns all-NaN -> the harness
        # silently takes zero signals and reports a garbage AUC. (Upgrade path:
        # per-event-type models or explicit missing-indicators.)
        with warnings.catch_warnings():
            # All-NaN / single-value column slices warn; we handle them below.
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(X, axis=0)
            std = np.nanstd(X, axis=0)
        self.mean_ = np.where(np.isfinite(mean), mean, 0.0)
        self.std_ = np.where(np.isfinite(std) & (std >= 1e-9), std, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mean_) / self.std_
        # Mean-impute any remaining NaN: 0 IS the standardized column mean, so a
        # missing/type-specific feature reads as neutral. No NaN reaches the model.
        return np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


@dataclass
class LogisticModel:
    """
    L2-regularized logistic regression via batch gradient descent (numpy only).

    Standardizes inputs (scaler fit on training data only, so no leakage), then
    fits weights minimizing log-loss + l2 * ||w||^2. predict_proba returns the
    probability of the positive class (a profitable bracket). Interpretable: the
    standardized coefficients show which structural features drive the signal.
    """

    l2: float = 1.0
    lr: float = 0.1
    epochs: int = 500
    names: list[str] = field(default_factory=list)

    scaler: StandardScaler = field(default_factory=StandardScaler)
    w_: np.ndarray | None = None
    b_: float = 0.0

    @property
    def feature_names(self) -> list[str]:
        return self.names

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticModel:
        Xs = self.scaler.fit_transform(X.astype(float))
        n, d = Xs.shape
        w = np.zeros(d)
        b = 0.0
        y = y.astype(float)
        for _ in range(self.epochs):
            z = Xs @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            err = p - y
            grad_w = (Xs.T @ err) / n + self.l2 * w / n
            grad_b = float(np.mean(err))
            w -= self.lr * grad_w
            b -= self.lr * grad_b
        self.w_ = w
        self.b_ = b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X.astype(float))
        z = Xs @ self.w_ + self.b_
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def coefficients(self) -> dict[str, float]:
        """Standardized coefficients keyed by feature name (interpretability)."""
        if self.w_ is None:
            return {}
        names = self.names or [f"x{i}" for i in range(len(self.w_))]
        return {n: float(c) for n, c in zip(names, self.w_)}


@dataclass
class GBTModel:
    """Gradient-boosted trees behind the same SignalModel interface (research).

    Wraps sklearn's HistGradientBoostingClassifier, which handles NaN natively —
    so it consumes the heterogeneous event features with no imputation. It is an
    OPTIONAL model (the `ml` extra); the core install stays numpy-only. The
    forward-test gate judges it exactly like the logistic — a more expressive
    model that overfits in-sample will simply show larger forward decay and not be
    promoted. Defaults are conservative (shallow, regularized) for small data.
    """

    names: list[str] = field(default_factory=list)
    max_depth: int = 3
    learning_rate: float = 0.05
    max_iter: int = 200
    l2_regularization: float = 1.0
    min_samples_leaf: int = 20
    seed: int = 0
    clf_: object = None
    const_: float | None = None

    @property
    def feature_names(self) -> list[str]:
        return self.names

    def fit(self, X: np.ndarray, y: np.ndarray) -> GBTModel:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "GBTModel needs scikit-learn — install the ml extra: `uv sync --extra ml`"
            ) from exc
        y = np.asarray(y).astype(int)
        if np.unique(y).size < 2:  # HistGBT errors on a single class; degrade gracefully
            self.const_ = float(y.mean()) if y.size else 0.5
            self.clf_ = None
            return self
        clf = HistGradientBoostingClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            l2_regularization=self.l2_regularization,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
        )
        clf.fit(np.asarray(X, dtype=float), y)  # NaN handled internally
        self.clf_ = clf
        self.const_ = None
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xa = np.asarray(X, dtype=float)
        if self.clf_ is None:
            return np.full(len(Xa), 0.5 if self.const_ is None else self.const_)
        return self.clf_.predict_proba(Xa)[:, 1]


def make_model(kind: str = "logistic", names: list[str] | None = None, **kw):
    """Factory so the harness/scanner request models by name, not class."""
    if kind == "logistic":
        return LogisticModel(names=names or [], **kw)
    if kind == "gbt":
        return GBTModel(names=names or [], **kw)
    raise ValueError(f"unknown model kind: {kind}")
