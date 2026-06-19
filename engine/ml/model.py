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
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-9] = 1.0  # guard constant columns
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

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


def make_model(kind: str = "logistic", names: list[str] | None = None, **kw):
    """Factory so the harness/scanner request models by name, not class."""
    if kind == "logistic":
        return LogisticModel(names=names or [], **kw)
    raise ValueError(f"unknown model kind: {kind}")
