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


# Model zoo (the ml extra). A real, model-agnostic edge should be found by MANY
# of these, not one — agreement across diverse inductive biases is strong evidence
# it is not an artifact of a single algorithm. NaN-imputed + standardized so even
# distance/linear models consume the heterogeneous event features.
_SKLEARN_KINDS = (
    "rf",
    "extratrees",
    "gbm",
    "adaboost",
    "lda",
    "qda",
    "gnb",
    "knn",
    "mlp",
    "tree2",
    "tree3",
)


@dataclass
class SklearnModel:
    """Generic SignalModel over a scikit-learn classifier (research model zoo)."""

    names: list[str] = field(default_factory=list)
    kind: str = "rf"
    seed: int = 0
    est_: object = None
    const_: float | None = None
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    @property
    def feature_names(self) -> list[str]:
        return self.names

    def _estimator(self):
        from sklearn.discriminant_analysis import (
            LinearDiscriminantAnalysis,
            QuadraticDiscriminantAnalysis,
        )
        from sklearn.ensemble import (
            AdaBoostClassifier,
            ExtraTreesClassifier,
            GradientBoostingClassifier,
            RandomForestClassifier,
        )
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.tree import DecisionTreeClassifier

        s = self.seed
        builders = {
            "rf": lambda: RandomForestClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=s, n_jobs=-1
            ),
            "extratrees": lambda: ExtraTreesClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=s, n_jobs=-1
            ),
            "gbm": lambda: GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05, random_state=s
            ),
            "adaboost": lambda: AdaBoostClassifier(
                n_estimators=200, learning_rate=0.5, random_state=s
            ),
            "lda": lambda: LinearDiscriminantAnalysis(),
            "qda": lambda: QuadraticDiscriminantAnalysis(reg_param=0.1),
            "gnb": lambda: GaussianNB(),
            "knn": lambda: KNeighborsClassifier(n_neighbors=50),
            "mlp": lambda: MLPClassifier(
                hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=400, random_state=s
            ),
            "tree2": lambda: DecisionTreeClassifier(
                max_depth=2, min_samples_leaf=50, random_state=s
            ),
            "tree3": lambda: DecisionTreeClassifier(
                max_depth=3, min_samples_leaf=50, random_state=s
            ),
        }
        return builders[self.kind]()

    def _prep(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if fit:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                m = np.nanmean(X, axis=0)
                sd = np.nanstd(X, axis=0)
            self.mean_ = np.where(np.isfinite(m), m, 0.0)
            self.std_ = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
        Xi = np.where(np.isfinite(X), X, self.mean_)  # impute NaN with train mean
        return (Xi - self.mean_) / self.std_

    def fit(self, X: np.ndarray, y: np.ndarray) -> SklearnModel:
        try:
            self._estimator  # noqa: B018 - ensure sklearn import path is reachable
        except ImportError as exc:  # pragma: no cover
            raise ImportError("model zoo needs scikit-learn: `uv sync --extra ml`") from exc
        y = np.asarray(y).astype(int)
        Xs = self._prep(X, fit=True)
        if np.unique(y).size < 2:
            self.const_ = float(y.mean()) if y.size else 0.5
            self.est_ = None
            return self
        est = self._estimator()
        est.fit(Xs, y)
        self.est_ = est
        self.const_ = None
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self._prep(X, fit=False)
        if self.est_ is None:
            return np.full(len(Xs), 0.5 if self.const_ is None else self.const_)
        return self.est_.predict_proba(Xs)[:, 1]


def make_model(kind: str = "logistic", names: list[str] | None = None, **kw):
    """Factory so the harness/scanner request models by name, not class."""
    if kind == "logistic":
        return LogisticModel(names=names or [], **kw)
    if kind == "gbt":
        return GBTModel(names=names or [], **kw)
    if kind in _SKLEARN_KINDS:
        return SklearnModel(names=names or [], kind=kind, **kw)
    raise ValueError(f"unknown model kind: {kind}")
