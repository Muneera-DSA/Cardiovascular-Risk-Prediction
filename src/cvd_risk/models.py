"""Model factories, imbalance strategies and a small Cox model.

Candidates (complexity must earn its place):
  age_sex     logistic regression on a spline of age + sex  (the "how old are you?" baseline)
  pce_refit   logistic regression on the Pooled Cohort Equation inputs, refitted to this outcome
  logistic    logistic regression on all baseline features (age as a spline; median imputation
              with missing-value flags fitted inside each training fold)
  hgb         histogram gradient boosting (native missing-value handling; finds interactions)

Imbalance strategies (10-year CVD death is a ~3-5% outcome): none, class_weight, oversample, smote.
All resampling happens inside fit(), so in cross-validation it only touches training folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from . import config

BINARY = {"female", "race_black", "race_hispanic", "race_other", "bp_treated", "lipid_treated", "smoker", "former_smoker",
          "diabetes", "education_college"}
FEATURE_SETS = {
    "age_sex": ["age", "female"],
    "pce_refit": ["age", "female", "race_black", "tchol", "hdl", "sbp", "bp_treated", "smoker", "diabetes"],
    "logistic": config.FEATURES,
    "hgb": config.FEATURES,
}


def logistic_pipeline(features: list[str], C: float = 1.0, class_weight=None) -> Pipeline:
    other = [f for f in features if f != "age"]
    num_other = [f for f in other if f not in BINARY]
    binary = [f for f in other if f in BINARY]
    parts = [("age", SplineTransformer(n_knots=4, degree=3), ["age"])]
    if num_other:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), num_other))
    if binary:
        parts.append(("bin", SimpleImputer(strategy="most_frequent", add_indicator=True), binary))
    return Pipeline([("pre", ColumnTransformer(parts)), ("lr", LogisticRegression(C=C, max_iter=5000, class_weight=class_weight))])


def hgb(learning_rate: float = 0.05, max_leaf_nodes: int = 15, min_samples_leaf: int = 40, l2_regularization: float = 1.0,
        max_iter: int = 300, class_weight=None, seed: int = config.SEED) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                                          l2_regularization=l2_regularization, max_iter=max_iter, early_stopping=True,
                                          validation_fraction=0.15, n_iter_no_change=30, class_weight=class_weight, random_state=seed)


# ----------------------------------------------------------------------- resampling

def random_oversample(X, y, seed=0):
    rng = np.random.default_rng(seed)
    cls, cnt = np.unique(y, return_counts=True)
    minority = cls[cnt.argmin()]
    extra = rng.choice(np.flatnonzero(y == minority), cnt.max() - cnt.min(), replace=True)
    return X.iloc[np.r_[np.arange(len(X)), extra]].reset_index(drop=True), np.concatenate([y, y[extra]])


def smote_nc(X: pd.DataFrame, y, k: int = 5, seed: int = 0):
    """SMOTE-NC style on a DataFrame: interpolate continuous columns, majority vote of the k neighbours
    for binary ones (plain SMOTE would invent '0.4 of a smoker'). Missing values are imputed with the
    minority-class median only for the neighbour search; synthetic rows inherit the base row's missingness."""
    rng = np.random.default_rng(seed)
    cls, cnt = np.unique(y, return_counts=True)
    minority = cls[cnt.argmin()]
    need = int(cnt.max() - cnt.min())
    Xm = X[y == minority].reset_index(drop=True)
    if need == 0 or len(Xm) <= k:
        return X, y
    filled = Xm.fillna(Xm.median(numeric_only=True)).to_numpy(float)
    sd = filled.std(axis=0)
    sd[sd == 0] = 1
    neigh = NearestNeighbors(n_neighbors=k + 1).fit(filled / sd).kneighbors(filled / sd, return_distance=False)[:, 1:]
    i = rng.integers(0, len(Xm), need)
    j = neigh[i, rng.integers(0, k, need)]
    lam = rng.random((need, 1))
    new = filled[i] + lam * (filled[j] - filled[i])
    cols = list(X.columns)
    for c, name in enumerate(cols):
        if name in BINARY:
            new[:, c] = (filled[neigh[i]][:, :, c].mean(axis=1) >= 0.5).astype(float)
    # A synthetic patient keeps the missing values of the patient it was generated from. Otherwise
    # "has a missing value" would become a spurious signal for the majority class.
    new[Xm.isna().to_numpy()[i]] = np.nan
    Xn = pd.DataFrame(new, columns=cols)
    return pd.concat([X.reset_index(drop=True), Xn], ignore_index=True), np.concatenate([y, np.full(need, minority)])


class Resampled(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None, method: str = "oversample", seed: int = 0):
        self.estimator, self.method, self.seed = estimator, method, seed

    def fit(self, X, y):
        y = np.asarray(y)
        X = X.reset_index(drop=True)
        X, y = (random_oversample(X, y, self.seed) if self.method == "oversample" else smote_nc(X, y, seed=self.seed))
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


def build(name: str, imbalance: str = "none", params: dict | None = None, seed: int = config.SEED):
    params = dict(params or {})
    cw = "balanced" if imbalance == "class_weight" else None
    if name == "hgb":
        base = hgb(class_weight=cw, seed=seed, **params)
    else:
        base = logistic_pipeline(FEATURE_SETS[name], class_weight=cw, **params)
    if imbalance in ("oversample", "smote"):
        return Resampled(base, imbalance, seed)
    return base


# ----------------------------------------------------------------------- Cox model

class CoxPH:
    """Cox proportional hazards by Newton-Raphson (Breslow ties), with a small ridge penalty for stability.
    Used for the cause-specific hazard of CVD death (other deaths censored) on cohorts whose follow-up
    is too short for a complete 10-year binary outcome."""

    def __init__(self, penalty: float = 0.01, max_iter: int = 50):
        self.penalty, self.max_iter = penalty, max_iter

    def fit(self, X: np.ndarray, time: np.ndarray, event: np.ndarray):
        X = np.asarray(X, float)
        self.mean_, self.sd_ = X.mean(axis=0), X.std(axis=0)
        self.sd_[self.sd_ == 0] = 1
        Z = (X - self.mean_) / self.sd_
        order = np.argsort(-time, kind="stable")
        Z, t, e = Z[order], np.asarray(time, float)[order], np.asarray(event, int)[order]
        beta = np.zeros(Z.shape[1])
        for _ in range(self.max_iter):
            eta = Z @ beta
            w = np.exp(eta - eta.max())
            S0 = np.cumsum(w)
            S1 = np.cumsum(w[:, None] * Z, axis=0)
            S2 = np.cumsum(w[:, None, None] * Z[:, :, None] * Z[:, None, :], axis=0)
            # Breslow: risk set at time t_i = everyone with time >= t_i -> last index of the tie block
            last = np.searchsorted(-t, -t, side="right") - 1
            ev = e == 1
            mu = S1[last[ev]] / S0[last[ev], None]
            grad = (Z[ev] - mu).sum(axis=0) - self.penalty * beta
            H = (S2[last[ev]] / S0[last[ev], None, None] - mu[:, :, None] * mu[:, None, :]).sum(axis=0) + self.penalty * np.eye(len(beta))
            step = np.linalg.solve(H, grad)
            beta = beta + step
            if np.max(np.abs(step)) < 1e-7:
                break
        self.coef_ = beta
        self.se_ = np.sqrt(np.diag(np.linalg.inv(H)))
        # Breslow baseline cumulative hazard
        eta = Z @ beta
        w = np.exp(eta)
        ut = np.unique(t[e == 1])
        risk = np.array([w[t >= u].sum() for u in ut])
        d = np.array([((t == u) & (e == 1)).sum() for u in ut])
        self.times_, self.cumhaz_ = ut, np.cumsum(d / risk)
        return self

    def linear_predictor(self, X) -> np.ndarray:
        return ((np.asarray(X, float) - self.mean_) / self.sd_) @ self.coef_

    def risk_at(self, X, t: float) -> np.ndarray:
        H0 = self.cumhaz_[np.searchsorted(self.times_, t, side="right") - 1] if t >= self.times_[0] else 0.0
        return 1 - np.exp(-H0 * np.exp(self.linear_predictor(X)))

    def hazard_ratios(self, names: list[str]) -> list[dict]:
        """Per one standard deviation of each (imputed) predictor."""
        return [{"feature": n, "hr_per_sd": float(np.exp(b)), "ci": (float(np.exp(b - 1.96 * s)), float(np.exp(b + 1.96 * s))),
                 "sd": float(sd)} for n, b, s, sd in zip(names, self.coef_, self.se_, self.sd_)]


def harrell_c(time, event, risk) -> float:
    """Concordance for right-censored data: among comparable pairs (the earlier time is an event),
    the share where the earlier event had the higher predicted risk. Ties in risk count half."""
    t, e, r = np.asarray(time, float), np.asarray(event, int), np.asarray(risk, float)
    order = np.argsort(t, kind="stable")
    t, e, r = t[order], e[order], r[order]
    conc = disc = ties = 0.0
    for i in np.flatnonzero(e == 1):
        later = t > t[i]
        if not later.any():
            continue
        rl = r[later]
        conc += (r[i] > rl).sum()
        disc += (r[i] < rl).sum()
        ties += (r[i] == rl).sum()
    total = conc + disc + ties
    return float((conc + 0.5 * ties) / total) if total else float("nan")


def kaplan_meier(time, event, at: float) -> float:
    """Kaplan-Meier cumulative incidence 1 - S(at) (other causes treated as censoring)."""
    t, e = np.asarray(time, float), np.asarray(event, int)
    s = 1.0
    for u in np.unique(t[(e == 1) & (t <= at)]):
        n = (t >= u).sum()
        d = ((t == u) & (e == 1)).sum()
        s *= 1 - d / n
    return float(1 - s)
