import json

import numpy as np
import pandas as pd
import synthetic
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

from cvd_risk import config, experiments, uci
from cvd_risk.models import CoxPH, build, harrell_c, kaplan_meier, smote_nc


def _surv(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    T = rng.exponential(1 / (0.05 * np.exp(X @ np.array([0.7, -0.4, 0.0]))))
    C = rng.uniform(0, 30, n)
    return X, np.minimum(T, C), (T <= C).astype(int)


def test_cox_matches_direct_partial_likelihood_optimisation():
    X, t, e = _surv()
    m = CoxPH(penalty=0, max_iter=200).fit(X, t, e)
    o = np.argsort(-t)
    Xs, es = X[o], e[o]
    def nll(b):
        eta = Xs @ b
        return -(eta[es == 1] - np.log(np.cumsum(np.exp(eta)))[es == 1]).sum()
    ref = minimize(nll, np.zeros(3), method="BFGS").x
    np.testing.assert_allclose(m.coef_ / m.sd_, ref, atol=2e-3)


def test_cox_absolute_risk_matches_kaplan_meier_overall():
    X, t, e = _surv()
    m = CoxPH(penalty=0).fit(X, t, e)
    assert abs(m.risk_at(X, 10).mean() - kaplan_meier(t, e, 10)) < 0.02


def test_c_index_equals_auc_without_censoring_or_ties():
    rng = np.random.default_rng(0)
    r = rng.random(500)
    y = (rng.random(500) < r).astype(int)
    t = np.where(y == 1, 1.0, 2.0) + rng.random(500) * 0.01 * (y == 0)
    assert abs(harrell_c(t, y, r) - roc_auc_score(y, r)) < 1e-9


def test_smote_nc_keeps_binary_features_binary():
    c = synthetic.cohort(800)
    X, y = c[config.FEATURES], c.cvd_death_10y.to_numpy()
    Xr, yr = smote_nc(X, y)
    assert yr.mean() == 0.5
    for b in ("smoker", "diabetes", "female", "bp_treated"):
        assert set(np.unique(Xr[b].dropna())) <= {0.0, 1.0}
    assert abs(Xr.iloc[len(X):].hba1c.isna().mean() - X[y == 1].hba1c.isna().mean()) < 0.1   # missingness inherited


def test_models_fit_with_missing_values():
    c = synthetic.cohort(1500)
    for name in ("age_sex", "pce_refit", "logistic", "hgb"):
        m = build(name).fit(c[config.FEATURES if name in ("logistic", "hgb") else
                               {"age_sex": ["age", "female"], "pce_refit": ["age", "female", "race_black", "tchol", "hdl", "sbp", "bp_treated", "smoker", "diabetes"]}[name]],
                              c.cvd_death_10y)
        assert hasattr(m, "predict_proba")


def test_experiment_stages_run_and_respect_splits(tmp_path, monkeypatch):
    c = synthetic.cohort(5000)
    path = tmp_path / "cohort.csv.gz"
    c.to_csv(path, index=False)
    monkeypatch.setattr(config, "COHORT_CSV", path)
    monkeypatch.setattr(config, "HARMONISED_CSV", tmp_path / "absent.csv")
    monkeypatch.setattr(config, "CV_REPEATS", 1)
    monkeypatch.setattr(experiments, "benchmark_models", lambda seed: {"Age + sex (spline)": build("age_sex"), "Logistic": build("logistic")})
    experiments.run(["quality", "eda", "benchmark", "select", "temporal", "survival", "leakage", "subgroups", "explain"], out=tmp_path, n_boot=20)
    t = json.loads((tmp_path / "temporal.json").read_text())
    dev = c[(c.split == "development") & (c.complete_10y == 1)]
    test = c[(c.split == "temporal_test") & (c.complete_10y == 1)]
    assert t["n_dev"] == len(dev) and t["n_test"] == len(test)
    assert t["models"]["pce"]["auc"] > 0.6                      # the synthetic outcome follows PCE risk
    leak = json.loads((tmp_path / "leakage.json").read_text())
    key = "+ death-certificate diabetes & hypertension flags (recorded after death)"
    assert leak[key]["auc"] > leak["correct (baseline variables, temporal split)"]["auc"]
    ex = json.loads((tmp_path / "explain.json").read_text())
    assert ex["logistic"]["additivity_max_error"] < 1e-8


def test_uci_recoding_and_site_matching():
    raw = pd.DataFrame({"site": ["cleveland", "switzerland"], "age": [63, 50], "sex": [1, 0], "cp": [1, 4], "trestbps": [145, 120],
                        "chol": [233, 0], "fbs": [1, 0], "restecg": [2, 0], "thalach": [150, 130], "exang": [0, 1], "oldpeak": [2.3, 1.0],
                        "slope": [3, 2], "ca": [0, np.nan], "thal": [6, np.nan], "num": [0, 2]})
    k = uci.to_kaggle_coding(raw)
    assert list(k.ChestPainType) == ["TA", "ASY"] and list(k.ST_Slope) == ["Down", "Flat"] and list(k.HeartDisease) == [0, 1]
    m = uci.match_sites(k.drop(columns="site"), k)
    assert list(m.site) == ["cleveland", "switzerland"]


def test_extra_models_merge_into_benchmark_on_the_same_folds(tmp_path, monkeypatch):
    """XGBoost / LightGBM run elsewhere (Colab); their fold AUCs join the table and its tests."""
    c = synthetic.cohort(3000)
    path = tmp_path / "cohort.csv.gz"
    c.to_csv(path, index=False)
    monkeypatch.setattr(config, "COHORT_CSV", path)
    monkeypatch.setattr(config, "HARMONISED_CSV", tmp_path / "absent.csv")
    monkeypatch.setattr(config, "CV_REPEATS", 1)
    monkeypatch.setattr(experiments, "benchmark_models", lambda seed: {"Age + sex (spline)": build("age_sex")})
    extra = {"versions": {"xgboost": "9.9", "lightgbm": "9.9"},
             "rows": [{"model": "XGBoost", "cv_auc": 0.7, "cv_auc_sd": 0.01, "cv_pr_auc": 0.1, "cv_brier": 0.03, "seconds": 1, "fold_auc": [0.70] * 5}]}
    (tmp_path / "benchmark_extra.json").write_text(json.dumps(extra))
    experiments.run(["benchmark"], out=tmp_path, n_boot=10)
    tab = {r["model"]: r for r in json.loads((tmp_path / "benchmark.json").read_text())["table"]}
    assert "XGBoost (9.9)" in tab and "p_vs_best_holm" in tab["XGBoost (9.9)"]
    assert "fold_auc" not in tab["XGBoost (9.9)"]
