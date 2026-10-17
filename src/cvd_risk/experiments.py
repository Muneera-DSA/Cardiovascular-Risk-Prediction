"""Part 2 experiments on the NHANES cohort. Every stage writes results/<stage>.json.

    python -m cvd_risk.experiments                 # all stages
    python -m cvd_risk.experiments --stages temporal explain

Stages
  quality     validity ranges, missingness, included-vs-excluded comparison (selection bias), drift by cycle
  eda         baseline table, event rates by age / sex / risk category, survey-weighted prevalence
  benchmark   broad model screen by repeated stratified CV on the development cohort (LazyPredict-style, done properly)
  imbalance   none / class weights / oversampling / SMOTE-NC inside the folds
  select      tuning of the logistic and gradient-boosting candidates by CV on development data only
  temporal    fit on 1999-2004, test once on 2005-2008: discrimination, calibration, decision curves, vs the PCE
  survival    cause-specific Cox model; C-index and 5-year calibration on 2009-2018 (incomplete follow-up)
  leakage     what death-certificate flags, prevalent heart disease and random splitting would do to the numbers
  subgroups   sex, race / ethnicity, age band, diabetes
  explain     SHAP for the chosen model (exact for logistic; TreeExplainer for gradient boosting when shap is installed)
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from . import config, pce, stats
from .models import FEATURE_SETS, CoxPH, build, harrell_c, kaplan_meier

STAGES = ["quality", "eda", "benchmark", "imbalance", "select", "temporal", "survival", "leakage", "subgroups", "explain"]
Y = "cvd_death_10y"
RANGES = {"sbp": (70, 270), "dbp": (20, 150), "tchol": (60, 600), "hdl": (10, 200), "hba1c": (3, 20), "bmi": (12, 90),
          "waist": (40, 200), "egfr": (3, 200), "pir": (0, 5), "age": (40, 79)}


def _json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else (None if o is pd.NA else str(o))))


def load(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or config.COHORT_CSV)


def binary_cohort(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Development / temporal-test rows with a complete 10-year outcome."""
    d = df[(df.split == split) & (df.complete_10y == 1)]
    return d.reset_index(drop=True)


def metrics(y, p, n_boot: int = 0, seed: int = 0) -> dict:
    y, p = np.asarray(y), np.asarray(p, float)
    auc = stats.delong_auc(y, p)
    cal = stats.calibration_intercept_slope(y, p)
    out = {"n": int(len(y)), "events": int(y.sum()), "auc": auc["value"], "auc_ci": auc["ci"], "pr_auc": float(average_precision_score(y, p)),
           "brier": stats.brier(y, p), "ece": stats.expected_calibration_error(y, p, 10), "cal_intercept": cal["intercept"], "cal_slope": cal["slope"],
           "observed": cal["observed_rate"], "mean_predicted": cal["mean_predicted"], "o_e": cal["observed_rate"] / max(cal["mean_predicted"], 1e-9)}
    if n_boot:
        b = stats.bootstrap_metric(lambda i: average_precision_score(y[i], p[i]) if y[i].sum() else np.nan, len(y), n_boot, seed)
        out["pr_auc_ci"] = b["ci"]
    return out


def smd(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


# ------------------------------------------------------------------------------ quality / eda

def stage_quality(df: pd.DataFrame, adults: pd.DataFrame | None, out: Path) -> dict:
    res = {"validity": {}, "missing_share": {}, "drift_smd_vs_development": {}}
    for v, (lo, hi) in RANGES.items():
        s = df[v].dropna()
        res["validity"][v] = {"n": int(len(s)), "share_in_range": float(s.between(lo, hi).mean()), "min": float(s.min()), "max": float(s.max())}
    for v in config.FEATURES:
        res["missing_share"][v] = float(df[v].isna().mean())
    # uniqueness: one row per person, and person IDs never reused across cycles
    res["uniqueness"] = {"rows": int(len(df)), "duplicate_person_ids": int(df.duplicated(["cycle", "seqn"]).sum()),
                         "ids_shared_across_cycles": int(df.seqn.duplicated().sum()), "duplicate_rows_all_columns": int(df.duplicated().sum())}
    # internal consistency: related fields must agree (share of rows passing)
    both = lambda a, b: df[a].notna() & df[b].notna()   # noqa: E731
    m1, m2 = both("dbp", "sbp"), both("hdl", "tchol")
    res["consistency"] = {
        "diastolic < systolic": float((df.loc[m1, "dbp"] < df.loc[m1, "sbp"]).mean()),
        "HDL < total cholesterol": float((df.loc[m2, "hdl"] < df.loc[m2, "tchol"]).mean()),
        "not both current and former smoker": float(1 - ((df.smoker == 1) & (df.former_smoker == 1)).mean()),
        "derivation: HbA1c >= 6.5% always coded as diabetes": float((df.loc[df.hba1c >= 6.5, "diabetes"] == 1).mean()),
        "CVD death implies death": float((df.loc[df.cvd_death == 1, "died"] == 1).mean()),
        "deaths have a recorded cause": float(df.loc[df.died == 1, "ucod_leading"].notna().mean()),
        "follow-up time not negative": float((df.months >= 0).mean()),
        "deaths occur within the follow-up of survivors (same cycle)": float(
            (df.loc[df.died == 1, "months"] <= df.loc[df.died == 1, "cycle"].map(df[df.died == 0].groupby("cycle").months.max())).mean()),
    }
    # accuracy against independent information in the same record
    diab = df[df.hba1c.notna()]
    res["accuracy_cross_checks"] = {
        "median HbA1c, diabetes vs not": [float(diab.loc[diab.diabetes == 1, "hba1c"].median()), float(diab.loc[diab.diabetes == 0, "hba1c"].median())],
        "median SBP, BP-treated vs not": [float(df.loc[df.bp_treated == 1, "sbp"].median()), float(df.loc[df.bp_treated == 0, "sbp"].median())],
        "BMI-waist correlation": float(df[["bmi", "waist"]].corr().iloc[0, 1]),
    }
    res["class_balance"] = {split: {"n": int(len(g)), "cvd_death_10y_rate": float(g.loc[g.complete_10y == 1, Y].mean())}
                            for split, g in df.groupby("split") if split != "recent"}
    dev = df[df.split == "development"]
    for cyc, g in df.groupby("cycle"):
        res["drift_smd_vs_development"][int(cyc)] = {v: smd(g[v], dev[v]) for v in ["age", "sbp", "tchol", "hdl", "bmi", "hba1c", "smoker", "bp_treated", "lipid_treated", "diabetes"]}
    if adults is not None:
        base = adults[(adults.examined == 1) & adults.age.between(config.AGE_MIN, config.AGE_MAX) & (adults.pregnant == 0)
                      & (adults.linkage_eligible == 1) & (adults.prior_cvd == 0)]
        inc = base.seqn.isin(df.seqn)
        res["excluded_for_missing_core"] = {"n_excluded": int((~inc).sum()), "share": float((~inc).mean()),
                                            "smd_excluded_vs_included": {v: smd(base.loc[~inc, v], base.loc[inc, v])
                                                                         for v in ["age", "female", "race_black", "race_hispanic", "pir", "bmi", "sbp"]},
                                            "cvd_death_10y_excluded": float(base.loc[~inc & (base.complete_10y == 1), Y].mean()),
                                            "cvd_death_10y_included": float(base.loc[inc & (base.complete_10y == 1), Y].mean())}
    _json(res, out / "quality.json")
    return res


def stage_eda(df: pd.DataFrame, out: Path) -> dict:
    res = {}
    num = ["age", "sbp", "dbp", "tchol", "hdl", "hba1c", "bmi", "egfr", "pir"]
    binv = ["female", "race_black", "race_hispanic", "smoker", "former_smoker", "diabetes", "bp_treated", "lipid_treated", "education_college"]
    table = {}
    for split, g in df.groupby("split"):
        col = {"n": int(len(g))}
        for v in num:
            col[v] = f"{g[v].median():.1f} ({g[v].quantile(.25):.1f}-{g[v].quantile(.75):.1f})"
        for v in binv:
            col[v] = f"{g[v].mean() * 100:.1f}%"
        col["10-year CVD death (complete follow-up)"] = f"{g.loc[g.complete_10y == 1, Y].mean() * 100:.2f}%"
        col["median follow-up, years"] = f"{g.months.median() / 12:.1f}"
        table[split] = col
    res["baseline_table"] = table
    b = pd.concat([binary_cohort(df, "development"), binary_cohort(df, "temporal_test")], ignore_index=True)
    res["n_binary"] = int(len(b))
    res["event_rate"] = float(b[Y].mean())
    res["noncvd_death_rate"] = float(b.noncvd_death_10y.mean())
    res["by_age_sex"] = (b.assign(age_band=pd.cut(b.age, [39, 49, 59, 69, 79]).astype(str), sex=np.where(b.female == 1, "female", "male"))
                         .groupby(["age_band", "sex"])[Y].agg(["mean", "size", "sum"]).reset_index().to_dict(orient="records"))
    b = b.assign(pce=pce.risk_10y(b))
    cats = pd.cut(b.pce, [0, 0.05, 0.075, 0.20, 1], labels=["<5%", "5-7.5%", "7.5-20%", ">=20%"])
    by_cat = b.groupby(cats, observed=True).agg(n=(Y, "size"), cvd_death_rate=(Y, "mean"), mean_pce=("pce", "mean")).reset_index()
    res["by_pce_category"] = by_cat.astype({"pce": str}).to_dict(orient="records")
    w = b.weight_mec.fillna(0)
    res["weighted_vs_unweighted"] = {v: {"unweighted": float(b[v].mean()), "survey_weighted": float(np.average(b[v].fillna(b[v].mean()), weights=w))}
                                     for v in ["smoker", "diabetes", "bp_treated", "female", "race_black", Y]}
    from scipy import stats as sps
    tests = {}
    for v in num:
        a0, a1 = b.loc[b[Y] == 0, v].dropna(), b.loc[b[Y] == 1, v].dropna()
        u = sps.mannwhitneyu(a1, a0)
        tests[v] = {"test": "Mann-Whitney U", "p": float(u.pvalue), "effect_P(event>non-event)": float(u.statistic / (len(a0) * len(a1)))}
    for v in binv:
        ct = pd.crosstab(b[v], b[Y])
        if ct.shape == (2, 2):
            chi2, p, *_ = sps.chi2_contingency(ct)
            r1, r0 = b.loc[b[v] == 1, Y].mean(), b.loc[b[v] == 0, Y].mean()
            tests[v] = {"test": "chi-square", "p": float(p), "risk_ratio": float(r1 / r0) if r0 else None}
    adj = stats.holm({k: t["p"] for k, t in tests.items()})
    for k in tests:
        tests[k]["p_holm"] = adj[k]
    res["univariable_tests"] = tests
    _json(res, out / "eda.json")
    return res


# ------------------------------------------------------------------------------ CV helpers

def cv_scores(est, X, y, folds, seed) -> dict:
    aucs, prs, briers, preds = [], [], [], np.zeros(len(y))
    for k, (tr, te) in enumerate(folds):
        m = clone(est)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        prs.append(average_precision_score(y[te], p))
        briers.append(stats.brier(y[te], p))
        if k < config.CV_FOLDS:
            preds[te] = p
    return {"auc": np.array(aucs), "pr_auc": np.array(prs), "brier": np.array(briers), "oof_first_repeat": preds}


def corrected_t(diffs: np.ndarray, n_train: int, n_test: int) -> float:
    from scipy import stats as sps
    k, var = len(diffs), np.var(diffs, ddof=1)
    if var == 0:
        return 1.0
    return float(2 * sps.t.sf(abs(diffs.mean() / np.sqrt((1 / k + n_test / n_train) * var)), df=k - 1))


def benchmark_models(seed: int) -> dict:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    imp = lambda: SimpleImputer(strategy="median", add_indicator=True)   # noqa: E731
    m = {
        "Prior only (no information)": DummyClassifier(strategy="prior"),
        "Age + sex (spline)": build("age_sex"),
        "PCE variables, refitted": build("pce_refit"),
        "Logistic regression (all features, age spline)": build("logistic"),
        "Logistic regression (L1, C=0.05)": make_pipeline(imp(), StandardScaler(), LogisticRegression(penalty="l1", C=0.05, solver="liblinear")),
        "LDA": make_pipeline(imp(), StandardScaler(), LinearDiscriminantAnalysis()),
        "Gaussian naive Bayes": make_pipeline(imp(), GaussianNB()),
        "k-nearest neighbours (k=100)": make_pipeline(imp(), StandardScaler(), KNeighborsClassifier(n_neighbors=100)),
        "SVM (RBF, Platt)": make_pipeline(imp(), StandardScaler(), SVC(probability=True, random_state=seed)),
        "Decision tree (depth 4)": make_pipeline(imp(), DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=seed)),
        "Random forest": make_pipeline(imp(), RandomForestClassifier(n_estimators=400, min_samples_leaf=20, n_jobs=-1, random_state=seed)),
        "Extra trees": make_pipeline(imp(), ExtraTreesClassifier(n_estimators=400, min_samples_leaf=20, n_jobs=-1, random_state=seed)),
        "AdaBoost": make_pipeline(imp(), AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=seed)),
        "Histogram gradient boosting": build("hgb"),
        "Neural network (MLP 16)": make_pipeline(imp(), StandardScaler(), MLPClassifier((16,), alpha=1e-2, early_stopping=True, max_iter=500, random_state=seed)),
    }
    return m


def extra_benchmark_models(seed: int) -> dict:
    """Gradient-boosting libraries that need their own install. Run with `--stages benchmark_extra`
    (e.g. on Colab); the results are merged into the benchmark table on the same folds."""
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    return {"XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, verbosity=0, random_state=seed),
            "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                       verbose=-1, random_state=seed)}


def stage_benchmark_extra(dev: pd.DataFrame, out: Path, seed: int) -> dict:
    import lightgbm
    import sklearn
    import xgboost
    X, y = dev[config.FEATURES], dev[Y].to_numpy()
    folds = _folds(y, seed)
    rows = []
    for name, est in extra_benchmark_models(seed).items():
        t0 = time.time()
        r = cv_scores(est, X, y, folds, seed)
        rows.append({"model": name, "cv_auc": float(r["auc"].mean()), "cv_auc_sd": float(r["auc"].std(ddof=1)), "cv_pr_auc": float(r["pr_auc"].mean()),
                     "cv_brier": float(r["brier"].mean()), "seconds": round(time.time() - t0, 1), "fold_auc": r["auc"].tolist()})
        print(f"    {name}: AUC {rows[-1]['cv_auc']:.3f}", flush=True)
    res = {"rows": rows, "versions": {"xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__, "scikit-learn": sklearn.__version__},
           "fold_check": [int(te.sum()) for _, te in folds[:3]]}
    _json(res, out / "benchmark_extra.json")
    return res


def _folds(y, seed, repeats: int | None = None):
    repeats = config.CV_REPEATS if repeats is None else repeats
    return list(RepeatedStratifiedKFold(n_splits=config.CV_FOLDS, n_repeats=repeats, random_state=seed).split(np.zeros(len(y)), y))


def stage_benchmark(dev: pd.DataFrame, out: Path, seed: int) -> dict:
    X, y = dev[config.FEATURES], dev[Y].to_numpy()
    folds = _folds(y, seed)
    rows, fold_auc = [], {}
    for name, est in benchmark_models(seed).items():
        t0 = time.time()
        r = cv_scores(est, X, y, folds, seed)
        fold_auc[name] = r["auc"]
        rows.append({"model": name, "cv_auc": float(r["auc"].mean()), "cv_auc_sd": float(r["auc"].std(ddof=1)), "cv_pr_auc": float(r["pr_auc"].mean()),
                     "cv_brier": float(r["brier"].mean()), "seconds": round(time.time() - t0, 1)})
        print(f"    {name}: AUC {rows[-1]['cv_auc']:.3f}", flush=True)
    extra = out / "benchmark_extra.json"
    if extra.exists():    # XGBoost / LightGBM, run separately on the same folds (see stage_benchmark_extra)
        ex = json.loads(extra.read_text())
        for r in ex["rows"]:
            fold_auc[r["model"]] = np.array(r.pop("fold_auc"))
            rows.append({**r, "model": r["model"] + f" ({ex['versions'][r['model'].lower()]})"})
            fold_auc[rows[-1]["model"]] = fold_auc.pop(r["model"])
    # the clinical benchmark needs no fitting: score it on the same folds
    p_pce = pce.risk_10y(dev)
    fold_auc["Pooled Cohort Equations (published, not refitted)"] = np.array([roc_auc_score(y[te], p_pce[te]) for _, te in folds])
    rows.append({"model": "Pooled Cohort Equations (published, not refitted)", "cv_auc": float(fold_auc["Pooled Cohort Equations (published, not refitted)"].mean()),
                 "cv_auc_sd": float(fold_auc["Pooled Cohort Equations (published, not refitted)"].std(ddof=1)),
                 "cv_pr_auc": float(np.mean([average_precision_score(y[te], p_pce[te]) for _, te in folds])), "cv_brier": float(stats.brier(y, p_pce)), "seconds": 0})
    tab = pd.DataFrame(rows).sort_values("cv_auc", ascending=False).reset_index(drop=True)
    best = tab.model[0]
    n_te = len(y) // config.CV_FOLDS
    raw = {m: (1.0 if m == best else corrected_t(fold_auc[best] - fold_auc[m], len(y) - n_te, n_te)) for m in tab.model}
    adj = stats.holm({m: p for m, p in raw.items() if m != best})
    tab["p_vs_best"] = tab.model.map(raw)
    tab["p_vs_best_holm"] = tab.model.map(lambda m: adj.get(m, 1.0))
    res = {"table": tab.to_dict(orient="records"), "best": best, "folds": f"{config.CV_REPEATS}x{config.CV_FOLDS} stratified",
           "n": int(len(y)), "events": int(y.sum())}
    _json(res, out / "benchmark.json")
    return res


def stage_imbalance(dev: pd.DataFrame, out: Path, seed: int) -> dict:
    X, y = dev[config.FEATURES], dev[Y].to_numpy()
    folds = _folds(y, seed, repeats=2)
    res = {}
    for name in ("logistic", "hgb"):
        for imb in ("none", "class_weight", "oversample", "smote"):
            est = build(name, imb, seed=seed)
            r = cv_scores(est, X, y, folds, seed)
            oof = r["oof_first_repeat"]
            cal = stats.calibration_intercept_slope(y, oof)
            res[f"{name} / {imb}"] = {"cv_auc": float(r["auc"].mean()), "cv_pr_auc": float(r["pr_auc"].mean()), "cv_brier": float(r["brier"].mean()),
                                      "cal_intercept": cal["intercept"], "mean_predicted": cal["mean_predicted"], "observed": cal["observed_rate"]}
            print(f"    {name} / {imb}: AUC {res[f'{name} / {imb}']['cv_auc']:.3f}", flush=True)
    _json(res, out / "imbalance.json")
    return res


def stage_select(dev: pd.DataFrame, out: Path, seed: int) -> dict:
    X, y = dev[config.FEATURES], dev[Y].to_numpy()
    folds = _folds(y, seed, repeats=3)
    grid = {"logistic": [{"C": c} for c in (0.01, 0.03, 0.1, 0.3, 1.0)],
            "hgb": [{"learning_rate": lr, "max_leaf_nodes": ml, "min_samples_leaf": ms} for lr in (0.03, 0.1) for ml in (4, 15) for ms in (40, 150)]}
    res = {}
    for name, cands in grid.items():
        scores = []
        for params in cands:
            r = cv_scores(build(name, params=params, seed=seed), X, y, folds, seed)
            scores.append({"params": params, "cv_auc": float(r["auc"].mean()), "cv_brier": float(r["brier"].mean()), "fold_auc": r["auc"].tolist()})
        best = max(scores, key=lambda s: s["cv_auc"])
        res[name] = {"grid": scores, "best_params": best["params"], "best_cv_auc": best["cv_auc"], "best_fold_auc": best["fold_auc"]}
    n_te = len(y) // config.CV_FOLDS
    diffs = np.array(res["hgb"]["best_fold_auc"]) - np.array(res["logistic"]["best_fold_auc"])
    p = corrected_t(diffs, len(y) - n_te, n_te)
    # Parsimony: gradient boosting must beat logistic regression clearly to be chosen.
    chosen = "hgb" if (diffs.mean() > 0.005 and p < 0.05) else "logistic"
    res["comparison"] = {"hgb_minus_logistic_auc": float(diffs.mean()), "p_corrected_t": p, "chosen": chosen,
                         "rule": "choose gradient boosting only if it beats logistic regression by > 0.005 AUC with p < 0.05 (corrected resampled t-test)"}
    _json(res, out / "select.json")
    return res


# ------------------------------------------------------------------------------ temporal validation

def recalibrate_pce(dev: pd.DataFrame) -> tuple[float, float]:
    """Logistic recalibration of the PCE linear predictor to 10-year CVD death on development data."""
    from scipy.special import logit
    lp = logit(np.clip(pce.risk_10y(dev), 1e-6, 1 - 1e-6))
    beta, _ = stats._fit_logistic(np.column_stack([np.ones(len(dev)), lp]), dev[Y].to_numpy())
    return float(beta[0]), float(beta[1])


def apply_recal(df, ab):
    from scipy.special import expit, logit
    return expit(ab[0] + ab[1] * logit(np.clip(pce.risk_10y(df), 1e-6, 1 - 1e-6)))


def fit_final(dev: pd.DataFrame, sel: dict, seed: int) -> dict:
    fitted = {}
    for name in ("age_sex", "pce_refit", "logistic", "hgb"):
        params = sel[name]["best_params"] if name in sel else {}
        m = build(name, params=params, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(dev[FEATURE_SETS[name]], dev[Y].to_numpy())
        fitted[name] = m
    return fitted


def predict_all(fitted: dict, df: pd.DataFrame, recal) -> dict:
    p = {name: m.predict_proba(df[FEATURE_SETS[name]])[:, 1] for name, m in fitted.items()}
    p["pce"] = pce.risk_10y(df)
    p["pce_recalibrated"] = apply_recal(df, recal)
    return p


LABELS = {"pce": "Pooled Cohort Equations (published)", "pce_recalibrated": "PCE, recalibrated to CVD death", "age_sex": "Age + sex",
          "pce_refit": "PCE variables, refitted", "logistic": "Logistic regression (all features)", "hgb": "Gradient boosting (all features)"}


def stage_temporal(df: pd.DataFrame, out: Path, sel: dict, seed: int, n_boot: int) -> dict:
    dev, test = binary_cohort(df, "development"), binary_cohort(df, "temporal_test")
    recal = recalibrate_pce(dev)
    fitted = fit_final(dev, sel, seed)
    p = predict_all(fitted, test, recal)
    y = test[Y].to_numpy()
    chosen = sel["comparison"]["chosen"]
    res = {"n_dev": int(len(dev)), "events_dev": int(dev[Y].sum()), "n_test": int(len(test)), "events_test": int(y.sum()),
           "pce_recalibration": {"intercept": recal[0], "slope": recal[1]}, "chosen": chosen, "models": {}, "vs_pce": {}, "dca": {}, "capture": {}}
    for k, v in p.items():
        res["models"][k] = {"label": LABELS[k], **metrics(y, v, n_boot, seed)}
        res["models"][k]["reliability"] = stats.reliability_curve(y, v, 10)
    for k in ("age_sex", "pce_refit", "logistic", "hgb"):       # recalibration is monotone: identical ranking to the PCE
        res["vs_pce"][k] = stats.delong_paired(y, p[k], p["pce"])
    adj = stats.holm({k: v["p"] for k, v in res["vs_pce"].items()})
    for k in res["vs_pce"]:
        res["vs_pce"][k]["p_holm"] = adj[k]
    res["chosen_vs_logistic"] = stats.delong_paired(y, p["hgb"], p["logistic"])
    res["chosen_vs_pce_refit"] = stats.delong_paired(y, p[chosen], p["pce_refit"])
    for k in ("pce", "pce_recalibrated", "pce_refit", chosen):
        res["dca"][k] = stats.net_benefit(y, p[k], config.DCA_THRESHOLDS)
    for share in (0.1, 0.2):
        res["capture"][str(share)] = {k: stats.capture_at_capacity(y, p[k], share) for k in ("pce", chosen, "age_sex")}
    pred = pd.DataFrame({"seqn": test.seqn, "cycle": test.cycle, "y": y, **{k: v for k, v in p.items()}})
    pred.to_csv(out / "temporal_test_predictions.csv.gz", index=False, compression="gzip", float_format="%.5f")
    _json(res, out / "temporal.json")
    import joblib
    joblib.dump({"models": fitted, "pce_recalibration": recal, "features": FEATURE_SETS, "chosen": chosen}, out / "models.joblib")
    return res


# ------------------------------------------------------------------------------ survival

def cox_matrix(df: pd.DataFrame, ref: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Linear Cox design: age, age^2 and the features, missing values imputed with development medians plus flags."""
    X = df[config.FEATURES].copy()
    names = list(config.FEATURES)
    flags = []
    for c in config.FEATURES:
        if ref[c].isna().any() or X[c].isna().any():
            if ref[c].isna().mean() > 0.01:
                X[c + "_missing"] = X[c].isna().astype(float)
                flags.append(c + "_missing")
            X[c] = X[c].fillna(ref[c].median())
    X["age_sq"] = (X.age - 60) ** 2 / 100
    return X[names + flags + ["age_sq"]].to_numpy(float), names + flags + ["age_sq"]


def stage_survival(df: pd.DataFrame, out: Path, seed: int) -> dict:
    """Cause-specific Cox model for CVD death (non-CVD deaths censored), fitted on development cycles with
    their full follow-up, then evaluated on 2005-2008 (10 years) and on 2009-2018 (incomplete follow-up)."""
    dev = df[df.split == "development"].reset_index(drop=True)
    Xd, names = cox_matrix(dev, dev)
    cox = CoxPH(penalty=0.01).fit(Xd, dev.months.to_numpy(), dev.cvd_death.to_numpy())
    res = {"hazard_ratios": cox.hazard_ratios(names), "evaluation": {}}
    for label, g in (("temporal test 2005-2008", df[df.split == "temporal_test"]), ("recent 2009-2018", df[df.split == "recent"])):
        g = g.reset_index(drop=True)
        X, _ = cox_matrix(g, dev)
        lp = cox.linear_predictor(X)
        ev = {"n": int(len(g)), "cvd_deaths": int(g.cvd_death.sum()), "median_followup_years": float(g.months.median() / 12),
              "c_index_cox": harrell_c(g.months, g.cvd_death, lp), "c_index_pce": harrell_c(g.months, g.cvd_death, pce.risk_10y(g)),
              "c_index_age": harrell_c(g.months, g.cvd_death, g.age)}
        horizon = 60 if label.startswith("recent") else 120
        risk = cox.risk_at(X, horizon)
        q = pd.qcut(risk, 5, labels=False, duplicates="drop")
        ev["calibration_by_quintile"] = [{"quintile": int(k) + 1, "predicted": float(risk[q == k].mean()),
                                          "observed_km": kaplan_meier(g.months[q == k], g.cvd_death[q == k], horizon)} for k in np.unique(q)]
        ev["horizon_months"] = horizon
        ev["overall_predicted"], ev["overall_observed_km"] = float(risk.mean()), kaplan_meier(g.months, g.cvd_death, horizon)
        res["evaluation"][label] = ev
    _json(res, out / "survival.json")
    return res


# ------------------------------------------------------------------------------ leakage

def stage_leakage(df: pd.DataFrame, adults: pd.DataFrame | None, out: Path, sel: dict, seed: int) -> dict:
    dev, test = binary_cohort(df, "development"), binary_cohort(df, "temporal_test")
    feats = config.FEATURES
    def fit_eval(tr, te, fs, name="logistic"):
        from .models import logistic_pipeline
        m = logistic_pipeline(fs, **sel["logistic"]["best_params"]) if name == "logistic" else build("hgb", params=sel["hgb"]["best_params"], seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(tr[fs], tr[Y])
        p = m.predict_proba(te[fs])[:, 1]
        return {"auc": float(roc_auc_score(te[Y], p)), "pr_auc": float(average_precision_score(te[Y], p))}
    res = {"correct (baseline variables, temporal split)": fit_eval(dev, test, feats)}
    d2, t2 = dev.copy(), test.copy()
    for c in config.LEAKY_FEATURES:
        d2[c], t2[c] = d2[c].fillna(0), t2[c].fillna(0)
    res["+ death-certificate diabetes & hypertension flags (recorded after death)"] = fit_eval(d2, t2, feats + config.LEAKY_FEATURES)
    both = pd.concat([dev, test], ignore_index=True)
    tr, te = train_test_split(both, test_size=len(test) / len(both), stratify=both[Y], random_state=seed)
    res["random split instead of temporal (same features)"] = fit_eval(tr, te, feats)
    if adults is not None:
        a = adults[(adults.examined == 1) & adults.age.between(config.AGE_MIN, config.AGE_MAX) & (adults.pregnant == 0) & (adults.linkage_eligible == 1)
                   & adults[config.CORE_REQUIRED].notna().all(axis=1) & (adults.complete_10y == 1) & adults.prior_cvd.notna()]
        a = a.assign(split=np.where(a.cycle.isin(config.DEV_CYCLES), "development", np.where(a.cycle.isin(config.TEST_CYCLES), "temporal_test", "recent")))
        ad, at = a[a.split == "development"], a[a.split == "temporal_test"]
        res["+ people who already had heart disease or stroke (prior CVD as a feature)"] = fit_eval(ad, at, feats + ["prior_cvd"])
        res["note_prior_cvd"] = {"share_with_prior_cvd": float(a.prior_cvd.mean()), "event_rate_prior_cvd": float(a.loc[a.prior_cvd == 1, Y].mean()),
                                 "event_rate_no_prior_cvd": float(a.loc[a.prior_cvd == 0, Y].mean())}
    _json(res, out / "leakage.json")
    return res


# ------------------------------------------------------------------------------ subgroups / explain

def stage_subgroups(out: Path, sel: dict) -> dict:
    pred = pd.read_csv(out / "temporal_test_predictions.csv.gz")
    df = load()
    pred = pred.merge(df[["seqn", "female", "race_black", "race_hispanic", "age", "diabetes"]], on="seqn")
    chosen = sel["comparison"]["chosen"]
    groups = {"women": pred.female == 1, "men": pred.female == 0, "non-Hispanic Black": pred.race_black == 1, "Hispanic": pred.race_hispanic == 1,
              "other (mostly non-Hispanic White)": (pred.race_black == 0) & (pred.race_hispanic == 0), "age 40-59": pred.age < 60, "age 60-79": pred.age >= 60,
              "diabetes": pred.diabetes == 1, "no diabetes": pred.diabetes == 0}
    res = {}
    for g, m in groups.items():
        s = pred[m]
        if s.y.sum() < 5:
            continue
        res[g] = {"n": int(len(s)), "events": int(s.y.sum()),
                  **{f"{k}_auc": float(roc_auc_score(s.y, s[k])) for k in ("pce", chosen)},
                  **{f"{k}_o_e": float(s.y.mean() / s[k].mean()) for k in ("pce_recalibrated", chosen)}}
    _json(res, out / "subgroups.json")
    return res


def linear_shap(pipe, X: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, float]:
    """Exact SHAP on the log-odds scale for a linear model on transformed features, summed back to each
    original feature (spline bases, missing flags)."""
    pre, lr = pipe.named_steps["pre"], pipe.named_steps["lr"]
    Z = pre.transform(X)
    phi = (Z - Z.mean(axis=0)) * lr.coef_[0]
    names = pre.get_feature_names_out()
    owner = []
    for n in names:
        core = n.split("__", 1)[1]
        core = core.replace("missingindicator_", "")
        owner.append(next(f for f in sorted(features, key=len, reverse=True) if core.startswith(f)))
    agg = pd.DataFrame(phi, columns=names).T.groupby(np.array(owner)).sum().T
    lp = pipe.decision_function(X)
    err = float(np.max(np.abs(agg.sum(axis=1).to_numpy() + lp.mean() - lp)))
    return agg, err


def stage_explain(df: pd.DataFrame, out: Path, sel: dict, seed: int) -> dict:
    import joblib
    bundle = joblib.load(out / "models.joblib")
    test = binary_cohort(df, "temporal_test")
    res = {}
    agg, err = linear_shap(bundle["models"]["logistic"], test[config.FEATURES], config.FEATURES)
    res["logistic"] = {"mean_abs_shap": agg.abs().mean().sort_values(ascending=False).to_dict(), "additivity_max_error": err,
                       "dependence": {f: {"x": test[f].tolist()[:2000], "shap": agg[f].tolist()[:2000]} for f in ("age", "sbp", "hdl", "hba1c", "egfr")}}
    try:
        import shap
        m = bundle["models"]["hgb"]
        ex = shap.TreeExplainer(m)
        sv = ex.shap_values(test[config.FEATURES])
        sv = sv[1] if isinstance(sv, list) else sv
        raw = m.decision_function(test[config.FEATURES]) if hasattr(m, "decision_function") else None
        base = ex.expected_value if np.ndim(ex.expected_value) == 0 else ex.expected_value[-1]
        res["hgb"] = {"mean_abs_shap": dict(sorted(zip(config.FEATURES, np.abs(sv).mean(axis=0).tolist()), key=lambda kv: -kv[1])),
                      "additivity_max_error": float(np.max(np.abs(sv.sum(axis=1) + base - raw))) if raw is not None else None}
    except ImportError:
        res["hgb"] = {"note": "shap not installed; TreeExplainer SHAP runs on Colab"}
    _json(res, out / "explain.json")
    return res


# ------------------------------------------------------------------------------ run

def run(stages: list[str], out: Path = config.RESULTS, seed: int = config.SEED, n_boot: int = config.N_BOOT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df = load()
    adults = pd.read_csv(config.HARMONISED_CSV) if config.HARMONISED_CSV.exists() else None
    dev = binary_cohort(df, "development")
    sel = json.loads((out / "select.json").read_text()) if (out / "select.json").exists() else None
    for s in ["benchmark_extra"] + STAGES:
        if s not in stages:
            continue
        t0 = time.time()
        print(f"[{s}]", flush=True)
        if s == "quality":
            stage_quality(df, adults, out)
        elif s == "eda":
            stage_eda(df, out)
        elif s == "benchmark_extra":
            stage_benchmark_extra(dev, out, seed)
        elif s == "benchmark":
            stage_benchmark(dev, out, seed)
        elif s == "imbalance":
            stage_imbalance(dev, out, seed)
        elif s == "select":
            sel = stage_select(dev, out, seed)
        elif s == "temporal":
            stage_temporal(df, out, sel, seed, n_boot)
        elif s == "survival":
            stage_survival(df, out, seed)
        elif s == "leakage":
            stage_leakage(df, adults, out, sel, seed)
        elif s == "subgroups":
            stage_subgroups(out, sel)
        elif s == "explain":
            stage_explain(df, out, sel, seed)
        print(f"[{s}] done in {time.time() - t0:.0f} s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="*", default=STAGES)
    ap.add_argument("--n-boot", type=int, default=config.N_BOOT)
    a = ap.parse_args(argv)
    run(a.stages, n_boot=a.n_boot)


if __name__ == "__main__":
    main()
