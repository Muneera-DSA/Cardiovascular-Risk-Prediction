"""Part 1: audit of the original 918-patient heart-disease notebook, then a corrected analysis.

    python -m cvd_risk.audit918

Stages
1. data quality       impossible zeros, duplicates, hospital mix and prevalence by hospital
2. shortcut           cholesterol = 0 is a hospital marker, not a measurement
3. original pipeline  the notebook's own split, encoding, scaling and models, reproduced exactly, to show
                      (a) its "test_score" was not ROC-AUC, (b) the test set was reused for selection,
                      (c) the final "best model" cell trained on the 9 dummy columns only
4. corrected          many-model screen by repeated cross-validation (inside-fold preprocessing),
                      paired tests, then leave-one-hospital-out external validation
5. statistics         association tests with effect sizes and Holm correction; adjusted odds ratios; SHAP
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config, stats
from .uci import KEY, SITES, match_sites, to_kaggle_coding

NUMERIC = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
TARGET = "HeartDisease"


# ------------------------------------------------------------------------------- data

def load(kaggle_csv: Path | None = None, uci_csv: Path | None = None) -> pd.DataFrame:
    """The analysis table: the 918 Kaggle rows when available (site attached by exact match to UCI),
    otherwise the UCI four-hospital files recoded to the Kaggle layout."""
    from .uci import UCI_CSV
    uci_csv = uci_csv or UCI_CSV
    uci = to_kaggle_coding(pd.read_csv(uci_csv))
    if kaggle_csv and Path(kaggle_csv).exists():
        k = pd.read_csv(kaggle_csv)
        return match_sites(k, uci)
    return uci


def quality(df: pd.DataFrame) -> dict:
    d = df
    by_site = (d.groupby("site", dropna=False)
                .agg(patients=(TARGET, "size"), disease_rate=(TARGET, "mean"),
                     cholesterol_zero=("Cholesterol", lambda s: float((s == 0).mean())),
                     resting_bp_zero=("RestingBP", lambda s: float((s == 0).mean())))
                .reset_index())
    by_site["site"] = by_site.site.map(lambda s: SITES.get(s, "unmatched"))
    return {
        "rows": len(d), "columns": len(KEY),
        "missing_values": int(d[KEY].isna().sum().sum()),
        "exact_duplicates": int(d[KEY].duplicated().sum()),
        "cholesterol_zero": int((d.Cholesterol == 0).sum()),
        "resting_bp_zero": int((d.RestingBP == 0).sum()),
        "oldpeak_negative": int((d.Oldpeak < 0).sum()),
        "disease_rate": float(d[TARGET].mean()),
        "site_matched": int(d.site.notna().sum()) if "site" in d else None,
        "site_assignment": d.site_assigned_by.value_counts().to_dict() if "site_assigned_by" in d else None,
        "by_site": by_site.to_dict(orient="records"),
    }


# ------------------------------------------------------------------------------- shortcut

def cholesterol_shortcut(df: pd.DataFrame) -> dict:
    d = df
    zero = d.Cholesterol == 0
    rate0, rate1 = d.loc[zero, TARGET].mean(), d.loc[~zero, TARGET].mean()
    mean_chol = d.groupby(TARGET).Cholesterol.mean()
    mean_chol_measured = d[~zero].groupby(TARGET).Cholesterol.mean()
    u = sps.mannwhitneyu(d.loc[~zero & (d[TARGET] == 1), "Cholesterol"], d.loc[~zero & (d[TARGET] == 0), "Cholesterol"])
    return {
        "disease_rate_when_cholesterol_zero": float(rate0), "disease_rate_when_measured": float(rate1),
        "auc_of_the_zero_flag_alone": float(roc_auc_score(d[TARGET], zero.astype(int))),
        "mean_cholesterol_by_class_with_zeros": {int(k): float(v) for k, v in mean_chol.items()},
        "mean_cholesterol_by_class_measured_only": {int(k): float(v) for k, v in mean_chol_measured.items()},
        "measured_only_mannwhitney_p": float(u.pvalue),
        "zeros_by_site": d[zero].site.value_counts(dropna=False).to_dict() if "site" in d else None,
    }


# ------------------------------------------------------------------------------- original

def _original_split(df: pd.DataFrame):
    """The notebook's encoding, split and scaling, exactly."""
    cols = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]
    X = pd.get_dummies(df[cols], columns=CATEGORICAL, drop_first=True).astype(float)
    y = df[TARGET].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=10, stratify=y)
    sc = StandardScaler().fit(Xtr[NUMERIC])
    Xtr_s, Xte_s = Xtr.copy(), Xte.copy()
    Xtr_s[NUMERIC], Xte_s[NUMERIC] = sc.transform(Xtr[NUMERIC]), sc.transform(Xte[NUMERIC])
    return Xtr, Xte, ytr, yte, Xtr_s, Xte_s


def _original_row(name, m, Xtr_s, ytr, Xte_s, yte) -> dict:
    cv = cross_validate(clone(m), Xtr_s, ytr, cv=5, scoring="roc_auc", return_train_score=True)
    m.fit(Xtr_s, ytr)
    hard = m.predict(Xte_s)
    prob = m.predict_proba(Xte_s)[:, 1]
    ci = stats.delong_auc(yte, prob)
    return {"model": name, "cv_train_auc": float(cv["train_score"].mean()), "cv_auc": float(cv["test_score"].mean()),
            "notebook_test_score (AUC of 0/1 predictions)": float(roc_auc_score(yte, hard)),
            "balanced_accuracy": float(balanced_accuracy_score(yte, hard)),
            "test_auc (probabilities)": ci["value"], "test_auc_ci": ci["ci"]}


EXTRA_JSON = config.RESULTS / "part1_extra.json"


def run_extra(df: pd.DataFrame, out: Path = EXTRA_JSON, seed: int = config.SEED, repeats: int = 10) -> dict:
    """XGBoost and LightGBM (need their own install, e.g. on Colab): the original notebook's XGBClassifier
    on its own split, and both libraries in the corrected screen on the same repeated CV folds."""
    import lightgbm
    import sklearn
    import xgboost
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    Xtr, Xte, ytr, yte, Xtr_s, Xte_s = _original_split(df)
    res = {"n_rows": int(len(df)), "versions": {"xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__, "scikit-learn": sklearn.__version__},
           "original": _original_row("XGBClassifier", XGBClassifier(verbosity=0, random_state=0), Xtr_s, ytr, Xte_s, yte), "screen": []}
    X, y = prepare(df), df[TARGET].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    for name, est in (("XGBoost", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, verbosity=0, random_state=seed)),
                      ("LightGBM", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=seed))):
        r = cross_validate(Pipeline([("pre", preprocessor()), ("m", est)]), X, y, cv=cv,
                           scoring={"auc": "roc_auc", "brier": "neg_brier_score", "acc": "accuracy"}, return_train_score=True, n_jobs=-1)
        res["screen"].append({"model": f"{name} ({res['versions'][name.lower()]})", "cv_auc": float(r["test_auc"].mean()), "cv_auc_sd": float(r["test_auc"].std(ddof=1)),
                              "train_auc": float(r["train_auc"].mean()), "brier": float(-r["test_brier"].mean()), "accuracy": float(r["test_acc"].mean()),
                              "fold_auc": r["test_auc"].tolist()})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return res


def original_pipeline(df: pd.DataFrame) -> dict:
    """The notebook, step for step: get_dummies(drop_first) -> split(random_state=10, stratify)
    -> StandardScaler on the 6 numeric columns -> six default models."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    Xtr, Xte, ytr, yte, Xtr_s, Xte_s = _original_split(df)
    models = {"LogisticRegression": LogisticRegression(max_iter=1000), "KNeighborsClassifier": KNeighborsClassifier(),
              "DecisionTreeClassifier": DecisionTreeClassifier(random_state=0), "RandomForestClassifier": RandomForestClassifier(random_state=0),
              "SVC": SVC(probability=True, random_state=0)}
    rows = [_original_row(name, m, Xtr_s, ytr, Xte_s, yte) for name, m in models.items()]
    if EXTRA_JSON.exists() and json.loads(EXTRA_JSON.read_text()).get("n_rows") == len(df):   # XGBClassifier, run where xgboost is installed
        rows.append(json.loads(EXTRA_JSON.read_text())["original"])
    # The "EZRA CODE" cell: X_train had been overwritten with the 9 dummy columns only, unscaled.
    dummies = [c for c in Xtr.columns if c not in NUMERIC]
    lr_dummies = LogisticRegression(C=10, max_iter=1000)
    acc_dummies = cross_validate(lr_dummies, Xtr[dummies], ytr, cv=5, scoring="accuracy")["test_score"].mean()
    acc_full = cross_validate(make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=1000)), Xtr, ytr, cv=5, scoring="accuracy")["test_score"].mean()
    return {"split": {"train": len(ytr), "test": len(yte), "test_positive": int(yte.sum())},
            "models": rows,
            "final_cell_bug": {"columns_used": dummies, "cv_accuracy_dummies_only": float(acc_dummies),
                               "cv_accuracy_all_features": float(acc_full)}}


# ------------------------------------------------------------------------------- corrected

def preprocessor():
    """Inside-fold preprocessing. Cholesterol/BP of 0 -> missing -> median imputation + a missing flag."""
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="ignore", drop="if_binary")
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def prepare(df: pd.DataFrame, zero_as_missing: bool = True) -> pd.DataFrame:
    X = df[NUMERIC + CATEGORICAL].copy()
    if zero_as_missing:
        X.loc[X.Cholesterol == 0, "Cholesterol"] = np.nan
        X.loc[X.RestingBP == 0, "RestingBP"] = np.nan
    return X


def screen_models(seed: int = config.SEED) -> dict:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    m = {
        "Prior only (no information)": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(max_iter=2000),
        "Logistic regression (L1, C=0.1)": LogisticRegression(penalty="l1", C=0.1, solver="liblinear"),
        "LDA": LinearDiscriminantAnalysis(),
        "Gaussian naive Bayes": GaussianNB(),
        "k-nearest neighbours (k=15)": KNeighborsClassifier(n_neighbors=15),
        "SVM (RBF)": SVC(probability=True, random_state=seed),
        "Decision tree (depth 4)": DecisionTreeClassifier(max_depth=4, random_state=seed),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1),
        "Extra trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=seed),
        "Gradient boosting": GradientBoostingClassifier(random_state=seed),
        "Histogram gradient boosting": HistGradientBoostingClassifier(max_depth=3, random_state=seed),
        "Neural network (MLP 16)": MLPClassifier((16,), alpha=1e-2, max_iter=2000, random_state=seed),
    }
    return m


def corrected_t(diffs: np.ndarray, n_train: int, n_test: int) -> float:
    """Nadeau & Bengio corrected resampled t-test (two-sided p) for repeated CV differences."""
    k = len(diffs)
    var = np.var(diffs, ddof=1)
    if var == 0:
        return 1.0
    t = diffs.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return float(2 * sps.t.sf(abs(t), df=k - 1))


def model_screen(df: pd.DataFrame, seed: int = config.SEED, repeats: int = 10) -> dict:
    X, y = prepare(df), df[TARGET].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    rows, fold_auc = [], {}
    for name, est in screen_models(seed).items():
        pipe = Pipeline([("pre", preprocessor()), ("m", est)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = cross_validate(pipe, X, y, cv=cv, scoring={"auc": "roc_auc", "brier": "neg_brier_score", "acc": "accuracy"},
                               return_train_score=True, n_jobs=-1)
        fold_auc[name] = r["test_auc"]
        rows.append({"model": name, "cv_auc": float(r["test_auc"].mean()), "cv_auc_sd": float(r["test_auc"].std(ddof=1)),
                     "train_auc": float(r["train_auc"].mean()), "brier": float(-r["test_brier"].mean()), "accuracy": float(r["test_acc"].mean())})
    if EXTRA_JSON.exists() and json.loads(EXTRA_JSON.read_text()).get("n_rows") == len(df):
        for r in json.loads(EXTRA_JSON.read_text())["screen"]:
            if len(r["fold_auc"]) == len(next(iter(fold_auc.values()))):
                fold_auc[r["model"]] = np.array(r.pop("fold_auc"))
                rows.append(r)
    tab = pd.DataFrame(rows).sort_values("cv_auc", ascending=False).reset_index(drop=True)
    best = tab.model[0]
    n_test = len(y) // 5
    tab["p_vs_best (corrected t)"] = [1.0 if m == best else corrected_t(fold_auc[best] - fold_auc[m], len(y) - n_test, n_test) for m in tab.model]
    tab["p_vs_best (Holm)"] = tab.model.map(stats.holm({m: p for m, p in zip(tab.model, tab["p_vs_best (corrected t)"]) if m != best})).fillna(1.0)
    return {"table": tab.to_dict(orient="records"), "best": best, "repeats": repeats,
            "indistinguishable_from_best": tab.loc[tab["p_vs_best (Holm)"] >= 0.05, "model"].tolist()}


def imbalance_check(df: pd.DataFrame, seed: int = config.SEED) -> dict:
    """55% of patients have heart disease, so the classes are close to balanced. Checked, not assumed:
    class weighting is compared with no weighting on the same repeated CV folds."""
    from sklearn.ensemble import RandomForestClassifier
    X, y = prepare(df), df[TARGET].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    out = {"positive_share": float(y.mean()), "by_site": df.groupby("site")[TARGET].mean().to_dict() if "site" in df else None, "cv": {}}
    for name, est in (("Logistic regression", LogisticRegression(max_iter=2000)), ("Random forest", RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1))):
        for cw in (None, "balanced"):
            pipe = Pipeline([("pre", preprocessor()), ("m", clone(est).set_params(class_weight=cw))])
            r = cross_validate(pipe, X, y, cv=cv, scoring={"auc": "roc_auc", "bal": "balanced_accuracy"}, n_jobs=-1)
            out["cv"][f"{name} / {'class weights' if cw else 'none'}"] = {"auc": float(r["test_auc"].mean()), "balanced_accuracy": float(r["test_bal"].mean())}
    return out


def leave_one_hospital_out(df: pd.DataFrame, model_names: list[str], seed: int = config.SEED) -> dict:
    """Train on three hospitals, test on the fourth. The honest test of 'will it work somewhere else?'"""
    d = df[df.site.notna()].reset_index(drop=True)
    models = screen_models(seed)
    out = {}
    for site in SITES:
        tr, te = d[d.site != site], d[d.site == site]
        if te[TARGET].nunique() < 2:
            continue
        res = {}
        for name in model_names:
            pipe = Pipeline([("pre", preprocessor()), ("m", clone(models[name]))])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(prepare(tr), tr[TARGET])
                p = pipe.predict_proba(prepare(te))[:, 1]
            auc = stats.delong_auc(te[TARGET], p)
            cal = stats.calibration_intercept_slope(te[TARGET], p)
            res[name] = {"auc": auc["value"], "auc_ci": auc["ci"], "calibration_intercept": cal["intercept"],
                         "calibration_slope": cal["slope"], "observed_rate": cal["observed_rate"], "mean_predicted": cal["mean_predicted"],
                         "accuracy_at_0.5": float(accuracy_score(te[TARGET], p >= 0.5))}
        out[SITES[site]] = {"n": len(te), "disease_rate": float(te[TARGET].mean()), "models": res}
    return out


# ------------------------------------------------------------------------------- statistics

def association_tests(df: pd.DataFrame) -> list[dict]:
    rows = []
    y = df[TARGET]
    for c in CATEGORICAL + ["FastingBS"]:
        ct = pd.crosstab(df[c], y)
        chi2, p, dof, _ = sps.chi2_contingency(ct)
        v = np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1)))
        rows.append({"feature": c, "test": "chi-square", "p": float(p), "effect": "Cramér's V", "effect_size": float(v)})
    X = prepare(df)
    for c in ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]:
        a, b = X.loc[y == 1, c].dropna(), X.loc[y == 0, c].dropna()
        u = sps.mannwhitneyu(a, b)
        rows.append({"feature": c + (" (zeros as missing)" if c in ("Cholesterol", "RestingBP") else ""), "test": "Mann-Whitney U",
                     "p": float(u.pvalue), "effect": "P(diseased > healthy)", "effect_size": float(u.statistic / (len(a) * len(b)))})
    adj = stats.holm({r["feature"]: r["p"] for r in rows})
    for r in rows:
        r["p_holm"] = adj[r["feature"]]
    return rows


def odds_ratios(df: pd.DataFrame) -> list[dict]:
    X = prepare(df)
    X["Cholesterol_missing"] = X.Cholesterol.isna().astype(float)
    X["Cholesterol"] = X.Cholesterol.fillna(X.Cholesterol.median())
    X["RestingBP"] = X.RestingBP.fillna(X.RestingBP.median())
    Z = pd.get_dummies(X, columns=CATEGORICAL, drop_first=True).astype(float)
    scale = {"Age": 10, "RestingBP": 10, "Cholesterol": 40, "MaxHR": 10, "Oldpeak": 1}
    for c, s in scale.items():
        Z[c] = Z[c] / s
    names = [f"{c} (per {scale[c]})" if c in scale else c for c in Z.columns]
    return stats.logistic_odds_ratios(Z.to_numpy(), df[TARGET].to_numpy(), names)


def linear_shap(df: pd.DataFrame) -> dict:
    """Exact SHAP for the logistic model (log-odds scale): phi_j = w_j (x_j - mean x_j),
    summed back to the original feature for one-hot and missing-flag columns."""
    X, y = prepare(df), df[TARGET].to_numpy()
    pipe = Pipeline([("pre", preprocessor()), ("m", LogisticRegression(max_iter=2000))]).fit(X, y)
    Z = pipe.named_steps["pre"].transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    w = pipe.named_steps["m"].coef_[0]
    phi = (Z - Z.mean(axis=0)) * w
    names = pipe.named_steps["pre"].get_feature_names_out()
    base = lambda n: next(f for f in NUMERIC + CATEGORICAL if f in n)   # noqa: E731
    groups = pd.Series([base(n) for n in names])
    agg = pd.DataFrame(phi, columns=names).T.groupby(groups.values).sum().T
    lp = pipe.decision_function(X)
    err = float(np.max(np.abs(agg.sum(axis=1) + (lp.mean()) - lp)))
    imp = agg.abs().mean().sort_values(ascending=False)
    return {"mean_abs_shap": imp.to_dict(), "additivity_max_error": err}


# ------------------------------------------------------------------------------- run

def run(df: pd.DataFrame, out: Path, seed: int = config.SEED, repeats: int = 10) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    res = {"quality": quality(df), "cholesterol_shortcut": cholesterol_shortcut(df), "original_pipeline": original_pipeline(df)}
    res["class_balance"] = imbalance_check(df, seed)
    res["model_screen"] = model_screen(df, seed, repeats)
    local = screen_models(seed)                  # XGBoost / LightGBM rows come from a separate run; refit only what is installed here
    top = [m for m in pd.DataFrame(res["model_screen"]["table"]).model if m != "Prior only (no information)" and m in local][:3]
    lohoo_models = list(dict.fromkeys(top + ["Logistic regression"]))
    res["leave_one_hospital_out"] = leave_one_hospital_out(df, lohoo_models, seed)
    res["association_tests"] = association_tests(df)
    res["odds_ratios"] = odds_ratios(df)
    res["shap_logistic"] = linear_shap(df)
    (out / "part1_audit.json").write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", type=Path, default=config.HEART918_CSV)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--extra-only", action="store_true", help="only the XGBoost / LightGBM runs (needs both installed)")
    a = ap.parse_args(argv)
    df = load(a.kaggle)
    if a.extra_only:
        if len(df) != 918:
            raise SystemExit(f"--extra-only needs the 918-row Kaggle file at {a.kaggle} (see data/README.md)")
        run_extra(df, repeats=a.repeats)
        print("wrote", EXTRA_JSON)
        return
    run(df, config.RESULTS, repeats=a.repeats)
    print("wrote", config.RESULTS / "part1_audit.json")


if __name__ == "__main__":
    main()
