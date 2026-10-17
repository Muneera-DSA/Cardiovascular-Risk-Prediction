"""Turn results/*.json into reports/results.md and reports/figures/*.png.

    python -m cvd_risk.report
Every number in the README comes from the files this writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

BLUE, ORANGE, GREY, GREEN = "#2a78d6", "#e8743b", "#8c8c8c", "#3a9a5b"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def _load(name: str, res: Path):
    p = res / name
    return json.loads(p.read_text()) if p.exists() else None


def _ci(c) -> str:
    return f"{c[0]:.3f}–{c[1]:.3f}"


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _table(rows: list[list], header: list[str]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


# ------------------------------------------------------------------------------------ Part 1

def part1(res: Path, fig: Path) -> list[str]:
    a = _load("part1_audit.json", res)
    if a is None:
        return []
    q, sh, orig = a["quality"], a["cholesterol_shortcut"], a["original_pipeline"]
    L = ["## Part 1 — Audit of the original 918-patient notebook", ""]
    L += ["### 1.1 Data quality", "",
          f"- {q['rows']} patients, {q['missing_values']} missing cells, {q['exact_duplicates']} exact duplicates.",
          f"- **Cholesterol recorded as 0 for {q['cholesterol_zero']} patients** and resting blood pressure 0 for {q['resting_bp_zero']}: physiologically impossible, so these are missing values.",
          f"- Hospital identified for {q['site_matched']} of {q['rows']} rows by exact match to the UCI source files.", ""]
    L += _table([[s["site"], s["patients"], f"{s['disease_rate']:.0%}", f"{s['cholesterol_zero']:.0%}"] for s in q["by_site"]],
                ["Hospital", "Patients", "Heart disease", "Cholesterol = 0"]) + [""]
    L += ["### 1.2 Cholesterol = 0 is a hospital marker, not a measurement", "",
          f"- Heart disease in patients with cholesterol 0: **{sh['disease_rate_when_cholesterol_zero']:.0%}**, versus {sh['disease_rate_when_measured']:.0%} when measured.",
          f"- The 0/1 flag *alone* gives ROC-AUC {sh['auc_of_the_zero_flag_alone']:.3f}.",
          f"- Mean cholesterol with the zeros left in: {sh['mean_cholesterol_by_class_with_zeros']['1']:.0f} (disease) vs {sh['mean_cholesterol_by_class_with_zeros']['0']:.0f} (no disease) — the "
          f"notebook's 'heart patients have lower cholesterol'. With zeros treated as missing: {sh['mean_cholesterol_by_class_measured_only']['1']:.0f} vs "
          f"{sh['mean_cholesterol_by_class_measured_only']['0']:.0f} (Mann–Whitney p = {_p(sh['measured_only_mannwhitney_p'])}). The paradox was an artefact.", ""]
    L += ["### 1.3 The original pipeline, reproduced exactly", "",
          f"Same encoding, split (`random_state=10`, stratified; {orig['split']['train']} train / {orig['split']['test']} test) and scaling as the notebook.", ""]
    rows = [[m["model"], f"{m['cv_auc']:.3f}", f"{m['notebook_test_score (AUC of 0/1 predictions)']:.3f}", f"{m['balanced_accuracy']:.3f}",
             f"**{m['test_auc (probabilities)']:.3f}** ({_ci(m['test_auc_ci'])})"] for m in orig["models"]]
    L += _table(rows, ["Model", "CV ROC-AUC", "Notebook 'test_score'", "Balanced accuracy", "Actual test ROC-AUC (95% CI)"]) + [""]
    fb = orig["final_cell_bug"]
    L += ["- The notebook's 'test_score' is identical to balanced accuracy: `roc_auc_score` was given 0/1 predictions, not probabilities.",
          f"- The final 'best classifier' cell trained on `X_train` after it had been overwritten with the {len(fb['columns_used'])} dummy columns only. "
          f"It reproduces the notebook's 0.856 exactly ({fb['cv_accuracy_dummies_only']:.4f}); with all features: {fb['cv_accuracy_all_features']:.4f}.",
          "- With 184 test patients, the 95% intervals above overlap almost completely: the ranking between models is noise.", ""]
    ms = pd.DataFrame(a["model_screen"]["table"])
    if "class_balance" in a:
        cb = a["class_balance"]
        L += ["### 1.3b Class balance", "", f"{cb['positive_share']:.0%} of patients have heart disease (near-balanced overall, but 36%–93% by hospital). "
              "Class weighting tested on the same repeated CV folds:", ""]
        L += _table([[k, f"{v['auc']:.3f}", f"{v['balanced_accuracy']:.3f}"] for k, v in cb["cv"].items()], ["Model / weighting", "CV ROC-AUC", "CV balanced accuracy"]) + [""]
    L += ["### 1.4 Corrected model comparison", "",
          f"{len(ms)} models, {a['model_screen']['repeats']}×5-fold repeated stratified cross-validation, all preprocessing inside the folds "
          "(zeros → missing → median + missing flag). p-values: Nadeau–Bengio corrected resampled t-test against the best model, Holm-adjusted.", ""]
    L += _table([[r.model, f"{r.cv_auc:.3f} ± {r.cv_auc_sd:.3f}", f"{r.train_auc:.3f}", f"{r.brier:.3f}", f"{r.accuracy:.3f}", _p(r["p_vs_best (Holm)"])]
                 for _, r in ms.iterrows()], ["Model", "CV ROC-AUC ± SD", "Train ROC-AUC", "Brier", "Accuracy", "p vs best (Holm)"]) + [""]
    L += [f"**{len(a['model_screen']['indistinguishable_from_best'])} models are statistically indistinguishable from the best.**", ""]
    L += ["### 1.5 External validation: train on three hospitals, test on the fourth", ""]
    rows = []
    for site, v in a["leave_one_hospital_out"].items():
        for m, r in v["models"].items():
            rows.append([site, v["n"], f"{v['disease_rate']:.0%}", m, f"{r['auc']:.3f} ({_ci(r['auc_ci'])})", f"{r['mean_predicted']:.0%}", f"{r['calibration_intercept']:+.2f}", f"{r['accuracy_at_0.5']:.0%}"])
    L += _table(rows, ["Held-out hospital", "n", "Disease rate", "Model", "ROC-AUC (95% CI)", "Mean predicted", "Calibration intercept", "Accuracy"]) + [""]
    L += ["### 1.6 Statistical tests and odds ratios", ""]
    L += _table([[t["feature"], t["test"], _p(t["p_holm"]), f"{t['effect']} = {t['effect_size']:.2f}"] for t in a["association_tests"]],
                ["Feature", "Test", "p (Holm)", "Effect size"]) + [""]
    L += _table([[o["feature"], f"{o['odds_ratio']:.2f}", f"{o['ci'][0]:.2f}–{o['ci'][1]:.2f}", _p(o["p"])] for o in a["odds_ratios"]],
                ["Adjusted logistic model", "Odds ratio", "95% CI", "p"]) + [""]
    shp = a["shap_logistic"]
    L += ["SHAP (exact, logistic regression, mean |log-odds contribution|): " + ", ".join(f"{k} {v:.2f}" for k, v in shp["mean_abs_shap"].items())
          + f". Additivity error {shp['additivity_max_error']:.1e}.", ""]
    _fig_part1(a, fig)
    L += ["![Part 1](figures/part1_external.png)", ""]
    return L


def _fig_part1(a: dict, fig: Path) -> None:
    fig.mkdir(parents=True, exist_ok=True)
    lo = a["leave_one_hospital_out"]
    sites = list(lo)
    cv_best = max(r["cv_auc"] for r in a["model_screen"]["table"])
    f, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    m = "Logistic regression"
    auc = [lo[s]["models"][m]["auc"] for s in sites]
    err = np.array([[lo[s]["models"][m]["auc"] - lo[s]["models"][m]["auc_ci"][0], lo[s]["models"][m]["auc_ci"][1] - lo[s]["models"][m]["auc"]] for s in sites]).T
    ax[0].barh(sites, auc, xerr=err, color=BLUE)
    ax[0].axvline(cv_best, color=ORANGE, ls="--", label=f"pooled cross-validation ({cv_best:.3f})")
    ax[0].set(xlim=(0.5, 1), xlabel="ROC-AUC on the held-out hospital", title="Logistic regression trained on the other three hospitals")
    ax[0].legend(loc="lower left", fontsize=8)
    obs = [lo[s]["disease_rate"] for s in sites]
    pred = [lo[s]["models"][m]["mean_predicted"] for s in sites]
    y = np.arange(len(sites))
    ax[1].barh(y + 0.2, obs, 0.4, color=GREY, label="observed")
    ax[1].barh(y - 0.2, pred, 0.4, color=ORANGE, label="predicted (mean)")
    ax[1].set(yticks=y, yticklabels=sites, xlabel="share with heart disease", title="Calibration-in-the-large by hospital", xlim=(0, 1))
    ax[1].legend(fontsize=8)
    f.tight_layout()
    f.savefig(fig / "part1_external.png", bbox_inches="tight")
    plt.close(f)


# ------------------------------------------------------------------------------------ Part 2

def part2(res: Path, fig: Path) -> list[str]:
    t = _load("temporal.json", res)
    if t is None:
        return []
    cb, q, e, bm = _load("cohort_build.json", res), _load("quality.json", res), _load("eda.json", res), _load("benchmark.json", res)
    imb, sel, sv, lk = _load("imbalance.json", res), _load("select.json", res), _load("survival.json", res), _load("leakage.json", res)
    sg, ex = _load("subgroups.json", res), _load("explain.json", res)
    L = ["## Part 2 — 10-year cardiovascular death in US adults (NHANES 1999–2018 + National Death Index)", ""]
    if cb:
        L += ["### 2.1 Cohort", ""] + _table([[f"{s['remaining']:,}", s["step"], f"−{s['removed']:,}" if s["removed"] else ""] for s in cb["steps"]], ["Remaining", "Step", "Removed"]) + [""]
    if e:
        tab = e["baseline_table"]
        keys = list(next(iter(tab.values())).keys())
        L += ["Baseline characteristics (median (IQR) or %):", ""] + _table([[k] + [tab[s][k] for s in tab] for k in keys], ["", *tab.keys()]) + [""]
    if q:
        L += ["### 2.2 Data quality", ""]
        L += _table([[v, r["n"], f"{r['share_in_range']:.2%}", f"{r['min']:.1f}–{r['max']:.1f}"] for v, r in q["validity"].items()], ["Variable", "n", "Within plausible range", "Observed range"]) + [""]
        if "uniqueness" in q:
            L += ["Uniqueness: " + ", ".join(f"{k.replace('_', ' ')} {v:,}" for k, v in q["uniqueness"].items()) + ".", ""]
            L += _table([[k, f"{v:.2%}"] for k, v in q["consistency"].items()], ["Consistency check", "Share passing"]) + [""]
            ac = q["accuracy_cross_checks"]
            L += [f"Cross-checks: median HbA1c {ac['median HbA1c, diabetes vs not'][0]:.1f}% with diabetes vs {ac['median HbA1c, diabetes vs not'][1]:.1f}% without; "
                  f"median systolic BP {ac['median SBP, BP-treated vs not'][0]:.0f} treated vs {ac['median SBP, BP-treated vs not'][1]:.0f} untreated; "
                  f"BMI-waist correlation {ac['BMI-waist correlation']:.2f}.", ""]
        L += ["Missing values (share): " + ", ".join(f"{k} {v:.1%}" for k, v in q["missing_share"].items() if v > 0) + ".", ""]
        if "excluded_for_missing_core" in q:
            x = q["excluded_for_missing_core"]
            L += [f"Excluded for missing core risk factors: {x['n_excluded']:,} ({x['share']:.1%}). Standardised differences excluded vs included: "
                  + ", ".join(f"{k} {v:+.2f}" for k, v in x["smd_excluded_vs_included"].items())
                  + f". 10-year CVD death: excluded {x['cvd_death_10y_excluded']:.2%}, included {x['cvd_death_10y_included']:.2%}.", ""]
    if e:
        L += ["### 2.3 Exploratory analysis", "", f"10-year CVD death {e['event_rate']:.2%}; non-CVD death (competing) {e['noncvd_death_rate']:.2%} (n = {e['n_binary']:,}).", ""]
        L += _table([[r["pce"], r["n"], f"{r['mean_pce']:.1%}", f"{r['cvd_death_rate']:.2%}"] for r in e["by_pce_category"]],
                    ["PCE risk category", "n", "Mean PCE risk (fatal + non-fatal ASCVD)", "Observed 10-year CVD death"]) + [""]
        ut = e["univariable_tests"]
        L += _table([[k, v["test"], _p(v["p_holm"]), (f"P(event>non-event) {v['effect_P(event>non-event)']:.2f}" if "effect_P(event>non-event)" in v else f"risk ratio {v['risk_ratio']:.2f}")]
                     for k, v in ut.items()], ["Variable", "Test", "p (Holm)", "Effect size"]) + [""]
    if bm:
        tb = pd.DataFrame(bm["table"])
        L += ["### 2.4 Model screen (development cohort, cross-validation)", "", f"{bm['folds']} CV on {bm['n']:,} people ({bm['events']} events).", ""]
        L += _table([[r.model, f"{r.cv_auc:.3f} ± {r.cv_auc_sd:.3f}", f"{r.cv_pr_auc:.3f}", f"{r.cv_brier:.4f}", _p(r.p_vs_best_holm)] for _, r in tb.iterrows()],
                    ["Model", "CV ROC-AUC ± SD", "CV PR-AUC", "Brier", "p vs best (Holm)"]) + [""]
        _fig_benchmark(tb, fig)
        L += ["![benchmark](figures/benchmark.png)", ""]
    if imb:
        L += ["### 2.5 Class imbalance", ""] + _table([[k, f"{v['cv_auc']:.3f}", f"{v['cv_pr_auc']:.3f}", f"{v['cv_brier']:.4f}", f"{v['mean_predicted']:.1%}", f"{v['observed']:.1%}"]
                                                         for k, v in imb.items()], ["Model / strategy", "CV ROC-AUC", "CV PR-AUC", "Brier", "Mean predicted", "Observed"]) + [""]
    if sel:
        c = sel["comparison"]
        L += ["### 2.6 Model selection", "", f"Tuned logistic {sel['logistic']['best_params']} CV AUC {sel['logistic']['best_cv_auc']:.3f}; tuned gradient boosting "
              f"{sel['hgb']['best_params']} CV AUC {sel['hgb']['best_cv_auc']:.3f}. Difference {c['hgb_minus_logistic_auc']:+.4f} (corrected t p = {_p(c['p_corrected_t'])}). "
              f"Rule: {c['rule']}. **Chosen: {c['chosen']}.**", ""]
    L += ["### 2.7 Temporal validation (fit on 1999–2004, test once on 2005–2008)", "",
          f"Development {t['n_dev']:,} ({t['events_dev']} CVD deaths); test {t['n_test']:,} ({t['events_test']} CVD deaths). "
          f"PCE recalibration on development data: intercept {t['pce_recalibration']['intercept']:+.2f}, slope {t['pce_recalibration']['slope']:.2f}.", ""]
    rows = [[v["label"], f"{v['auc']:.3f} ({_ci(v['auc_ci'])})", f"{v['pr_auc']:.3f}", f"{v['brier']:.4f}", f"{v['cal_intercept']:+.2f}", f"{v['cal_slope']:.2f}",
             f"{v['o_e']:.2f}"] for v in t["models"].values()]
    L += _table(rows, ["Model", "ROC-AUC (95% CI)", "PR-AUC", "Brier", "Cal. intercept", "Cal. slope", "Observed / expected"]) + [""]
    L += _table([[t["models"][k]["label"], f"{d['diff']:+.3f} ({d['ci'][0]:+.3f} to {d['ci'][1]:+.3f})", _p(d["p"]), _p(d["p_holm"])] for k, d in t["vs_pce"].items()],
                ["vs published PCE", "ΔAUC (95% CI)", "DeLong p", "Holm-adjusted p"]) + [""]
    d = t["chosen_vs_pce_refit"]
    L += [f"Chosen model vs refitted PCE variables: ΔAUC {d['diff']:+.3f} ({d['ci'][0]:+.3f} to {d['ci'][1]:+.3f}), p = {_p(d['p'])}.", ""]
    cap = t["capture"]
    L += _table([[f"top {float(s):.0%}", *[f"{cap[s][k]['deaths_captured']:.0%}" for k in cap[s]]] for s in cap], ["Flag highest-risk", *[t["models"][k]["label"] for k in next(iter(cap.values()))]]) + [""]
    _fig_temporal(t, fig)
    L += ["![calibration and decision curves](figures/temporal_validation.png)", ""]
    if sv:
        L += ["### 2.8 Survival analysis (cause-specific Cox; other deaths censored)", ""]
        L += _table([[h["feature"], f"{h['hr_per_sd']:.2f}", f"{h['ci'][0]:.2f}–{h['ci'][1]:.2f}"] for h in sv["hazard_ratios"]], ["Predictor", "HR per SD", "95% CI"]) + [""]
        L += _table([[k, f"{v['n']:,}", v["cvd_deaths"], f"{v['median_followup_years']:.1f}", f"{v['c_index_cox']:.3f}", f"{v['c_index_pce']:.3f}", f"{v['c_index_age']:.3f}",
                      f"{v['overall_predicted']:.2%} vs {v['overall_observed_km']:.2%} ({v['horizon_months'] // 12}y)"] for k, v in sv["evaluation"].items()],
                    ["Cohort", "n", "CVD deaths", "Median follow-up (y)", "C-index Cox", "C-index PCE", "C-index age", "Predicted vs Kaplan–Meier"]) + [""]
    if lk:
        base = lk["correct (baseline variables, temporal split)"]["auc"]
        L += ["### 2.9 Leakage and design checks", ""] + _table([[k, f"{v['auc']:.3f}", f"{v['auc'] - base:+.3f}", f"{v['pr_auc']:.3f}"] for k, v in lk.items() if k != "note_prior_cvd"],
                                                                 ["Design", "ROC-AUC", "vs correct", "PR-AUC"]) + [""]
    if sg:
        ch = t["chosen"]
        L += ["### 2.10 Subgroups (temporal test)", ""] + _table([[g, v["n"], v["events"], f"{v['pce_auc']:.3f}", f"{v[ch + '_auc']:.3f}", f"{v['pce_recalibrated_o_e']:.2f}", f"{v[ch + '_o_e']:.2f}"]
                                                                  for g, v in sg.items()], ["Group", "n", "Events", "PCE AUC", "Chosen AUC", "O/E recalibrated PCE", "O/E chosen"]) + [""]
    if ex:
        L += ["### 2.11 Explanations (SHAP)", "", "Logistic regression, exact SHAP (log-odds), mean |contribution|: "
              + ", ".join(f"{k} {v:.3f}" for k, v in list(ex["logistic"]["mean_abs_shap"].items())[:12]) + f". Additivity error {ex['logistic']['additivity_max_error']:.1e}.", ""]
        if "mean_abs_shap" in ex.get("hgb", {}):
            L += ["Gradient boosting, TreeExplainer: " + ", ".join(f"{k} {v:.3f}" for k, v in list(ex["hgb"]["mean_abs_shap"].items())[:12]) + ".", ""]
        _fig_shap(ex, fig)
        L += ["![SHAP](figures/shap.png)", ""]
    return L


def _fig_benchmark(tb: pd.DataFrame, fig: Path) -> None:
    tb = tb[tb.model != "Prior only (no information)"].iloc[::-1]
    f, ax = plt.subplots(figsize=(7, 0.32 * len(tb) + 1))
    colors = [ORANGE if "Pooled" in m else BLUE for m in tb.model]
    ax.errorbar(tb.cv_auc, range(len(tb)), xerr=tb.cv_auc_sd, fmt="none", ecolor=GREY, lw=1)
    ax.scatter(tb.cv_auc, range(len(tb)), c=colors, zorder=3)
    ax.set(yticks=range(len(tb)), yticklabels=tb.model, xlabel="cross-validated ROC-AUC (± SD across folds)", title="Model screen, development cohort")
    f.tight_layout()
    f.savefig(fig / "benchmark.png", bbox_inches="tight")
    plt.close(f)


def _fig_temporal(t: dict, fig: Path) -> None:
    f, ax = plt.subplots(1, 2, figsize=(11, 4))
    ch = t["chosen"]
    for k, c in (("pce", GREY), ("pce_recalibrated", ORANGE), (ch, BLUE)):
        r = t["models"][k]["reliability"]
        ax[0].plot(np.array(r["predicted"]) * 100, np.array(r["observed"]) * 100, "o-", color=c, label=t["models"][k]["label"], ms=4)
    lim = max(max(t["models"][ch]["reliability"]["observed"]), max(t["models"][ch]["reliability"]["predicted"])) * 100 * 1.1
    ax[0].plot([0, lim], [0, lim], "--", color="black", lw=0.8)
    ax[0].set(xlim=(0, lim * 1.6), ylim=(0, lim), xlabel="predicted 10-year risk (%)", ylabel="observed 10-year CVD death (%)", title="Calibration (deciles), 2005–2008")
    ax[0].legend(fontsize=7)
    for k, c in (("pce", GREY), ("pce_recalibrated", ORANGE), (ch, BLUE)):
        dca = t["dca"].get(k)
        if dca:
            ax[1].plot(np.array(dca["threshold"]) * 100, np.array(dca["model"]) * 1000, "o-", color=c, label=t["models"][k]["label"], ms=3)
    any_dca = next(iter(t["dca"].values()))
    ax[1].plot(np.array(any_dca["threshold"]) * 100, np.array(any_dca["treat_all"]) * 1000, ":", color="black", label="treat everyone")
    ax[1].axhline(0, color="black", lw=0.6)
    ax[1].set(ylim=(-5, None), xlabel="risk threshold for action (%)", ylabel="net benefit per 1,000 people", title="Decision curves")
    ax[1].legend(fontsize=7)
    f.tight_layout()
    f.savefig(fig / "temporal_validation.png", bbox_inches="tight")
    plt.close(f)


def _fig_shap(ex: dict, fig: Path) -> None:
    s = pd.Series(ex["logistic"]["mean_abs_shap"]).sort_values().tail(12)
    f, ax = plt.subplots(figsize=(6, 4))
    ax.barh(s.index, s.values, color=BLUE)
    ax.set(xlabel="mean |SHAP| (log-odds)", title="What drives predicted risk (logistic model)")
    f.tight_layout()
    f.savefig(fig / "shap.png", bbox_inches="tight")
    plt.close(f)


def main(res: Path = config.RESULTS, rep: Path = config.REPORTS) -> None:
    fig = rep / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    L = ["# Results (generated by `python -m cvd_risk.report`; do not edit by hand)", ""]
    L += part1(res, fig)
    L += part2(res, fig)
    (rep / "results.md").write_text("\n".join(L) + "\n")
    print("wrote", rep / "results.md")


if __name__ == "__main__":
    main()
