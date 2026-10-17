# Model card: 10-year cardiovascular death risk (NHANES)

## Model
- **Type:** logistic regression (L2, C = 1.0) on 20 baseline predictors.
- **Preprocessing:**
  - age enters as a cubic spline (4 knots);
  - continuous predictors are standardised;
  - missing values get median imputation plus missing-value flags, fitted on training data only.
- **One scikit-learn `Pipeline`**, saved as `results/models.joblib`. It is not committed; the code regenerates it.
- **Why this model:** chosen by a rule fixed before testing. Gradient boosting would replace logistic regression only if it beat it by more than 0.005 cross-validated ROC-AUC with p < 0.05. It didn't (−0.002, p = 0.87), and on the test period it was significantly worse (−0.024, p = 0.007).

## Intended use
- **Purpose:** ranking and estimating 10-year risk of **death from heart disease or stroke** for adults aged 40–79 **without** known cardiovascular disease, as a research and portfolio demonstration of risk-model development and validation.
- **Not for:**
  - clinical decisions about individual patients;
  - people with existing heart disease or stroke;
  - ages outside 40–79;
  - predicting non-fatal events (it was never trained on them);
  - populations outside the US.

## Training and evaluation data
- **Source:** NHANES 1999–2018 linked to the National Death Index (NCHS public-use files, follow-up to 31 December 2019).
- **Development:** 1999–2004 cycles. 6,081 people, 188 cardiovascular deaths within 10 years (3.1%).
- **Temporal test:** 2005–2008 cycles, used once. 4,724 people, 133 deaths (2.8%).
- **Recent check:** 2009–2018 cycles. 13,083 people, used for a survival analysis because follow-up is incomplete (median 5.8 years).

## Performance (temporal test, 2005–2008)

| Metric | Value |
|---|---|
| ROC-AUC | 0.822 (95% CI 0.791–0.854); published Pooled Cohort Equations 0.799 |
| PR-AUC | 0.140 (prevalence 0.028) |
| Calibration | observed/expected 1.01, intercept +0.01, slope 1.00 |
| Brier score | 0.0258 |
| Top 10% by risk | captures 47% of cardiovascular deaths (PCE: 36%) |
| 2009–2018 (Cox model, same predictors) | C-index 0.807; predicted 5-year risk 1.01% vs 0.93% observed |

The ranking gain over the Pooled Cohort Equations is **not statistically robust**: Holm-adjusted p = 0.13.

## Subgroup performance (temporal test)

| Group | ROC-AUC | Observed / expected |
|---|---|---|
| Women | 0.851 | 1.10 |
| Men | 0.791 | 0.95 |
| Non-Hispanic Black | 0.830 | 0.97 |
| Hispanic | 0.760 | **0.71** (over-predicts) |
| Other, mostly non-Hispanic White | 0.834 | 1.16 |
| Age 60–79 | 0.744 | 1.00 |
| Diabetes | 0.745 | **0.84** (over-predicts) |

Subgroups have few events (21–103), so these estimates are imprecise.

## Ethical considerations
- **Race and ethnicity are predictors**, as in the Pooled Cohort Equations. Here they mostly adjust for calibration differences, such as lower observed cardiovascular mortality among Hispanic adults. Using race in clinical algorithms is contested; any deployment would need to examine that trade-off explicitly.
- **Income and education are predictors.** They improve prediction but encode social disadvantage. A model like this should inform outreach and support, never deny care.
- **Complete-case analysis excluded 9% of eligible adults.** They were disproportionately Black and lower-income, and had higher mortality, so performance may be overstated for those groups.

## Caveats
- **Explanations:** SHAP explains the model, not biology. Correlated predictors (waist and BMI) share and trade credit.
- **Data perturbation:** the public-use mortality files are perturbed for confidentiality, which slightly blurs cause and timing of death for some people.
- **Retraining:** predictor definitions, measurement methods and treatment patterns change over time; lipid and BP-treatment rates rose across cycles. Recalibrate before reusing the model in a new period.
