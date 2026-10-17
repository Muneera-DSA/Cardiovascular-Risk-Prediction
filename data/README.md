# Data

## Part 2: NHANES 1999–2018 linked to the National Death Index (public domain)

| File | What it is | Rows |
|---|---|---|
| `nhanes_cvd_cohort.csv.gz` | Analytic cohort: adults 40–79, examined, no prior heart disease or stroke, eligible for mortality linkage, core risk factors measured | see `results/cohort_build.json` |
| `nhanes_adults_harmonised.csv.gz` | Every interviewed adult (18+) after harmonisation and linkage, before exclusions (used for the selection-bias and leakage checks) | |

Both files are built by `python -m cvd_risk.build_data` (or `notebooks/build_cohort_on_colab.ipynb`) from these official sources:

- **NHANES** (National Health and Nutrition Examination Survey), CDC / National Center for Health Statistics, continuous cycles 1999–2000 to 2017–2018:
  - Demographics (`DEMO`).
  - Examinations: blood pressure (`BPX`), body measures (`BMX`).
  - Laboratory:
    - Total cholesterol: `LAB13` / `L13_B` / `L13_C`, then `TCHOL_x`.
    - HDL: same files, then `HDL_x`.
    - HbA1c: `LAB10` / `L10_B` / `L10_C`, then `GHB_x`.
    - Serum creatinine: `LAB18` / `L40_B` / `L40_C`, then `BIOPRO_x`.
  - Questionnaires: blood pressure and cholesterol (`BPQ`), smoking (`SMQ`), diabetes (`DIQ`), medical conditions (`MCQ`).
  - https://wwwn.cdc.gov/nchs/nhanes/
- **NCHS 2019 Public-Use Linked Mortality Files.** NHANES participants linked to the National Death Index, with follow-up to 31 December 2019, the underlying cause of death (`UCOD_LEADING`) and months of follow-up from the examination. https://www.cdc.gov/nchs/data-linkage/mortality-public.htm

NHANES and the linked mortality files are US-government works in the public domain.

**Outcome:** death from cardiovascular disease within 10 years of the examination. That is, `UCOD_LEADING` 001 (diseases of the heart, ICD-10 I00–I09, I11, I13, I20–I51) or 005 (cerebrovascular diseases, I60–I69).

**Things a user of these files must know**

- **Public-use mortality files are perturbed.** For a small share of decedents, NCHS replaced cause and date of death with synthetic values to protect confidentiality. Aggregate analyses are designed to be unaffected, but individual records may not be exact.
- **The diabetes and hypertension flags in the mortality file come from the death certificate.** They are recorded after death, so they are never used as predictors. Section 2.9 of `reports/results.md` shows what using them would do.
- **NHANES has a complex survey design.** Sample weights (`weight_mec`) and design variables (`psu`, `stratum`) are kept. Model development and validation are unweighted, as is usual for prediction modelling; survey-weighted prevalences are reported for comparison.
- **Serum creatinine is recalibrated** for 1999–2000 and 2005–2006 as CDC recommends, before computing eGFR with the race-free CKD-EPI 2021 equation.

## Part 1: UCI Heart Disease, four hospitals (CC BY 4.0)

`uci_heart_4_hospitals.csv` contains the processed files from the UCI Machine Learning Repository (Cleveland, Hungarian, Switzerland, Long Beach VA), with a `site` column added. Downloaded by `python -m cvd_risk.build_data`.

> Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X. Licence: CC BY 4.0.

The original notebook used the Kaggle file *Heart Failure Prediction* (fedesoriano, 918 rows). That file merges these four hospitals and Statlog (a duplicate of Cleveland), then removes duplicates, but it drops the hospital column. Its licence could not be confirmed, so it is **not** redistributed here.

To reproduce Part 1 on exactly the 918 rows, download `heart.csv` from https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction, save it as `data/heart_failure_918.csv` and run `python -m cvd_risk.audit918`. Each row is matched to its hospital by exact comparison with the UCI files.
