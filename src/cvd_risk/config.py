"""Every analytical choice in one place, so a reviewer can audit it without reading the code."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"                              # downloaded CDC files (not committed)
COHORT_CSV = DATA / "nhanes_cvd_cohort.csv.gz"  # analytic cohort (public-domain, committed)
HARMONISED_CSV = DATA / "nhanes_adults_harmonised.csv.gz"  # all examined adults, before exclusions
HEART918_CSV = DATA / "heart_failure_918.csv"
CLEVELAND_CSV = DATA / "uci_cleveland_297.csv"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
SEED = 42

# ----------------------------------------------------------------------------- NHANES files
# Continuous NHANES 1999-2018: ten two-year cycles. The file suffix letter changes per cycle.
CYCLES = {1999: "", 2001: "_B", 2003: "_C", 2005: "_D", 2007: "_E",
          2009: "_F", 2011: "_G", 2013: "_H", 2015: "_I", 2017: "_J"}

# Laboratory files were renamed in 2005. Candidates are tried in order; the first that exists is used.
LAB_FILES = {
    "tchol": {1999: ["LAB13"], 2001: ["L13_B"], 2003: ["L13_C"]},      # later: TCHOL_x
    "hdl": {1999: ["LAB13"], 2001: ["L13_B"], 2003: ["L13_C"]},        # later: HDL_x
    "ghb": {1999: ["LAB10"], 2001: ["L10_B"], 2003: ["L10_C"]},        # later: GHB_x
    "biopro": {1999: ["LAB18"], 2001: ["L40_B"], 2003: ["L40_C"]},     # later: BIOPRO_x
}
LAB_LATER = {"tchol": "TCHOL", "hdl": "HDL", "ghb": "GHB", "biopro": "BIOPRO"}
STANDARD_FILES = ["DEMO", "BPX", "BPQ", "SMQ", "DIQ", "MCQ", "BMX"]

# CDC moved NHANES files in 2024; the old location is kept as a fallback.
URL_PATTERNS = [
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{start}/DataFiles/{name}.xpt",
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{start}/DataFiles/{name}.XPT",
    "https://wwwn.cdc.gov/Nchs/Nhanes/{start}-{end}/{name}.XPT",
]

# NCHS public-use Linked Mortality Files (National Death Index linkage, follow-up to 31 Dec 2019).
LMF_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_{start}_{end}_MORT_2019_PUBLIC.dat"
LMF_READIN_R = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/R_ReadInProgramAllSurveys.R"
# Fixed-width layout of the 2019 public-use NHANES files (from the NCHS R read-in program).
# The pipeline re-reads the layout from NCHS's own program when it can, and checks these against it.
LMF_LAYOUT = {"seqn": (1, 6), "eligstat": (15, 15), "mortstat": (16, 16), "ucod_leading": (17, 19),
              "diabetes_flag": (20, 20), "hyperten_flag": (21, 21), "permth_int": (43, 45), "permth_exm": (46, 48)}
FOLLOWUP_END = "2019-12-31"
# UCOD_LEADING: 001 = diseases of heart (I00-I09, I11, I13, I20-I51), 005 = cerebrovascular (I60-I69)
CVD_CAUSES = (1, 5)

# ----------------------------------------------------------------------------- cohort
AGE_MIN, AGE_MAX = 40, 79           # the age range the Pooled Cohort Equations were built for
HORIZON_MONTHS = 120                # 10-year risk
DEV_CYCLES = (1999, 2001, 2003)     # development: >= 15 years of follow-up
TEST_CYCLES = (2005, 2007)          # temporal test: complete 10-year follow-up by end of 2019
RECENT_CYCLES = (2009, 2011, 2013, 2015, 2017)  # incomplete follow-up: survival analysis only

# Predictors measured at the baseline examination (nothing from after it).
PCE_INPUTS = ["age", "female", "black", "tchol", "hdl", "sbp", "bp_treated", "smoker", "diabetes"]
FEATURES = ["age", "female", "race_black", "race_hispanic", "race_other",
            "tchol", "hdl", "sbp", "dbp", "bp_treated", "lipid_treated", "smoker", "former_smoker",
            "diabetes", "hba1c", "bmi", "waist", "egfr", "pir", "education_college"]
CORE_REQUIRED = ["tchol", "hdl", "sbp", "bp_treated", "smoker", "diabetes"]

# Death-certificate flags from the mortality file: recorded AFTER death, so using them as
# predictors is leakage. They are kept only for the leakage demonstration.
LEAKY_FEATURES = ["diabetes_flag_dc", "hyperten_flag_dc"]

N_BOOT = 1000
CV_REPEATS, CV_FOLDS = 5, 5
DCA_THRESHOLDS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]
