"""Download, read and harmonise NHANES 1999-2018 and link it to the NCHS public-use mortality files.

Pipeline:  download()  ->  harmonise()  (one row per examined adult, same variable names in every cycle)
           ->  link_mortality()  ->  build_cohort()  (exclusions counted step by step)

Design rules
- Variables are harmonised explicitly per cycle (CDC renamed several files and variables over time);
  the cycle and source file of every variable are recorded in a provenance table.
- Survey answers coded 7 / 9 (refused / don't know) become missing, never "no".
- Serum creatinine is recalibrated for 1999-2000 and 2005-2006, as CDC recommends.
- Only information available at the baseline examination is used as a predictor.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# --------------------------------------------------------------------------------- download

def file_candidates(start: int, key: str) -> list[str]:
    suf = config.CYCLES[start]
    if key in config.LAB_FILES:
        return config.LAB_FILES[key].get(start, [config.LAB_LATER[key] + suf])
    return [key + suf]


def _fetch(url: str, timeout: int = 120) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research download)"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 * (attempt + 1))
    return None


def _is_xpt(blob: bytes | None) -> bool:
    return bool(blob) and blob[:80].startswith(b"HEADER RECORD")


def download_xpt(start: int, key: str, raw: Path = config.RAW) -> tuple[Path | None, str]:
    """Return (local path, source URL). Cached: an existing valid file is never downloaded again."""
    for name in file_candidates(start, key):
        local = raw / str(start) / f"{name}.xpt"
        if local.exists() and _is_xpt(local.read_bytes()[:80]):
            return local, "cache"
        for pattern in config.URL_PATTERNS:
            url = pattern.format(start=start, end=start + 1, name=name)
            blob = _fetch(url)
            if _is_xpt(blob):      # CDC sometimes answers a missing file with an HTML page and status 200
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(blob)
                return local, url
    return None, ""


def download_lmf(start: int, raw: Path = config.RAW) -> Path:
    local = raw / "mortality" / f"NHANES_{start}_{start + 1}_MORT_2019_PUBLIC.dat"
    if local.exists() and local.stat().st_size > 1000:
        return local
    blob = _fetch(config.LMF_URL.format(start=start, end=start + 1))
    if not blob or blob[:20].lstrip().startswith(b"<"):
        raise RuntimeError(f"Could not download the mortality file for {start}-{start + 1} from {config.LMF_URL.format(start=start, end=start + 1)}")
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(blob)
    return local


def download(raw: Path = config.RAW, cycles=tuple(config.CYCLES)) -> dict:
    """Download every file needed. Returns a manifest (file -> URL) and fails loudly on gaps."""
    manifest, missing = {}, []
    keys = config.STANDARD_FILES + list(config.LAB_FILES)
    for start in cycles:
        for key in keys:
            path, url = download_xpt(start, key, raw)
            manifest[f"{start}/{key}"] = {"file": path.name if path else None, "url": url}
            if path is None:
                missing.append(f"{start}-{start + 1} {key} (tried {file_candidates(start, key)})")
        manifest[f"{start}/mortality"] = {"file": download_lmf(start, raw).name}
        print(f"  {start}-{start + 1}: downloaded", flush=True)
    rprog = _fetch(config.LMF_READIN_R)
    if rprog:
        (raw / "mortality" / "R_ReadInProgramAllSurveys.R").write_bytes(rprog)
    if missing:
        raise RuntimeError("Some NHANES files could not be found:\n  " + "\n  ".join(missing))
    return manifest


# --------------------------------------------------------------------------------- reading

def read_xpt(path: Path, columns: list[str]) -> pd.DataFrame:
    df = pd.read_sas(path, format="xport", encoding="latin-1")
    df.columns = [c.upper() for c in df.columns]
    # SAS transport files store 0 in a way pandas reads back as 5.4e-79; restore true zeros so that
    # checks such as "diastolic reading of 0 = invalid" work.
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-30, 0.0)
    keep = ["SEQN"] + [c for c in columns if c in df.columns]
    out = df[keep].copy()
    for c in columns:
        if c not in out.columns:
            out[c] = np.nan
    out["SEQN"] = out["SEQN"].astype("int64")
    return out


VARS = {
    "DEMO": ["RIDSTATR", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2", "INDFMPIR", "RIDEXPRG", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"],
    "BPX": ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"],
    "BPQ": ["BPQ020", "BPQ040A", "BPQ050A", "BPQ080", "BPQ090D", "BPQ100D"],
    "SMQ": ["SMQ020", "SMQ040"],
    "DIQ": ["DIQ010", "DIQ050", "DIQ070"],
    "MCQ": ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"],
    "BMX": ["BMXBMI", "BMXWAIST"],
    "tchol": ["LBXTC"],
    "hdl": ["LBDHDL", "LBXHDD", "LBDHDD"],    # HDL variable was renamed twice
    "ghb": ["LBXGH"],
    "biopro": ["LBXSCR"],
}


def read_cycle(start: int, raw: Path = config.RAW) -> tuple[pd.DataFrame, dict]:
    """Merge all files of one cycle on SEQN (left join on demographics). Returns data and provenance."""
    prov = {}
    base = None
    for key, cols in VARS.items():
        path = next((raw / str(start) / f"{n}.xpt" for n in file_candidates(start, key)
                     if (raw / str(start) / f"{n}.xpt").exists()), None)
        if path is None:
            raise FileNotFoundError(f"{start}: no file for {key}; run download() first")
        df = read_xpt(path, cols)
        prov[key] = {"file": path.name, "found": [c for c in cols if df[c].notna().any()]}
        if base is None:
            base = df
        else:
            dup = [c for c in cols if c in base.columns]
            base = base.merge(df.drop(columns=dup), on="SEQN", how="left")
    base["cycle"] = start
    return base, prov


# --------------------------------------------------------------------------------- harmonise

def _yes_no(s: pd.Series) -> pd.Series:
    """1 = yes, 2 = no; 7 / 9 (refused / don't know) and blanks -> missing."""
    return s.map({1: 1.0, 2: 0.0})


def recalibrate_creatinine(scr: pd.Series, cycle: int) -> pd.Series:
    """CDC-recommended corrections to the standardised (IDMS-traceable) creatinine method."""
    if cycle == 1999:
        return 1.013 * scr + 0.147
    if cycle == 2005:
        return -0.016 + 0.978 * scr
    return scr


def egfr_ckd_epi_2021(scr: pd.Series, age: pd.Series, female: pd.Series) -> pd.Series:
    """CKD-EPI 2021 creatinine equation (race-free), mL/min/1.73 m^2."""
    kappa = np.where(female == 1, 0.7, 0.9)
    alpha = np.where(female == 1, -0.241, -0.302)
    ratio = scr / kappa
    e = 142 * np.minimum(ratio, 1) ** alpha * np.maximum(ratio, 1) ** -1.200 * 0.9938 ** age * np.where(female == 1, 1.012, 1.0)
    return pd.Series(e, index=scr.index)


def harmonise(raw_cycle: pd.DataFrame) -> pd.DataFrame:
    """Raw NHANES columns -> analysis variables with the same meaning in every cycle."""
    r = raw_cycle
    cycle = int(r["cycle"].iloc[0])
    out = pd.DataFrame({"seqn": r["SEQN"], "cycle": r["cycle"]})
    out["examined"] = (r["RIDSTATR"] == 2).astype(int)
    out["age"] = r["RIDAGEYR"]
    out["female"] = (r["RIAGENDR"] == 2).astype(int)
    eth = r["RIDRETH1"]
    out["race_black"] = (eth == 4).astype(int)
    out["race_hispanic"] = eth.isin([1, 2]).astype(int)
    out["race_other"] = (eth == 5).astype(int)
    out["black"] = out["race_black"]
    out["education_college"] = r["DMDEDUC2"].map({1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0, 5: 1.0})
    out["pir"] = r["INDFMPIR"]
    out["pregnant"] = (r["RIDEXPRG"] == 1).astype(int)
    out["weight_mec"] = r["WTMEC2YR"]
    out["psu"], out["stratum"] = r["SDMVPSU"], r["SDMVSTRA"]

    sys_ = r[["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]].where(lambda d: d > 0)
    dia = r[["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]].where(lambda d: d > 0)   # DBP of 0 = sounds heard to zero
    out["sbp"], out["dbp"] = sys_.mean(axis=1), dia.mean(axis=1)
    out["n_bp_readings"] = sys_.notna().sum(axis=1)

    told_htn, rx_told, rx_now = _yes_no(r["BPQ020"]), _yes_no(r["BPQ040A"]), _yes_no(r["BPQ050A"])
    out["bp_treated"] = np.where(rx_now == 1, 1.0, np.where((told_htn == 0) | (rx_told == 0) | (rx_now == 0), 0.0, np.nan))
    told_chol, lrx_told, lrx_now = _yes_no(r["BPQ080"]), _yes_no(r["BPQ090D"]), _yes_no(r["BPQ100D"])
    out["lipid_treated"] = np.where(lrx_now == 1, 1.0, np.where((told_chol == 0) | (lrx_told == 0) | (lrx_now == 0), 0.0, np.nan))

    ever = _yes_no(r["SMQ020"])
    now = r["SMQ040"].map({1: 1.0, 2: 1.0, 3: 0.0})
    out["smoker"] = np.where(ever == 0, 0.0, np.where((ever == 1) & now.notna(), now, np.nan))
    out["former_smoker"] = np.where(ever == 0, 0.0, np.where((ever == 1) & (now == 0), 1.0, np.where(now == 1, 0.0, np.nan)))

    out["tchol"] = r["LBXTC"]
    out["hdl"] = r["LBDHDL"].fillna(r["LBXHDD"]).fillna(r["LBDHDD"])
    out["hba1c"] = r["LBXGH"]
    dm_told = r["DIQ010"].map({1: 1.0, 2: 0.0, 3: 0.0})          # 3 = borderline -> not diabetes
    insulin, pills = _yes_no(r["DIQ050"]), _yes_no(r["DIQ070"])
    dm_yes = (dm_told == 1) | (insulin == 1) | (pills == 1) | (out["hba1c"] >= 6.5)
    dm_known = dm_told.notna() | out["hba1c"].notna()
    out["diabetes"] = np.where(dm_yes, 1.0, np.where(dm_known, 0.0, np.nan))

    mcq = r[["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]].apply(_yes_no)
    out["prior_cvd"] = np.where((mcq == 1).any(axis=1), 1.0, np.where(mcq.notna().any(axis=1), 0.0, np.nan))

    out["bmi"], out["waist"] = r["BMXBMI"], r["BMXWAIST"]
    scr = recalibrate_creatinine(r["LBXSCR"], cycle)
    out["creatinine"] = scr
    out["egfr"] = egfr_ckd_epi_2021(scr, out["age"], out["female"])
    return out


# --------------------------------------------------------------------------------- mortality

def read_lmf(path: Path, layout: dict = config.LMF_LAYOUT) -> pd.DataFrame:
    names = list(layout)
    colspecs = [(a - 1, b) for a, b in layout.values()]
    df = pd.read_fwf(path, colspecs=colspecs, names=names, na_values=["."], dtype=str)
    for c in names:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["seqn"] = df["seqn"].astype("int64")
    return df


def check_lmf(lmf: pd.DataFrame, seqns: pd.Series) -> dict:
    """The layout is right only if these hold. Fail loudly instead of producing silent nonsense."""
    checks = {
        "linked to a survey participant": float(lmf["seqn"].isin(seqns).mean()),
        "eligstat in {1,2,3}": float(lmf["eligstat"].isin([1, 2, 3]).mean()),
        "eligible: mortstat in {0,1}": float(lmf.loc[lmf.eligstat == 1, "mortstat"].isin([0, 1]).mean()),
        "deaths: cause code 1-10": float(lmf.loc[lmf.mortstat == 1, "ucod_leading"].between(1, 10).mean()),
        "eligible: follow-up months 0-300": float(lmf.loc[lmf.eligstat == 1, "permth_exm"].dropna().between(0, 300).mean()),
    }
    bad = {k: v for k, v in checks.items() if v < 0.99}
    if bad:
        raise ValueError(f"Mortality file layout checks failed: {bad}")
    return checks


def link_mortality(adults: pd.DataFrame, lmf: pd.DataFrame) -> pd.DataFrame:
    m = lmf.rename(columns={"diabetes_flag": "diabetes_flag_dc", "hyperten_flag": "hyperten_flag_dc"})
    out = adults.merge(m[["seqn", "eligstat", "mortstat", "ucod_leading", "diabetes_flag_dc", "hyperten_flag_dc", "permth_exm"]],
                       on="seqn", how="left")
    out["linkage_eligible"] = (out["eligstat"] == 1).astype(int)
    out["months"] = out["permth_exm"]
    out["died"] = (out["mortstat"] == 1).astype(int)
    out["cvd_death"] = ((out["mortstat"] == 1) & out["ucod_leading"].isin(config.CVD_CAUSES)).astype(int)
    h = config.HORIZON_MONTHS
    out["cvd_death_10y"] = ((out["cvd_death"] == 1) & (out["months"] <= h)).astype(int)
    out["noncvd_death_10y"] = ((out["died"] == 1) & (out["cvd_death"] == 0) & (out["months"] <= h)).astype(int)
    out["complete_10y"] = ((out["months"] >= h) | ((out["died"] == 1) & (out["months"] <= h))).astype(int)
    return out


# --------------------------------------------------------------------------------- cohort

def build_cohort(linked: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the eligibility rules in order, counting what each removes."""
    steps = []
    d = linked
    def keep(mask, label):
        nonlocal d
        before = len(d)
        d = d[mask(d)]
        steps.append({"step": label, "removed": before - len(d), "remaining": len(d)})
    steps.append({"step": "adults in NHANES 1999-2018 (all ages, interviewed)", "removed": 0, "remaining": len(d)})
    keep(lambda x: x["examined"] == 1, "examined at the mobile examination centre")
    keep(lambda x: x["age"].between(config.AGE_MIN, config.AGE_MAX), f"aged {config.AGE_MIN}-{config.AGE_MAX}")
    keep(lambda x: x["pregnant"] == 0, "not pregnant")
    keep(lambda x: x["linkage_eligible"] == 1, "eligible for mortality linkage")
    keep(lambda x: x["prior_cvd"] == 0, "no self-reported heart disease or stroke (prevention population)")
    keep(lambda x: x[config.CORE_REQUIRED].notna().all(axis=1), "core risk factors measured (cholesterol, HDL, BP, BP treatment, smoking, diabetes)")
    keep(lambda x: x["months"].notna(), "follow-up time recorded")
    d = d.copy()
    d["split"] = np.select([d.cycle.isin(config.DEV_CYCLES), d.cycle.isin(config.TEST_CYCLES)], ["development", "temporal_test"], "recent")
    return d.reset_index(drop=True), steps


def build_all(raw: Path = config.RAW, cycles=tuple(config.CYCLES)) -> dict:
    """Read every cycle, harmonise, link and build the cohort. Writes the committed CSVs."""
    frames, provenance, lmf_checks = [], {}, {}
    for start in cycles:
        cyc, prov = read_cycle(start, raw)
        provenance[start] = prov
        h = harmonise(cyc)
        lmf = read_lmf(raw / "mortality" / f"NHANES_{start}_{start + 1}_MORT_2019_PUBLIC.dat")
        lmf_checks[start] = check_lmf(lmf, h["seqn"])
        frames.append(link_mortality(h, lmf))
    rfile = raw / "mortality" / "R_ReadInProgramAllSurveys.R"
    layout_check = None
    if rfile.exists():
        nchs = lmf_layout_from_r(rfile.read_text(errors="ignore"))
        alias = {"diabetes_flag": "diabetes", "hyperten_flag": "hyperten"}     # NCHS names for the same columns
        layout_check = {k: {"ours": list(v), "nchs_program": nchs.get(alias.get(k, k), [])} for k, v in config.LMF_LAYOUT.items()}
    adults = pd.concat(frames, ignore_index=True)
    adults = adults[adults["age"] >= 18].reset_index(drop=True)
    cohort, steps = build_cohort(adults)
    return {"adults": adults, "cohort": cohort, "steps": steps, "provenance": provenance, "lmf_checks": lmf_checks,
            "lmf_layout_vs_nchs": layout_check}


def save(built: dict, data: Path = config.DATA, results: Path = config.RESULTS) -> None:
    data.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    built["adults"].to_csv(data / config.HARMONISED_CSV.name, index=False, compression="gzip", float_format="%.4g")
    built["cohort"].to_csv(data / config.COHORT_CSV.name, index=False, compression="gzip", float_format="%.4g")
    meta = {"steps": built["steps"], "provenance": {str(k): v for k, v in built["provenance"].items()},
            "lmf_checks": {str(k): v for k, v in built["lmf_checks"].items()},
            "lmf_layout_vs_nchs": built.get("lmf_layout_vs_nchs")}
    (results / "cohort_build.json").write_text(json.dumps(meta, indent=2))


def lmf_layout_from_r(text: str) -> dict:
    """Read the NHANES column layout from NCHS's own R read-in program (to cross-check LMF_LAYOUT)."""
    import re
    found = {}
    for name, a, b in re.findall(r"(\w+)\s*=\s*c\((\d+)\s*,\s*(\d+)\)", text):
        found.setdefault(name.lower(), set()).add((int(a), int(b)))
    return {k: sorted(v) for k, v in found.items()}
