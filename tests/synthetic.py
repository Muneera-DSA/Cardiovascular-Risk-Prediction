"""Synthetic NHANES-like files (same column names and codes as the CDC files) for offline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cvd_risk import config, nhanes


def raw_cycle(start: int, n: int = 400, seed: int = 0) -> dict[str, pd.DataFrame]:
    """One cycle as the per-file frames read_xpt would return (upper-case CDC names)."""
    rng = np.random.default_rng(seed + start)
    seqn = np.arange(start * 100, start * 100 + n)
    age = rng.integers(18, 86, n)
    female = rng.random(n) < 0.5
    files = {
        "DEMO": pd.DataFrame({"SEQN": seqn, "RIDSTATR": rng.choice([1, 2], n, p=[0.05, 0.95]), "RIAGENDR": np.where(female, 2, 1),
                              "RIDAGEYR": age, "RIDRETH1": rng.choice([1, 2, 3, 4, 5], n), "DMDEDUC2": rng.choice([1, 2, 3, 4, 5, 9], n),
                              "INDFMPIR": rng.uniform(0, 5, n), "RIDEXPRG": np.where(female & (age < 45), rng.choice([1, 2], n, p=[0.05, 0.95]), np.nan),
                              "WTMEC2YR": rng.uniform(5000, 50000, n), "SDMVPSU": rng.choice([1, 2], n), "SDMVSTRA": rng.integers(1, 15, n)}),
        "BPX": pd.DataFrame({"SEQN": seqn, **{f"BPXSY{k}": rng.normal(125 + 0.4 * (age - 50), 15, n).round() for k in (1, 2, 3)},
                             "BPXSY4": np.nan, **{f"BPXDI{k}": rng.normal(75, 10, n).round() for k in (1, 2, 3)}, "BPXDI4": np.nan}),
        "BPQ": pd.DataFrame({"SEQN": seqn, "BPQ020": rng.choice([1, 2, 9], n, p=[0.35, 0.63, 0.02]), "BPQ040A": rng.choice([1, 2], n),
                             "BPQ050A": rng.choice([1, 2], n), "BPQ080": rng.choice([1, 2], n), "BPQ090D": rng.choice([1, 2], n), "BPQ100D": rng.choice([1, 2], n)}),
        "SMQ": pd.DataFrame({"SEQN": seqn, "SMQ020": rng.choice([1, 2, 7], n, p=[0.45, 0.54, 0.01]), "SMQ040": rng.choice([1, 2, 3], n)}),
        "DIQ": pd.DataFrame({"SEQN": seqn, "DIQ010": rng.choice([1, 2, 3, 9], n, p=[0.1, 0.85, 0.04, 0.01]), "DIQ050": rng.choice([1, 2], n, p=[0.05, 0.95]),
                             "DIQ070": rng.choice([1, 2], n, p=[0.08, 0.92])}),
        "MCQ": pd.DataFrame({"SEQN": seqn, **{f"MCQ160{c}": rng.choice([1, 2, 9], n, p=[0.03, 0.96, 0.01]) for c in "BCDEF"}}),
        "BMX": pd.DataFrame({"SEQN": seqn, "BMXBMI": rng.normal(28, 5, n), "BMXWAIST": rng.normal(98, 12, n)}),
        "tchol": pd.DataFrame({"SEQN": seqn, "LBXTC": rng.normal(200, 35, n).round()}),
        "hdl": pd.DataFrame({"SEQN": seqn, ("LBDHDL" if start <= 2001 else "LBXHDD" if start == 2003 else "LBDHDD"): rng.normal(52, 14, n).clip(15).round()}),
        "ghb": pd.DataFrame({"SEQN": seqn, "LBXGH": rng.normal(5.6, 0.7, n).round(1)}),
        "biopro": pd.DataFrame({"SEQN": seqn, "LBXSCR": rng.normal(0.9, 0.2, n).clip(0.4)}),
    }
    for f in files.values():   # a few blanks everywhere, like the real files
        cols = [c for c in f.columns if c != "SEQN"]
        f.loc[rng.random(n) < 0.03, cols[0]] = np.nan
    return files


def lmf_text(seqns, start: int, seed: int = 0) -> str:
    """Fixed-width mortality file in the 2019 public-use layout."""
    rng = np.random.default_rng(seed + start)
    lines = []
    max_months = (2019 - start) * 12 + 6
    for s in seqns:
        line = [" "] * 48
        def put(name, value):
            a, b = config.LMF_LAYOUT[name]
            text = str(value).rjust(b - a + 1)
            line[a - 1:b] = list(text)
        elig = rng.choice([1, 2], p=[0.97, 0.03])
        put("seqn", s)
        put("eligstat", elig)
        if elig == 1:
            died = rng.random() < 0.2
            put("mortstat", int(died))
            months = int(rng.integers(1, max_months)) if died else max_months
            put("permth_exm", months)
            put("permth_int", months + 1)
            if died:
                put("ucod_leading", f"{rng.choice([1, 2, 5, 10]):03d}")
                put("diabetes_flag", rng.choice([0, 1]))
                put("hyperten_flag", rng.choice([0, 1]))
        else:
            for name in ("mortstat", "permth_exm", "permth_int"):
                put(name, ".")
        lines.append("".join(line))
    return "\n".join(lines) + "\n"


def built_dataset(tmp_path, monkeypatch, cycles=(1999, 2001, 2003, 2005, 2007, 2009), n: int = 400) -> dict:
    """Run the real build_all() on synthetic files (read_xpt is redirected to the synthetic frames)."""
    frames = {c: raw_cycle(c, n) for c in cycles}
    for c in cycles:
        d = tmp_path / str(c)
        d.mkdir(parents=True, exist_ok=True)
        for key in nhanes.VARS:
            for name in nhanes.file_candidates(c, key):
                (d / f"{name}.xpt").write_bytes(b"HEADER RECORD*******")
        m = tmp_path / "mortality"
        m.mkdir(exist_ok=True)
        (m / f"NHANES_{c}_{c + 1}_MORT_2019_PUBLIC.dat").write_text(lmf_text(frames[c]["DEMO"].SEQN, c))

    def fake_read(path, columns):
        cyc = int(path.parent.name)
        key = next(k for k in nhanes.VARS if path.stem in nhanes.file_candidates(cyc, k) and set(columns) == set(nhanes.VARS[k]))
        df = frames[cyc][key].copy()
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        return df[["SEQN"] + columns]

    monkeypatch.setattr(nhanes, "read_xpt", fake_read)
    return nhanes.build_all(tmp_path, cycles)


def cohort(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    """A cohort-format table with a known risk signal (for experiment tests; not NHANES)."""
    from cvd_risk import pce
    rng = np.random.default_rng(seed)
    cycles = np.array(list(config.CYCLES))
    d = pd.DataFrame({"seqn": np.arange(n), "cycle": rng.choice(cycles, n), "age": rng.integers(40, 80, n).astype(float),
                      "female": rng.integers(0, 2, n), "race_black": (rng.random(n) < 0.2).astype(int)})
    d["race_hispanic"] = ((rng.random(n) < 0.2) & (d.race_black == 0)).astype(int)
    d["race_other"] = ((rng.random(n) < 0.05) & (d.race_black == 0) & (d.race_hispanic == 0)).astype(int)
    d["black"] = d.race_black
    d["tchol"] = rng.normal(205, 38, n).round()
    d["hdl"] = rng.normal(52, 15, n).clip(20).round()
    d["sbp"] = rng.normal(118 + 0.5 * (d.age - 40), 17).round()
    d["dbp"] = rng.normal(74, 11, n).round()
    d["bp_treated"] = (rng.random(n) < 0.25).astype(float)
    d["lipid_treated"] = np.where(rng.random(n) < 0.05, np.nan, (rng.random(n) < 0.15).astype(float))
    d["smoker"] = (rng.random(n) < 0.2).astype(float)
    d["former_smoker"] = np.where(d.smoker == 1, 0.0, (rng.random(n) < 0.3).astype(float))
    d["diabetes"] = (rng.random(n) < 0.12).astype(float)
    d["hba1c"] = np.where(rng.random(n) < 0.05, np.nan, rng.normal(5.6, 0.6, n) + d.diabetes)
    d["bmi"] = rng.normal(29, 6, n)
    d["waist"] = rng.normal(99, 14, n)
    d["egfr"] = np.where(rng.random(n) < 0.04, np.nan, rng.normal(95 - 0.8 * (d.age - 40), 15))
    d["pir"] = np.where(rng.random(n) < 0.08, np.nan, rng.uniform(0, 5, n))
    d["education_college"] = (rng.random(n) < 0.5).astype(float)
    d["weight_mec"] = rng.uniform(5000, 40000, n)
    d["pregnant"] = 0
    d["prior_cvd"] = 0.0
    d["examined"] = 1
    d["linkage_eligible"] = 1
    risk = pce.risk_10y(d) * 0.45
    max_months = (2019 - d.cycle) * 12 + 6
    die_cvd = rng.random(n) < risk
    die_other = rng.random(n) < 0.08
    t_cvd = rng.uniform(1, 120, n)
    d["months"] = np.where(die_cvd, t_cvd, np.where(die_other, rng.uniform(1, 200, n), max_months)).clip(None, max_months).round()
    d["died"] = (die_cvd | (die_other & (d.months < max_months))).astype(int)
    d["cvd_death"] = (die_cvd & (t_cvd <= max_months)).astype(int)
    d["ucod_leading"] = np.where(d.cvd_death == 1, 1, np.where(d.died == 1, 2, np.nan))
    d["cvd_death_10y"] = ((d.cvd_death == 1) & (d.months <= 120)).astype(int)
    d["noncvd_death_10y"] = ((d.died == 1) & (d.cvd_death == 0) & (d.months <= 120)).astype(int)
    d["complete_10y"] = ((d.months >= 120) | ((d.died == 1) & (d.months <= 120))).astype(int)
    d["diabetes_flag_dc"] = np.where(d.died == 1, d.diabetes, np.nan)
    d["hyperten_flag_dc"] = np.where(d.died == 1, d.bp_treated, np.nan)
    d["split"] = np.select([d.cycle.isin(config.DEV_CYCLES), d.cycle.isin(config.TEST_CYCLES)], ["development", "temporal_test"], "recent")
    return d
