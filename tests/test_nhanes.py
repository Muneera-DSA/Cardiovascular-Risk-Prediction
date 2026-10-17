import numpy as np
import pandas as pd
import pytest
from synthetic import built_dataset, lmf_text, raw_cycle

from cvd_risk import config, nhanes, pce


def _raw(start=2005, n=50):
    files = raw_cycle(start, n)
    base = files["DEMO"]
    for k, f in files.items():
        if k != "DEMO":
            base = base.merge(f, on="SEQN", how="left")
    for cols in nhanes.VARS.values():
        for c in cols:
            if c not in base:
                base[c] = np.nan
    base["cycle"] = start
    return base


def test_file_names_follow_cdc_conventions():
    assert nhanes.file_candidates(1999, "DEMO") == ["DEMO"]
    assert nhanes.file_candidates(2017, "BPX") == ["BPX_J"]
    assert nhanes.file_candidates(1999, "hdl") == ["LAB13"]
    assert nhanes.file_candidates(2003, "biopro") == ["L40_C"]
    assert nhanes.file_candidates(2005, "tchol") == ["TCHOL_D"]
    assert nhanes.file_candidates(2015, "ghb") == ["GHB_I"]


def test_refused_and_dont_know_are_missing_not_no():
    r = _raw()
    r = r.astype({"SMQ020": float, "SMQ040": float})
    r.loc[:3, "SMQ020"] = [7, 9, 1, 2]
    r.loc[:3, "SMQ040"] = [1, 1, 3, np.nan]
    h = nhanes.harmonise(r)
    assert np.isnan(h.smoker[0]) and np.isnan(h.smoker[1])
    assert h.smoker[2] == 0 and h.former_smoker[2] == 1     # ever smoked, not now
    assert h.smoker[3] == 0 and h.former_smoker[3] == 0     # never smoked


def test_blood_pressure_ignores_zero_and_missing_readings():
    r = _raw()
    r.loc[0, ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]] = [120, 130, 0, np.nan]
    r.loc[0, ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]] = [0, 80, 70, np.nan]
    h = nhanes.harmonise(r)
    assert h.sbp[0] == 125 and h.dbp[0] == 75 and h.n_bp_readings[0] == 2


def test_diabetes_combines_self_report_medication_and_hba1c():
    r = _raw()
    r.loc[:3, "DIQ010"] = [2, 2, 1, 3]
    r.loc[:3, "DIQ050"] = [2, 2, 2, 2]
    r.loc[:3, "DIQ070"] = [2, 1, 2, 2]
    r.loc[:3, "LBXGH"] = [5.4, 5.4, 5.4, 6.7]
    h = nhanes.harmonise(r)
    assert list(h.diabetes[:4]) == [0, 1, 1, 1]          # none / pills / told / undiagnosed (HbA1c >= 6.5)


def test_hdl_uses_whichever_name_the_cycle_has():
    for start in (1999, 2003, 2009):
        h = nhanes.harmonise(_raw(start, n=1000))
        assert h.hdl.notna().mean() > 0.9


def test_creatinine_recalibration_only_where_cdc_requires_it():
    s = pd.Series([1.0])
    assert nhanes.recalibrate_creatinine(s, 1999)[0] == pytest.approx(1.16)
    assert nhanes.recalibrate_creatinine(s, 2005)[0] == pytest.approx(0.962)
    assert nhanes.recalibrate_creatinine(s, 2009)[0] == 1.0


def test_egfr_matches_ckd_epi_2021_reference():
    # 50-year-old woman, creatinine 0.8 mg/dL -> about 89 mL/min/1.73m2 (published calculator)
    e = nhanes.egfr_ckd_epi_2021(pd.Series([0.8]), pd.Series([50]), pd.Series([1]))[0]
    assert e == pytest.approx(89, abs=1)


def test_mortality_file_layout_round_trip(tmp_path):
    seqns = np.arange(1000, 1200)
    p = tmp_path / "lmf.dat"
    p.write_text(lmf_text(seqns, 2005))
    lmf = nhanes.read_lmf(p)
    assert len(lmf) == 200 and set(lmf.eligstat.unique()) <= {1, 2}
    checks = nhanes.check_lmf(lmf, pd.Series(seqns))
    assert all(v >= 0.99 for v in checks.values())


def test_wrong_layout_fails_loudly(tmp_path):
    p = tmp_path / "lmf.dat"
    p.write_text(lmf_text(np.arange(1000, 1100), 2005))
    bad = dict(config.LMF_LAYOUT, eligstat=(16, 16), mortstat=(15, 15))
    with pytest.raises(ValueError):
        nhanes.check_lmf(nhanes.read_lmf(p, bad), pd.Series(np.arange(1000, 1100)))


def test_layout_matches_nchs_read_in_program_syntax():
    text = "fwf_cols(seqn = c(1,6), eligstat = c(15,15), mortstat = c(16,16), ucod_leading = c(17,19))"
    found = nhanes.lmf_layout_from_r(text)
    for k in ("seqn", "eligstat", "mortstat", "ucod_leading"):
        assert tuple(found[k][0]) == config.LMF_LAYOUT[k]


def test_end_to_end_build_on_synthetic_files(tmp_path, monkeypatch):
    built = built_dataset(tmp_path, monkeypatch)
    c = built["cohort"]
    assert c.age.between(40, 79).all() and (c.prior_cvd == 0).all() and (c.pregnant == 0).all()
    assert c[config.CORE_REQUIRED].notna().all().all()
    assert set(c.split) <= {"development", "temporal_test", "recent"}
    assert (c[c.split == "development"].cycle.isin(config.DEV_CYCLES)).all()
    removed = sum(s["removed"] for s in built["steps"])
    assert built["steps"][0]["remaining"] - removed == len(c)
    # outcome logic: a 10-year CVD death is a CVD death within 120 months
    ev = c[c.cvd_death_10y == 1]
    assert (ev.months <= 120).all() and ev.ucod_leading.isin(config.CVD_CAUSES).all()


def test_death_certificate_flags_are_never_model_features():
    assert not set(config.LEAKY_FEATURES) & set(config.FEATURES)


def test_pooled_cohort_equations_reproduce_the_guideline_example():
    d = pd.DataFrame({"age": [55] * 4, "tchol": [213] * 4, "hdl": [50] * 4, "sbp": [120] * 4, "bp_treated": [0] * 4,
                      "smoker": [0] * 4, "diabetes": [0] * 4, "female": [1, 1, 0, 0], "black": [0, 1, 0, 1]})
    np.testing.assert_allclose(pce.risk_10y(d) * 100, [2.1, 3.0, 5.3, 6.1], atol=0.1)


def test_pce_risk_increases_with_each_risk_factor():
    base = dict(age=60, tchol=200, hdl=50, sbp=130, bp_treated=0, smoker=0, diabetes=0, female=0, black=0)
    r0 = pce.risk_10y(pd.DataFrame([base]))[0]
    for change in ({"smoker": 1}, {"diabetes": 1}, {"sbp": 160}, {"age": 70}, {"hdl": 35}):
        assert pce.risk_10y(pd.DataFrame([{**base, **change}]))[0] > r0


def test_reader_restores_zeros_that_pandas_reads_as_tiny_floats(monkeypatch, tmp_path):
    fake = pd.DataFrame({"SEQN": [1.0, 2.0], "BPXDI1": [5.397605346934028e-79, 70.0], "BPXDI2": [72.0, 74.0]})
    monkeypatch.setattr(pd, "read_sas", lambda *a, **k: fake.copy())
    df = nhanes.read_xpt(tmp_path / "BPX.xpt", ["BPXDI1", "BPXDI2"])
    assert df.BPXDI1[0] == 0.0
    r = _raw()
    r.loc[0, ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]] = [df.BPXDI1[0], 70, 72, np.nan]
    assert nhanes.harmonise(r).dbp[0] == 71                # the zero reading is excluded from the average
