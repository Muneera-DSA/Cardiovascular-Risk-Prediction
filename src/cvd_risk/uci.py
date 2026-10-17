"""UCI Heart Disease (Janosi, Steinbrunn, Pfisterer & Detrano, 1988; CC BY 4.0), four hospitals.

The popular Kaggle "Heart Failure Prediction" file (918 patients) is a merge of these four files
(plus Statlog, which duplicates Cleveland). The original files keep what the Kaggle file lost:
**which hospital each patient came from**, which is what makes an external (unseen-hospital) test possible.

Coding follows the Kaggle file so results are comparable:
    cp 1-4 -> TA, ATA, NAP, ASY      restecg 0-2 -> Normal, ST, LVH      slope 1-3 -> Up, Flat, Down
    target  HeartDisease = 1 if angiographic diameter narrowing > 50% in any major vessel (num > 0)
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
SITES = {"cleveland": "Cleveland (USA)", "hungarian": "Budapest (Hungary)",
         "switzerland": "Zurich & Basel (Switzerland)", "va": "Long Beach VA (USA)"}
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
UCI_CSV = config.DATA / "uci_heart_4_hospitals.csv"


def parse_processed(text: str, site: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), header=None, names=COLUMNS, na_values=["?"])
    df.insert(0, "site", site)
    return df


def download(dest: Path = UCI_CSV) -> pd.DataFrame:
    from .nhanes import _fetch
    blob = _fetch(URL)
    if not blob:
        raise RuntimeError(f"Could not download {URL}")
    z = zipfile.ZipFile(io.BytesIO(blob))
    frames = []
    for site in SITES:
        name = next((n for n in z.namelist() if n.endswith(f"processed.{site}.data")), None)
        if name is None:
            raise RuntimeError(f"processed.{site}.data not in the UCI archive: {z.namelist()}")
        frames.append(parse_processed(z.read(name).decode("latin-1"), site))
    df = pd.concat(frames, ignore_index=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return df


def to_kaggle_coding(uci: pd.DataFrame) -> pd.DataFrame:
    """The 11 Kaggle features + target + site, with the Kaggle category labels."""
    d = pd.DataFrame({
        "site": uci.site,
        "Age": uci.age,
        "Sex": uci.sex.map({1: "M", 0: "F"}),
        "ChestPainType": uci.cp.map({1: "TA", 2: "ATA", 3: "NAP", 4: "ASY"}),
        "RestingBP": uci.trestbps,
        "Cholesterol": uci.chol,
        "FastingBS": uci.fbs,
        "RestingECG": uci.restecg.map({0: "Normal", 1: "ST", 2: "LVH"}),
        "MaxHR": uci.thalach,
        "ExerciseAngina": uci.exang.map({1: "Y", 0: "N"}),
        "Oldpeak": uci.oldpeak,
        "ST_Slope": uci.slope.map({1: "Up", 2: "Flat", 3: "Down"}),
        "HeartDisease": (uci.num > 0).astype(int),
    })
    return d


KEY = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope", "HeartDisease"]


def match_sites(kaggle: pd.DataFrame, uci_coded: pd.DataFrame) -> pd.DataFrame:
    """Attach the hospital to each Kaggle row: a UCI row matches if every value it records equals the
    Kaggle value (the UCI files have missing values that the Kaggle file filled in, so a missing UCI value
    matches anything). Rows matching patients from more than one hospital, or none, are left unassigned."""
    k = kaggle.reset_index(drop=True).copy()
    u = uci_coded.reset_index(drop=True)
    eq = np.ones((len(k), len(u)), bool)
    for col in KEY:
        kv, uv = k[col].to_numpy(object), u[col].to_numpy(object)
        umiss = pd.isna(u[col]).to_numpy()
        try:
            same = np.isclose(k[col].astype(float).to_numpy()[:, None], u[col].astype(float).to_numpy()[None, :])
        except (TypeError, ValueError):
            same = kv[:, None] == uv[None, :]
        eq &= same | umiss[None, :]
    sites = u.site.to_numpy()
    cands = [sorted(set(sites[row])) for row in eq]
    k["site_candidates"] = cands
    k["n_matching_uci_rows"] = eq.sum(axis=1)
    site = [c[0] if len(c) == 1 else None for c in cands]
    # The Kaggle file keeps the hospitals in blocks; an ambiguous row takes the hospital of its uniquely
    # matched neighbours when that hospital is one of its candidates.
    k["site_assigned_by"] = ["exact match" if s else None for s in site]
    for i, c in enumerate(cands):
        if site[i] is None and len(c) > 1:
            prev = next((site[j] for j in range(i - 1, -1, -1) if site[j]), None)
            nxt = next((site[j] for j in range(i + 1, len(site)) if site[j]), None)
            for s in (prev, nxt):
                if s in c and pd.isna(k.at[i, "site_assigned_by"]):
                    site[i] = s
                    k.at[i, "site_assigned_by"] = "neighbouring rows (ambiguous match)"
    k["site"] = site
    return k
