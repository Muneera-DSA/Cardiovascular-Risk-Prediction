"""Download NHANES 1999-2018 + linked mortality and build the analytic cohort.

    python -m cvd_risk.build_data            # download (cached) and build
    python -m cvd_risk.build_data --no-download

Writes data/nhanes_adults_harmonised.csv.gz, data/nhanes_cvd_cohort.csv.gz and results/cohort_build.json,
then prints a short summary that should be checked before any modelling.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import config, nhanes, uci


def summary(built: dict) -> str:
    c = built["cohort"]
    lines = ["Cohort construction:"]
    for s in built["steps"]:
        lines.append(f"  {s['remaining']:>7,}  {s['step']}" + (f"  (-{s['removed']:,})" if s["removed"] else ""))
    lines.append("")
    for split, g in c.groupby("split"):
        if split == "recent":   # follow-up too short for a complete 10-year outcome: survival analysis only
            lines.append(f"{split:>14}: {len(g):,} people, cycles {sorted(int(x) for x in g.cycle.unique())}, CVD deaths during follow-up {int(g.cvd_death.sum()):,}; "
                         f"median follow-up {g.months.median() / 12:.1f} years")
            continue
        complete = g[g.complete_10y == 1]
        lines.append(f"{split:>14}: {len(g):,} people, cycles {sorted(int(x) for x in g.cycle.unique())}, "
                     f"10-year CVD deaths {int(complete.cvd_death_10y.sum()):,} of {len(complete):,} with complete follow-up "
                     f"({complete.cvd_death_10y.mean():.2%}); median follow-up {g.months.median() / 12:.1f} years")
    lines.append("")
    lines.append("Mortality-file layout checks (share passing, by cycle):")
    for cyc, chk in built["lmf_checks"].items():
        lines.append(f"  {cyc}: " + ", ".join(f"{k} {v:.3f}" for k, v in chk.items()))
    if built.get("lmf_layout_vs_nchs"):
        lines.append("Layout vs NCHS read-in program: " + json.dumps(built["lmf_layout_vs_nchs"]))
    return "\n".join(lines)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=config.RAW)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    if not args.no_download:
        print("Downloading NHANES 1999-2018 and the linked mortality files (cached after the first run)...", flush=True)
        manifest = nhanes.download(args.raw)
        config.RESULTS.mkdir(parents=True, exist_ok=True)
        (config.RESULTS / "download_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("Harmonising and linking...", flush=True)
    built = nhanes.build_all(args.raw)
    nhanes.save(built)
    text = summary(built)
    (config.RESULTS / "cohort_summary.txt").write_text(text + "\n")
    print(text)
    try:
        u = uci.download()
        print(f"\nUCI Heart Disease (4 hospitals): {len(u)} patients -> {uci.UCI_CSV.name}: " + ", ".join(f"{k} {v}" for k, v in u.site.value_counts().items()))
    except Exception as e:   # Part 1 only; never block the NHANES build
        print("UCI download failed:", e)
    print(f"\nDone in {time.time() - t0:.0f} s.")
    return built


if __name__ == "__main__":
    main()
