"""Summarise prediction files written by evaluate_field.py.

Reports, per dataset: accuracy (restricted / unrestricted) as mean +- sample SD over runs,
Welch t-tests between architectures (n is small: interpret with caution), the majority-class baseline,
and the species-level / disease-level error split (unrestricted predictions).

Example:
    python src/analyze.py --out-dir runs
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

from common import ARCHITECTURES, arch_from_tag, load_classes


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    sd = values.std(ddof=1) if len(values) > 1 else float("nan")
    return values.mean(), sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs")
    args = ap.parse_args()
    out = Path(args.out_dir)
    classes = load_classes()
    species = np.array([c.split("___")[0] for c in classes])

    for dataset in ("plantdoc", "plantwild"):
        files = sorted(out.glob(f"preds_{dataset}_*.npz"))
        if not files:
            continue
        runs = {}
        for f in files:
            tag = f.stem[len(f"preds_{dataset}_"):]
            runs.setdefault(arch_from_tag(tag), {})[tag] = np.load(f)
        any_run = next(iter(next(iter(runs.values())).values()))
        y = any_run["y"]
        majority = np.bincount(y).max() / len(y) * 100
        tomato = (species[y] == "Tomato").mean() * 100
        print(f"\n=== {dataset}: {len(y)} images | majority-class baseline {majority:.1f}% | "
              f"'Tomato' species baseline {tomato:.1f}% ===")

        restricted, unrestricted = {}, {}
        for arch in ARCHITECTURES:
            if arch not in runs:
                continue
            tags = sorted(runs[arch])
            restricted[arch] = np.array([(runs[arch][t]["p_restricted"] == y).mean() * 100 for t in tags])
            unrestricted[arch] = np.array([(runs[arch][t]["p_all"] == y).mean() * 100 for t in tags])
            r, u = mean_sd(restricted[arch]), mean_sd(unrestricted[arch])
            print(f"{arch:16s} runs {len(tags)} | restricted {r[0]:.2f} +- {r[1]:.2f} "
                  f"(runs: {', '.join(f'{v:.1f}' for v in restricted[arch])}) | "
                  f"unrestricted {u[0]:.2f} +- {u[1]:.2f}")

        print("-- Welch t-tests on restricted accuracy (no multiple-comparison correction; low power for n=3) --")
        for a, b in combinations([x for x in ARCHITECTURES if x in restricted], 2):
            t, p = stats.ttest_ind(restricted[a], restricted[b], equal_var=False)
            print(f"{a} vs {b}: difference {restricted[a].mean() - restricted[b].mean():+.2f} points, p = {p:.3f}")

        print("-- species / disease split (unrestricted predictions) --")
        for arch in ARCHITECTURES:
            if arch not in runs:
                continue
            sp_acc, dis_given_sp, species_share = [], [], []
            for tag in sorted(runs[arch]):
                p = runs[arch][tag]["p_all"]
                same_species = species[y] == species[p]
                wrong = p != y
                sp_acc.append(same_species.mean() * 100)
                dis_given_sp.append((y[same_species] == p[same_species]).mean() * 100)
                species_share.append((~same_species)[wrong].mean() * 100)
            s, d, e = mean_sd(sp_acc), mean_sd(dis_given_sp), mean_sd(species_share)
            print(f"{arch:16s} species acc {s[0]:.1f} +- {s[1]:.1f} | disease acc given correct species "
                  f"{d[0]:.1f} +- {d[1]:.1f} | share of errors that are species errors {e[0]:.1f}%")


if __name__ == "__main__":
    main()
