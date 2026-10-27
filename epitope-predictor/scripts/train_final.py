#!/usr/bin/env python3
"""
Fit the final model on every training fold and persist it for inference.

The cross-validated runs produce FIVE models, each trained on 4/5 of the data
and valid only on its own held-out fold.  None of them is "the model", so none
can be applied to a new protein.  This script fits one model on all of it.

WHAT IS SAVED, AND WHY EACH PIECE MATTERS

    model          the fitted estimator
    columns        the feature names IN ORDER.  A gradient-boosted model takes a
                   bare float array and cannot tell that column 40 is now
                   hse_down rather than hse_up -- it will predict confidently
                   and wrongly.  Inference_And_Calibration.md section 1 calls
                   train/inference feature skew "the single biggest production
                   failure mode"; saving the order is what makes it detectable.
    params         surface cutoff, pLDDT mask, smoothing radius, patch rules.
                   Inference that smooths at a different radius than evaluation
                   is measuring something the reported numbers do not describe.
    provenance     what it was trained on, when, and what was held out.

THE HOLDOUT STAYS SEALED.  Proteins in the holdout split are excluded here, as
they are everywhere else.  That costs ~16% of the training data and keeps one
honest reading available for later.  Pass --include-holdout only once you have
stopped making decisions, and expect to never get that reading back.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_COLS = {"resnum", "resname", "aa", "protein", "is_interface", "class",
              "n_partners", "source", "plddt", "rsa_complex", "delta_rsa",
              "max_dsasa", "buried_frac", "y", "fold", "component", "split",
              "n_residues"}

# Inference defaults, recorded with the model so prediction matches evaluation.
# SMOOTH_RADIUS is a Class B parameter; 10 A is the value Inference_And_Calibration.md
# specifies a priori.  Per-fold nested selection chose 8 or 12 depending on the
# fold, so 10 sits inside the range it explored rather than being tuned on test.
DEFAULTS = dict(surface_rsa=0.20, min_plddt=70.0, smooth_radius=10.0,
                top_frac=0.25, patch_cutoff=8.0, min_patch=4, n_hotspot=4)

GBM = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
           min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def load(pilot: Path, include_holdout: bool):
    lab = pd.read_csv(pilot / "training_table.tsv", sep="\t")
    lab = lab[lab["class"].isin(["positive", "negative"])].copy()
    lab["y"] = (lab["class"] == "positive").astype(int)

    sp = pd.read_csv(pilot / "splits.tsv", sep="\t")
    held = set(sp.loc[sp["split"] == "holdout", "protein"])
    if not include_holdout:
        sp = sp[sp["split"] != "holdout"]
    lab = lab.merge(sp[["protein", "component"]], on="protein", how="inner")

    cols = []
    for sub in ("features", "physchem"):
        d = pilot / sub
        frames = [pd.read_csv(d / f"{a}.tsv", sep="\t").assign(protein=a)
                  for a in lab["protein"].unique() if (d / f"{a}.tsv").exists()]
        if not frames:
            sys.exit(f"no feature files in {d}")
        t = pd.concat(frames, ignore_index=True)
        keep = [c for c in t.columns
                if c not in LABEL_COLS and pd.api.types.is_numeric_dtype(t[c])]
        lab = lab.merge(t[["protein", "resnum"] + keep],
                        on=["protein", "resnum"], how="inner")
        cols += keep
    return lab.sort_values(["protein", "resnum"]).reset_index(drop=True), cols, held


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pilot", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output .joblib (default <pilot>/model_final.joblib)")
    ap.add_argument("--include-holdout", action="store_true",
                    help="spend the sealed holdout as training data -- irreversible")
    ap.add_argument("--smooth-radius", type=float, default=DEFAULTS["smooth_radius"])
    args = ap.parse_args()

    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier

    out = args.out or (args.pilot / "model_final.joblib")
    t0 = time.time()
    d, cols, held = load(args.pilot, args.include_holdout)
    X = np.nan_to_num(d[cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = d["y"].to_numpy()
    t_load = time.time() - t0

    print(f"training on {d['protein'].nunique()} proteins, "
          f"{d['component'].nunique()} components, {len(y)} surface residues "
          f"({y.mean():.1%} positive), {len(cols)} features")
    if args.include_holdout:
        print(f"  *** --include-holdout: the {len(held)}-protein holdout is being")
        print(f"      SPENT as training data.  No clean reading remains after this.")
    else:
        print(f"  holdout: {len(held)} proteins excluded and still sealed")

    t0 = time.time()
    model = HistGradientBoostingClassifier(**GBM).fit(X, y)
    t_fit = time.time() - t0

    params = dict(DEFAULTS, smooth_radius=args.smooth_radius)
    bundle = {
        "model": model,
        "columns": cols,                 # ORDER MATTERS -- see the docstring
        "params": params,
        "gbm": GBM,
        "provenance": {
            "trained": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pilot": str(args.pilot),
            "n_proteins": int(d["protein"].nunique()),
            "n_components": int(d["component"].nunique()),
            "n_residues": int(len(y)),
            "positive_rate": float(y.mean()),
            "holdout_proteins": sorted(held),
            "holdout_spent": bool(args.include_holdout),
            "sklearn": __import__("sklearn").__version__,
        },
    }
    joblib.dump(bundle, out, compress=3)
    size = out.stat().st_size / 1e6

    print(f"\n  load  {t_load:5.1f} s")
    print(f"  fit   {t_fit:5.1f} s")
    print(f"  wrote {out}  ({size:.1f} MB)")
    print(f"\n  inference parameters recorded with the model:")
    for k, v in params.items():
        print(f"    {k:<16} {v}")
    print(f"\n  NOTE: this model has never been evaluated -- it is fitted on all")
    print(f"  the data it was given.  The reported performance comes from the")
    print(f"  cross-validated runs, not from this object.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
