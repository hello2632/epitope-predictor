# Model Card — `model_final.joblib`

## What it does

Takes **one unbound protein chain** and returns a ranked list of candidate
epitope patches, each with 4 hotspot residues, in the form a binder-design tool
(BindCraft, BoltzGen) consumes.

It is **partner-agnostic**: it answers "which surface here could bind
something", not "where does partner X bind".

## Training

| | |
|---|---|
| Source | HumanPPI AlphaFold2 dimer models, PDB files only |
| Proteins | 9,806 |
| Independent components | 1,219 |
| Surface residues | 1,423,543 |
| Positive rate | 25.3% |
| Features | 170 (geometric + physicochemical, computed on the monomer) |
| Estimator | `HistGradientBoostingClassifier`, 400 iter, lr 0.06, 31 leaves, min 40/leaf, L2 1.0 |
| Trained | 2026-10-17 |
| scikit-learn | 1.8.0 |

**Labels.** A residue is interface if any heavy atom lies within 5.0 Å of any
heavy atom of the partner chain (the PeSTo convention). Labels are the **union
across all modelled partners**. Only surface residues (RSA ≥ 0.20) are trained
on or scored.

**Held out.** 1,863 proteins (17.8% of residues) were excluded from training and
from every tuning decision. As of this release the holdout has **never been
read** — `holdout_spent: false`. The numbers below are 5-fold cross-validated,
not holdout numbers.

## Measured performance

Out-of-fold, full corpus, 2026-10-17. n = 3,529 proteins after excluding
partner-contaminated test cases.

| Measure | Model | Random | Lift |
|---|---|---|---|
| Top patch hits a real epitope | **64%** | 41% | **1.57×** |
| One of top 3 hits | 79% | 59% | 1.33× |
| Patch precision (all patches) | 46% | 28% | 1.64× |
| Top 4 hotspots are true contacts | **49%** | 28% | **1.74×** |
| Median buried area of correct hotspots | 50 Å² | 43 Å² | — |
| Targets yielding any patch | 99% | 99% | — |

**54% of predicted patches land on nothing.** That is the wasted-design-run
rate and the number to plan around.

By site count: single-epitope targets are the hard and common case.

| Epitopes on target | n | hit@1 | hit@3 |
|---|---|---|---|
| 1 | 3,043 | 63% | 77% |
| 2 | 420 | 70% | 86% |
| 3+ | 66 | 74% | 91% |

Random here means compact patches of matched size placed at random on the same
surface, scored under identical rules — not a uniform coin flip. **Compare lift,
not absolute rates**: the rates move with how easy the protein population is.

## Limitations

- **Trained on predicted structures, not experimental ones.** Labels come from
  AlphaFold2 dimer models. Published hand-inspection of comparable AF2 interface
  sets puts the artefact rate near 15%.
- **Negatives mean "no partner was modelled", not "does not bind".** Absence of
  a label is absence of evidence.
- **Bound-conformation bias.** Features are computed on the chain as it appears
  in the complex. PeSTo measured ~7 ROC points of difference between bound and
  unbound inputs; expect some loss on a genuinely unbound structure.
- **No conservation, no MSA, no pretrained channel.** Geometry and
  physicochemistry only.
- **Human proteins.** Behaviour on non-human targets is uncharacterised.
- **One split scheme.** Homology + partner components. An interface-cluster
  split and an experimental-PDB split were specified but never run.
- A residual partner-contamination effect of **+3.1 points hit@1** was measured
  but is confounded with epitope count; it is not a clean leakage estimate.

## Intended use

Ranking candidate sites for binder or VHH design campaigns. **Test two or three
patches, not one.** It is not a binding-site detector and not a verdict on
druggability.

## Security note

This file is a Python pickle. `joblib.load` executes code during
deserialisation. Load it only from a source you trust, and pin scikit-learn to
the version above — loading across versions can change behaviour silently.
