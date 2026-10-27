# Applying the model to a new target protein

Two questions, with very different answers:

1. **Where is the interface?** — solved by the pipeline below. Straightforward.
2. **What is the probability it is a true binding site?** — the model does *not* give you this
   directly, and the gap between what it outputs and what you asked is the substance of §3.

---

## 1. Inference pipeline

```
new protein
   │
   ├─ 1. structure            AlphaFold DB entry, or predict, or experimental
   ├─ 2. QC                   mask pLDDT < 70; reject if >50% of chain is masked
   ├─ 3. features             SAME code path as training (see warning below)
   ├─ 4. per-residue score    LightGBM ensemble → raw score + spread
   ├─ 5. calibration          isotonic/Platt → per-residue probability
   ├─ 6. smoothing            mean over 10 Å spatial neighbours
   ├─ 7. patch extraction     connected components of high-scoring surface residues
   └─ 8. patch ranking        size × confidence × buriedness → ranked patch list
```

### The single biggest production failure mode

**Train/inference feature skew.** If features are computed by one script at training and a
reimplemented one at inference, small differences — a different DSSP version, a different RSA
reference table, residues ordered differently, a missing neighbour-mean column — silently destroy
performance and are very hard to debug.

Mitigation, non-negotiable: **one `features.py` module, imported by both paths.** Write a unit
test that computes features for a fixed reference PDB and asserts a stored checksum. Run it in CI.

### Practical notes

- **No partner required.** The model is partner-agnostic by construction (see
  `Interface_Model_Design.md` §1), so a single chain is the entire input.
- **Oligomeric state matters.** Score the biological unit. Scoring a monomer that is natively a
  dimer will show the dimer interface as a predicted site — correct, but perhaps not what you want.
- **Masked regions produce no prediction**, rather than a low one. A disordered region is not a
  confident non-interface; say "no call".

---

## 2. Getting the position — patch extraction

Residue-level scores are not the deliverable. Interfaces are **patches**, and a scattered set of
high-scoring residues across the surface is almost always noise.

```
1. keep surface residues only            RSA ≥ 20%, pLDDT ≥ 70
2. threshold at calibrated p ≥ τ         τ from the PR curve, not 0.5 (see §3.1)
3. build graph: Cα within 8 Å
4. connected components → candidate patches
5. drop patches with < 4 residues        the Foldseek-Interface incidental-contact filter
6. score each patch                      n_residues, mean p, max p, buried area, mean pLDDT
7. rank                                  patch score (§3.2)
```

Output per protein: a ranked list of patches, each with residue numbers, a patch-level
probability, and the supporting evidence. Not a single number, and not a raw per-residue column.

`../scripts/extract_patches.py` implements steps 1–7.

---

## 3. Getting a probability that means something

### 3.1 Raw model output is not a probability

LightGBM's output is a score, not a calibrated probability, and it is systematically distorted by
whatever class weighting was used for the imbalance. Calibrate it:

- **Isotonic regression** if you have ≥ ~500 held-out *proteins*; otherwise **Platt scaling**.
- Critically: the effective sample size is **the number of proteins, not residues**. Residues
  within a protein are heavily correlated, so a million residues from 50 proteins calibrates about
  as well as 50 independent samples. Calibrate on held-out proteins, grouped.
- Validate with a **reliability diagram**, **Brier score** and **expected calibration error**.
  A reliability diagram that bends away from the diagonal means the number on your report is lying.
- Pick the operating threshold τ from the PR curve at your tolerable precision — **not 0.5**.
  Under 1:10 imbalance, 0.5 is an arbitrary and usually bad choice.

### 3.2 Patch-level probability is not the mean of residue probabilities

Residue scores inside a patch are strongly correlated — they share neighbours, burial and
secondary structure. Averaging them, or worse multiplying them, badly overstates confidence.

Fit a **separate patch-level calibrator**:

- Patch label: does this patch overlap a true interface by ≥ 50% of its residues?
- Patch features: size, mean score, max score, buried area, mean pLDDT, compactness
- Model: logistic regression or a small GBM, trained on held-out proteins
- Output: a genuine `P(patch overlaps a real interface)`

This is the number to put in front of a biologist.

### 3.3 Uncertainty — two kinds, both worth reporting

| Kind | How | What it tells you |
|---|---|---|
| **Epistemic** | Bag 5–10 models (different seeds/folds); report the spread | How much the model disagrees with itself. High spread = don't trust the mean |
| **Applicability domain** | Foldseek search of the query against the training structures; flag if no hit above TM 0.5. Also flag unusual feature-space position | Whether the query resembles anything the model was trained on |

**A calibrated probability is only valid inside the applicability domain.** A novel fold, a
membrane protein when training was soluble, or a low-pLDDT model all put the query outside it, and
the calibrated number should then be reported as unreliable rather than quietly used.

### 3.4 What the number actually means — read this before quoting it

Trained on HumanPPI, a calibrated patch probability means:

> **P(AlphaFold2 would predict an interface here, given the observed human PPI set)**

It does **not** mean P(this is a real binding site in a cell). Three gaps:

1. **Label provenance.** The labels are AF2 predictions, so the model inherits AF2's biases. It
   cannot be evidence that AF2 is right.
2. **Negatives are "unobserved", not "non-binding"** (`Interface_Model_Design.md` §2). The
   calibration set contains false negatives, which makes the calibrated probability
   **conservative** — a high-scoring "false positive" may be a real interface whose partner was
   never modelled. Good news for discovery, bad news for taking the number literally.
3. **Patch ≠ druggable site.** A real PPI interface can be large, flat and completely intractable
   for a small molecule. Interface prediction and druggability are different questions.

To make a claim about *real* binding sites, you need the experimental-PDB evaluation split
(`Interface_Model_Design.md` §3) — that is the only one measuring biology rather than AF2 agreement.

---

## 4. Raising confidence with orthogonal evidence

The model's score should be one column among several, not the verdict. For a shortlisted target:

| Evidence | What it adds | Cost |
|---|---|---|
| **Pretrained PeSTo agreement** | An independently trained model, different architecture, trained on experimental PDB. Agreement is meaningful precisely because the training data differs | seconds on CPU |
| **Conservation** | Interfaces are usually conserved; a non-conserved patch is suspicious | MSA already available in omicMSAs |
| **Foldseek-Interface search** | Does this patch resemble a known interface? A hit gives a mechanistic hypothesis and a candidate partner | fast |
| **AF-Multimer / AF3 with candidate partners** | The natural confirmation: if the patch is real, co-folding with a plausible partner should place the interface there. ipTM ≥ 0.7 is the usual confidence bar | GPU, slow — shortlist only |
| **Known PPI databases** | BioGRID/STRING partners whose predicted interface maps to this patch | cheap |
| **Pocket detection** (fpocket) | Only if the goal is small-molecule tractability, which the interface model says nothing about | cheap |

**Consensus across independently-trained methods is the strongest cheap signal available.** Where
PeSTo and our model disagree, the honest report says so rather than quoting whichever is higher.

Experimental validation remains the only real answer. Mutagenesis of predicted patch residues
followed by a binding assay is the standard test.

---

## 5. Suggested report format

Per target, one page:

```
Target: Q9BX79 (STRA6)          applicability: IN DOMAIN (nearest training TM 0.71)
Structure: AF-Q9BX79-F1         mean pLDDT 84.2    masked 11%

Patch 1   P(interface) = 0.81 [0.74–0.86]      residues 142-149, 187-193, 221-224
          23 residues · buried 890 Å² · mean pLDDT 88
          PeSTo agreement: 0.76 (concordant)
          conservation: high (JSD 0.72, top decile)
          Foldseek-Interface: matches cluster #4471 (SLC-family dimerisation)
          → strongest candidate; consistent across all orthogonal evidence

Patch 2   P(interface) = 0.44 [0.21–0.68]      residues 310-318
          9 residues · buried 310 Å² · mean pLDDT 71
          PeSTo agreement: 0.19 (DISCORDANT)
          conservation: low
          → weak; model ensemble spread is wide and PeSTo disagrees. Not recommended.
```

Report the interval, the applicability flag and the disagreements. A single number with no
uncertainty attached invites exactly the over-interpretation this project should avoid.

---

## 6. Prerequisites

1. `features.py` shared between training and inference, with a checksum test
2. A calibration hold-out set of proteins, never used in training
3. The patch-level calibrator, fitted separately from the residue model
4. Applicability-domain check (Foldseek DB of training structures)
5. Model ensemble (5–10 bags) rather than a single fit
