# Plan — Structure-based interface prediction for novel target identification

**Goal:** train a per-residue binding-interface classifier on our human PPI dimer database, then
scan unannotated human proteins to rank new candidate targets by the quality of their predicted
interface patches.

**Settings chosen:** per-residue output · structure-based (geometric) features · CPU-only server.

---

## 0. The CPU constraint, stated honestly

PeSTo-style atom transformers and MaSIF-style surface CNNs need GPUs to *train*. On CPU only,
training one from scratch on ~10⁵ chains is not realistic (weeks to months).

Two things remain fully practical on CPU:

1. **Pretrained PeSTo inference.** The paper reports 5.3 ± 2.8 s per protein on CPU. Running it
   over our dimers and over the AlphaFold human proteome is a few CPU-days, trivially
   parallelised. This gives us a strong **baseline and a feature channel for free**.
2. **Training a lightweight geometric model.** Hand-crafted per-residue geometric descriptors
   + gradient boosting (LightGBM) or a small residue-graph GNN. Trains in minutes to hours on CPU.

So the plan is: **engineer geometry explicitly instead of learning it**, and use pretrained PeSTo
as both a baseline to beat and an input feature. If a GPU becomes available later, Stage 4 swaps
in a learned geometric encoder without changing anything upstream.

---

## Stage 1 — Build a clean dataset (the step that decides whether any of this works)

Most of the risk in this project is here, not in the model.

### 1.1 Define interface residues

Pick one definition and keep it fixed:

- **Primary:** heavy-atom distance < 5.0 Å to any atom of the partner chain.
- **Alternative (used by ProSiteHunter for nucleic acids):** atom distance < 0.5 Å + sum of
  van der Waals radii.
- Record ΔSASA per residue as well (monomer SASA − complex SASA); it lets us switch to a
  burial-based definition without recomputing contacts.

Keep only chain pairs with **≥ 4 interface residues per chain** (the Foldseek-Interface cutoff) —
fewer than that is usually an incidental contact.

### 1.2 Filter out crystal artifacts — do not skip this

A large fraction of PDB dimers are **crystal packing contacts, not biological interfaces**.
Training on them teaches the model noise.

- Use **biological assemblies** (`.pdb1` / `assembly 1`), not the asymmetric unit.
- Cross-check against **PISA** and/or **ProtCID** where available.
- Drop interfaces with buried surface area below ~400 Å² per side unless they are
  SLiM-mediated and supported by independent evidence.
- Flag, don't silently keep, dimers extracted from much larger assemblies — Foldseek-Interface
  showed these behave differently and AlphaFold-Multimer predicts them poorly (median DockQ 0.02).

### 1.3 Define negatives correctly — the most common source of fake performance

Residues come in three classes, not two:

| Class | Treatment |
|---|---|
| Interface (contacts partner) | **positive** |
| Surface, non-interface (RSA ≥ 15–20%) | **negative** |
| Buried core (RSA < 15%) | **exclude from training and from evaluation** |

If buried residues are counted as negatives, the model partly learns "is this residue exposed?"
rather than "is this an interface". **Report all metrics on surface residues only.**

> **Corrected 2026-10-17.** This paragraph previously claimed the inflation reached "ROC AUC ~0.9".
> That figure was never measured. Measured on pilot2000: **0.679** including buried residues versus
> **0.624** on the surface subset — an inflation of 0.055, not 0.3. The direction of the argument
> holds and the surface-only rule stands; the magnitude was wrong.
Caveat: a protein's "non-interface" surface may simply be an interface whose partner has not been
observed yet, so negatives are noisy by nature — this caps achievable precision and is worth
stating in any writeup.

Expected imbalance after this filtering: roughly **1:4 to 1:10** positive:negative.

### 1.4 Split the data to prevent leakage — two levels

Sequence-identity splitting alone is **not sufficient** here, because unrelated folds can share
interface geometry (Foldseek-Interface found 721 clusters containing multiple distinct CATH
domain pairs).

1. **Sequence level:** MMseqs2 cluster at **30% identity**; no cluster spans train and test.
2. **Interface level:** cluster the training interfaces with **Foldseek-Interface** at interface
   TM-score **0.4** (their optimal cutoff); hold out whole interface clusters.

Report test performance **both ways**. The gap between the two is the real measure of how much
the model generalises to genuinely new binding modes — and that gap is exactly what matters for
finding *new* targets.

Also keep a **temporal split** (structures deposited after a cutoff date) as a third, honest test set.

### 1.5 Expected scale

Human-only dimers will be far smaller than the 3.1M PDB dimers in the Foldseek-Interface paper —
realistically **10³–10⁴ non-redundant human interfaces** after filtering. That is enough for a
feature-based model, and is another argument against a large deep net.

---

## Stage 2 — Per-residue geometric features (all CPU-computable)

Computed on the **unbound/monomer** structure, never the complex — otherwise we leak the answer.

**Surface and shape**
- RSA (DSSP or FreeSASA), absolute and relative
- Protrusion index (CX), depth index, distance to protein centroid
- Local surface curvature (mean + Gaussian, from a Connolly/MSMS mesh or an α-sphere proxy)
- Half-sphere exposure (HSE-up / HSE-down), coordination number at 8/12/16 Å

**Local environment** (the explicit stand-in for PeSTo's learned neighbourhoods)
- Amino acid composition of residues within 8 Å and 12 Å (20-dim each)
- Mean hydrophobicity, net charge, aromatic fraction in those shells
- Secondary structure (DSSP 8-state, one-hot) + SS composition of the neighbourhood
- Backbone φ/ψ, Cα–Cβ orientation relative to the local surface normal

**Structural alphabet**
- **Foldseek 3Di state** of the residue and of its sequence/spatial neighbours — cheap, and it
  encodes local geometry in exactly the representation the interface papers rely on

**Conservation** (optional but usually the single strongest non-geometric signal)
- Per-residue conservation from an HHblits/MMseqs2 MSA (Shannon entropy, Jensen–Shannon
  divergence vs background). Caveat: this is the slow step; budget for it or drop it.

**Pretrained model channel**
- PeSTo generalised-model scores for protein / nucleic-acid / ion / ligand / lipid interfaces.
  Five extra columns, obtained by CPU inference, no training.

**Smoothing**
- For each feature, add the mean over spatial neighbours within 10 Å. Interfaces are *patches*;
  patch-level context is what makes isolated residues interpretable.

---

## Stage 3 — Model

**Primary: LightGBM** on the feature table above.

- Handles mixed-scale heterogeneous features, trains on CPU in minutes, gives feature importances
  we can actually interrogate.
- Class imbalance: `scale_pos_weight`, or focal loss (ProSiteHunter used α=0.75/0.25, γ=2).
- Group-aware CV using the interface clusters from 1.4.

**Secondary: small residue-graph GNN** (PyTorch Geometric, 3–4 layers, ~100k params).

- Nodes = residues, edges = Cα within 10 Å, edge features = distance + relative orientation.
- Trainable on CPU at this dataset size; captures patch structure the GBDT can only approximate.
- Worth it only if it beats LightGBM on the interface-cluster-held-out split.

**Post-processing — predict patches, not residues**
- Spatially smooth per-residue scores over 10 Å neighbourhoods.
- Cluster high-scoring residues into connected patches; score each patch by size, mean
  confidence, and buried area.
- A protein's final score = its best patch, not its best single residue.

**Baselines that must be reported**
1. RSA alone (the trivial baseline — a surprising number of published methods barely beat it)
2. Conservation alone
3. Pretrained PeSTo, unmodified
4. Our model

If we do not beat #3 we should say so plainly and consider simply using PeSTo plus our
patch-ranking layer, which would still be a useful result.

---

## Stage 4 — Evaluation

**Metrics** (surface residues only, per-protein then median across proteins):

- **PR AUC — the headline metric.** ROC AUC is inflated under imbalance; ProSiteHunter's gains
  were 6–8% ROC AUC but 26–79% PR AUC, which shows how differently the two behave.
- MCC and F1 at a calibrated threshold
- **Top-10% recovery rate** (PeSTo's definition: are the top 10% ranked residues all true interface) —
  closest to how the model will actually be used
- Patch-level: does the predicted top patch overlap the true interface by ≥ 50%?

**Report on all three splits:** sequence-split, interface-cluster-split, temporal-split.

**Error analysis to run, not skip**
- Performance vs interface size, vs disorder content, vs RSA
- Ordered vs disorder-mediated interfaces (disorder-mediated is where every method degrades)
- Which features drive predictions (SHAP) — if conservation dominates, we have built a
  conservation predictor with extra steps

---

## Stage 5 — Scanning for new targets

1. **Candidate set:** AlphaFold DB human models. Filter on quality as PeSTo did — they kept
   7,464 of 20,504 entries on pLDDT/PAE. Low-pLDDT regions are disordered and will produce
   confident-looking garbage, so mask pLDDT < 70 before scoring.
2. **Score** every residue, build patches, rank proteins by best-patch quality.
3. **Subtract what is already known.** Remove proteins whose top patch corresponds to an
   interface already in our database or in the PDB. Foldseek-Interface search is the right tool:
   a patch with no structural neighbour is the genuinely novel case.
4. **Prioritise** the remainder by orthogonal evidence:
   - Disease association (Open Targets), expression specificity
   - Patch size and buriedness (small flat patches are rarely tractable)
   - Absence of an existing binder/ligand
   - Pathway context
5. **Deliverable:** a ranked table — protein, patch residues, confidence, novelty flag,
   supporting evidence — not a list of raw scores.

### Known failure mode to watch for

The Foldseek-Interface authors manually inspected 39 "novel" clusters from predicted human
complexes and found **most were artifacts**: wrong-partner docking (23), poorly predicted
contacts (10), artificial termini from fragmentation (2), extracellular–intracellular contacts
(1). Only 6 were of real interest. **Budget for manual inspection of the top hits — the
automated novelty call is a filter, not a conclusion.**

---

## Suggested sequence of work

| Step | Output | Rough effort |
|---|---|---|
| 1 | Dataset: filtered dimers, labels, RSA, splits | 1–2 weeks — the bulk of the work |
| 2 | Feature pipeline, parallelised over CPU cores | 1 week |
| 3 | PeSTo CPU inference over dimers + proteome | few CPU-days, runs in background |
| 4 | LightGBM + baselines, three-way evaluation | 3–5 days |
| 5 | GNN variant, only if step 4 justifies it | 1 week |
| 6 | Proteome scan, novelty filter, manual triage | 1–2 weeks |

**Decision point after step 4:** if the model does not beat pretrained PeSTo on the
interface-cluster-held-out split, stop model development and invest the remaining time in the
patch-ranking and novelty-filtering layer, which is where the target-identification value
actually sits.

---

## Open questions for us

1. What is in the database exactly — experimental PDB dimers, AlphaFold-predicted human
   complexes, or both? If predicted, the artifact rate above applies to the *training* data too.
2. Are there negative/non-interacting pairs available, or only positives?
3. Do we have MSAs precomputed, or does conservation need to be generated from scratch?
   On CPU this is the single biggest time cost.
4. "New target" — novel *protein*, or novel *interface on a known protein*? The second is a more
   tractable and arguably more valuable question.
5. Is there in-house binding data to use as a final held-out validation set? That would be worth
   more than any public benchmark.
