# Feature specification

What to compute per residue, in what order of priority, and what to leave out.

---

## 0. Two principles that decide most of the design

### 0.1 Burial is the strongest single feature — and that is a trap

Relative solvent accessibility separates interface from non-interface better than anything else
you can compute cheaply. It is also the **trivial baseline**. A model dominated by RSA has learned
"is this residue exposed?", which is not a useful answer.

Consequences, both already in the plan but worth restating as the reason they are there:

- **Train and evaluate on surface residues only** (RSA ≥ 20%). Including buried residues inflates
  every metric while teaching nothing.
- **The job of every other feature is to discriminate among exposed residues.** Judge a candidate
  feature by whether it adds signal *within the surface subset*, not overall.
- Always report the RSA-only baseline alongside the model. If the gap is small, say so.

### 0.2 Interfaces are patches, so neighbourhood aggregation is where the signal lives

A single residue's own properties are weakly informative. The composition and geometry of its
surroundings are much more so. PeSTo makes this explicit: it uses **only element names and
coordinates** — no mass, radius, charge or hydrophobicity — and beats parametrised methods, with
its context growing from 8 nearest neighbours (~3.4 Å) to 64 (~8.2 Å). All the signal came from
geometry at multiple scales.

We cannot learn that on CPU, so we hand-code it: **every feature appears in both per-residue and
neighbourhood-aggregated form** (mean over spatial neighbours at 8 Å and 12 Å). This is message
passing done by hand, and it is the single highest-value design choice in the feature set.

---

## 1. The bound-conformation problem — specific to your data, and easy to miss

Training chains are extracted from **dimer models**. Their backbones and side chains are in the
**bound** conformation: rotamers already arranged for binding, loops already positioned. At
inference you will score **unbound** structures — AlphaFold monomer models of proteins with no
modelled partner. That is a distribution shift, and it is measurable.

PeSTo quantified exactly this on PPDB5: **median ROC AUC 0.85 on bound structures, 0.78 on the
corresponding unbound ones.** A ~7-point drop from conformational state alone, for a model that
never saw explicit side-chain parameters.

Three mitigations, in order of preference:

1. **Train on monomer models, transfer labels from the dimer.** Compute features on the
   AlphaFold *monomer* model of each protein; map interface labels from the dimer by residue
   number (already in full-protein numbering after `build_interface_labels.py`). This removes the
   leak at the source and matches train to inference exactly. **Recommended.**
2. **Prefer conformation-robust features.** Backbone and Cβ-based geometry (half-sphere exposure,
   Cα coordination, protrusion from Cβ) is far less sensitive to rotamer state than all-atom
   side-chain descriptors. Weight the feature set accordingly.
3. **At minimum, measure the gap.** Score a held-out set both ways — chain-from-dimer and
   AlphaFold monomer — and report the difference. If it resembles PeSTo's 7 points, say so.

Option 1 costs one extra structural mapping step and removes a whole class of silent failure.

---

## 2. Tier 1 — geometric (the core; compute these first)

All from coordinates alone. No external dependencies beyond a SASA tool.

| Feature | Notes |
|---|---|
| **RSA** (relative solvent accessibility) | FreeSASA or DSSP. The anchor feature and the baseline |
| **Absolute SASA** | Complements RSA; large residues behave differently |
| **Half-sphere exposure (HSE-up / HSE-down)** | Direction-aware burial. Strong, cheap, rotamer-robust |
| **Coordination number** at 8 / 12 / 16 Å | Cα counts. Multi-scale density, the cheap PeSTo analogue |
| **Protrusion index (CX)** | Interfaces sit on protrusions. Computed from Cβ |
| **Depth index** | Distance from the molecular surface |
| **Distance to centroid**, normalised by radius of gyration | Crude but informative position descriptor |
| **Local surface curvature** (mean + Gaussian) | From a surface mesh, or an α-sphere proxy. Flat patches favour PPIs |
| **Secondary structure** (DSSP 8-state, one-hot) | See §6 — the direction of this signal is known |
| **Backbone φ / ψ** | Cheap, captures local conformation |
| **Cα–Cβ orientation vs local surface normal** | Does the side chain point outward? |
| **Foldseek 3Di state** (20-state categorical) | Learned local-geometry descriptor; trees handle categoricals natively |
| **Virtual-centre neighbour features** (`vc_*`) | Foldseek's tertiary-partner geometry: distance, sequence separation, backbone-fallback flag. Backbone-only so rotamer-robust. **Hypothesis — validate before relying on it**, see `Foldseek_3Di_Notes.md` §3.2 |

Plus, for each of the continuous ones, the **8 Å and 12 Å neighbourhood mean** (§0.2).

**Neighbourhood composition** deserves its own line: the 20-dim amino-acid composition of
residues within 8 Å and within 12 Å, plus the 3Di-state composition. These capture "what kind of
surface is this" and are among the most useful features in the whole set.

---

## 3. Tier 2 — evolutionary (highest value per unit of effort, given your data)

Interfaces are usually conserved. And **you already have the MSAs** — `segment_omicMSAs.tar.gz`
removes the single largest CPU cost that would otherwise make this tier expensive.

| Feature | Notes |
|---|---|
| **Shannon entropy** per column | Simplest conservation measure |
| **Jensen–Shannon divergence** vs background | Better behaved than entropy; the usual choice |
| **Relative conservation within the neighbourhood** | A conserved patch beats a conserved residue |
| **Alignment depth (Neff)** at that position | Confidence in the conservation estimate |
| **Taxonomic breadth** | The omicMSA headers carry taxonomy; conservation across phyla means more |

Use the omicMSA **`mask` track** (`*` = high quality, `-` = low) to exclude unreliable columns —
the authors already computed it, and it is free quality control.

One caution: conservation is informative about *function generally*, not interfaces specifically.
Catalytic sites, folding nuclei and allosteric sites are conserved too. Check with SHAP that
conservation is not simply dominating — if it is, you have built a conservation predictor.

---

## 4. Tier 3 — physicochemical and statistical (cheap, modest payoff)

| Feature | Notes |
|---|---|
| Amino acid identity (one-hot or target-encoded) | |
| Hydrophobicity (Kyte–Doolittle), charge, polarity, volume, polarizability, isoelectric point | ProSiteHunter used exactly this set |
| BLOSUM62 row | 20 columns; marginal but free |
| Interface propensity scale | Literature-derived per-residue PPI propensity |
| Neighbourhood means of the above | Mean hydrophobicity / net charge / aromatic fraction in 8 and 12 Å shells |

ProSiteHunter's ablation found statistical features contributed least (≈0.23% ROC AUC) with
geometric features also auxiliary (≈2.18%) — but that was a **sequence-only** model where
language-model embeddings carried the structure information. With real structures the balance
shifts toward Tier 1. Keep Tier 3 because it costs nothing, but do not expect much.

Relevant from the same paper: RSA correlated only **9%** with their language-model embedding,
which is good evidence that structural and evolutionary/sequence signals are genuinely
complementary rather than redundant.

---

## 5. Tier 4 — pretrained model outputs (strong, nearly free)

| Feature | Notes |
|---|---|
| **PeSTo scores** (protein / nucleic-acid / ion / ligand / lipid) | 5 columns, CPU inference ~5 s per protein. Trained on experimental PDB with a different architecture — genuinely complementary to our AF2-derived labels |
| **ProstT5 / ESM-2 embedding** (optional) | Large; only if a GPU appears. Reduce dimensionality (PCA to ~32) before feeding a tree model |

Including PeSTo as a feature has a cost worth naming: it makes "beat PeSTo" ambiguous as a
success criterion. **Train two versions** — with and without the PeSTo columns — and report both.
The without-PeSTo model is the honest comparison; the with-PeSTo model is probably the better
predictor.

---

## 6. What the papers tell us to expect

- **Secondary structure has a known direction.** Foldseek-Interface found interfaces — both
  ordered and disorder-mediated — are **enriched in turns, bends and unassigned conformations**
  and **depleted in helices and strands** relative to CATH folds. So DSSP state is not just a
  generic categorical; expect turns/bends/coil to carry positive weight.
- **Amino acid composition recapitulates biochemistry.** PeSTo's interface residue distributions
  matched known preferences (Arg/Lys at nucleic-acid interfaces, hydrophobics at lipid sites).
  A sanity check for us: if predicted interface composition does not look like known PPI
  propensity, something is wrong upstream.
- **Disorder-mediated interfaces are the hard case** for every method in all three papers. Record
  a per-residue disorder prediction (or use low pLDDT as a proxy) as a **stratification variable**
  for error analysis, even if it is not a model input.

---

## 7. What NOT to include — leakage

| Do not use | Why |
|---|---|
| **Anything computed from the complex** | ΔSASA on complexation, inter-chain contact counts, partner-derived anything. This is the label |
| **pLDDT from the dimer model** | AF2 confidence at a predicted interface is correlated with the interface itself. Use pLDDT from the **monomer** model, and only as a mask, not a feature |
| **Residue number / chain length as raw values** | Invites memorising specific proteins. Normalised position is fine |
| **Anything derived from the PPI database annotation** | Partner count, STRING score — these are labels wearing a feature costume |
| **Features computed on a different structure than inference will use** | The §1 bound-conformation problem |

The pLDDT one is subtle and worth repeating: in a **dimer** prediction, residues at the predicted
interface frequently have systematically different confidence. Feeding that in leaks the answer.

---

## 8. Concrete starting set

Roughly 70 columns before neighbourhood expansion, ~150 after. That is appropriate for an
effective sample size of 10³–10⁴ *proteins* (not residues — residues within a protein are
correlated, so proteins are what count).

```
geometric      RSA, SASA, HSE-up, HSE-down, coord-8/12/16, CX, depth,
               dist-centroid-norm, curvature-mean, curvature-gauss,
               phi, psi, CB-normal-angle, ss8 (one-hot), 3di (categorical)      ~30
composition    aa-composition-8Å (20), aa-composition-12Å (20),
               3di-composition-8Å (20)                                          ~60  → PCA to ~20
evolution      entropy, JSD, neff, taxonomic-breadth, neighbourhood-JSD          ~5
physchem       aa one-hot (20), hydrophobicity, charge, volume,
               polarizability, pI, interface-propensity                         ~26
pretrained     pesto-protein, -nucleic, -ion, -ligand, -lipid                     5
neighbourhood  8 Å and 12 Å means of all continuous features above            ×2 on ~45
```

**Start smaller than this.** Tier 1 + conservation + PeSTo, roughly 50 columns, gets most of the
way. Add Tier 3 only after establishing the baseline. More features on a small effective sample
size buys variance, not accuracy.

---

## 9. How to validate the feature set

1. **Baselines first.** RSA alone, conservation alone, PeSTo alone. Know what you must beat before
   adding anything.
2. **Ablate by tier**, not by individual feature. Drop Tier 2, Tier 3, Tier 4 in turn and report
   the PR AUC change — the same design ProSiteHunter used.
3. **SHAP on the surface subset only.** If RSA still dominates after buried residues are excluded,
   the geometry features are not doing their job.
4. **Check feature stability across the bound/unbound split** (§1). A feature that is highly
   important on dimer-extracted chains but shifts distribution on monomer models is a liability
   regardless of its training importance.
5. **Correlation audit.** Tree models tolerate correlated features but SHAP attribution becomes
   unreadable. Cluster features by correlation and keep one representative per cluster for the
   interpretability analysis, even if the model keeps all of them.

---

## 10. Next artifact

`../scripts/geometric_features.py` computes the dependency-free part of Tier 1 — HSE,
coordination numbers, protrusion, depth proxy, neighbourhood composition, normalised position —
from coordinates alone. RSA, DSSP and 3Di need FreeSASA/DSSP/Foldseek and belong in the full
`features.py`, which must be the **single shared module** used by both training and inference
(`Inference_And_Calibration.md` §1).
