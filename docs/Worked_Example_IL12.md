# Worked example — `humanPPI/P29459_P29460`

Feature extraction end to end on one real dimer, plus a validation check that **changes the
plan**. Outputs in `../examples/`.

The pair is **IL-12**: P29459 = IL-12 subunit α (p35), P29460 = IL-12 subunit β (p40).
File: `P29459_S0__P29460_S0__AF.pdb`.

---

## 1. What the file actually contains

| | chain A (P29459) | chain B (P29460) |
|---|---|---|
| Residues | 184 | 300 |
| Residue range | **36–219** | **21–325** |
| Contiguous | yes | **no** (5 gaps) |
| B-factor | **all 0.00** | all 0.00 |

Three facts settled by looking:

**Numbering is already full-protein.** Chain A runs 36–219 and P29459 is 219 aa, so the PDB
carries full-protein numbering, not segment-local 1..L. `build_interface_labels.py --numbering
full` is correct for this release — no `segment_def` offset needed. This was the blocking question
from `HumanPPI_Dataset_Notes.md`.

**`.contacts` and `.npz` are absent.** Only PDBs were downloaded. Labels must be derived
geometrically (§2), and the soft-label idea from `Interface_Model_Design.md` §3 is unavailable
until those files are fetched. Worth getting them — the npz confidence matrix is the better target.

**pLDDT is missing, and it depends on the model type.** Scanning 400 files:

| Model type | files | with pLDDT |
|---|---|---|
| `AF` | 63 | **0** |
| `AF1`–`AF5` | 77 | 55 |
| `AFMM1`–`AFMM5` | 260 | 260 |

`__AF` files (AlphaFold2 model 3) carry **no** B-factor data at all. This example is an `AF` file,
so the pLDDT ≥ 70 mask cannot be applied to it. Either prefer `AFMM` models where both exist, or
make the confidence filter conditional. A pipeline that assumes pLDDT is always present will
silently treat every `AF` residue as pLDDT 0 and discard the whole chain.

*(Also: 252 of 300 sampled pair-folders are empty — the download looks partial.)*

---

## 2. Labels — and why the cutoff choice matters

Interface residues on chain A, by heavy-atom distance < 5.0 Å to any chain-B atom:

```
26 of 184 residues (14%)
```

Applying Foldseek's **10 Å Cα** criterion to the same chain instead:

```
38 residues  — 46% more positives
```

That is the concrete version of the warning in `Parameter_Selection.md` §1. Foldseek's 10 Å Cα
cutoff selects residues to *encode and align*, where over-inclusion is cheap. As an ML label it
adds 12 second-shell residues that never touch p40, and the model would learn "near an interface".

**Of the 26 interface residues, 5 are buried (RSA < 0.20).** Those are excluded from training
entirely — they are neither positives nor negatives (`Feature_Specification.md` §0.1). That leaves
**21 positives among 119 surface residues**, roughly 1:5.

---

## 3. Features — computed on the monomer, never the complex

Chain A was written out alone to `examples/IL12a_P29459.pdb`, then:

```bash
python3 scripts/geometric_features.py examples/IL12a_P29459.pdb -o geom_IL12a.tsv   # 103 columns
```
plus RSA from FreeSASA on the monomer. Result: `examples/IL12a_features.tsv`, 184 rows × 177 columns — the 103 geometric
columns above, 70 physicochemical columns, and the label columns. Partner-derived
quantities (`rsa_complex`, `delta_rsa`) live in `examples/IL12a_labels.tsv`, not here.

Sample rows (`cd8` = coord_8, `hseU/D` = half-sphere exposure, `cx` = protrusion, `vcD/vcSep/vcFB`
= virtual-centre distance / sequence separation / backbone-fallback, `compL` = Leu fraction within
8 Å, `cd8nb` = coord_8 averaged over 8 Å neighbours):

```
resnum  aa   rsa  cd8 hseU hseD   cx   vcD  vcSep vcFB compL cd8nb   y
    42   Q  0.34    9    7   26  0.57  5.85     1    1  0.33 10.50   1   <- interface
    45   L  0.21   12   17   24  0.47  3.35   158    0  0.08 10.92   1   <- interface
    66   S  0.26    9    7   20  0.68  5.47     1    1  0.00  9.00   1   <- interface
    36   P  0.84    5    9   14  0.70  5.18   129    0  0.20  8.67   0   <- surface, non-interface
    38   L  0.21   10    9   25  0.60  3.33   172    0  0.00  9.45   0   <- surface, non-interface
    39   H  0.48    7    4   25  0.68  5.79     1    1  0.14  9.25   0   <- surface, non-interface
    41   S  0.00   13   21   24  0.33  1.75   165    0  0.31 10.79   0   <- BURIED, excluded
    43   N  0.15    9   16   18  0.55  5.49   115    0  0.22 10.60   0   <- BURIED, excluded
```

Look at residues 42 (interface) and 39 (non-interface surface): similar RSA, similar coordination,
both in virtual-centre backbone fallback. **The rows do not obviously separate.** That is the
honest picture, and §5 quantifies it.

### The leakage demonstration

Computing RSA on the **complex** instead of the monomer and taking ΔRSA:

| | interface | non-interface | AUC (surface only) |
|---|---|---|---|
| `delta_rsa` (monomer − complex) | 0.176 | 0.0008 | **0.999** |

ΔRSA is not a strong feature — it *is* the label, restated. Any feature touching the partner chain
will look spectacular in cross-validation and be worthless at inference, when there is no partner.
This is the concrete form of the leakage table in `Feature_Specification.md` §7.

---

## 4. Single-protein results — which turned out to be misleading

On IL-12 chain A, AUC within the surface subset:

| feature | AUC (surface) |
|---|---|
| `vc_unsatisfied` | 0.682 |
| `coord_8_nb8` | 0.656 |
| `vc_nn_dist` | 0.655 |
| `coord_8` | 0.649 |
| `rsa_monomer` | 0.354 |
| `dist_centroid_norm` | 0.323 |

Encouraging for the virtual-centre hypothesis — and, as it turned out, **a fluke**.

Note `rsa_monomer` at 0.354: within the surface subset, interface residues are *less* exposed than
other surface residues. RSA's apparent power comes almost entirely from separating buried from
exposed, which is the split we deliberately removed. Exactly the §0.1 prediction, confirmed.

---

## 5. Validation across 25 chains — the finding that matters

The single-protein numbers rest on 21 positives, so I repeated the measurement over 25 chains from
different pairs (median 20 interface residues on the surface each):

| feature | median AUC | IQR | % of chains > 0.5 |
|---|---|---|---|
| `rsa_monomer` | 0.537 | [0.459, 0.580] | 56% |
| `protrusion_cx` | 0.524 | [0.474, 0.588] | 72% |
| `vc_backbone_fallback` | 0.527 | [0.431, 0.571] | 56% |
| `vc_nn_dist` | 0.508 | [0.446, 0.569] | 56% |
| `coord_8` | 0.507 | [0.421, 0.609] | 52% |
| `vc_unsatisfied` | 0.499 | [0.442, 0.597] | 44% |
| `dist_centroid_norm` | 0.443 | [0.372, 0.566] | 36% |

**No single hand-crafted geometric feature separates interface from non-interface within the
surface subset.** Every median sits at chance and every IQR straddles 0.5. IL-12's 0.682 for
`vc_unsatisfied` was sampling noise; across 25 chains its median is 0.499.

### What this does and does not mean

It does **not** mean the model will fail. Tree models exploit interactions among features, and no
individual feature needs to be informative alone. But it does mean three things:

1. **Lower the expectation for hand-crafted geometry.** PeSTo needed a learned encoder over
   neighbourhoods from 8 to 64 residues to extract geometric signal. Summary statistics of those
   neighbourhoods evidently discard most of it. The CPU-only constraint costs more than hoped.
2. **Conservation and PeSTo scores move up in priority.** Tier 2 and Tier 4 of
   `Feature_Specification.md` are now likely to carry the signal, not Tier 1. Get
   `segment_omicMSAs.tar.gz` and run PeSTo early rather than late.
3. **The virtual-centre hypothesis is not supported** on this evidence. Keep the features — they
   cost nothing and may contribute in combination — but `Foldseek_3Di_Notes.md` §3.2 should be
   read as "tested, did not replicate". Do not build a story on it.

### Caveats on the caveat

25 chains with ~20 positives each is itself a small study, and per-chain AUC is noisy at that size.
The labels are geometric calls on AF2 models, not experimental interfaces. A properly pooled
analysis over thousands of chains could still show modest effects that per-chain medians miss.
The conclusion to carry forward is **"no large single-feature effect"**, not "geometry is useless".

---

## 6. What to do next, revised

1. **Run the Tier 1 + conservation + PeSTo comparison early**, on a few hundred chains, before
   building the full pipeline. If geometry alone cannot beat the RSA baseline by a clear margin,
   that decides the architecture.
2. **Fetch `.contacts` and `.npz`** — the per-residue-pair AF2 probabilities are better labels
   than geometric cutoffs, and they were not in this download.
3. **Handle the `AF` vs `AFMM` pLDDT difference** explicitly in the feature code.
4. **Pool properly** when evaluating features: per-chain AUC at n≈20 is too noisy to rank features.
   Use pooled AUC with chain as a grouping variable, or a mixed model.

## 7. Reproducing this

```bash
# labels + monomer extraction + features  (see section 3)
python3 scripts/geometric_features.py examples/IL12a_P29459.pdb -o geom_IL12a.tsv
pip install freesasa
```
Files: `examples/IL12a_P29459.pdb`, `examples/IL12a_labels.tsv`,
`examples/IL12a_features.tsv`, `examples/IL12a_epitopes_by_partner.tsv`.
