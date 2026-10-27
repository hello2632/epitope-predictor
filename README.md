# Epitope Predictor

Predicts which surface patches of a target protein could be bound, and hands
back hotspot residues for minibinder and VHH design.

Input is **one unbound chain**. Output is a ranked patch list with 4 hotspots
each — the form BindCraft and BoltzGen take.

```
patch 1: 10 residues, mean score 0.272
   hotspots  THR208  VAL207  ARG203  LEU45
   residues  42 45 46 49 50 203 204 207 208 211
```

## Install

Tested on **Python 3.14.4**. The pinned `pandas 3.x` / `numpy 2.4` require a
recent interpreter.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`freesasa` is a compiled extension and the most likely install failure on a
fresh machine; if pip cannot build it, `conda install -c conda-forge freesasa`.

**scikit-learn is pinned exactly.** The shipped model is a pickle, and loading
it under a different minor version can fail or change behaviour silently.

## Predict

```bash
python scripts/predict.py examples/IL12a_P29459.pdb \
    --model models/model_final.joblib --top 3
```

The chain must be **unbound** — the partner must not be present in the file.
Scoring a chain with its partner still attached leaks the answer into the
features and the result is meaningless.

Useful flags: `--residues-out` writes per-residue scores, `--top N` controls how
many patches print. The full patch table is always written to a `.patches.tsv`.

Runtime is about 3 seconds for a 184-residue chain.

## What the numbers mean

Out-of-fold on 3,529 proteins:

| | Model | Random | Lift |
|---|---|---|---|
| Top patch hits a real epitope | **64%** | 41% | **1.57×** |
| One of top 3 hits | 79% | 59% | 1.33× |
| Top 4 hotspots are true contacts | **49%** | 28% | **1.74×** |

**Test two or three patches, not one.** 54% of predicted patches land on
nothing — that is the wasted-design-run rate. Single-epitope targets are the
hard case (63% hit@1) and are 75% of proteins; targets with three or more sites
reach 74%.

A high score means *"AlphaFold2 would place a human protein interface here"*,
not *"this is a real binding site"*. See [models/MODEL_CARD.md](models/MODEL_CARD.md)
for training data, limitations, and the security note on loading pickles.

## Training data — getting HumanPPI

The model is trained on the Cong-lab HumanPPI set of predicted human dimer
structures. It is **not included here**; download it yourself:

```bash
mkdir -p data && cd data
wget https://conglab.swmed.edu/humanPPI/downloads/best_models.tar.gz
tar xzf best_models.tar.gz
```

Only `best_models.tar.gz` is needed — the other bulk downloads (MSAs, contact
maps, full model sets) are not used. Expect roughly 15 GB unpacked.

The result is one directory per interacting pair, named `ACC1_ACC2` by UniProt
accession, each holding predicted dimer PDBs:

```
data/humanPPI/
├── O00116_O15228/
│   ├── AFMM1.pdb        AlphaFold-Multimer models, ranked
│   ├── AF1.pdb ...      other predictors
└── O00124_O43462/ ...
```

Every script that needs it takes `--humanppi data/humanPPI`. Nothing is
downloaded at runtime.

**Model-quality note.** Model types differ in whether they carry pLDDT in the
B-factor column: `AFMM1–5` always do, `AF1–5` sometimes, plain `AF` never. The
pipeline prefers models that carry it (`AFMM > AF1–5 > AF`) and skips the
confidence filter for chains that do not, rather than discarding them — the
no-pLDDT models are ~17% of the corpus.

## Retraining

Nine steps, in order. Every step writes into one working directory (`WORK`
below), so a run is resumable — if a step fails, fix it and re-run from there.

```bash
WORK=run1
HPPI=data/humanPPI
```

### 1 · Choose the seed proteins

```bash
python scripts/make_pilot_subset.py $HPPI -o $WORK     --n-seeds 999999 --min-partners 1 --max-partners 999999 --prefix ''
```

A *seed* is a protein whose epitopes you want to learn; its *partners* are the
other chains it is modelled with. The flags above take **everything** (12,296
seeds). To rehearse on a small subset, use `--n-seeds 500 --min-partners 3
--max-partners 8`, which is what the original pilots did.

> **That filter changes the answer, not just the runtime.** Proteins with 3–8
> partners carry several epitopes over much of their surface, so they are
> *easier*: random guessing scores 52% hit@1 there against 41% on the
> unfiltered set. Never compare absolute rates between a filtered and an
> unfiltered run — compare lift over the matched random baseline.

Writes `manifest.tsv`, `seeds.txt`, `folders.txt`.

### 2–3 · Labels

```bash
python scripts/build_pilot_labels.py $WORK --humanppi $HPPI
python scripts/graded_labels.py      $WORK --humanppi $HPPI -j 48
```

**What a label is.** A residue is **interface** if any of its heavy atoms lies
within **5.0 Å** of any heavy atom of the partner chain — the PeSTo convention,
not Foldseek-Interface's 10 Å Cα rule, which over-includes (38 residues vs 26 on
IL-12α chain A, the extra twelve never touching the partner).

Labels are the **union across every partner** a protein was modelled with. That
is deliberate: the question is "which residues *can* bind something", not "where
does partner X bind". A protein seen with three partners contributes all three
sites as positives.

`graded_labels.py` adds ΔSASA on complexation per residue — how much surface the
partner actually buries — which separates a real contact from a residue scraping
one atom. Step 4 consumes it; skip it and `max_dsasa` silently becomes 0.

Writes `labels.tsv`, `per_pair.tsv`, `labels_graded.tsv`.

### 4 · Training table

```bash
python scripts/build_training_table.py $WORK --humanppi $HPPI -j 48
```

Extracts each seed as an isolated monomer into `$WORK/monomers/` and assigns
classes:

| Residue | Class |
|---|---|
| Surface (RSA ≥ 0.20) and interface | **positive** |
| Surface and not interface | **negative** |
| Buried (RSA < 0.20) | **excluded entirely** |

**Buried residues are dropped from training *and* from every metric.** Keep them
as negatives and the model partly learns "is this residue exposed?" instead of
"is this an interface", which flatters every number you report.

Negatives mean *"no partner was modelled here"*, not *"does not bind"* — the
noise floor this imposes is why patch precision tops out where it does.

### 5–6 · Features

Both run on the monomers from step 4, never on the complex.

```bash
mkdir -p $WORK/features
ls $WORK/monomers/*.pdb | xargs -P 48 -I{} sh -c   'b=$(basename "{}" .pdb); python scripts/geometric_features.py "{}" -o '$WORK'/features/"$b".tsv'

python scripts/physchem_features.py $WORK/monomers/*.pdb -o $WORK/physchem -j 48
```

`geometric_features.py` takes one structure at a time (hence `xargs`);
`physchem_features.py` takes them all at once. Together they produce **170
numeric columns** per residue: packing density, directional exposure, shape and
position, virtual-centre neighbour geometry, sequence locality, neighbourhood
composition, amino-acid identity, physicochemistry, and neighbourhood means of
the last two.

No MSA, no conservation, no pretrained embedding — coordinates and chemistry
only, which is what keeps this CPU-only. See
[docs/Feature_Specification.md](docs/Feature_Specification.md).

> **Bound-conformation caveat.** Features come from the chain as it sits in the
> predicted complex, so side chains are already arranged for binding. At
> inference you give it an *unbound* structure. PeSTo measured ~7 ROC points
> between the two; expect some loss.

### 7 · Splits

```bash
python scripts/make_splits.py $WORK -j 48 --prefilter-above 9999999
```

Three things can put the same information on both sides of a split: the same
protein twice, homologues, and both sides of one interface. `make_splits.py`
handles all three. Two details matter if you change it:

- **Identity requires coverage.** Scoring identity as `matches / min(length)`
  alone makes a 10-residue chain read as a homologue of thousands of proteins.
  Alignments must cover ≥50% of **both** sequences (`--min-coverage`).
- **Clustering is greedy set-cover, not single-linkage.** Single-linkage chains
  A~B~C into runaway components. Set-cover defines membership against a
  centroid, so it cannot chain.

Writes `splits.tsv`, plus `identity_pairs.tsv` caching every alignment at or
above 25% identity. **Keep that cache.** The alignments are the slow part — about
2 hours for 12,296 proteins on 48 cores — while re-clustering from the cache
takes 35 seconds, so the identity threshold and clustering mode can be changed
without paying for the alignments again.

### 8–9 · Evaluate, then fit

```bash
python scripts/evaluate_retrieval.py $WORK --humanppi $HPPI
python scripts/train_final.py $WORK
```

`evaluate_retrieval.py` reports cross-validated retrieval metrics — hit@1,
hit@3, patch precision, hotspot precision — each against a random baseline
measured on the same proteins under identical rules. `train_final.py` then fits
one model on all folds and writes `model_final.joblib` with its feature column
order, inference parameters and provenance.

**The holdout stays sealed.** `train_final.py` excludes it by default. Pass
`--include-holdout` only when you have stopped making decisions; there is no way
back.

### Runtime

Full corpus on 48 cores, ~4 hours end to end: subset 13 min, labels 20 min,
graded 6 min, table 3 min, features 8 min, **splits 2 h**, evaluate 87 min,
fit 4 min. Peak memory ~23 GB.

### One rule worth keeping

`predict.py` calls the feature scripts as **subprocesses** rather than
reimplementing them, so training and inference cannot drift apart. Keep
`scripts/` intact as a unit; `predict.py` will not work copied out on its own.

## Documentation

**[models/MODEL_CARD.md](models/MODEL_CARD.md)** — training data, metrics,
limitations, intended use. Start here.

`docs/` holds the four notes the scripts cite directly:

| File | Why it is here |
|---|---|
| `Feature_Specification.md` | what is computed per residue, and what is deliberately left out |
| `Inference_And_Calibration.md` | patch extraction and what the score does and does not mean |
| `Worked_Example_IL12.md` | the IL-12α reference values the pipeline regression-checks against |
| `Target_ID_ML_Plan.md` | overall plan — dataset, features, evaluation |

These were written during development and describe exploratory work, not all of
which is shipped here. Where their numbers predate 2026-10-17 they are superseded
by [models/MODEL_CARD.md](models/MODEL_CARD.md); corrections are marked inline.

## License

MIT — see [LICENSE](LICENSE). The license covers this code and model; it does
not cover HumanPPI or any third-party data you supply.
