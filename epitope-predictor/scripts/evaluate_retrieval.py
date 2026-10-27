#!/usr/bin/env python3
"""
Retrieval-style evaluation: can the model hand you a site worth designing against?

The earlier evaluation asked "can you rank every surface residue correctly" --
a whole-surface question that is not the one being asked.  For binder design the
question is:

    if I take the top N predicted patches and design against them,
    how often does at least one land on a real binding site?

That reframes this as retrieval, and changes four things.

1. EPITOPES ARE KEPT SEPARATE.  The training label unions every partner's
   contacts, which is correct -- "which residues can bind something" genuinely is
   the union.  But for evaluation the union destroys the structure that matters:
   measured on this pilot, 42% of proteins have two or more distinct sites, a
   median of 29.6 A apart.  Per-partner contact sets are therefore clustered by
   overlap coefficient (|A and B| / min(|A|,|B|)) at 0.25, single linkage.

2. A HIT IS LENIENT.  A patch hits an epitope when they share at least 3
   residues -- roughly the hotspot count BindCraft takes, so a hit means "this
   patch yields usable hotspots", not "this patch reproduces the interface".

3. TOP N, NOT TOP 1.  You would realistically test two to five sites, so hit
   rate is reported at N = 1, 3, 5.

4. PER-PROTEIN THRESHOLDING.  A global score cutoff leaves proteins with
   uniformly low scores producing no patch at all -- 28 of 401 under the old
   scheme.  For retrieval that is a hard failure, so each protein contributes its
   own top fraction of residues instead.

EPITOPE RECALL IS A DIAGNOSTIC, NOT A TARGET.  Foldseek-Interface's hand
inspection of this dataset found "different partners or fragments docked to the
same or multiple different surfaces" to be the largest artifact class, 23 of 39.
That is the same signal multi-epitope structure is built from, so some fraction
of the second sites are docking artifacts rather than alternative binding sites.
Optimising to find all of them risks training toward the artifact.

Every metric is reported for random and depth_proxy baselines under identical
rules: a 3-residue overlap on a surface that is 32% interface is hit by chance
more often than intuition suggests, and that number belongs on the page.
"""

import argparse
import sys
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pilot_labels import read_chains, interface_residues, family, FAMILY_RANK

LABEL_COLS = {"resnum", "resname", "aa", "protein", "is_interface", "class",
              "n_partners", "source", "plddt", "rsa_complex", "delta_rsa",
              "max_dsasa", "buried_frac", "y", "fold", "component", "split",
              "n_residues"}
OVERLAP_THRESH = 0.25     # Class C: fixed a priori, never tuned
MIN_SHARED = 3            # residues shared for a hit
MIN_FRAC = 0.25           # ... AND this fraction of the patch must be in it
PATCH_CUTOFF = 8.0        # Ca, Angstrom
MIN_PATCH = 4             # residues
TOP_FRAC = 0.25           # per-protein fraction of surface kept as candidate
N_HOTSPOT = 4             # residues handed to a design tool


def ca_coords(pdb: Path):
    out = {}
    for l in pdb.open():
        if l.startswith("ATOM") and l[12:16].strip() == "CA" and l[16] in " A":
            out[int(l[22:26])] = (float(l[30:38]), float(l[38:46]), float(l[46:54]))
    return out


def smooth(score, resnums, coords, radius):
    """Mean score over Ca neighbours within `radius`, self included.

    Inference_And_Calibration.md section 2 specifies this step and the first
    version of this script skipped it.  Interfaces are patches, so an isolated
    high-scoring residue is almost always noise; smoothing also stops a granular
    score from fragmenting one real site into several small patches, each too
    small to survive the >= 4 residue filter.
    """
    if radius <= 0:
        return score
    P = np.array([coords[r] for r in resnums])
    D = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    W = (D <= radius).astype(np.float64)
    return (W @ score) / W.sum(1)


def components(resnums, coords, keep, cutoff=PATCH_CUTOFF, min_size=MIN_PATCH):
    idx = [i for i, k in enumerate(keep) if k]
    if not idx:
        return []
    P = np.array([coords[resnums[i]] for i in idx])
    D = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    seen, out = set(), []
    for a in range(len(idx)):
        if a in seen:
            continue
        q, comp = deque([a]), []
        seen.add(a)
        while q:
            u = q.popleft(); comp.append(u)
            for v in np.where(D[u] <= cutoff)[0]:
                v = int(v)
                if v != u and v not in seen:
                    seen.add(v); q.append(v)
        if len(comp) >= min_size:
            out.append([idx[c] for c in comp])
    return out


def build_epitopes(pilot, humanppi, seeds, surface_of):
    """-> {protein: [set(resnum), ...]} distinct sites, restricted to surface."""
    lines = (pilot / "manifest.tsv").read_text().splitlines()
    col = {n: i for i, n in enumerate(lines[0].split("\t"))}
    by_pair = defaultdict(list)
    for l in lines[1:]:
        if l:
            r = l.split("\t")
            by_pair[r[col["pair"]]].append(r)

    per_partner = defaultdict(dict)
    for pair, cands in by_pair.items():
        cands.sort(key=lambda r: (r[col["has_plddt"]] != "yes",
                                  FAMILY_RANK[family(r[col["model_type"]])],
                                  r[col["model_type"]]))
        r = cands[0]
        chains = read_chains(humanppi / pair / r[col["pdb"]])
        if len(chains) != 2:
            continue
        cids = sorted(chains)
        iface = interface_residues(chains)
        if min(len(iface[c]) for c in cids) < 4:
            continue
        accs = (r[col["protein1"]], r[col["protein2"]])
        for cid, acc in zip(cids, accs):
            if acc not in seeds:
                continue
            other = accs[1] if acc == accs[0] else accs[0]
            s = iface[cid] & surface_of.get(acc, set())
            if s:
                per_partner[acc][other] = s

    epitopes = {}
    for acc, d in per_partner.items():
        names = sorted(d)
        parent = {n: n for n in names}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in combinations(names, 2):
            A, B = d[a], d[b]
            if A and B and len(A & B) / min(len(A), len(B)) >= OVERLAP_THRESH:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
        groups = defaultdict(set)
        for n in names:
            groups[find(n)] |= d[n]
        eps = [g for g in groups.values() if len(g) >= MIN_SHARED]
        if eps:
            epitopes[acc] = eps
    return epitopes


def evaluate(score, prot, resnum, y, dsasa, coords_of, epitopes,
             top_frac=TOP_FRAC, smooth_radius=0.0):
    """-> per-protein records of patch retrieval performance."""
    recs = []
    for p in np.unique(prot):
        m = prot == p
        eps = epitopes.get(p)
        if not eps or p not in coords_of:
            continue
        coords = coords_of[p]
        rn = resnum[m]
        if not all(r in coords for r in rn):
            continue
        s = smooth(score[m], rn, coords, smooth_radius)
        k = max(MIN_PATCH, int(round(top_frac * m.sum())))
        keep = np.zeros(m.sum(), bool)
        keep[np.argsort(-s)[:k]] = True
        comps = components(rn, coords, keep)
        comps.sort(key=lambda c: -(len(c) * s[c].mean()))

        hits = []          # per patch: set of epitope indices hit
        for c in comps:
            res = set(rn[c])
            # Two conditions, not one.  The absolute count alone is far too
            # easy to satisfy: a smoothed random field produces one large blob,
            # and since ~32% of the surface is interface it shares 3 residues
            # with an epitope almost every time -- random scored 61% hit@1 under
            # the count-only rule.  Requiring a quarter of the PATCH to lie in
            # one epitope keeps small-overlap tolerance while making a blob that
            # covers everything fail, which is correct: you cannot pick hotspots
            # out of a 60-residue patch that is only 8% real.
            hits.append({i for i, e in enumerate(eps)
                         if len(res & e) >= MIN_SHARED
                         and len(res & e) / len(res) >= MIN_FRAC})

        rec = dict(protein=p, n_epitopes=len(eps), n_patches=len(comps))
        for N in (1, 3, 5):
            found = set().union(*hits[:N]) if hits[:N] else set()
            rec[f"hit@{N}"] = len(found) > 0
            rec[f"recall@{N}"] = len(found) / len(eps)

        # PATCH PRECISION -- the number hit@N cannot express.
        # hit@N asks "did any suggestion land on a real site"; it says nothing
        # about how many suggestions were wrong.  Five patches with one correct
        # scores as a success at hit@5 while four design runs target nothing.
        # This is the waste figure: of the patches handed over, what fraction
        # are real.  Counted over all patches and over the top 3, since in
        # practice only a handful get tested.
        rec["n_patch_hit"] = sum(1 for h in hits if h)
        rec["patch_prec"] = (rec["n_patch_hit"] / len(comps)) if comps else 0.0
        top3 = hits[:3]
        rec["patch_prec@3"] = (sum(1 for h in top3 if h) / len(top3)) if top3 else 0.0

        # Hotspots are read off the TOP-RANKED patch whether or not it hits.
        # Taking them from the first *hitting* patch conditions the metric on
        # success, which is why random previously scored above the model: its
        # few accidental hits land on proteins where almost anything is right.
        # This is also what happens in practice -- patch 1 goes to the design tool.
        rec["hotspot_prec"] = 0.0
        rec["hotspot_dsasa"] = np.nan
        if comps:
            c = comps[0]
            order = np.array(c)[np.argsort(-s[c])][:N_HOTSPOT]
            rec["hotspot_prec"] = float(y[m][order].mean())
            true = order[y[m][order] == 1]
            if len(true):
                rec["hotspot_dsasa"] = float(np.median(dsasa[m][true]))
        recs.append(rec)
    return pd.DataFrame(recs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pilot", type=Path)
    ap.add_argument("--humanppi", type=Path, required=True)
    ap.add_argument("--smooth", default="0,8,10,12",
                    help="comma-separated smoothing radii in Angstrom; 0 = off")
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier

    lab = pd.read_csv(args.pilot / "training_table.tsv", sep="\t")
    lab = lab[lab["class"].isin(["positive", "negative"])].copy()
    lab["y"] = (lab["class"] == "positive").astype(int)
    sp = pd.read_csv(args.pilot / "splits.tsv", sep="\t")
    sp = sp[sp["split"] != "holdout"].copy()
    sp["fold"] = sp["split"].str.replace("fold", "", regex=False).astype(int)
    # LEAKY PROTEINS: an interface the model already saw from the partner's side
    # (make_splits could not keep the pair in one fold).  They stay in TRAINING
    # -- the data is real -- but they are not sound TEST cases, so they are
    # dropped from the metrics below.  Older splits.tsv files have no such
    # column; treat them as all-clean and say so.
    if "leaky" not in sp.columns:
        sp["leaky"] = 0
        print("note: splits.tsv predates leakage marking -- metrics include "
              "partner-contaminated proteins (regenerate with make_splits.py)")
    d = lab.merge(sp[["protein", "component", "fold", "leaky"]], on="protein")

    cols = []
    for sub in ("features", "physchem"):
        fr = [pd.read_csv(args.pilot / sub / f"{a}.tsv", sep="\t").assign(protein=a)
              for a in d["protein"].unique() if (args.pilot / sub / f"{a}.tsv").exists()]
        t = pd.concat(fr, ignore_index=True)
        keep = [c for c in t.columns
                if c not in LABEL_COLS and pd.api.types.is_numeric_dtype(t[c])]
        d = d.merge(t[["protein", "resnum"] + keep], on=["protein", "resnum"])
        cols += keep

    d = d.sort_values(["protein", "resnum"], kind="mergesort").reset_index(drop=True)
    X = np.nan_to_num(d[cols].to_numpy(np.float32), nan=0., posinf=0., neginf=0.)
    y = d["y"].to_numpy()
    fold = d["fold"].to_numpy()
    prot = d["protein"].to_numpy()
    resnum = d["resnum"].to_numpy()
    dsasa = d["max_dsasa"].to_numpy()

    oof = np.full(len(y), np.nan)
    for k in range(fold.max() + 1):
        tr, te = fold != k, fold == k
        m = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=.06, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]

    leaky_set = set(d.loc[d["leaky"] == 1, "protein"].unique())
    if leaky_set:
        print(f"excluding {len(leaky_set)} partner-contaminated proteins from the "
              f"metrics ({len(leaky_set) / max(1, d['protein'].nunique()):.1%} of "
              f"the evaluable set); they remain in training")

    surface_of = d.groupby("protein")["resnum"].apply(set).to_dict()
    coords_of = {}
    for p in surface_of:
        f = args.pilot / "monomers" / f"{p}.pdb"
        if f.exists():
            coords_of[p] = ca_coords(f)

    epitopes = build_epitopes(args.pilot, args.humanppi, set(surface_of), surface_of)
    ne = np.array([len(v) for v in epitopes.values()])
    print(f"{d['protein'].nunique()} proteins, {len(epitopes)} with epitopes mapped")
    print(f"epitopes per protein: 1 site {np.mean(ne==1):.0%}, "
          f"2 {np.mean(ne==2):.0%}, 3+ {np.mean(ne>=3):.0%}, mean {ne.mean():.2f}")
    print(f"hit rule: patch shares >= {MIN_SHARED} residues with an epitope; "
          f"top {TOP_FRAC:.0%} of each protein's surface kept\n")

    rng = np.random.default_rng(0)
    runs = [("GBM", oof),
            ("depth_proxy", d["depth_proxy"].to_numpy()),
            ("random", rng.random(len(y)))]
    radii = [float(x) for x in args.smooth.split(",")]
    fold_of = d.groupby("protein")["fold"].first().to_dict()
    nfold = int(d["fold"].max()) + 1

    # Evaluate every radius once, then select per outer fold.  The smoothing
    # radius is a Class B parameter -- it changes X, not y -- so it may be
    # tuned, but only on training folds.  Sweeping it and reading the test
    # numbers, which the previous run did, is tuning on test.
    cache = {}
    for name, sc in runs:
        for rad in radii:
            cache[(name, rad)] = evaluate(sc, prot, resnum, y, dsasa, coords_of,
                                          epitopes, smooth_radius=rad)

    # Drop partner-contaminated proteins from EVERY metric table at once, so the
    # exclusion applies to the radius selection and the sensitivity sweep too,
    # not only the headline.  Filtering later would leave leaky proteins
    # steering the Class B radius choice.
    # DOES PARTNER CONTAMINATION ACTUALLY INFLATE THE SCORE?  Excluding every
    # contaminated protein costs most of the evaluation set at full scale, which
    # would shift the population far more than the leak it removes.  So measure
    # the effect instead of assuming it: the model is partner-agnostic and sees
    # only the monomer, so seeing an interface from the other side is a much
    # weaker leak than homology.  Headline stays on the CLEAN subset; the
    # comparison below says what excluding them bought.
    if leaky_set:
        t = cache[("GBM", radii[len(radii) // 2])]
        lk = t[t.protein.isin(leaky_set)]
        cl = t[~t.protein.isin(leaky_set)]
        if len(lk) and len(cl):
            print(f"\n--- partner-contamination check (GBM, "
                  f"{radii[len(radii)//2]:.0f}A) ---")
            print(f"  leaky  n={len(lk):>5}  hit@1 {lk['hit@1'].mean():.0%}  "
                  f"hit@3 {lk['hit@3'].mean():.0%}  "
                  f"patch_prec {lk['patch_prec'].mean():.0%}")
            print(f"  clean  n={len(cl):>5}  hit@1 {cl['hit@1'].mean():.0%}  "
                  f"hit@3 {cl['hit@3'].mean():.0%}  "
                  f"patch_prec {cl['patch_prec'].mean():.0%}")
            print(f"  difference in hit@1: "
                  f"{lk['hit@1'].mean() - cl['hit@1'].mean():+.1%}  "
                  f"(positive = contamination inflates the score)")
        before = cache[(runs[0][0], radii[0])].protein.nunique()
        cache = {k: t[~t.protein.isin(leaky_set)].reset_index(drop=True)
                 for k, t in cache.items()}
        after = cache[(runs[0][0], radii[0])].protein.nunique()
        print(f"  headline computed on the clean subset: "
              f"{before} -> {after} proteins\n")

    def nested(name, criterion="hit@3"):
        """Per outer fold: pick the radius on the other folds, apply it here."""
        picked, parts = {}, []
        for k in range(nfold):
            best, best_v = radii[0], -1.0
            for rad in radii:
                t = cache[(name, rad)]
                tr = t[t.protein.map(fold_of) != k]
                v = tr[criterion].mean() if len(tr) else 0.0
                if v > best_v:
                    best, best_v = rad, v
            picked[k] = best
            t = cache[(name, best)]
            parts.append(t[t.protein.map(fold_of) == k])
        return pd.concat(parts, ignore_index=True), picked

    print("--- radius selected per outer fold (on training folds only) ---")
    results, chosen = {}, {}
    for name, _ in runs:
        results[name], chosen[name] = nested(name)
        print(f"  {name:<14}" + "  ".join(f"fold{k}:{int(v)}A"
                                          for k, v in chosen[name].items()))

    rand_ref = results["random"]
    print(f"\n{'model':<14}{'cover':>7}{'hit@1':>8}{'lift':>7}{'hit@3':>8}"
          f"{'lift':>7}{'rec@5':>8}{'Pprec':>7}{'P@3':>6}{'hotspot':>9}{'lift':>7}{'dSASA':>8}")
    print("-" * 98)
    for name, _ in runs:
        r = results[name]
        l1 = r['hit@1'].mean() / max(1e-9, rand_ref['hit@1'].mean())
        l3 = r['hit@3'].mean() / max(1e-9, rand_ref['hit@3'].mean())
        lh = r.hotspot_prec.mean() / max(1e-9, rand_ref.hotspot_prec.mean())
        # pooled over patches, not averaged over proteins: a protein with 8
        # patches should weigh more than one with 2, since that is where the
        # wasted runs actually are
        pp = r.n_patch_hit.sum() / max(1, r.n_patches.sum())
        print(f"{name:<14}{(r.n_patches>0).mean():>6.0%} {r['hit@1'].mean():>7.0%} "
              f"{l1:>6.2f}x {r['hit@3'].mean():>7.0%} {l3:>6.2f}x "
              f"{r['recall@5'].mean():>7.0%} {pp:>6.0%} {r['patch_prec@3'].mean():>5.0%} "
              f"{r.hotspot_prec.mean():>8.0%} {lh:>6.2f}x {r.hotspot_dsasa.median():>7.0f}")

    g0 = results["GBM"]
    print(f"\n  Pprec = patch precision, pooled over all "
          f"{int(g0.n_patches.sum())} predicted patches: the fraction that land on a")
    print(f"          real epitope.  {1 - g0.n_patch_hit.sum()/max(1,g0.n_patches.sum()):.0%} "
          f"of patches hit nothing -- that is the wasted-design-run rate.")
    print(f"  P@3   = the same over each protein's top 3 patches only.")
    print(f"\n  hit rule: >= {MIN_SHARED} shared residues AND >= {MIN_FRAC:.0%} "
          f"of the patch inside one epitope")
    print(f"  lift     : rate divided by the random baseline under identical rules")

    print("\n--- sensitivity: fixed radius, no selection (not the headline) ---")
    print(f"{'radius':>8}" + "".join(f"{n:>14}" for n, _ in runs))
    for rad in radii:
        row = "".join(f"{cache[(n, rad)]['hit@3'].mean():>13.0%} " for n, _ in runs)
        print(f"{rad:>7.0f}A{row}")

    g = results["GBM"]
    print(f"\n  n = {len(g)} proteins.")
    print(f"  'hotspot' = fraction of the top {N_HOTSPOT} residues of the "
          f"TOP-RANKED patch that are true contacts (0 if no patch)")
    print(f"  'dSASA'   = median buried area of those that are, A^2")

    print("\n--- single-site vs multi-site proteins (GBM) ---")
    for lo, hi, lab_ in ((1, 2, "1 epitope"), (2, 3, "2 epitopes"), (3, 99, "3+ epitopes")):
        b = g[(g.n_epitopes >= lo) & (g.n_epitopes < hi)]
        if len(b):
            print(f"  {lab_:<12} n={len(b):>4}  hit@1 {b['hit@1'].mean():>4.0%}  "
                  f"hit@3 {b['hit@3'].mean():>4.0%}  hit@5 {b['hit@5'].mean():>4.0%}  "
                  f"recall@5 {b['recall@5'].mean():>4.0%}")

    g.to_csv(args.pilot / "retrieval_gbm.tsv", sep="\t", index=False,
             float_format="%.4f")
    print(f"\nwrote {args.pilot/'retrieval_gbm.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
