#!/usr/bin/env python3
"""
Leak-free train/test splits, computed in-house from the HumanPPI structures.

Three things can put the same information on both sides of a split.  Grouping by
protein, which is what the pilot did, closes only the first:

  1. the same protein appearing twice                  -- group by protein
  2. HOMOLOGUES of each other                           -- needs sequence identity
  3. BOTH SIDES OF ONE INTERFACE.  If protein A is in train and its partner B is
     in test, the model has already seen that exact interface from the other
     side.  In the 500-seed pilot this affects 139 proteins, 28% of the set.
     Grouping by protein does not catch it; grouping by interacting component
     does.

All three are handled by clustering proteins into components and assigning whole
components to folds.

HOMOLOGY IS CLUSTERED BY GREEDY SET-COVER, NOT SINGLE-LINKAGE.  Single-linkage
merges A with B and B with C, so a chain of weak 30% links runs away even though
A and C share nothing.  Measured on the full 12,296-protein set, that produced
ten components each pinned at the size cap, together holding 52% of all
proteins -- not families, just one percolated blob sliced into cap-sized pieces,
which then scatter across folds and leak.

Set-cover cannot chain: repeatedly take the protein with the most unassigned
neighbours as a centroid and make a cluster of it plus everything at or above
the identity threshold TO THAT CENTROID.  Every member is similar to the
representative, which is what "a 30% identity cluster" is supposed to mean, and
is what MMseqs2 cluster and CD-HIT do by default.  The algorithm is implemented
here; no external tool is called.

SEQUENCE IDENTITY WITHOUT MMSEQS2.  Sequences are read from the monomer PDBs and
aligned with Bio.Align.PairwiseAligner (Smith-Waterman, BLOSUM62, C-implemented).
For a few thousand proteins an all-against-all pass is affordable across a
process pool.  Above --prefilter-above it switches to a k-mer prefilter and only
aligns candidate pairs, which is the same prefilter-then-align shape MMseqs2
uses, just slower and without spaced seeds.

HONEST LIMITATION: exact-kmer prefiltering loses sensitivity for remote
homology.  All-against-all mode (the default at pilot scale) has no such gap, but
in prefilter mode expect pairs near 30% identity to be missed.  Report which mode
was used.
"""

import argparse
import os
import sys
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
    "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

_SEQS = None            # set per worker process


class Union:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def join(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def read_sequence(path: Path) -> str:
    seen, seq = set(), []
    for line in path.open():
        if not line.startswith("ATOM") or line[16] not in (" ", "A"):
            continue
        key = line[22:27]
        if key in seen:
            continue
        seen.add(key)
        seq.append(THREE_TO_ONE.get(line[17:20].strip(), "X"))
    return "".join(seq)


def _init(seqs):
    global _SEQS
    _SEQS = seqs


def _identity(pair):
    """Local alignment identity AND coverage of both sequences.

    IDENTITY ALONE IS NOT ENOUGH.  Normalising matches by the shorter sequence
    makes any short chain that matches a motif inside a long protein score near
    1.0.  Measured on the full set, a 10-residue chain (P60331) came out ">= 30%
    identical" to 5,502 proteins, and 26% of all >= 30% pairs had a shorter
    sequence under 50 aa.  That is not homology, and it drove every component
    pathology seen so far.

    Coverage is returned as the SMALLER of the two per-sequence coverages, so a
    short-inside-long hit scores low on the long side and is rejected.  This is
    what MMseqs2 (--cov-mode 0) and CD-HIT enforce; only the idea is reused.
    """
    from Bio import Align
    from Bio.Align import substitution_matrices
    a, b = pair
    sa, sb = _SEQS[a], _SEQS[b]
    al = Align.PairwiseAligner(mode="local", open_gap_score=-11,
                               extend_gap_score=-1)
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    try:
        aln = al.align(sa, sb)[0]
    except Exception:
        return a, b, 0.0, 0.0
    ta, tb = aln[0], aln[1]
    matches = sum(1 for x, y in zip(ta, tb) if x == y and x != "-")
    na = sum(1 for x in ta if x != "-")        # residues of a inside the alignment
    nb = sum(1 for x in tb if x != "-")
    ident = matches / max(1, min(len(sa), len(sb)))
    cov = min(na / max(1, len(sa)), nb / max(1, len(sb)))
    return a, b, ident, cov


def setcover_clusters(names, hits, thresh):
    """Greedy set-cover on the identity graph.  Returns a list of name sets.

    Each round takes the still-unassigned protein with the most unassigned
    neighbours as the centroid, and clusters it with every unassigned protein at
    or above `thresh` identity TO THAT CENTROID.  Because membership is defined
    against the centroid rather than transitively, no chain of weak links can
    grow a component -- the failure mode that single-linkage has here.
    """
    import heapq
    adj = defaultdict(set)
    for ident, a, b, _cov in hits:
        if ident >= thresh:
            adj[a].add(b)
            adj[b].add(a)

    remaining = set(names)
    heap = [(-len(adj[p]), p) for p in names]
    heapq.heapify(heap)
    clusters = []
    while heap:
        negd, p = heapq.heappop(heap)
        if p not in remaining:
            continue
        live = adj[p] & remaining
        # lazy heap: degrees only shrink, so a stale entry is re-pushed, never lost
        if len(live) != -negd:
            heapq.heappush(heap, (-len(live), p))
            continue
        members = live | {p}
        clusters.append(members)
        remaining -= members
    clusters.extend({p} for p in sorted(remaining))
    return clusters


def load_pairs(path):
    """Read a cached identity table.  Coverage is NaN in pre-coverage caches."""
    out, has_cov = [], False
    with open(path) as fh:
        head = next(fh, "").rstrip("\n").split("\t")
        has_cov = len(head) >= 4
        for line in fh:
            f = line.rstrip("\n").split("\t")
            out.append((float(f[2]), f[0], f[1],
                        float(f[3]) if has_cov else float("nan")))
    return out, has_cov


def candidate_pairs(seqs, k, min_shared):
    """k-mer prefilter: pairs sharing at least `min_shared` distinct k-mers."""
    idx = defaultdict(set)
    for name, s in seqs.items():
        for i in range(len(s) - k + 1):
            idx[s[i:i + k]].add(name)
    shared = defaultdict(int)
    for names in idx.values():
        if len(names) < 2 or len(names) > 400:      # skip ubiquitous k-mers
            continue
        for a, b in combinations(sorted(names), 2):
            shared[(a, b)] += 1
    return [p for p, c in shared.items() if c >= min_shared]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pilot", type=Path)
    ap.add_argument("--identity", type=float, default=0.30,
                    help="merge proteins at or above this identity (default 0.30)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--holdout", type=float, default=0.15,
                    help="fraction of residues sealed as a final test set")
    ap.add_argument("--prefilter-above", type=int, default=3000,
                    help="protein count above which to use the k-mer prefilter")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--min-shared", type=int, default=3)
    ap.add_argument("-j", "--workers", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--cluster-mode", choices=("setcover", "single"),
                    default="setcover",
                    help="homology clustering: greedy set-cover (default, cannot "
                         "chain) or single-linkage (percolates; kept to reproduce "
                         "the 2026-10-17 full run)")
    ap.add_argument("--pairs-from", type=Path, default=None,
                    help="reuse a cached identity_pairs.tsv instead of realigning")
    ap.add_argument("--min-coverage", type=float, default=0.50,
                    help="an alignment must cover at least this fraction of BOTH "
                         "sequences (default 0.50).  Without it a 10-residue "
                         "chain reads as a homologue of 5,502 proteins.")
    ap.add_argument("--pair-floor", type=float, default=0.25,
                    help="cache every pair at or above this identity so the "
                         "threshold can be varied without realigning (default 0.25)")
    ap.add_argument("--cap", type=float, default=0.05,
                    help="no component may exceed this fraction of all residues "
                         "(default 0.05); prevents the partner graph percolating")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    tbl = pd.read_csv(args.pilot / "training_table.tsv", sep="\t")
    usable = tbl[tbl["class"].isin(["positive", "negative"])]
    nres = usable.groupby("protein").size().to_dict()
    prots = sorted(nres)
    print(f"{len(prots)} proteins with usable residues")

    seqs = {}
    for p in prots:
        f = args.pilot / "monomers" / f"{p}.pdb"
        if f.exists():
            seqs[p] = read_sequence(f)
    names = sorted(seqs)

    # ---- identity pairs: reuse the cache, or align and write one -----------
    #
    # The all-against-all pass is the most expensive step in the pipeline (118
    # minutes for 12,296 proteins).  The first full run discarded its result, so
    # changing the CLUSTERING -- which costs seconds -- meant paying for the
    # ALIGNMENTS again.  Everything at or above --pair-floor is now cached, so
    # the identity threshold and the clustering mode can both be varied for free.
    cache = args.pairs_from or (args.pilot / "identity_pairs.tsv")
    if cache.exists():
        hits, has_cov = load_pairs(cache)
        print(f"identity: reusing {len(hits)} cached pairs from {cache}")
        if not has_cov:
            # Backfill coverage by realigning ONLY the cached candidate pairs --
            # a few hundred thousand, not the full N^2.  Seconds, not hours.
            print(f"  cache predates coverage; realigning {len(hits)} cached "
                  f"pairs to add it (not the full all-against-all)")
            todo = [(a, b) for _i, a, b, _c in hits]
            redone = []
            with Pool(args.workers, initializer=_init, initargs=(seqs,)) as pool:
                for a, b, ident, cov in pool.imap_unordered(_identity, todo,
                                                            chunksize=256):
                    redone.append((ident, a, b, cov))
            hits = redone
            tmp = cache.with_suffix(".tmp")
            with tmp.open("w") as fh:
                fh.write("a\tb\tidentity\tcoverage\n")
                for ident, a, b, cov in sorted(hits, reverse=True):
                    fh.write(f"{a}\t{b}\t{ident:.4f}\t{cov:.4f}\n")
            tmp.replace(cache)
            print(f"  cache upgraded with coverage -> {cache}")
    else:
        if len(names) <= args.prefilter_above:
            pairs = list(combinations(names, 2))
            mode = "all-against-all"
        else:
            pairs = candidate_pairs(seqs, args.k, args.min_shared)
            mode = f"k-mer prefilter (k={args.k}, >={args.min_shared} shared)"
        print(f"identity mode: {mode}  ->  {len(pairs)} pairs to align")
        hits = []
        if pairs:
            with Pool(args.workers, initializer=_init, initargs=(seqs,)) as pool:
                for a, b, ident, cov in pool.imap_unordered(_identity, pairs,
                                                            chunksize=256):
                    if ident >= args.pair_floor:
                        hits.append((ident, a, b, cov))
        tmp = cache.with_suffix(".tmp")
        with tmp.open("w") as fh:
            fh.write("a\tb\tidentity\tcoverage\n")
            for ident, a, b, cov in sorted(hits, reverse=True):
                fh.write(f"{a}\t{b}\t{ident:.4f}\t{cov:.4f}\n")
        tmp.replace(cache)          # atomic: a crash never leaves a partial cache
        print(f"  cached {len(hits)} pairs at or above "
              f"{args.pair_floor:.0%} identity -> {cache}")

    if args.pairs_from and args.pair_floor > args.identity:
        sys.exit(f"cache floor {args.pair_floor:.0%} is above the requested "
                 f"identity {args.identity:.0%}; pairs between them are missing")

    total_res = sum(nres.get(p, 0) for p in prots)
    uf = Union(prots)
    sizes = {p: nres.get(p, 0) for p in prots}
    comp_size = dict(sizes)
    budget = args.cap * max(1, total_res)

    refused_pairs = []

    def try_merge(a, b):
        """Join a and b unless the result would exceed the component cap."""
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            return None                       # already together
        if comp_size[ra] + comp_size[rb] > budget:
            refused_pairs.append((a, b))      # may become residual leakage
            return False                      # refused
        uf.join(a, b)
        merged = comp_size.pop(ra) + comp_size.pop(rb)
        comp_size[uf.find(a)] = merged
        return True

    # HOMOLOGY CLUSTERING.  30% identity is the field convention and a Class C
    # parameter, so it is not tuned here; what changed is the ALGORITHM.
    #
    # Single-linkage chains: A~B, B~C, and the component runs away although A and
    # C share nothing.  On the full set that gave ten components all pinned at
    # the cap, holding 52% of the proteins between them.  The cap did not stop
    # the percolation, it only cut the blob into arbitrary pieces, and pieces of
    # one blob land in different folds -- which is the leak.
    #
    # Set-cover defines membership against a centroid instead, so it cannot
    # chain.  Single-linkage is kept behind --cluster-mode for reproducing the
    # earlier run.
    n_before = sum(1 for h in hits if h[0] >= args.identity)
    hits = [h for h in hits
            if h[0] >= args.identity and h[3] >= args.min_coverage]
    hits.sort(reverse=True)                   # highest identity first
    print(f"  {n_before} pairs at or above {args.identity:.0%} identity; "
          f"{len(hits)} also cover >= {args.min_coverage:.0%} of BOTH sequences "
          f"({n_before - len(hits)} dropped as short-inside-long artefacts)")
    n_hom_refused = 0
    if args.cluster_mode == "setcover":
        clusters = setcover_clusters(names, hits, args.identity)
        multi = [c for c in clusters if len(c) > 1]
        n_hom = 0
        for c in multi:
            it = sorted(c)
            for other in it[1:]:
                r = try_merge(it[0], other)
                if r is True:
                    n_hom += 1
                elif r is False:
                    n_hom_refused += 1
        big = max((len(c) for c in clusters), default=0)
        print(f"  set-cover: {len(clusters)} homology clusters "
              f"({len(multi)} with >1 member), largest {big} proteins")
    else:
        n_hom = 0
        for ident, a, b, _cov in hits:
            r = try_merge(a, b)
            if r is True:
                n_hom += 1
            elif r is False:
                n_hom_refused += 1
        print(f"  {len(hits)} pairs at or above {args.identity:.0%} identity: "
              f"single-linkage merged {n_hom}, {n_hom_refused} refused by the cap")
    if n_hom_refused:
        print(f"  {n_hom_refused} homology merges refused by the {args.cap:.0%} cap")

    # ---- merge interacting partners, with a size cap ----------------------
    #
    # Homology is transitive, so union-find is the right tool for it.  PARTNERSHIP
    # IS NOT: A binds B and B binds C does not make A and C related, yet union-find
    # merges all three.  In a dense interactome that percolates.  Measured:
    #
    #     500 seeds   0.17 partner edges/seed   largest component   15 proteins (3%)
    #    2000 seeds   0.44 partner edges/seed   largest component 1067 proteins (56%)
    #
    # The giant component made one fold hold 53% of the data and the other four
    # ~5% each, which is not a usable cross-validation.  It gets worse with scale.
    #
    # So partner merges are applied greedily and REFUSED when the result would
    # exceed `cap` of all residues.  Most partner pairs still land together --
    # the leak they create is closed -- while no component can swallow the split.
    # Refused pairs are counted and reported, because they are residual leakage
    # and the number belongs on the page rather than hidden.
    n_part = n_refused = 0
    for line in (args.pilot / "folders.txt").read_text().split():
        a, _, b = line.partition("_")
        if a not in uf.p or b not in uf.p:
            continue
        r = try_merge(a, b)
        if r is True:
            n_part += 1
        elif r is False:
            n_refused += 1
    print(f"  {n_part} partner pairs merged (both sides of one interface)")
    if n_refused:
        print(f"  {n_refused} refused: merging would exceed the {args.cap:.0%} "
              f"component cap -- residual leakage, reported not hidden")

    comps = defaultdict(list)
    for p in prots:
        comps[uf.find(p)].append(p)
    comps = sorted(comps.values(), key=lambda c: -sum(nres.get(p, 0) for p in c))
    print(f"\n{len(comps)} independent components "
          f"(was {len(prots)} if grouping by protein alone)")
    sizes = [len(c) for c in comps]
    print(f"  component size: median {int(np.median(sizes))}, "
          f"max {max(sizes)} proteins")

    # ---- seal a held-out set, then fold the rest --------------------------
    total = total_res
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(comps))
    held, acc = set(), 0
    for ci in order:
        if acc >= args.holdout * total:
            break
        for p in comps[ci]:
            held.add(p)
        acc += sum(nres.get(p, 0) for p in comps[ci])

    rest = [c for i, c in enumerate(comps) if not (set(c) & held)]
    load = [0] * args.folds
    fold_of = {}
    for c in sorted(rest, key=lambda c: -sum(nres.get(p, 0) for p in c)):
        f = int(np.argmin(load))               # greedy balance by residue count
        for p in c:
            fold_of[p] = f
        load[f] += sum(nres.get(p, 0) for p in c)

    # ---- residual leakage, REMOVED from the measurement -------------------
    #
    # PARTNERSHIP IS NOT TRANSITIVE -- A binds B and B binds C does not relate A
    # and C -- so union-find over partner edges percolates, and the cap only
    # truncates the runaway at an arbitrary point.  Capping alone leaves pairs
    # split across folds: protein A trains while its partner B is tested, and the
    # model has already seen that interface from A's side.
    #
    # Those pairs cannot all be kept together (one protein gets exactly one
    # fold), so instead of reporting the leak we EXCLUDE the contaminated
    # proteins from the metrics.  They remain valid TRAINING data -- only their
    # use as test cases is unsound.  This bounds the damage by a counted number
    # rather than leaving an unquantified bias in the headline.
    split_of = {p: ("holdout" if p in held else f"fold{fold_of.get(p, 0)}")
                for p in prots}
    leaky = set()
    for a, b in refused_pairs:
        if a in split_of and b in split_of and split_of[a] != split_of[b]:
            leaky.add(a)
            leaky.add(b)
    evaluable = [p for p in prots if p not in leaky and split_of[p] != "holdout"]
    print(f"\n  {len(refused_pairs)} merges refused by the {args.cap:.0%} cap")
    print(f"  {len(leaky)} proteins land in a different split from a refused "
          f"partner/homologue -- marked leaky=1, excluded from metrics, still "
          f"usable for training")
    print(f"  {len(evaluable)} proteins remain evaluable out of "
          f"{len(prots) - len(held)} non-holdout")

    out = args.pilot / "splits.tsv"
    with out.open("w") as fh:
        fh.write("protein\tcomponent\tsplit\tn_residues\tleaky\n")
        comp_id = {p: i for i, c in enumerate(comps) for p in c}
        for p in prots:
            sp = split_of[p]
            fh.write(f"{p}\t{comp_id[p]}\t{sp}\t{nres.get(p, 0)}"
                     f"\t{1 if p in leaky else 0}\n")

    print(f"\nheld out : {len(held)} proteins, {acc} residues ({acc/total:.1%}) "
          f"-- never train, never tune, look once")
    for f in range(args.folds):
        ps = [p for p, v in fold_of.items() if v == f]
        print(f"  fold {f}  : {len(ps):>4} proteins, {load[f]:>6} residues")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
