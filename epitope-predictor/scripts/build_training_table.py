#!/usr/bin/env python3
"""
Turn the aggregated interface labels into a trainable per-residue table.

Everything comes from HumanPPI; nothing is fetched or predicted.

For each seed protein one source chain is chosen -- the longest modelled segment
across all folders that protein appears in, preferring models that carry pLDDT.
That chain is written out ALONE, with the partner deleted, and all features are
computed on it.  The partner is used only to make labels, never to make features:
anything computed in the presence of the partner is the label restated, which
Worked_Example_IL12.md section 3 measured at AUC 0.999 for delta-RSA.

Residues are then sorted into four classes, following Target_ID_ML_Plan.md 1.3:

  positive          surface, contacts some partner in some model
  negative          surface, modelled but never seen contacting anything
  excluded_buried   RSA below the surface cutoff -- trivially non-interface, and
                    including them teaches the model "is this residue exposed?"
  excluded_plddt    low-confidence region; disordered predictions are confident
                    looking nonsense.  __AF models carry no B-factor at all, so
                    this filter is skipped for them rather than failing them all.

PERFORMANCE.  SASA dominates everything else, so it uses freesasa (C) rather than
Bio.PDB.SASA.ShrakeRupley (pure Python): profiled at 98 ms vs 509 ms per chain.
Each pair PDB is read once, not once per chain, and the per-protein work runs
across a process pool.  Measured on 500 proteins: 5 m 18 s -> about 12 s.

KNOWN LIMITATION: the chain is extracted from a dimer, so it is in the BOUND
conformation -- side chains already arranged for binding, loops already placed.
PeSTo measured a 7-point ROC AUC drop between bound and unbound inputs. Numbers
from this table are therefore optimistic.
"""

import argparse
import os
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SURFACE_RSA = 0.20
MIN_PLDDT = 70.0
FAMILY_RANK = {"AFMM": 0, "AFn": 1, "AF": 2}
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def family(mt: str) -> str:
    return "AFMM" if mt.startswith("AFMM") else ("AF" if mt == "AF" else "AFn")


def split_chains(path: Path):
    """One pass over a PDB -> {chain_id: [atom lines]}, filtered to heavy atoms."""
    out = defaultdict(list)
    try:
        with path.open() as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                if line[16] not in (" ", "A") or line[26] != " ":
                    continue
                el = line[76:78].strip() or line[12:16].strip()[0]
                if el in ("H", "D"):
                    continue
                out[line[21]].append(line)
    except OSError:
        return {}
    return out


def scan_pair(task):
    """Worker: return (acc, n_residues, has_plddt, pair, pdb, chain_id, lines)."""
    pair, pdb_name, has_pl, p1, p2, root = task
    chains = split_chains(root / pair / pdb_name)
    ids = sorted(chains)
    if len(ids) != 2:
        return []
    res = []
    for cid, acc in zip(ids, (p1, p2)):
        lines = chains[cid]
        n = len({l[22:26] for l in lines})
        res.append((acc, n, has_pl, pair, pdb_name, cid, lines))
    return res


def score_protein(acc, rec, outdir, surface_rsa, min_plddt, labels):
    """Worker: write the monomer, compute RSA and pLDDT, classify every residue."""
    import freesasa
    freesasa.setVerbosity(freesasa.silent)

    _, has_pl, pair, pdb, cid, lines = rec
    mono = outdir / f"{acc}.pdb"
    mono.write_text("".join(lines) + "END\n")

    try:
        st = freesasa.Structure(str(mono))
        areas = freesasa.calc(st).residueAreas()
    except Exception as e:                      # keep one bad chain from killing the run
        return acc, [], f"freesasa failed: {type(e).__name__}"

    # per-residue mean B-factor = pLDDT, where the model carries one
    bsum, bcnt, rname = defaultdict(float), defaultdict(int), {}
    for l in lines:
        n = int(l[22:26])
        rname[n] = l[17:20].strip()
        try:
            bsum[n] += float(l[60:66])
            bcnt[n] += 1
        except ValueError:
            pass

    rows, mismatch = [], 0
    for ch in areas.values():
        for num_s, a in ch.items():
            num = int(num_s)
            rn = rname.get(num)
            if rn is None or rn not in STANDARD_AA:
                continue
            rsa = a.relativeTotal
            if rsa is None or not np.isfinite(rsa):
                continue
            pl = (bsum[num] / bcnt[num]) if (has_pl and bcnt[num]) else float("nan")

            lab = labels.get(num)
            y = None if lab is None else lab[0]
            if lab is not None and lab[1] != rn:
                mismatch += 1

            if rsa < surface_rsa:
                cls = "excluded_buried"
            elif has_pl and pl < min_plddt:
                cls = "excluded_plddt"
            elif y is None:
                cls = "unmodelled"
            else:
                cls = "positive" if y == 1 else "negative"
            rows.append((acc, num, rn, y, rsa, pl, cls, f"{pair}:{cid}"))
    rows.sort(key=lambda r: r[1])
    return acc, rows, mismatch


def _score_star(item, **kw):
    return score_protein(item[0], item[1], **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pilot", type=Path)
    ap.add_argument("--humanppi", type=Path, required=True)
    ap.add_argument("--surface-rsa", type=float, default=SURFACE_RSA)
    ap.add_argument("--min-plddt", type=float, default=MIN_PLDDT)
    ap.add_argument("-j", "--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = ap.parse_args()

    seeds = set((args.pilot / "seeds.txt").read_text().split())
    lines = (args.pilot / "manifest.tsv").read_text().splitlines()
    col = {n: i for i, n in enumerate(lines[0].split("\t"))}
    rows = [l.split("\t") for l in lines[1:] if l]

    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r[col["pair"]]].append(r)
    tasks = []
    for pair, cands in by_pair.items():
        cands.sort(key=lambda r: (r[col["has_plddt"]] != "yes",
                                  FAMILY_RANK[family(r[col["model_type"]])],
                                  r[col["model_type"]]))
        r = cands[0]
        tasks.append((pair, r[col["pdb"]], r[col["has_plddt"]] == "yes",
                      r[col["protein1"]], r[col["protein2"]], args.humanppi))

    # ---- choose one source chain per seed: longest segment, pLDDT preferred ----
    best = {}
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(scan_pair, tasks, chunksize=16):
            for acc, n, has_pl, pair, pdb, cid, ls in res:
                if acc not in seeds:
                    continue
                key = (n, has_pl)
                if acc not in best or key > (best[acc][0], best[acc][1]):
                    best[acc] = (n, has_pl, pair, pdb, cid, ls)
    print(f"{len(tasks)} pairs scanned, {len(best)} seed chains selected")

    # ---- labels, grouped per protein -----------------------------------------
    lab_of = defaultdict(dict)
    npart = {}
    for line in (args.pilot / "labels.tsv").read_text().splitlines()[1:]:
        if not line:
            continue
        p, n, rn, y, np_, _nc = line.split("\t")
        lab_of[p][int(n)] = (int(y), rn)
        npart[p] = int(np_)

    # graded labels (buried area on complexation), if graded_labels.py has run
    graded = {}
    gpath = args.pilot / "labels_graded.tsv"
    if gpath.exists():
        for line in gpath.read_text().splitlines()[1:]:
            if line:
                p_, n_, ds, bf = line.split("\t")
                graded[(p_, int(n_))] = (float(ds), float(bf))
        print(f"joined {len(graded)} graded labels from {gpath.name}")
    else:
        print("no labels_graded.tsv; max_dsasa / buried_frac will be 0")

    mono = args.pilot / "monomers"
    mono.mkdir(exist_ok=True)

    work = [(acc, best[acc]) for acc in sorted(best)]
    fn = partial(_score_star, outdir=mono, surface_rsa=args.surface_rsa,
                 min_plddt=args.min_plddt, labels={})
    # labels differ per protein, so they are bound per task rather than in partial
    results = []
    with Pool(args.workers) as pool:
        jobs = [pool.apply_async(score_protein,
                                 (acc, rec, mono, args.surface_rsa,
                                  args.min_plddt, lab_of.get(acc, {})))
                for acc, rec in work]
        for j in jobs:
            results.append(j.get())

    out_rows, tot, per_prot, mismatch, failed = [], defaultdict(int), {}, 0, []
    for acc, rows_, mm in results:
        if isinstance(mm, str):
            failed.append((acc, mm))
            continue
        mismatch += mm
        counts = defaultdict(int)
        for acc_, num, rn, y, rsa, pl, cls, src in rows_:
            counts[cls] += 1
            tot[cls] += 1
            ds, bf = graded.get((acc_, num), (0.0, 0.0))
            out_rows.append("\t".join([
                acc_, str(num), rn, "" if y is None else str(y),
                f"{rsa:.4f}", "" if not np.isfinite(pl) else f"{pl:.1f}",
                cls, str(npart.get(acc_, 0)), src,
                f"{ds:.2f}", f"{bf:.4f}"]))
        per_prot[acc] = counts

    tbl = args.pilot / "training_table.tsv"
    tbl.write_text(
        "protein\tresnum\tresname\tis_interface\trsa\tplddt\tclass"
        "\tn_partners\tsource\tmax_dsasa\tburied_frac\n"
        + "\n".join(out_rows) + "\n")

    n_all = sum(tot.values())
    pos, neg = tot["positive"], tot["negative"]
    if failed:
        print(f"\n{len(failed)} proteins failed: "
              + ", ".join(f"{a} ({m})" for a, m in failed[:5]))
    print(f"\nresidue identity mismatches between label and structure: {mismatch}"
          + ("   <-- numbering problem, stop here" if mismatch else "   (clean)"))
    print(f"\nresidues total            : {n_all}")
    for k in ("positive", "negative", "excluded_buried", "excluded_plddt", "unmodelled"):
        if tot[k]:
            print(f"  {k:<22}: {tot[k]:>7}  ({tot[k]/n_all:5.1%})")
    if pos + neg:
        print("\nSURFACE SUBSET (what the model sees)")
        print(f"  positives               : {pos}")
        print(f"  negatives               : {neg}")
        print(f"  class balance           : 1:{neg/pos:.1f}   "
              f"({pos/(pos+neg):.1%} positive)")
        fr = [c["positive"] / max(1, c["positive"] + c["negative"])
              for c in per_prot.values()]
        print(f"  per-protein positive rate : median {np.median(fr):.1%}, "
              f"min {min(fr):.1%}, max {max(fr):.1%}")

    ref = "P29459"
    if ref in per_prot:
        c = per_prot[ref]
        print(f"\nIL-12 check ({ref}): {c['positive']} positives among "
              f"{c['positive'] + c['negative']} surface residues"
              f"   (Worked_Example_IL12.md: 21 of 119)")
    print(f"\nwrote {tbl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
