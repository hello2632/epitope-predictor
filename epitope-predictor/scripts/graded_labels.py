#!/usr/bin/env python3
"""
Graded interface labels: buried surface area per residue on complexation.

A binary 5 A contact call treats a residue scraping one atom of the partner the
same as one burying 80 A^2 at the core of the interface.  Buried area separates
them, and it is free -- the dimers are already on disk.

    dSASA(residue) = SASA in the isolated chain  -  SASA in the dimer

aggregated per protein as the MAXIMUM over every partner it was modelled with,
matching the partner-agnostic union used for the binary labels: a residue counts
as deeply buried if ANY partner buries it deeply.

Reported two ways:
    max_dsasa      absolute, A^2
    buried_frac    max_dsasa / Tien reference area for that residue type,
                   clipped to [0, 1] -- dimensionless and comparable between a
                   glycine and a tryptophan

NOTE ON LEAKAGE.  dSASA is computed WITH the partner present, so it must never
become a feature: Worked_Example_IL12.md section 3 measured delta-RSA at AUC
0.999 because it IS the label.  That is exactly why it belongs here, in the
label table, and why `rsa_complex` and `delta_rsa` stay on the drop list for
every feature consumer.
"""

import argparse
import os
import sys
import tempfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

MAX_ASA = dict(zip("ARNDCQEGHILKMFPSTWYV",
                   [129, 274, 195, 193, 167, 225, 223, 104, 224, 197, 201, 236,
                    224, 240, 159, 155, 172, 285, 263, 174]))
THREE_TO_ONE = dict(zip(
    ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE",
     "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"],
    "ARNDCEQGHILKMFPSTWYV"))
FAMILY_RANK = {"AFMM": 0, "AFn": 1, "AF": 2}


def family(mt):
    return "AFMM" if mt.startswith("AFMM") else ("AF" if mt == "AF" else "AFn")


def split_chains(path):
    out = defaultdict(list)
    try:
        with path.open() as fh:
            for line in fh:
                if not line.startswith("ATOM") or line[16] not in (" ", "A"):
                    continue
                if line[26] != " ":
                    continue
                el = line[76:78].strip() or line[12:16].strip()[0]
                if el in ("H", "D"):
                    continue
                out[line[21]].append(line)
    except OSError:
        return {}
    return out


def residue_sasa(lines, tmpdir):
    """freesasa on an atom-line list -> {(chain, resnum): total SASA}.

    Keyed by CHAIN AND NUMBER, never by number alone: the two chains of a dimer
    routinely share residue numbers, and collapsing them lets one chain's areas
    overwrite the other's, which silently corrupts every dSASA."""
    import freesasa
    freesasa.setVerbosity(freesasa.silent)
    fd, name = tempfile.mkstemp(suffix=".pdb", dir=tmpdir)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write("".join(lines))
            fh.write("END\n")
        st = freesasa.Structure(name)
        res = freesasa.calc(st)
        areas = res.residueAreas()
    except Exception:
        return None
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass
    out = {}
    for chain_label, ch in areas.items():
        for num_s, a in ch.items():
            try:
                out[(chain_label, int(num_s))] = a.total
            except (ValueError, TypeError):
                continue
    return out


def process(task):
    pair, pdb, p1, p2, root, tmpdir = task
    chains = split_chains(root / pair / pdb)
    ids = sorted(chains)
    if len(ids) != 2:
        return []
    # one pass for the dimer; chain-aware keys make a second pass unnecessary
    in_cplx = residue_sasa(chains[ids[0]] + chains[ids[1]], tmpdir)
    if in_cplx is None:
        return []

    out = []
    for cid, acc in zip(ids, (p1, p2)):
        alone = residue_sasa(chains[cid], tmpdir)
        if alone is None:
            continue
        rname = {}
        for l in chains[cid]:
            rname[int(l[22:26])] = l[17:20].strip()
        for (ch_, num), a_alone in alone.items():
            rn3 = rname.get(num)
            aa = THREE_TO_ONE.get(rn3)
            if aa is None:
                continue
            d = a_alone - in_cplx.get((cid, num), a_alone)
            if d > 0.1:                       # ignore numerical noise
                out.append((acc, num, aa, float(d)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pilot", type=Path)
    ap.add_argument("--humanppi", type=Path, required=True)
    ap.add_argument("-j", "--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = ap.parse_args()

    seeds = set((args.pilot / "seeds.txt").read_text().split())
    lines = (args.pilot / "manifest.tsv").read_text().splitlines()
    col = {n: i for i, n in enumerate(lines[0].split("\t"))}
    rows = [l.split("\t") for l in lines[1:] if l]

    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r[col["pair"]]].append(r)

    with tempfile.TemporaryDirectory() as tmp:
        tasks = []
        for pair, cands in by_pair.items():
            cands.sort(key=lambda r: (r[col["has_plddt"]] != "yes",
                                      FAMILY_RANK[family(r[col["model_type"]])],
                                      r[col["model_type"]]))
            r = cands[0]
            tasks.append((pair, r[col["pdb"]], r[col["protein1"]],
                          r[col["protein2"]], args.humanppi, tmp))

        best = defaultdict(float)
        aa_of = {}
        nburied = defaultdict(set)
        done = 0
        with Pool(args.workers) as pool:
            for res in pool.imap_unordered(process, tasks, chunksize=8):
                done += 1
                for acc, num, aa, d in res:
                    if acc not in seeds:
                        continue
                    key = (acc, num)
                    if d > best[key]:
                        best[key] = d
                    aa_of[key] = aa
                    nburied[acc].add(num)

    out = args.pilot / "labels_graded.tsv"
    vals = []
    with out.open("w") as fh:
        fh.write("protein\tresnum\tmax_dsasa\tburied_frac\n")
        for (acc, num), d in sorted(best.items()):
            frac = min(1.0, d / MAX_ASA[aa_of[(acc, num)]])
            vals.append((d, frac))
            fh.write(f"{acc}\t{num}\t{d:.2f}\t{frac:.4f}\n")

    v = np.asarray([x[0] for x in vals])
    f = np.asarray([x[1] for x in vals])
    print(f"{done} pairs processed, {len(best)} residues with dSASA > 0.1 A^2 "
          f"on {len(nburied)} seed proteins")
    print(f"\nmax_dsasa (A^2) : median {np.median(v):6.1f}  "
          f"p90 {np.percentile(v,90):6.1f}  max {v.max():6.1f}")
    print(f"buried_frac     : median {np.median(f):6.3f}  "
          f"p90 {np.percentile(f,90):6.3f}")
    for t in (10, 25, 50, 100):
        print(f"  residues burying > {t:>3} A^2 : {int((v > t).sum()):>7}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
