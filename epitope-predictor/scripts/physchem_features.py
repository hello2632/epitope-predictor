#!/usr/bin/env python3
"""
Tier 3 per-residue features: amino-acid identity and physicochemistry.

The geometric feature set has neighbourhood COMPOSITION (aa_comp_8_*, aa_comp_12_*)
but drops the residue's own identity -- `aa` and `resname` are string columns and
were being excluded as non-numeric.  So the model has never been told what amino
acid it is looking at, which is a foundational input for interface prediction.

Emitted per residue:

    aa_onehot_[A..V]        20   identity
    blosum62_[A..V]         20   substitution profile, from Biopython
    hydropathy               1   Kyte-Doolittle
    volume                   1   Zamyatnin side-chain volume, A^3
    charge                   1   formal charge at pH 7
    max_asa                  1   Tien et al. reference area, A^2
    is_aromatic/polar/charged/small/donor/acceptor   6
    <scalar>_nb8 / _nb12    18   means over Ca neighbours at 8 and 12 A

ON INTERFACE PROPENSITY.  Feature_Specification.md section 4 lists a
"literature-derived per-residue PPI propensity" scale.  It is deliberately NOT
included: the published scales are easy to misremember, and deriving one from
this dataset would be target encoding, which leaks unless it is fitted per
training fold.  The 20-way one-hot gives any model everything a fixed propensity
scale would, learned from the data instead of asserted.
"""

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

AA3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
THREE_TO_ONE = dict(zip(AA3, "ARNDCEQGHILKMFPSTWYV"))
AA1 = sorted(THREE_TO_ONE.values())

# Kyte & Doolittle 1982
HYDROPATHY = dict(zip("ARNDCQEGHILKMFPSTWYV",
                      [1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2, 4.5,
                       3.8, -3.9, 1.9, 2.8, -1.6, -0.8, -0.7, -0.9, -1.3, 4.2]))
# Zamyatnin 1972, side-chain volume A^3
VOLUME = dict(zip("ARNDCQEGHILKMFPSTWYV",
                  [88.6, 173.4, 114.1, 111.1, 108.5, 143.8, 138.4, 60.1, 153.2,
                   166.7, 166.7, 168.6, 162.9, 189.9, 112.7, 89.0, 116.1, 227.8,
                   193.6, 140.0]))
# Tien et al. 2013 theoretical maximum ASA
MAX_ASA = dict(zip("ARNDCQEGHILKMFPSTWYV",
                   [129, 274, 195, 193, 167, 225, 223, 104, 224, 197, 201, 236,
                    224, 240, 159, 155, 172, 285, 263, 174]))
CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}
AROMATIC = set("FWYH")
POLAR = set("STNQCYH")
CHARGED = set("DEKRH")
SMALL = set("AGSCP")
DONOR = set("KRWNQHSTY")
ACCEPTOR = set("DENQSTY")

SCALARS = ["hydropathy", "volume", "charge", "max_asa", "is_aromatic",
           "is_polar", "is_charged", "is_small", "is_donor", "is_acceptor"]


def read_residues(path: Path):
    """-> list of (resnum, one-letter code, CA coordinate)."""
    out, seen = [], set()
    for line in path.open():
        if not line.startswith("ATOM") or line[16] not in (" ", "A"):
            continue
        if line[12:16].strip() != "CA" or line[26] != " ":
            continue
        rn = line[17:20].strip()
        if rn not in THREE_TO_ONE:
            continue
        num = int(line[22:26])
        if num in seen:
            continue
        seen.add(num)
        out.append((num, THREE_TO_ONE[rn],
                    (float(line[30:38]), float(line[38:46]), float(line[46:54]))))
    return out


def build(path: Path, blosum):
    res = read_residues(path)
    if not res:
        return None
    nums = np.array([r[0] for r in res], dtype=np.int32)
    seq = [r[1] for r in res]
    xyz = np.array([r[2] for r in res], dtype=np.float64)
    n = len(res)

    onehot = np.zeros((n, 20), np.float32)
    blos = np.zeros((n, 20), np.float32)
    scal = np.zeros((n, len(SCALARS)), np.float32)
    for i, a in enumerate(seq):
        onehot[i, AA1.index(a)] = 1.0
        for j, b in enumerate(AA1):
            blos[i, j] = blosum[a, b]
        scal[i] = [HYDROPATHY[a], VOLUME[a], CHARGE.get(a, 0.0), MAX_ASA[a],
                   a in AROMATIC, a in POLAR, a in CHARGED, a in SMALL,
                   a in DONOR, a in ACCEPTOR]

    # neighbourhood means over Ca neighbours, excluding self
    D = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    nb = {}
    for r in (8.0, 12.0):
        m = (D <= r).astype(np.float32)
        cnt = np.maximum(m.sum(1, keepdims=True), 1.0)
        nb[r] = (m @ scal) / cnt

    cols = (["resnum"]
            + [f"aa_onehot_{a}" for a in AA1]
            + [f"blosum62_{a}" for a in AA1]
            + SCALARS
            + [f"{s}_pc_nb8" for s in SCALARS]
            + [f"{s}_pc_nb12" for s in SCALARS])
    data = np.column_stack([nums, onehot, blos, scal, nb[8.0], nb[12.0]])
    return cols, data


def _one(task):
    path, outdir = task
    from Bio.Align import substitution_matrices
    blosum = substitution_matrices.load("BLOSUM62")
    r = build(path, blosum)
    if r is None:
        return path.stem, 0
    cols, data = r
    lines = ["\t".join(cols)]
    for row in data:
        lines.append(str(int(row[0])) + "\t"
                     + "\t".join(f"{v:.4g}" for v in row[1:]))
    (outdir / f"{path.stem}.tsv").write_text("\n".join(lines) + "\n")
    return path.stem, data.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdb", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("-j", "--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [(p, args.out) for p in args.pdb]
    tot = 0
    with Pool(args.workers) as pool:
        for name, n in pool.imap_unordered(_one, tasks, chunksize=8):
            tot += n
    ncol = 1 + 20 + 20 + len(SCALARS) * 3
    print(f"{len(tasks)} chains, {tot} residues, {ncol} columns "
          f"({ncol - 1} features) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
