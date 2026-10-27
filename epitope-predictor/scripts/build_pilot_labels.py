#!/usr/bin/env python3
"""
Build partner-agnostic interface labels for the pilot subset.

Labels come from the partner chain; features never do.  For each pair folder the
chosen model is read, interface residues are called by heavy-atom distance, and
the result is then aggregated PER PROTEIN as a union over every folder that
protein appears in -- a residue is positive if it contacts any partner in any
model.  Aggregating is what makes the labels partner-agnostic; scoring file by
file would label the same residue positive in one row and negative in another.

Deliberate choices, with the reasoning in docs/:

  * heavy-atom < 5.0 A, not Foldseek's 10 A C-alpha.  The 10 A C-alpha rule
    selects residues to encode and align, where over-inclusion is cheap; as a
    label it adds second-shell residues that never touch the partner and teaches
    the model "near an interface".  On IL-12 chain A: 26 residues vs 38.
  * >= 4 interface residues per chain (the Foldseek-Interface cutoff).  Fewer is
    usually an incidental contact, so the whole pair is dropped.
  * one model per pair, preferring those that carry pLDDT, since __AF files have
    no B-factor data at all and would otherwise look like pLDDT 0 everywhere.

Numbering is assumed to be full-protein, which is what the IL-12 worked example
found.  That assumption is CHECKED here, not trusted: if residue numbering were
segment-local the union across folders would silently corrupt every label, so the
distribution of chain start positions is reported and a warning is raised if it
looks like 1-based segment numbering.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

CONTACT_CUTOFF = 5.0          # heavy-atom, Angstrom
MIN_INTERFACE_RESIDUES = 4    # per chain

# model families, best first: AFMM carries pLDDT, plain AF never does
FAMILY_RANK = {"AFMM": 0, "AFn": 1, "AF": 2}


def family(model_type: str) -> str:
    if model_type.startswith("AFMM"):
        return "AFMM"
    if model_type == "AF":
        return "AF"
    return "AFn"                      # AF1..AF5


def read_chains(path: Path):
    """Parse a PDB into {chain_id: (resnums, resnames, coords, elements)}.

    Written directly rather than via Bio.PDB: these files are small and uniform,
    and this avoids the parser's warnings about discontinuous chains, which are
    expected here because the models are segment-based.
    """
    chains = defaultdict(lambda: {"res": {}, "coords": [], "resnum": []})
    seen = set()
    with path.open() as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue              # first altloc only
            element = line[76:78].strip() or line[12:16].strip()[0]
            if element == "H" or element == "D":
                continue              # heavy atoms only
            cid = line[21]
            try:
                resnum = int(line[22:26])
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            icode = line[26]
            if icode != " ":
                continue              # insertion codes would break integer numbering
            c = chains[cid]
            key = (cid, resnum)
            if key not in seen:
                seen.add(key)
                c["res"][resnum] = line[17:20].strip()
            c["coords"].append(xyz)
            c["resnum"].append(resnum)
    out = {}
    for cid, c in chains.items():
        if not c["coords"]:
            continue
        out[cid] = {
            "resnames": c["res"],
            "coords": np.asarray(c["coords"], dtype=np.float32),
            "resnum": np.asarray(c["resnum"], dtype=np.int32),
        }
    return out


def interface_residues(chains, cutoff=CONTACT_CUTOFF):
    """Return {chain_id: set(resnum)} of residues within `cutoff` of the other chain."""
    (ca, a), (cb, b) = chains.items()
    tree = cKDTree(a["coords"])
    hits = tree.query_ball_point(b["coords"], r=cutoff)
    ia, ib = set(), set()
    for j, idxs in enumerate(hits):
        if idxs:
            ib.add(int(b["resnum"][j]))
            for i in idxs:
                ia.add(int(a["resnum"][i]))
    return {ca: ia, cb: ib}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pilot", type=Path, help="pilot dir holding manifest.tsv and seeds.txt")
    ap.add_argument("--humanppi", type=Path, required=True, help="humanPPI root")
    ap.add_argument("-o", "--out", type=Path, default=None, help="output dir (default: pilot dir)")
    ap.add_argument("--cutoff", type=float, default=CONTACT_CUTOFF)
    ap.add_argument("--min-interface", type=int, default=MIN_INTERFACE_RESIDUES)
    args = ap.parse_args()

    out = args.out or args.pilot
    out.mkdir(parents=True, exist_ok=True)

    seeds = set((args.pilot / "seeds.txt").read_text().split())
    rows = (args.pilot / "manifest.tsv").read_text().splitlines()
    header, rows = rows[0].split("\t"), [r.split("\t") for r in rows[1:] if r]
    col = {name: i for i, name in enumerate(header)}

    # ---- one model per pair ------------------------------------------------
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r[col["pair"]]].append(r)

    chosen = {}
    for pair, cands in by_pair.items():
        cands.sort(key=lambda r: (
            r[col["has_plddt"]] != "yes",
            FAMILY_RANK[family(r[col["model_type"]])],
            r[col["model_type"]],
        ))
        chosen[pair] = cands[0]

    print(f"{len(by_pair)} pairs, one model chosen each "
          f"({sum(1 for r in chosen.values() if r[col['has_plddt']] == 'yes')} with pLDDT)")

    # ---- per-pair contacts -------------------------------------------------
    observed = defaultdict(set)       # protein -> resnums seen in any model
    interface = defaultdict(set)      # protein -> resnums contacting any partner
    resname = {}                      # (protein, resnum) -> 3-letter code
    partners = defaultdict(set)       # protein -> partners modelled with
    contacting = defaultdict(set)     # protein -> partners it actually contacts
    chain_starts, pair_rows = [], []
    n_ok = n_skip_chains = n_skip_small = 0

    for pair in sorted(chosen):
        r = chosen[pair]
        path = args.humanppi / pair / r[col["pdb"]]
        p1, p2 = r[col["protein1"]], r[col["protein2"]]
        chains = read_chains(path)

        if len(chains) != 2:
            n_skip_chains += 1
            pair_rows.append((pair, r[col["pdb"]], "", "", "", "", f"SKIP {len(chains)} chains"))
            continue

        # chain order in the file maps to the two accessions in folder-name order
        cids = sorted(chains)
        acc_of = {cids[0]: p1, cids[1]: p2}
        iface = interface_residues(chains, args.cutoff)

        sizes = {c: len(iface[c]) for c in cids}
        if min(sizes.values()) < args.min_interface:
            n_skip_small += 1
            pair_rows.append((pair, r[col["pdb"]], cids[0], cids[1],
                              str(sizes[cids[0]]), str(sizes[cids[1]]),
                              f"SKIP <{args.min_interface} per chain"))
            continue

        n_ok += 1
        for cid in cids:
            acc, ch = acc_of[cid], chains[cid]
            nums = sorted(ch["resnames"])
            chain_starts.append(nums[0])
            observed[acc].update(nums)
            interface[acc].update(iface[cid])
            for n in nums:
                resname[(acc, n)] = ch["resnames"][n]
            other = acc_of[cids[1] if cid == cids[0] else cids[0]]
            partners[acc].add(other)
            if iface[cid]:
                contacting[acc].add(other)
        pair_rows.append((pair, r[col["pdb"]], cids[0], cids[1],
                          str(sizes[cids[0]]), str(sizes[cids[1]]), "ok"))

    print(f"  usable {n_ok}   skipped: {n_skip_chains} not-2-chains, "
          f"{n_skip_small} below {args.min_interface} interface residues/chain")

    # ---- numbering sanity check -------------------------------------------
    starts = np.asarray(chain_starts)
    frac_at_1 = float((starts == 1).mean()) if starts.size else 0.0
    print(f"\nnumbering check: chain start residue  min {starts.min()}  "
          f"median {int(np.median(starts))}  max {starts.max()}   "
          f"starting at 1: {frac_at_1:.0%}")
    if frac_at_1 > 0.5:
        print("  *** WARNING: most chains start at residue 1 -- numbering may be")
        print("      SEGMENT-LOCAL, not full-protein.  Aggregating across folders")
        print("      would then corrupt every label.  Resolve before continuing.")
    else:
        print("  consistent with full-protein numbering (as Worked_Example_IL12.md found)")

    # ---- write labels ------------------------------------------------------
    lab = out / "labels.tsv"
    with lab.open("w") as fh:
        fh.write("protein\tresnum\tresname\tis_interface\tn_partners\tn_contacting\n")
        for acc in sorted(observed):
            if acc not in seeds:
                continue              # features are only computed for seeds
            for n in sorted(observed[acc]):
                fh.write(f"{acc}\t{n}\t{resname[(acc, n)]}\t"
                         f"{1 if n in interface[acc] else 0}\t"
                         f"{len(partners[acc])}\t{len(contacting[acc])}\n")

    with (out / "per_pair.tsv").open("w") as fh:
        fh.write("pair\tpdb\tchain1\tchain2\tn_iface1\tn_iface2\tstatus\n")
        for row in pair_rows:
            fh.write("\t".join(row) + "\n")

    # ---- summary -----------------------------------------------------------
    seed_accs = [a for a in sorted(observed) if a in seeds]
    fracs, tot_res, tot_pos = [], 0, 0
    for a in seed_accs:
        nres, npos = len(observed[a]), len(interface[a])
        tot_res += nres
        tot_pos += npos
        fracs.append(npos / nres if nres else 0.0)

    print(f"\nseed proteins labelled : {len(seed_accs)} / {len(seeds)}")
    print(f"residues               : {tot_res}")
    print(f"interface residues     : {tot_pos}  ({tot_pos/tot_res:.1%} before surface filter)")
    if fracs:
        print(f"per-protein interface fraction: median {np.median(fracs):.1%}, "
              f"min {min(fracs):.1%}, max {max(fracs):.1%}")
    pc = [len(partners[a]) for a in seed_accs]
    print(f"partners per seed      : median {int(np.median(pc))}, max {max(pc)}")

    # ---- IL-12 reference check --------------------------------------------
    ref = "P29459"
    if ref in interface:
        n = len(interface[ref])
        verdict = "MATCHES" if n == 26 else "DOES NOT MATCH"
        print(f"\nIL-12 check: {ref} chain A has {n} interface residues "
              f"(Worked_Example_IL12.md: 26) -- {verdict}")
        if n != 26:
            print("  note: the worked example scored the single __AF model of this pair;")
            print("  a different chosen model or the >=4 filter can legitimately differ.")

    print(f"\nwrote {lab} and {out/'per_pair.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
