#!/usr/bin/env python3
"""
Select a small pilot subset from the HumanPPI PDB extract.

The subset is chosen to exercise every code path the full pipeline needs, not to
be representative:

  * seed proteins have SEVERAL partners, so per-protein label aggregation
    (the union across folders) is actually tested -- a random sample of pairs
    would be mostly singletons and would never hit that path;
  * every folder containing a seed is included, INCLUDING folders whose other
    chain is not a seed.  Labels must be aggregated over all of a protein's
    partners or residues binding an unselected partner are silently mislabelled
    negative;
  * the manifest records model type and whether pLDDT is actually present, since
    __AF files carry no B-factor data at all.

Writes to the output directory:
    seeds.txt        one accession per line -- proteins features are computed for
    folders.txt      one pair folder per line -- all folders to read labels from
    manifest.tsv     one row per PDB file, with model type and pLDDT presence
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

# Pairs whose protein is modelled with no pLDDT at all; see Worked_Example_IL12.md
NO_PLDDT_MODEL = "AF"


def model_type(pdb_name: str) -> str:
    """'P29459_S0__P29460_S0__AFMM2.pdb' -> 'AFMM2'."""
    stem = pdb_name[:-4] if pdb_name.endswith(".pdb") else pdb_name
    return stem.rsplit("__", 1)[-1]


def has_plddt(path: Path) -> bool:
    """True if any ATOM record carries a non-zero B-factor."""
    try:
        with path.open() as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        if float(line[60:66]) != 0.0:
                            return True
                    except ValueError:
                        continue
    except OSError:
        return False
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="humanPPI directory of pair folders")
    ap.add_argument("-o", "--out", type=Path, required=True, help="output directory")
    ap.add_argument("--n-seeds", type=int, default=40,
                    help="number of seed proteins to select (default 40)")
    ap.add_argument("--min-partners", type=int, default=3,
                    help="seed must appear in at least this many folders (default 3)")
    ap.add_argument("--max-partners", type=int, default=12,
                    help="and at most this many, to keep the subset small (default 12)")
    ap.add_argument("--prefix", default="P",
                    help="restrict seeds to accessions with this prefix; '' for any")
    ap.add_argument("--include", default="",
                    help="comma-separated accessions to force into the seed set")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    args = ap.parse_args()

    if not args.root.is_dir():
        sys.exit(f"not a directory: {args.root}")

    # protein -> folders it appears in
    folders_of = defaultdict(list)
    n_folders = 0
    for d in args.root.iterdir():
        if not d.is_dir() or "_" not in d.name:
            continue
        n_folders += 1
        parts = d.name.split("_")
        if len(parts) != 2:
            continue                       # unexpected name; skip rather than guess
        for acc in parts:
            folders_of[acc].append(d.name)

    print(f"scanned {n_folders} pair folders, {len(folders_of)} distinct proteins")

    # candidate seeds: right partner count, right prefix
    candidates = [
        acc for acc, fs in folders_of.items()
        if args.min_partners <= len(fs) <= args.max_partners
        and (not args.prefix or acc.startswith(args.prefix))
    ]
    candidates.sort()                       # determinism before shuffling
    print(f"{len(candidates)} candidate seeds with "
          f"{args.min_partners}-{args.max_partners} partners"
          + (f" and prefix '{args.prefix}'" if args.prefix else ""))

    forced = [a for a in args.include.split(",") if a]
    missing = [a for a in forced if a not in folders_of]
    if missing:
        print(f"WARNING: forced accessions not found, ignored: {', '.join(missing)}")
    forced = [a for a in forced if a in folders_of]

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    pool = [a for a in candidates if a not in forced]
    seeds = forced + pool[: max(0, args.n_seeds - len(forced))]
    seeds.sort()

    # every folder touching a seed -- including non-seed partners
    folders = sorted({f for acc in seeds for f in folders_of[acc]})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "seeds.txt").write_text("\n".join(seeds) + "\n")
    (args.out / "folders.txt").write_text("\n".join(folders) + "\n")

    # manifest over every PDB in those folders
    rows, n_pdb, by_type, plddt_by_type = [], 0, defaultdict(int), defaultdict(int)
    multi_model_pairs = 0
    for name in folders:
        d = args.root / name
        pdbs = sorted(p for p in d.iterdir() if p.suffix == ".pdb")
        if len(pdbs) > 1:
            multi_model_pairs += 1
        a, b = name.split("_")
        for p in pdbs:
            mt = model_type(p.name)
            hp = has_plddt(p)
            by_type[mt] += 1
            plddt_by_type[mt] += int(hp)
            n_pdb += 1
            rows.append("\t".join([
                name, a, b, p.name, mt,
                "yes" if hp else "no",
                "yes" if a in seeds else "no",
                "yes" if b in seeds else "no",
            ]))

    header = ("pair\tprotein1\tprotein2\tpdb\tmodel_type\thas_plddt"
              "\tprotein1_is_seed\tprotein2_is_seed")
    (args.out / "manifest.tsv").write_text(header + "\n" + "\n".join(rows) + "\n")

    partners = [len(folders_of[a]) for a in seeds]
    non_seed = {p for f in folders for p in f.split("_")} - set(seeds)

    print()
    print(f"  seed proteins (features)   : {len(seeds)}")
    print(f"  partner count per seed     : min {min(partners)}, "
          f"median {sorted(partners)[len(partners)//2]}, max {max(partners)}")
    print(f"  pair folders (labels)      : {len(folders)}")
    print(f"  ... of which multi-model   : {multi_model_pairs}")
    print(f"  non-seed partner proteins  : {len(non_seed)}")
    print(f"  pdb files                  : {n_pdb}")
    print()
    print("  model type   files   with pLDDT")
    for mt in sorted(by_type):
        print(f"  {mt:<11} {by_type[mt]:>5}   {plddt_by_type[mt]:>10}")
    print()
    print(f"  wrote {args.out}/seeds.txt, folders.txt, manifest.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
