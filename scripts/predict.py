#!/usr/bin/env python3
"""
Score a protein structure and emit a ranked list of candidate epitope patches.

    python scripts/predict.py target.pdb --model pilotfull/model_final.joblib

Input is ONE CHAIN, unbound -- the partner must not be present.  Output is a
ranked patch table plus hotspot residues per patch, which is what a binder
design tool takes.

FEATURES ARE COMPUTED BY THE SAME SCRIPTS USED IN TRAINING, called as
subprocesses rather than reimplemented.  That is deliberate.  A second
implementation drifts -- a different SASA reference table, a different residue
ordering, one missing neighbour-mean column -- and the model still returns
confident numbers from the wrong inputs.  Shelling out guarantees one code path.
The saved column order is then checked explicitly, and inference aborts rather
than guessing if anything is missing.

WHAT THE SCORE MEANS.  Trained on HumanPPI, a high score means "AlphaFold2
would place a human protein interface here", not "this is a real binding site".
Negatives in training mean "no partner was modelled", not "does not bind".

MEASURED PERFORMANCE (full corpus, 2026-10-17, out-of-fold, leak-excluded
splits, n=3529): the top-ranked patch hits a real epitope 64% of the time
against a 41% floor for randomly placed compact patches -- a lift of 1.57x.
One of the top three hits 79% of the time.  Hotspots handed back are true
contacts 49% of the time against a 28% floor.

54% OF PREDICTED PATCHES LAND ON NOTHING.  That is the wasted-design-run rate
and the number to plan around: test two or three patches rather than trusting
the first.  Single-epitope targets are the hard and common case (63% hit@1,
75% of the corpus); targets with three or more sites reach 74%.

Earlier versions of this file quoted 72% against 54%.  Those came from the
500-protein pilot, whose seeds were filtered to 3-8 partners -- an easier
population where even random guessing scored 52%.  They do not describe this
model.
"""

import argparse
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def run_features(pdb: Path, tmp: Path):
    """Call the training feature scripts; never reimplement them."""
    geom = tmp / "geom.tsv"
    pc = tmp / "pc"
    pc.mkdir(exist_ok=True)
    for cmd in ([sys.executable, str(HERE / "geometric_features.py"), str(pdb),
                 "-o", str(geom)],
                [sys.executable, str(HERE / "physchem_features.py"), str(pdb),
                 "-o", str(pc), "-j", "1"]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"feature step failed:\n{' '.join(cmd)}\n{r.stderr[-2000:]}")
    pcf = pc / f"{pdb.stem}.tsv"
    if not (geom.exists() and pcf.exists()):
        sys.exit("feature scripts produced no output")
    return pd.read_csv(geom, sep="\t").merge(pd.read_csv(pcf, sep="\t"), on="resnum")


def rsa_and_plddt(pdb: Path):
    """RSA on the isolated chain, and mean B-factor per residue."""
    import freesasa
    freesasa.setVerbosity(freesasa.silent)
    st = freesasa.Structure(str(pdb))
    areas = freesasa.calc(st).residueAreas()
    rsa = {}
    for ch in areas.values():
        for num_s, a in ch.items():
            try:
                v = a.relativeTotal
                if v is not None and np.isfinite(v):
                    rsa[int(num_s)] = float(v)
            except (ValueError, TypeError):
                continue
    bsum, bcnt, name = {}, {}, {}
    for line in pdb.open():
        if not line.startswith("ATOM") or line[16] not in (" ", "A"):
            continue
        try:
            n = int(line[22:26])
        except ValueError:
            continue
        name[n] = line[17:20].strip()
        try:
            bsum[n] = bsum.get(n, 0.0) + float(line[60:66])
            bcnt[n] = bcnt.get(n, 0) + 1
        except ValueError:
            pass
    plddt = {n: bsum[n] / bcnt[n] for n in bsum if bcnt[n]}
    has_plddt = any(v > 0 for v in plddt.values())
    return rsa, plddt, name, has_plddt


def ca_coords(pdb: Path):
    out = {}
    for l in pdb.open():
        if l.startswith("ATOM") and l[12:16].strip() == "CA" and l[16] in " A":
            out[int(l[22:26])] = (float(l[30:38]), float(l[38:46]), float(l[46:54]))
    return out


def smooth(score, resnums, coords, radius):
    if radius <= 0:
        return score
    P = np.array([coords[r] for r in resnums])
    D = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    W = (D <= radius).astype(np.float64)
    return (W @ score) / W.sum(1)


def patches(resnums, coords, keep, cutoff, min_size):
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdb", type=Path, help="single-chain structure, partner removed")
    ap.add_argument("-m", "--model", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, default=None, help="patch table tsv")
    ap.add_argument("--residues-out", type=Path, default=None,
                    help="also write the per-residue scores")
    ap.add_argument("--top", type=int, default=5, help="patches to print")
    args = ap.parse_args()

    import joblib
    b = joblib.load(args.model)
    model, cols, P = b["model"], b["columns"], b["params"]
    prov = b.get("provenance", {})

    with tempfile.TemporaryDirectory() as td:
        feat = run_features(args.pdb, Path(td))
    rsa, plddt, resname, has_plddt = rsa_and_plddt(args.pdb)
    coords = ca_coords(args.pdb)

    feat = feat[feat["resnum"].isin(coords)].copy()
    feat["rsa"] = feat["resnum"].map(rsa)
    feat = feat.dropna(subset=["rsa"]).sort_values("resnum").reset_index(drop=True)
    feat = feat[feat["resnum"].map(lambda n: resname.get(n) in STANDARD_AA)]

    missing = [c for c in cols if c not in feat.columns]
    if missing:
        sys.exit("feature mismatch -- the model was trained on columns this "
                 f"structure did not produce:\n  {missing[:12]}"
                 f"{' ...' if len(missing) > 12 else ''}\n"
                 "Refusing to predict; the scores would be silently wrong.")

    # the surface subset -- the only residues the model was ever trained on
    surf = feat["rsa"] >= P["surface_rsa"]
    if has_plddt:
        pl = feat["resnum"].map(plddt).fillna(0.0)
        surf &= pl >= P["min_plddt"]
    sub = feat[surf].reset_index(drop=True)
    if len(sub) < P["min_patch"]:
        sys.exit(f"only {len(sub)} residues pass the surface filter -- nothing to score")

    X = np.nan_to_num(sub[cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    raw = model.predict_proba(X)[:, 1]
    rn = sub["resnum"].to_numpy()
    sc = smooth(raw, rn, coords, P["smooth_radius"])

    k = max(P["min_patch"], int(round(P["top_frac"] * len(rn))))
    keep = np.zeros(len(rn), bool)
    keep[np.argsort(-sc)[:k]] = True
    comps = patches(rn, coords, keep, P["patch_cutoff"], P["min_patch"])
    comps.sort(key=lambda c: -(len(c) * sc[c].mean()))

    print(f"{args.pdb.name}: {len(feat)} residues, {len(sub)} pass the surface filter"
          f"{'' if has_plddt else '  (no pLDDT in this file; confidence filter skipped)'}")
    print(f"model: {prov.get('n_proteins','?')} proteins, "
          f"{prov.get('n_components','?')} components, trained {prov.get('trained','?')}")
    print(f"{len(comps)} patches\n")

    rows = []
    for i, c in enumerate(comps, 1):
        res = np.sort(rn[c])
        order = np.array(c)[np.argsort(-sc[c])][:P["n_hotspot"]]
        hot = [int(rn[o]) for o in order]
        rows.append(dict(rank=i, n_residues=len(c), mean_score=float(sc[c].mean()),
                         max_score=float(sc[c].max()),
                         hotspots=",".join(str(h) for h in hot),
                         residues=",".join(str(r) for r in res)))
        if i <= args.top:
            print(f"  patch {i}: {len(c)} residues, mean score {sc[c].mean():.3f}")
            print(f"     hotspots  " + "  ".join(f"{resname.get(h,'?')}{h}" for h in hot))
            print(f"     residues  {' '.join(str(r) for r in res)}")

    tbl = pd.DataFrame(rows)
    out = args.out or args.pdb.with_suffix(".patches.tsv")
    tbl.to_csv(out, sep="\t", index=False, float_format="%.4f")
    print(f"\nwrote {out}")

    if args.residues_out:
        pd.DataFrame({"resnum": rn, "resname": [resname.get(r, "?") for r in rn],
                      "score_raw": raw, "score_smoothed": sc,
                      "rsa": sub["rsa"].to_numpy()}).to_csv(
            args.residues_out, sep="\t", index=False, float_format="%.4f")
        print(f"wrote {args.residues_out}")

    if rows:
        print(f"\nhotspot string for a design tool (patch 1): {rows[0]['hotspots']}")
    print("\nScores mean \"AF2 would place an interface here\", not \"this is a real")
    print("binding site\".  Top patch is right ~64% of the time against a 41% floor")
    print("(1.57x); one of the top three is right ~79% of the time.  About half of")
    print("all patches land on nothing, so test two or three, not just the first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
