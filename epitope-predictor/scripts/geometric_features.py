#!/usr/bin/env python3
"""
geometric_features.py — per-residue geometric features from coordinates alone.

The dependency-free part of Tier 1 in Feature_Specification.md: everything computable
from a PDB without FreeSASA, DSSP or Foldseek. Deliberately favours backbone/Cbeta
geometry over side-chain detail, because those features are far less sensitive to
bound-vs-unbound conformation (Feature_Specification.md section 1).

Computed per residue:
    coord_8 / coord_12 / coord_16   CA neighbour counts at 3 scales
    hse_up / hse_down               half-sphere exposure about the CA->CB vector
    hse_ratio                       hse_up / (hse_up + hse_down)
    protrusion_cx                   protrusion index proxy (free volume around CB)
    depth_proxy                     distance to the convex-hull-ish outer shell
    dist_centroid_norm              distance to centroid / radius of gyration
    cb_outward                      cos angle between CA->CB and CA->centroid
                                    (negative = side chain points outward)
    local_density_ratio             coord_8 / coord_16, local vs regional packing
    aa_comp_8_X / aa_comp_12_X      20-dim amino acid composition of the shell
    n_seq_neighbors_8               how many 8A neighbours are sequence-adjacent
                                    (low = spatially clustered from distant sequence)

IMPORTANT: this module is for TRAINING AND INFERENCE BOTH. Never reimplement it in the
scoring path -- import it. Run --selftest in CI to catch silent drift.

Usage:
    python3 geometric_features.py model.pdb -o features.tsv [--chain A]
    python3 geometric_features.py --selftest

Requires: numpy.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required:  pip install numpy")


AA3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
AA1 = dict(zip(AA3, "ARNDCQEGHILKMFPSTWYV"))


# ------------------------------------------------------------------ parsing

def parse_residues(path, chain=None):
    """-> list of dicts with resnum, resname, ca, cb (virtual for GLY), n, c."""
    res = {}
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if not line.startswith("ATOM  "):
                continue
            if line[16] not in (" ", "A"):
                continue
            cid = line[21]
            if chain is not None and cid != chain:
                continue
            try:
                num = int(line[22:26])
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            name = line[12:16].strip()
            r = res.setdefault(num, {"resnum": num, "resname": line[17:20].strip(),
                                     "chain": cid})
            if name in ("CA", "CB", "N", "C"):
                r[name.lower()] = xyz
    out = []
    for num in sorted(res):
        r = res[num]
        if "ca" not in r:
            continue
        r["ca"] = np.array(r["ca"], float)
        if "cb" in r:
            r["cb"] = np.array(r["cb"], float)
        elif "n" in r and "c" in r:
            r["cb"] = virtual_cb(np.array(r["n"], float), r["ca"],
                                 np.array(r["c"], float))
        else:
            r["cb"] = r["ca"]          # last resort; flagged via has_cb
        r["has_cb"] = "cb" in res[num] or ("n" in res[num] and "c" in res[num])
        out.append(r)
    return out


def virtual_cb(n, ca, c):
    """Standard tetrahedral CB reconstruction from backbone N, CA, C (used for
    glycine). Gives the CA->CB direction needed for HSE."""
    b = ca - n
    cc = c - ca
    a = np.cross(b, cc)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca


# ------------------------------------------------- Foldseek virtual center

# van Kempen et al. 2024, Methods, "Nearest-neighbor selection":
#   "The virtual center position is defined by the angle theta (V-CA-CB), the
#    dihedral angle tau (V-CA-CB-N) and the length l (|V - CA|). For each residue
#    i, we selected the residue j with the smallest distance between their virtual
#    centers. ... The virtual center defined by theta = 270 deg, tau = 0 deg and
#    l = 2 performed best in the SCOPe benchmark."
# l is in units of 1.53 A (the CA-CB distance), so l = 2 -> 3.06 A.
#
# NOTE ON THE ANGLE CONVENTION -- do not "fix" this.
# A bond angle reported in the usual [0, 180] range can never read 270. The paper
# searched theta over the full circle [0, 2pi], so (theta=270, tau=0) is expressed
# in a signed/full-circle convention. Under the NeRF placement below,
#     theta=270, tau=0   =>  d2 = [0, -l, 0]
#     theta=90,  tau=180 =>  d2 = [0, -l, 0]
# i.e. the two parameter pairs place V at exactly the same point. Measuring the
# result with standard [0,180] angle code therefore returns 90 / 180, which is
# correct, not a bug. The selftest asserts this equivalence.
VC_THETA_DEG = 270.0
VC_TAU_DEG = 0.0
VC_L = 2 * 1.53


def place_atom(a, b, c, bond, angle_rad, dihedral_rad):
    """NeRF placement: position D with |CD| = bond, angle(B,C,D) = angle,
    dihedral(A,B,C,D) = dihedral."""
    bc = c - b
    nbc = np.linalg.norm(bc)
    if nbc < 1e-8:
        return c.copy()
    bc = bc / nbc
    ab = b - a
    nrm = np.cross(ab, bc)
    nn = np.linalg.norm(nrm)
    if nn < 1e-8:                      # collinear: pick any perpendicular
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(bc @ tmp) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        nrm = np.cross(bc, tmp)
        nn = np.linalg.norm(nrm)
    nrm = nrm / nn
    m = np.stack([bc, np.cross(nrm, bc), nrm], axis=1)
    d2 = np.array([-bond * np.cos(angle_rad),
                   bond * np.sin(angle_rad) * np.cos(dihedral_rad),
                   bond * np.sin(angle_rad) * np.sin(dihedral_rad)])
    return c + m @ d2


def virtual_centers(residues):
    """One virtual center per residue, following the published geometry.
    Needs N; residues lacking it fall back to the CA->CB direction at length l."""
    th = np.radians(VC_THETA_DEG)
    ta = np.radians(VC_TAU_DEG)
    out = np.zeros((len(residues), 3))
    for i, r in enumerate(residues):
        ca, cb = r["ca"], r["cb"]
        n = r.get("n")
        if n is not None:
            # chain N - CB - CA - V, so angle(CB,CA,V)=theta, dihedral(N,CB,CA,V)=tau
            out[i] = place_atom(np.asarray(n, float), cb, ca, VC_L, th, ta)
        else:
            v = cb - ca
            nv = np.linalg.norm(v)
            out[i] = ca + (v / nv) * VC_L if nv > 1e-8 else ca
    return out


# ------------------------------------------------------------------ features

def compute(residues):
    n = len(residues)
    if n == 0:
        return []
    ca = np.array([r["ca"] for r in residues])
    cb = np.array([r["cb"] for r in residues])
    nums = np.array([r["resnum"] for r in residues])

    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)

    # Foldseek virtual-center nearest neighbour (see virtual_centers docstring).
    # The 3Di alphabet encodes residue i's TERTIARY interaction partner; the paper
    # notes it "only falls back to selecting interactions to i+1 or i-1 when no
    # other residues are nearby. In that case, the interaction captures only the
    # backbone conformation." A surface residue with no tertiary partner in its own
    # chain therefore has an unsatisfied interaction slot -- a plausible geometric
    # signature of an available binding surface, and cheap to compute.
    vc = virtual_centers(residues)
    dvc = np.linalg.norm(vc[:, None, :] - vc[None, :, :], axis=-1)
    np.fill_diagonal(dvc, np.inf)
    vc_nn = np.argmin(dvc, axis=1)

    centroid = ca.mean(0)
    rg = float(np.sqrt(((ca - centroid) ** 2).sum(1).mean())) or 1.0

    rows = []
    for i in range(n):
        f = {"resnum": int(nums[i]),
             "resname": residues[i]["resname"],
             "aa": AA1.get(residues[i]["resname"], "X")}

        for rad in (8.0, 12.0, 16.0):
            f[f"coord_{int(rad)}"] = int((d[i] <= rad).sum())

        # half-sphere exposure about the CA->CB direction, 13 A sphere
        v = cb[i] - ca[i]
        nv = np.linalg.norm(v)
        if nv < 1e-6:
            up = dn = 0
        else:
            v = v / nv
            nb = np.where(d[i] <= 13.0)[0]
            if len(nb):
                proj = (ca[nb] - ca[i]) @ v
                up = int((proj > 0).sum())
                dn = int((proj <= 0).sum())
            else:
                up = dn = 0
        f["hse_up"] = up
        f["hse_down"] = dn
        f["hse_ratio"] = round(up / (up + dn), 4) if (up + dn) else 0.0

        # protrusion: free volume around CB relative to a full 10 A sphere.
        # Higher = more protrusive = more exposed.
        nb10 = int((np.linalg.norm(cb - cb[i], axis=1) <= 10.0).sum()) - 1
        f["protrusion_cx"] = round(1.0 - nb10 / 40.0, 4)   # ~40 = packed interior

        dist_c = float(np.linalg.norm(ca[i] - centroid))
        f["dist_centroid_norm"] = round(dist_c / rg, 4)
        # depth proxy: how far inside the outer shell, in units of rg
        f["depth_proxy"] = round(max(0.0, (ca - centroid).dot(
            (ca[i] - centroid) / (dist_c or 1.0)).max() - dist_c) / rg, 4)

        # does the side chain point away from the core?
        to_c = centroid - ca[i]
        ntc = np.linalg.norm(to_c)
        f["cb_outward"] = round(float(v @ (to_c / ntc)), 4) if (nv > 1e-6 and ntc > 1e-6) else 0.0

        f["local_density_ratio"] = round(
            f["coord_8"] / f["coord_16"], 4) if f["coord_16"] else 0.0

        # --- virtual-center interaction-partner features ---
        j = int(vc_nn[i])
        seqsep = int(abs(nums[j] - nums[i]))
        f["vc_nn_dist"] = round(float(dvc[i, j]), 4)
        f["vc_nn_ca_dist"] = round(float(d[i, j]), 4)
        f["vc_nn_seqsep"] = seqsep
        # backbone fallback: partner is i+-1, i.e. no tertiary partner nearby
        f["vc_backbone_fallback"] = int(seqsep <= 1)
        f["vc_nn_is_tertiary"] = int(seqsep > 4)
        # how crowded is this residue's interaction space
        f["vc_density_8"] = int((dvc[i] <= 8.0).sum())
        # "unsatisfied" slot: distant partner and/or backbone fallback
        f["vc_unsatisfied"] = round(
            float(dvc[i, j]) * (1.5 if seqsep <= 1 else 1.0), 4)

        # sequence-locality of spatial neighbours
        nb8 = np.where(d[i] <= 8.0)[0]
        f["n_seq_neighbors_8"] = int((np.abs(nums[nb8] - nums[i]) <= 2).sum())
        f["frac_seq_local_8"] = round(
            f["n_seq_neighbors_8"] / len(nb8), 4) if len(nb8) else 0.0

        # amino acid composition of the 8 A and 12 A shells
        for rad in (8, 12):
            nb = np.where(d[i] <= rad)[0]
            counts = dict.fromkeys(AA3, 0)
            for j in nb:
                rn = residues[j]["resname"]
                if rn in counts:
                    counts[rn] += 1
            tot = max(1, len(nb))
            for a in AA3:
                f[f"aa_comp_{rad}_{AA1[a]}"] = round(counts[a] / tot, 4)

        rows.append(f)
    return rows


def neighbourhood_means(rows, residues, radii=(8.0, 12.0),
                        skip=("resnum", "resname", "aa")):
    """Add the spatial-neighbour mean of every numeric column -- the hand-coded
    message passing described in Feature_Specification.md section 0.2."""
    if not rows:
        return rows
    ca = np.array([r["ca"] for r in residues])
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    numeric = [k for k, v in rows[0].items()
               if k not in skip and isinstance(v, (int, float))
               and not k.startswith("aa_comp_")]   # composition already aggregates
    mat = np.array([[r[k] for k in numeric] for r in rows], float)
    for rad in radii:
        for i, row in enumerate(rows):
            nb = np.where(d[i] <= rad)[0]
            m = mat[nb].mean(0) if len(nb) else mat[i]
            for k, v in zip(numeric, m):
                row[f"{k}_nb{int(rad)}"] = round(float(v), 4)
    return rows


# ------------------------------------------------------------------ selftest

def selftest():
    """Synthetic sphere: surface residues must be less coordinated and more
    protrusive than core residues. Catches sign errors and silent drift."""
    rng = np.random.default_rng(0)
    pts = []
    for r in (4.0, 9.0, 14.0):            # three concentric shells
        k = int(12 * (r / 4.0) ** 2)
        v = rng.normal(size=(k, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        pts.append(v * r)
    pts = np.vstack(pts)
    residues = []
    for i, p in enumerate(pts, 1):
        ca = p
        cb = p * 1.08                      # CB points outward
        residues.append({"resnum": i, "resname": "ALA", "ca": ca, "cb": cb,
                         "has_cb": True, "chain": "A"})
    rows = compute(residues)
    rows = neighbourhood_means(rows, residues)

    radii = np.linalg.norm(pts, axis=1)
    core = radii < 6
    surf = radii > 12
    c8 = np.array([r["coord_8"] for r in rows])
    cx = np.array([r["protrusion_cx"] for r in rows])
    dn = np.array([r["dist_centroid_norm"] for r in rows])

    # virtual-center geometry: exact length, and the (270,0) == (90,180) identity
    rng2 = np.random.default_rng(11)
    vc_len_ok = vc_equiv_ok = True
    for _ in range(20):
        nn_, ca_, cb_ = rng2.normal(size=3), rng2.normal(size=3), rng2.normal(size=3)
        v1 = place_atom(nn_, cb_, ca_, VC_L,
                        np.radians(270.0), np.radians(0.0))
        v2 = place_atom(nn_, cb_, ca_, VC_L,
                        np.radians(90.0), np.radians(180.0))
        vc_len_ok &= abs(np.linalg.norm(v1 - ca_) - VC_L) < 1e-6
        vc_equiv_ok &= np.allclose(v1, v2, atol=1e-6)

    vcd = np.array([r["vc_nn_dist"] for r in rows])
    fallback = np.array([r["vc_backbone_fallback"] for r in rows])

    ok = True
    checks = [
        ("virtual center at exact length l=2x1.53", vc_len_ok),
        ("theta/tau (270,0) == (90,180)", vc_equiv_ok),
        ("vc_nn_dist finite and positive", np.all(np.isfinite(vcd)) and vcd.min() > 0),
        ("vc_backbone_fallback is binary", set(np.unique(fallback)) <= {0, 1}),
        # NOT TESTED HERE: whether backbone fallback is enriched on real surfaces.
        # This fixture places consecutive residue numbers at random sphere
        # positions, so it has no chain connectivity and cannot test that claim.
        # It is a hypothesis to validate on real structures -- see
        # Feature_Specification.md, "virtual-center features".
        ("core more coordinated than surface", c8[core].mean() > c8[surf].mean()),
        ("surface more protrusive than core", cx[surf].mean() > cx[core].mean()),
        ("dist_centroid_norm ordered", dn[core].mean() < dn[surf].mean()),
        ("no NaNs", not any(
            isinstance(v, float) and np.isnan(v) for r in rows for v in r.values())),
        ("neighbourhood columns present", any(k.endswith("_nb8") for k in rows[0])),
    ]
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    print(f"\n  {len(rows)} residues, {len(rows[0])} columns")
    print(f"  core coord_8={c8[core].mean():.1f}  surface coord_8={c8[surf].mean():.1f}")
    print(f"  core cx={cx[core].mean():.3f}  surface cx={cx[surf].mean():.3f}")
    return 0 if ok else 1


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structure", type=Path, nargs="?")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--chain", default=None)
    ap.add_argument("--no-neighbourhood", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.structure:
        ap.error("structure required (or use --selftest)")

    residues = parse_residues(args.structure, args.chain)
    if not residues:
        sys.exit("no residues parsed")
    print(f"{len(residues)} residues", file=sys.stderr)
    n_virtual = sum(1 for r in residues if not r["has_cb"])
    if n_virtual:
        print(f"note: {n_virtual} residues used a virtual CB", file=sys.stderr)

    rows = compute(residues)
    if not args.no_neighbourhood:
        rows = neighbourhood_means(rows, residues)

    cols = list(rows[0].keys())
    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}  ({len(rows)} rows x {len(cols)} cols)",
              file=sys.stderr)
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
