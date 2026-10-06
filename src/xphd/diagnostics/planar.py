#!/usr/bin/env python3
"""
validate_planar.py
==================
Checks to run after recomputing with the exactly planar structure. Each one
answers a specific question and prints PASS/FAIL with the number behind it, so
a failure names what to look at rather than just failing.

Run from the directory holding GI_ExcPh_Q*.npz, hw_fine.npy, matdyn.modes.

    python validate_planar.py --savepath ../ELPH/SAVE

Skip individual checks with --skip 3 5, or run one with --only 4.
"""
from __future__ import annotations
import argparse
import glob
import numpy as np
import xphd

BOHR = 0.529177210903
LABELS = ["ZA", "TA", "LA", "ZO", "TO", "LO"]
_results = []


def report(tag, ok, detail):
    _results.append((tag, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}")
    for line in detail.split("\n"):
        print(f"         {line}")


# ---------------------------------------------------------------- 1
def check_structure(a):
    """The relaxed cell must be planar to well below the force threshold."""
    print("\n1. STRUCTURE -- is the layer planar?")
    try:
        from yambopy import YamboLatticeDB
        lat = YamboLatticeDB.from_db_file(f"{a.savepath}/ns.db1")
        z = np.asarray(lat.car_atomic_positions, float)[:, 2] * BOHR
        nums = np.rint(lat.atomic_numbers).astype(int)
    except Exception as e:
        report("structure", False, f"could not read ns.db1: {e}")
        return
    dz = float(np.ptp(z))
    report("planar geometry", dz < 1e-3,
           f"z spread = {dz:.3e} Ang   (want << 1e-3, the placement\n"
           f"precision a 1e-5 Ry/Bohr force threshold can deliver)\n"
           f"atomic numbers {nums} -- must be distinct species")


# ---------------------------------------------------------------- 2
def check_symmetry(a):
    """With sigma_h present the group should double."""
    print("\n2. SYMMETRY -- how many operations does yambo report?")
    try:
        ops = xphd.symmetry.load_ops(a.savepath)
    except Exception as e:
        report("symmetry", False, f"could not load: {e}")
        return
    det = np.array([round(np.linalg.det(R[:2, :2]), 6) for R in ops])
    npro, nimp = int((det > 0).sum()), int((det < 0).sum())
    n1 = a.mesh
    i, j = np.meshgrid(np.arange(n1), np.arange(n1), indexing="ij")
    Q = np.stack([i.ravel() / n1, j.ravel() / n1, np.zeros(n1 * n1)], 1)
    reps, _ = xphd.symmetry.star_map(Q, ops)
    report("operation count", len(ops) >= 24,
           f"{len(ops)} operations ({npro} proper, {nimp} improper)\n"
           f"D3h x T gives 24; C3v x T gives 12. Fewer than 24 means QE\n"
           f"still sees the layer as buckled.\n"
           f"{len(reps)} stars over {n1*n1} points "
           f"({n1*n1/len(reps):.1f}x saving)")


# ---------------------------------------------------------------- 3
def check_archive(a):
    """Internal consistency of one exciton-phonon archive."""
    print("\n3. ARCHIVE -- internal consistency")
    f = sorted(glob.glob(a.archives))
    if not f:
        report("archive", False, "no GI_ExcPh_Q*.npz found")
        return
    arc = xphd.ExcPhArchive(f[0])
    c = arc.check(verbose=False)
    ok = (c["E_n_vs_E_m0"] < 1e-9 and c["g2_acoustic_at_Gamma"] < 1e-18
          and c.get("g2_vs_G2", 0) < 1e-10)
    report("archive checks", ok,
           f"E_n vs E_m(q=0)   {c['E_n_vs_E_m0']*1e3:.4f} meV\n"
           f"acoustic |G|^2(0) {c['g2_acoustic_at_Gamma']:.2e}\n"
           f"|G| median {c['G_median_meV']:.3f} meV, "
           f"max {c['G_max_meV']:.1f} meV\n"
           f"({len(f)} archives present)")
    return arc


# ---------------------------------------------------------------- 4
def check_parity(a, arc):
    """The eigenvectors should have exact sigma_h parity everywhere."""
    print("\n4. MIRROR PARITY -- are the modes cleanly in/out of plane?")
    try:
        q, freq, ev = xphd.modes.read_modes("matdyn.modes", verbose=False)
    except Exception as e:
        report("parity", False, f"could not read matdyn.modes: {e}\n"
                                "rerun matdyn with flvec enabled")
        return None
    from xphd.io.matdyn import source_indices
    nf = int(round(np.sqrt(len(q))))
    idx = source_indices(nf, arc.n1)
    e = ev[idx]
    nrm = np.sum(np.abs(e) ** 2, axis=(2, 3))
    z = np.sum(np.abs(e[..., 2]) ** 2, axis=2) / nrm
    top2 = np.sort(z, axis=1)[:, -2:].min()
    rest = np.sort(z, axis=1)[:, :-2].max()
    report("out-of-plane weight is 0 or 1", top2 > 0.99 and rest < 0.01,
           f"the two most-Z modes: min weight {top2:.4f}  (want > 0.99)\n"
           f"all other modes:      max weight {rest:.4f}  (want < 0.01)\n"
           f"a clean split means sigma_h is a good quantum number at every q")
    return e, q[idx], idx


# ---------------------------------------------------------------- 5
def check_selection_rule(a, arc, mode_data):
    """Flexural coupling must vanish in a planar layer."""
    print("\n5. SELECTION RULE -- does the flexural coupling vanish?")
    if mode_data is None:
        report("selection rule", False, "needs check 4")
        return
    e, qs, idx = mode_data
    lab = xphd.modes.classify(e, q_cart=qs, masses=a.masses, verbose=False)
    g2 = arc.grid("g2")
    r = np.zeros((arc.n1, arc.n2))
    r[arc._i, arc._j] = xphd.hex_norm(arc.q_red)

    # put the labels in the archive's q order
    qf = np.mod(arc.q_red, 1.0)
    qm = np.mod(qs, 1.0)
    d = np.linalg.norm(((qf[:, None, :2] - qm[None, :, :2] + .5) % 1) - .5,
                       axis=2)
    perm = np.argmin(d, axis=1)
    lab_q = lab[perm]                                  # (nq, nmod)
    labg = np.zeros((arc.n1, arc.n2, arc.nmod), int)
    labg[arc._i, arc._j] = lab_q
    odd = np.isin(labg, [0, 3])                        # ZA, ZO

    tot = g2.sum(axis=(3, 4))
    lines, worst = [], 0.0
    for lo, hi in ((0, 0.05), (0.05, 0.15), (0.15, 0.5)):
        m = (r >= lo) & (r < hi)
        fr = tot[m][odd[m]].sum() / max(tot[m].sum(), 1e-300)
        worst = max(worst, fr)
        lines.append(f"|q| in [{lo:.2f},{hi:.2f}): odd fraction {fr:.3e}")
    c = np.bincount(lab.ravel(), minlength=6)
    lines.append("labels: " + "  ".join(f"{LABELS[k]}={c[k]}" for k in range(6))
                 + f"   (expect {len(lab)} each)")
    report("flexural coupling vanishes", worst < 1e-4,
           "\n".join(lines) + "\n"
           "a planar layer forbids ZA/ZO coupling at first order")
    return lab_q


# ---------------------------------------------------------------- 6
def check_za_dispersion(a, arc, mode_data):
    """sigma_h protects the flexural branch as q^2."""
    print("\n6. ZA DISPERSION -- flexural or linear?")
    if mode_data is None:
        report("ZA exponent", False, "needs check 4")
        return
    e, qs, idx = mode_data
    lab = xphd.modes.classify(e, q_cart=qs, masses=a.masses, verbose=False)
    hw = arc.grid("hw")
    r = np.zeros((arc.n1, arc.n2))
    r[arc._i, arc._j] = xphd.hex_norm(arc.q_red)
    from xphd.core.mesh import shells
    sh = [s[0] for s in shells(r.ravel())[1:5]]

    # the ZA branch is the acoustic out-of-plane one, index 0 in LABELS
    lines, exps = [], []
    for ns in (2, 3, 4):
        rq = np.array(sh[:ns])
        y = []
        for qv in rq:
            m = np.abs(r - qv) < qv * 1e-6
            # lowest branch that is out-of-plane at these q
            y.append(hw[:, :, 0][m].mean())
        p = np.polyfit(np.log(rq), np.log(np.maximum(y, 1e-300)), 1)[0]
        exps.append(p)
        lines.append(f"{ns} shells, |q| <= {rq[-1]:.4f}: hw ~ q^{p:+.2f}")
    near = exps[0]
    report("ZA is quadratic", abs(near - 2.0) < 0.3,
           "\n".join(lines) + "\n"
           "a planar layer protects the flexural branch as q^2; the fit\n"
           "closest to Gamma is the one to read")


# ---------------------------------------------------------------- 7
def check_acoustic_cut(a, arc):
    """The linewidth should not depend on where the acoustic cut is set."""
    print("\n7. ACOUSTIC CUT -- is the sum rule doing its job?")
    try:
        hw = np.load("hw_fine.npy")
    except Exception as e:
        report("acoustic cut", False, f"no hw_fine.npy: {e}")
        return
    r = hw.shape[0] if hw.ndim == 3 else int(round(np.sqrt(hw.shape[0])))
    vals = []
    for cut in (5e-4, 1e-4, 1e-5):
        res = xphd.compute(arc, [77.0], hw_fine=hw, refine=r // arc.n1,
                           acoustic_cut=cut, verbose=False)
        vals.append(res.total[0, 0] * 1e3)
    spread = (max(vals) - min(vals)) / max(max(vals), 1e-300)
    report("insensitive to the cut", spread < 0.02,
           f"cut 5e-4 / 1e-4 / 1e-5 eV -> "
           f"{vals[0]:.4f} / {vals[1]:.4f} / {vals[2]:.4f} meV\n"
           f"relative spread {spread:.2%}\n"
           f"insensitivity over a 50x range is the evidence that the\n"
           f"acoustic coupling correctly vanishes at Gamma")


# ---------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(prog="xphd validate-planar")
    p.add_argument("--savepath", default="../ELPH/SAVE")
    p.add_argument("--mesh", type=int, default=24)
    p.add_argument("--masses", type=float, nargs=2, required=True,
                   metavar=("M1", "M2"),
                   help="atomic masses in amu, in the order of the cell. "
                        "No default: GaN is 69.723 14.007, hBN 10.81 14.007, "
                        "and a wrong pair silently corrupts the mode "
                        "classification rather than raising")
    p.add_argument("--archives", default="GI_ExcPh_Q*.npz",
                   help="glob for the exciton-phonon archives")
    p.add_argument("--only", type=int, nargs="*", default=None)
    p.add_argument("--skip", type=int, nargs="*", default=[])
    a = p.parse_args(argv)

    def run(n):
        return (a.only is None or n in a.only) and n not in a.skip

    if run(1):
        check_structure(a)
    if run(2):
        check_symmetry(a)
    arc = check_archive(a) if run(3) else None
    if arc is None and any(run(n) for n in (4, 5, 6, 7)):
        f = sorted(glob.glob(a.archives))
        arc = xphd.ExcPhArchive(f[0]) if f else None
    md = check_parity(a, arc) if (run(4) and arc) else None
    if run(5) and arc:
        check_selection_rule(a, arc, md)
    if run(6) and arc:
        check_za_dispersion(a, arc, md)
    if run(7) and arc:
        check_acoustic_cut(a, arc)

    print("\n" + "=" * 60)
    bad = [t for t, ok in _results if not ok]
    print(f"  {len(_results) - len(bad)} of {len(_results)} checks passed")
    if bad:
        print("  failed: " + ", ".join(bad))
    else:
        print("  everything consistent with an exactly planar D3h layer")


if __name__ == "__main__":
    main()
