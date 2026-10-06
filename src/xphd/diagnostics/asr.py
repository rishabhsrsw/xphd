#!/usr/bin/env python3
"""
check_parity_asr.py
===================
Two checks that decide how the flexural result should be written up.

(1) BAND MIRROR PARITY
    A sigma_h-odd phonon couples two exciton states only if their parities
    differ. Since the exciton parity is p_c(k+Q) p_v(k), that reduces to a
    question about the BANDS: if every band in the Bethe-Salpeter window
    carries the same sigma_h parity, flexural coupling is forbidden and a
    large odd fraction would signal an error. If the window mixes parities,
    flexural coupling is allowed and the odd fraction is simply correct.
    This is the graphene case inverted: there the pi bands share parity and
    the ZA rule holds; here it must be checked, not assumed.

(2) ACOUSTIC SUM RULE
    The dominant channel is the lowest branch, which is also where an
    unconverged sum rule leaves its residue. Imaginary frequencies at Gamma
    are the visible symptom. This checks how large the residue is, whether
    the dominant channel's weight sits inside it, and whether the reported
    contribution moves when the acoustic cut is raised past it.

    python check_parity_asr.py
"""
from __future__ import annotations

import argparse

import numpy as np
import xphd
from xphd.symmetry import load_ops

CM1 = 1.239841984e-4          # eV per cm^-1
LAB = ["ZA", "TA", "LA", "ZO", "TO", "LO"]


def sigma_h_matrix(save, ylat, D):
    """The sigma_h D-matrix and its k-map. In reduced coordinates sigma_h is
    the improper operation with trace +1 and R[2,2] = -1: it inverts z and
    leaves the plane alone, so its k-map is the identity."""
    R_all = np.asarray(load_ops(save))
    for iop in range(len(R_all) // 2):
        R = R_all[iop]
        if (round(np.linalg.det(R)) == -1 and round(np.trace(R)) == 1
                and round(R[2, 2]) == -1):
            return iop, D[iop]
    raise SystemExit("no horizontal mirror found; is the layer planar?")


def main(argv=None):
    p = argparse.ArgumentParser(prog="xphd check-parity-asr")
    p.add_argument("--save", default="../LELPH/SAVE")
    p.add_argument("--dmats", default="Dmats.npy")
    p.add_argument("--archive", default="GI_ExcPh_Q0001.npz")
    p.add_argument("--labels", default="mode_labels.npy")
    p.add_argument("--hw-fine", default="hw_fine.npy")
    p.add_argument("--nv", type=int, default=3)
    p.add_argument("--T", type=float, default=77.0)
    p.add_argument("--state", type=int, default=1)
    a = p.parse_args(argv)

    # =================================================================
    print("1. BAND MIRROR PARITY IN THE BSE WINDOW")
    print("=" * 62)
    # imported after argument parsing, so --help works without yambopy
    from yambopy import YamboLatticeDB
    ylat = YamboLatticeDB.from_db_file(filename=f"{a.save}/ns.db1")
    D = np.load(a.dmats)
    D = D[:, :, 0, :, :] if D.ndim == 5 else D
    iop, Dsh = sigma_h_matrix(a.save, ylat, D)
    nb = Dsh.shape[-1]
    print(f"   sigma_h is operation {iop}; {nb} bands, {Dsh.shape[0]} k-points")

    offd = np.abs(Dsh - np.diag(np.ones(nb))[None] * np.diagonal(
        Dsh, axis1=1, axis2=2)[:, :, None] * np.eye(nb)[None]).max()
    pn = np.diagonal(Dsh, axis1=1, axis2=2).real          # (nk, nb)
    print(f"   largest off-diagonal element of D(sigma_h): {offd:.2e}")
    if offd > 1e-4:
        print("   [warn] sigma_h MIXES bands, so a single parity per band is")
        print("   not defined and the argument below does not apply.")

    print(f"\n   {'band':>5} {'mean p':>9} {'min':>8} {'max':>8}  parity")
    for n in range(nb):
        kind = ("valence" if n < a.nv else "conduction")
        clean = np.abs(np.abs(pn[:, n]) - 1) < 0.1
        par = ("even" if pn[:, n].mean() > 0.5 else
               "ODD" if pn[:, n].mean() < -0.5 else "MIXED across k")
        print(f"   {n+1:5d} {pn[:, n].mean():+9.3f} {pn[:, n].min():+8.3f} "
              f"{pn[:, n].max():+8.3f}  {par:>14}  ({kind}, "
              f"{100*clean.mean():.0f}% clean)")

    pv, pc = pn[:, :a.nv], pn[:, a.nv:]
    same = np.sign(pv).mean() * np.sign(pc).mean()
    print(f"\n   valence mean parity {np.sign(pv).mean():+.3f}, "
          f"conduction {np.sign(pc).mean():+.3f}")
    if abs(same) > 0.8:
        print("   -> the window is dominated by ONE parity sector. A flexural")
        print("      phonon cannot connect two states within it, and a large")
        print("      odd coupling fraction would need explaining.")
    else:
        print("   -> the window MIXES parities. Flexural coupling between")
        print("      opposite-parity states is symmetry ALLOWED, and a large")
        print("      odd fraction is expected rather than anomalous. This is")
        print("      the point at which the graphene ZA rule fails to carry")
        print("      over: there the pi bands share parity, here they do not.")

    # =================================================================
    print("\n\n2. ACOUSTIC SUM RULE AT GAMMA")
    print("=" * 62)
    arc = xphd.ExcPhArchive(a.archive)
    hw = arc.grid("hw")                                  # (n1, n2, nmod)
    r = np.zeros((arc.n1, arc.n2))
    r[arc._i, arc._j] = xphd.hex_norm(arc.q_red)
    iG = np.unravel_index(np.argmin(r), r.shape)
    print(f"   frequencies at Gamma, from the archive (meV):")
    for nu in range(arc.nmod):
        w = hw[iG][nu] * 1e3
        print(f"     branch {nu+1}: {w:+9.4f}"
              + ("   <-- should be 0 for an acoustic branch" if nu < 3 else ""))
    res = np.abs(hw[iG][:3]).max() * 1e3
    print(f"   largest acoustic residue: {res:.4f} meV")
    print(f"   your acoustic cut must exceed this to exclude it")

    try:
        hwf = np.load(a.hw_fine)
        n = int(round(np.sqrt(hwf.shape[0])))
        hwf = hwf.reshape(n, n, -1)
        print(f"\n   matdyn frequencies at Gamma ({n}x{n} mesh), meV:")
        print(f"     {np.round(hwf[0, 0] * 1e3, 4)}")
        print(f"   largest acoustic residue there: "
              f"{np.abs(hwf[0, 0][:3]).max() * 1e3:.4f} meV")
    except Exception as e:
        print(f"\n   could not read {a.hw_fine}: {e}")

    # ---- where does the dominant channel's weight sit? --------------
    g2 = arc.grid("g2").sum(axis=(3, 4))                 # (n1, n2, nmod)
    print(f"\n   radial distribution of the coupling, by branch (% of that")
    print(f"   branch's total):")
    print(f"   {'|q|/|b|':>14} " + " ".join(f"{'nu' + str(k+1):>7}"
                                            for k in range(arc.nmod)))
    for lo, hi in ((0.0, 0.02), (0.02, 0.05), (0.05, 0.15), (0.15, 0.5)):
        m = (r >= lo) & (r < hi)
        row = f"   [{lo:.2f},{hi:.2f})   "
        for nu in range(arc.nmod):
            t = g2[:, :, nu]
            row += f" {100 * t[m].sum() / max(t.sum(), 1e-30):7.2f}"
        print(row)
    print("   if the dominant branch's weight piles up in the innermost bin,")
    print("   it is sitting on the sum-rule residue rather than on physics")

    # ---- does the answer move when the cut is raised? ---------------
    print(f"\n   contribution of each branch to Gamma(state {a.state}) at "
          f"{a.T:.0f} K,\n   as the acoustic cut is raised past the residue:")
    try:
        hwf = np.load(a.hw_fine)
        ref = hwf.shape[0]
        ref = int(round(np.sqrt(ref))) // arc.n1 if hwf.ndim == 2 else 1
    except Exception:
        hwf, ref = None, 1
    print(f"   {'cut (meV)':>11} {'Gamma (meV)':>12} " +
          " ".join(f"{'nu' + str(k+1):>7}" for k in range(arc.nmod)))
    for cut in (1e-4, 5e-4, 1e-3, 2e-3):
        res_ = xphd.compute(arc, [a.T], hw_fine=hwf, refine=ref,
                            acoustic_cut=cut, verbose=False)
        me, ma = res_.mode_em[0, a.state - 1], res_.mode_ab[0, a.state - 1]
        tot = me + ma
        row = f"   {cut*1e3:11.2f} {tot.sum()*1e3:12.3f} "
        for nu in range(len(tot)):
            row += f" {100 * tot[nu] / max(tot.sum(), 1e-30):7.2f}"
        print(row)
    print("   a contribution that collapses as the cut passes the residue was")
    print("   an artefact; one that is flat across the range is physical")


if __name__ == "__main__":
    main()
