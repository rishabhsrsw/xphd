"""
bte_snapshots.py
================
Plot exciton populations F_nQ(t) on the exciton band structure, with a spot
radius set by log(F), plus the energy distribution F(E) = sum_nQ F_nQ
delta(E - E_nQ) in a side panel.

The spots are the ACTUAL computed Q-points near the path, not an interpolation
of F onto a dense line. F varies over orders of magnitude between neighbouring
Q, so interpolating it would invent structure; the band lines are interpolated
(they are smooth) but the populations are not.

Usage:
    python rt_bte.py 'GI_ExcPh_Q*.npz' --T 300 --inject gamma \
           --t-end 400 --nt 40 --snapshots snaps.npz
    python bte_snapshots.py snaps.npz --times 30 200 400
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wrap(d):
    d = np.asarray(d, float)
    return d - np.rint(d)


def to_cart(qr, alat):
    """Reduced -> Cartesian (1/Ang) for a hexagonal cell."""
    b = 4 * np.pi / (np.sqrt(3) * alat)
    w = wrap(qr[:, :2])
    x = (w[:, 0] + 0.5 * w[:, 1]) * b
    y = (np.sqrt(3) / 2) * w[:, 1] * b
    return np.stack([x, y], 1)


def build_path(labels, alat, npts=400):
    """High-symmetry path in Cartesian coordinates, plus tick positions."""
    pts = {"G": (0.0, 0.0), "M": (0.5, 0.0), "K": (1/3, 1/3),
           "M'": (-0.5, 0.0), "K'": (-1/3, -1/3),
           "Q": (1/6, 1/6), "Q'": (-1/6, -1/6)}
    red = np.array([pts[l] for l in labels])
    car = to_cart(np.column_stack([red, np.zeros(len(red))]), alat)
    seg = np.linalg.norm(np.diff(car, axis=0), axis=1)
    ticks = np.concatenate([[0.0], np.cumsum(seg)])
    line, coord = [], []
    for i in range(len(car) - 1):
        n = max(2, int(npts * seg[i] / seg.sum()))
        t = np.linspace(0, 1, n, endpoint=(i == len(car) - 2))
        line.append(car[i][None, :] + t[:, None] * (car[i+1] - car[i])[None, :])
        coord.append(ticks[i] + t * seg[i])
    return np.vstack(line), np.concatenate(coord), ticks, car


def project(pts, car, ticks, tol):
    """Distance of each point to the path, and its coordinate along it."""
    best_d = np.full(len(pts), np.inf)
    best_s = np.zeros(len(pts))
    for i in range(len(car) - 1):
        a, b = car[i], car[i+1]
        ab = b - a
        L2 = ab @ ab
        if L2 < 1e-12:
            continue
        t = np.clip(((pts - a) @ ab) / L2, 0.0, 1.0)
        proj = a[None, :] + t[:, None] * ab[None, :]
        d = np.linalg.norm(pts - proj, axis=1)
        m = d < best_d
        best_d[m] = d[m]
        best_s[m] = ticks[i] + t[m] * np.sqrt(L2)
    return best_d, best_s, best_d < tol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("snaps")
    p.add_argument("--times", type=float, nargs="*", default=None,
                   help="fs; the nearest saved snapshots are used")
    p.add_argument("--alat", type=float, default=3.2, help="Angstrom")
    p.add_argument("--path", nargs="*",
                   default=["M'", "K'", "Q'", "G", "Q", "K", "M"])
    p.add_argument("--tol", type=float, default=0.035,
                   help="1/Ang; how close a Q must be to the path to be shown")
    p.add_argument("--floor", type=float, default=1e-8,
                   help="populations below this are not drawn")
    p.add_argument("--smax", type=float, default=260.0, help="largest marker")
    p.add_argument("--sigma", type=float, default=8.0,
                   help="meV, broadening for the F(E) panel")
    p.add_argument("--smooth", type=int, default=8,
                   help="Points per band segment for the spline through the "
                        "dispersion. 0 draws straight joins instead. Only the "
                        "band lines are smoothed; populations are never "
                        "interpolated.")
    p.add_argument("--out", default="bte_snapshots.png")
    a = p.parse_args()

    d = np.load(a.snaps)
    t, F, E, Qr = d["t"], d["F"], d["E"], d["Q_red"]
    # band index, written by rt-BTE.py alongside the snapshots. Needed to
    # join the dispersion into lines: sorting by energy instead would break
    # every band at a crossing.
    band = (np.asarray(d["band"], int) if "band" in d
            else np.zeros(len(E), dtype=int))
    print(f"{a.snaps}: {len(t)} snapshots, {len(E)} states, "
          f"t = {t[0]:.0f} .. {t[-1]:.0f} fs")

    car = to_cart(Qr, a.alat)
    line, lcoord, ticks, corners = build_path(a.path, a.alat)
    dist, s, keep = project(car, corners, ticks, a.tol)
    print(f"  {keep.sum()} of {len(E)} states within {a.tol} 1/Ang of the path")
    if keep.sum() < 20:
        print("  [warn] very few states near the path; raise --tol")

    want = a.times if a.times else [t[len(t)//4], t[len(t)//2], t[-1]]
    idx = [int(np.argmin(np.abs(t - w))) for w in want]
    # The solver stores a fixed set of times, so a requested value snaps to
    # the nearest. If the gap is large the label would be misleading, so say
    # so -- and the remedy is to save more snapshots, not to interpolate the
    # populations, which vary over orders of magnitude between states.
    for w, k in zip(want, idx):
        if abs(t[k] - w) > 0.02 * max(abs(w), 1.0):
            print(f"  [warn] asked for {w:.1f} fs, nearest saved is {t[k]:.1f} fs. "
                  f"Rerun rt-BTE.py with a larger --nt for finer spacing.")


    fig, axes = plt.subplots(len(idx), 2, figsize=(7.2, 3.0*len(idx)),
                             gridspec_kw={"width_ratios": [3.2, 1]},
                             squeeze=False)
    smooth = a.smooth
    Es = np.linspace(E.min() - 0.02, E.max() + 0.02, 600)
    sig = a.sigma * 1e-3

    for row, k in enumerate(idx):
        ax, axr = axes[row]
        f = F[k]
        # faint band structure: every state near the path
        # Band skeleton. Sort each band by position along the path and join
        # with a dashed line, so the dispersion reads as bands rather than as
        # scattered points. Bands are identified by the stored band index, so
        # no re-sorting by energy is done here -- that would break the lines
        # at every crossing.
        for bnd in np.unique(band[keep]):
            sel = keep & (band == bnd)
            if sel.sum() < 2:
                continue
            o = np.argsort(s[sel])
            xs, ys = s[sel][o], E[sel][o]
            # Average duplicates at the same path coordinate. Several
            # q-points can project onto one place along the path, and an
            # interpolator cannot pass through two values at one x.
            xu, inv = np.unique(np.round(xs, 10), return_inverse=True)
            yu = np.bincount(inv, weights=ys) / np.bincount(inv)
            if len(xu) >= 4 and smooth > 0:
                # Cubic spline through the band points. The dispersion is a
                # smooth function of q, so the kinks in a straight-line join
                # are an artefact of sampling, not physics. Populations are
                # NOT interpolated -- they vary over orders of magnitude
                # between neighbouring q and are drawn only at real points.
                from scipy.interpolate import make_interp_spline
                # named kspl, not k: the outer loop uses k for the
                # snapshot index and shadowing it silently breaks
                # the time labels
                kspl = 3 if len(xu) > 3 else 1
                xf = np.linspace(xu[0], xu[-1], max(200, smooth * len(xu)))
                try:
                    yf = make_interp_spline(xu, yu, k=kspl)(xf)
                    ax.plot(xf, yf, ls="--", lw=0.9, c="0.55",
                            zorder=1, dashes=(4, 2))
                    continue
                except Exception:
                    pass
            ax.plot(xu, yu, ls="--", lw=0.9, c="0.55",
                    zorder=1, dashes=(4, 2))
        ax.scatter(s[keep], E[keep], s=2.0, c="0.45", lw=0, zorder=1)
        m = keep & (f > a.floor)
        if m.any():
            lf = np.log10(f[m] / a.floor)
            size = a.smax * (lf / max(lf.max(), 1e-12))
            ax.scatter(s[m], E[m], s=size, c="#2b7bba", alpha=0.75,
                       lw=0, zorder=2)
        for x in ticks[1:-1]:
            ax.axvline(x, color="k", lw=0.7)
        ax.set_xticks(ticks)
        ax.set_xticklabels([l.replace("G", r"$\Gamma$") for l in a.path])
        ax.set_xlim(ticks[0], ticks[-1])
        ax.set_ylim(E.min() - 0.01, 5)
        ax.set_ylabel("Exciton energy (eV)")
        ax.text(0.03, 0.06, f"{t[k]:.0f} fs", transform=ax.transAxes,
                fontsize=11)

        # F(E), over ALL states, not just those near the path
        FE = np.zeros_like(Es)
        nz = f > 0
        if nz.any():
            FE = (f[nz][None, :] * np.exp(-0.5*((Es[:, None]-E[nz][None, :])/sig)**2)
                  ).sum(axis=1)
        axr.plot(FE, Es, color="magenta", lw=1.2)
        axr.set_ylim(*ax.get_ylim())
        axr.set_yticklabels([])
        axr.set_xticks([])
        axr.set_xlabel("F(E)")

    plt.tight_layout()
    plt.savefig(a.out, dpi=300)
    print(f"saved {a.out}")

    print("\n  spot radius ~ log10(F/floor), normalised per panel, so sizes")
    print("  are comparable WITHIN a snapshot but not between them. The F(E)")
    print("  panel carries the absolute scale.")


if __name__ == "__main__":
    main()