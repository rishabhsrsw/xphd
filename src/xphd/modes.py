"""Physical labels for phonon branches, from the eigenvectors.

Energy-sorted branch indices swap wherever two branches cross, so colouring a
band structure by index produces discontinuities that are bookkeeping rather
than physics. This module assigns each branch a CHARACTER -- acoustic or
optical, and out-of-plane, transverse or longitudinal -- which is a property
of the eigenvector and survives crossings.

    ZA TA LA   acoustic:  all sublattices move in phase
    ZO TO LO   optical:   sublattices move against each other
    Z          polarised out of plane
    L / T      in plane, parallel / perpendicular to q

This is NOT an irreducible-representation labelling. Proper irreps require
the little group of q and the characters of the mode under each of its
operations, matched against a character table -- a separate and considerably
larger job, and one that only pays off at the high-symmetry points. The
polarisation labels above are what is usually wanted for a figure legend, are
defined at every q, and can be computed from matdyn's eigenvector output
alone.
"""
from __future__ import annotations

import re

import numpy as np

__all__ = ["read_modes", "classify", "LABELS"]

LABELS = ["ZA", "TA", "LA", "ZO", "TO", "LO"]


def read_modes(path, verbose=True):
    """Read matdyn's `flvec` file (matdyn.modes).

    Returns (q, freq_cm1, evec) with evec shaped (nq, nmod, nat, 3), complex.
    Requires matdyn to have been run WITHOUT `flvec = ' '`.
    """
    txt = open(path, "rb").read().decode("latin-1")
    blocks = re.split(r"^\s*q\s*=\s*", txt, flags=re.M)[1:]
    if not blocks:
        raise SystemExit(f"{path}: no 'q =' blocks. Is this a matdyn.modes "
                         f"file? flvec must not have been blanked.")
    num = r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?"
    Q, W, V = [], [], []
    for b in blocks:
        head = b.split("\n", 1)
        Q.append([float(x) for x in re.findall(num, head[0])[:3]])
        w, v = [], []
        for m in re.finditer(r"freq\s*\(\s*\d+\s*\)\s*=\s*(" + num + r")"
                             r"\s*\[THz\]\s*=\s*(" + num + r")\s*\[cm-1\]"
                             r"(.*?)(?=freq\s*\(|\Z)", b, re.S):
            w.append(float(m.group(2)))
            vals = np.array([float(x) for x in re.findall(num, m.group(3))])
            v.append(vals.reshape(-1, 3, 2))          # (nat, 3, re/im)
        if w:
            W.append(w)
            V.append(np.array(v))
    q = np.array(Q)
    freq = np.array(W)
    ev = np.array(V)
    ev = ev[..., 0] + 1j * ev[..., 1]                 # (nq, nmod, nat, 3)
    if verbose:
        print(f"  {path}: {len(q)} q-points, {freq.shape[1]} branches, "
              f"{ev.shape[2]} atoms")
    return q, freq, ev


def classify(evec, q_cart=None, masses=None, freq=None, n_acoustic=3,
             z_tol=0.5, verbose=True):
    """Character label index for each branch. Returns (nq, nmod) into LABELS.

    evec   : (nq, nmod, nat, 3) complex, as from `read_modes`
    q_cart : (nq, 2 or 3) cartesian q, to separate L from T
    masses : (nat,) atomic masses, used only for the fallback ordering
    freq   : (nq, nmod) frequencies. STRONGLY PREFERRED -- see below.
    z_tol  : out-of-plane weight above which a mode is called flexural

    Assignment proceeds in two stages.

    First, out-of-plane character from the weight |e_z|^2. In a planar layer
    the horizontal mirror is a symmetry at every in-plane q, so every mode is
    either purely out-of-plane or purely in-plane, the weight is 1 or 0 to
    four decimals, and a threshold is unambiguous.

    Second, acoustic versus optical WITHIN each parity set. Given frequencies
    this is done by ordering them: the acoustic branch of a parity set is
    always the lower one, and the two cannot cross, since one goes to zero at
    Gamma and the other does not. Without frequencies the fallback is the
    overlap with a rigid translation -- correct in principle, but the margin
    separating the sets can collapse. In one real 24x24 case 38 of 576
    q-points had a margin below 0.01, where the split is decided by numerical
    noise rather than by physics.
    """
    ev = np.asarray(evec)
    nq, nmod, nat, _ = ev.shape
    m = np.ones(nat) if masses is None else np.asarray(masses, float)
    norm = np.maximum(np.sqrt(np.sum(np.abs(ev) ** 2, axis=(2, 3))), 1e-300)

    z = np.sum(np.abs(ev[..., 2]) ** 2, axis=2) / norm ** 2
    is_z = z > z_tol

    ac = None
    if freq is None:
        def overlap(w):
            w = w / np.sqrt(np.sum(w ** 2))
            o = np.zeros((nq, nmod))
            for dd in range(3):
                o = np.maximum(o, np.abs(np.einsum("qma,a->qm",
                                                   ev[..., dd], w)))
            return o / norm

        cands = {"mass-weighted": overlap(np.sqrt(m)),
                 "displacement": overlap(np.ones(nat))}
        best, score = None, -np.inf
        for name, o in cands.items():
            so = np.sort(o, axis=1)[:, ::-1]
            gap = float(np.median(so[:, n_acoustic - 1] - so[:, n_acoustic]))
            if verbose:
                print(f"    {name:>14}: median acoustic/optical gap {gap:+.3f}")
            if gap > score:
                best, score = name, gap
        if verbose:
            print(f"    using {best}  [no frequencies given; pass freq= for "
                  f"a more robust split]")
        ac = cands[best]

    lon = np.zeros((nq, nmod))
    if q_cart is not None:
        qc = np.asarray(q_cart, float)[:, :2]
        n = np.linalg.norm(qc, axis=1)
        good = n > 1e-8
        qh = np.zeros_like(qc)
        qh[good] = qc[good] / n[good, None]
        inpl = np.maximum(np.sqrt(np.sum(np.abs(ev[..., :2]) ** 2,
                                         axis=(2, 3))), 1e-300)
        lon = np.abs(np.einsum("qmad,qd->qm", ev[..., :2], qh)) / inpl
        lon[~good] = 0.0

    # one acoustic mode out of plane (ZA), the rest in plane (TA, LA)
    n_z_ac = 1
    n_p_ac = n_acoustic - n_z_ac
    lab = np.zeros((nq, nmod), int)
    for k in range(nq):
        jz = np.where(is_z[k])[0]
        jp = np.where(~is_z[k])[0]
        key = np.asarray(freq[k], float) if freq is not None else -ac[k]

        o = jz[np.argsort(key[jz])]
        lab[k, o[:n_z_ac]] = 0                       # ZA, the lower one
        lab[k, o[n_z_ac:]] = 3                       # ZO

        o = jp[np.argsort(key[jp])]
        for grp, base in ((o[:n_p_ac], 0), (o[n_p_ac:], 3)):
            if grp.size == 2:
                a_, b_ = grp[np.argsort(-lon[k, grp])]
                lab[k, a_] = base + 2                # L, more nearly || q
                lab[k, b_] = base + 1                # T
            else:
                for r_ in grp:
                    lab[k, r_] = base + (2 if lon[k, r_] > 0.707 else 1)

    if verbose:
        c = np.bincount(lab.ravel(), minlength=6)
        print("    counts: " + "  ".join(f"{LABELS[k]}={c[k]}"
                                         for k in range(6))
              + f"   (expect {nq} each)")
        nz = is_z.sum(axis=1)
        if not np.all(nz == 2):
            print(f"    [warn] {(nz != 2).sum()} q-points do not have exactly "
                  f"two out-of-plane modes;\n    the layer may not be planar, "
                  f"or z_tol needs adjusting")
        if freq is not None:
            g = np.array([np.sort(np.asarray(freq[k], float)
                                  [np.where(is_z[k])[0]])
                          for k in range(nq)])
            print(f"    ZA/ZO frequency gap: min {(g[:,1]-g[:,0]).min():.2f}, "
                  f"median {np.median(g[:,1]-g[:,0]):.2f} cm-1")
    return lab
