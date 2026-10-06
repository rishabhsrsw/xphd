"""Time- and temperature-resolved ARPES from exciton populations.

Photoemitting the electron out of an exciton leaves the hole behind, so the
feature appears at the exciton energy PLUS the valence energy at the hole's
momentum, and carries the momentum distribution of the exciton's electron
component \\cite{Perfetto2016}:

    I(k, E, t) = sum_{S,Q} N_SQ(t) sum_vc |M_ck|^2 |A^{SQ}_vck|^2
                 delta(E - E_SQ - eps_{v,k-Q}) .

E is the energy axis of the band structure itself, on the reference of the
valence energies: an ordinary valence state would appear at eps_v. Energy
conservation for removing the electron from an exciton, leaving a hole at v,

    E = -[ (E_GS - eps_v) - (E_GS + E_SQ) ] = E_SQ + eps_v ,

gives two checks that any implementation must pass:

  * at the valence maximum the signal sits at E_S + eps_v,top = eps_c - E_b,
    BELOW the conduction band by the binding energy -- the measured result;
  * the low-temperature signal is a REPLICA of the valence band, with the
    same dispersion shifted up by E_S -- not its mirror image.

An earlier version used E_SQ - eps_v. The two agree only at eps_v = 0, which
is why the error was not obvious; away from it the replica came out inverted.
The tests check both properties directly rather than a single number.

The occupations N_SQ are the input that makes this time resolved. Taken from
a real-time Boltzmann propagation they give genuine snapshots; taken as a
Boltzmann distribution they give the quasi-equilibrium spectrum at any
temperature. Most calculations assume the latter, so the former is the part
worth having.

The photoemission matrix element
-------------------------------
M_ck is the amplitude for the photoelectron to leave the crystal. Setting it
to unity is near universal and is the default here, but it discards all
polarisation and geometry dependence, which is often what an experiment
varies. The plane-wave final-state approximation retains the leading part,

    M_ck  ~  (eps . kappa) * u_ck(kappa) ,     kappa = k + G ,

with eps the photon polarisation and kappa the photoelectron momentum fixed
by the photon energy. The first factor carries the polarisation dependence
and needs only the geometry; the second is the plane-wave coefficient of the
Bloch state, available from the wavefunction database. This is the
approximation underlying orbital tomography.

What it does NOT include: final-state scattering, the inner potential,
photoelectron damping, and multiple scattering. Those require a one-step
treatment and are outside the scope of this module. Any claim about absolute
intensities should be avoided; relative intensities across k, E and t are
the meaningful output.

Exciton-exciton interaction
---------------------------
Not included, and not needed for the spectrum at low density: photoemission
probes one-electron removal, and independent excitons contribute additively.
Exciton-exciton annihilation instead removes pairs at a rate quadratic in
the population, so it belongs in the transport equation rather than here;
`xphd.bte` accepts a phenomenological coefficient for that purpose. A
microscopic treatment requires the four-point exciton-exciton scattering
amplitude and is a separate undertaking.
"""
from __future__ import annotations

import numpy as np

__all__ = ["arpes_map", "exciton_arpes", "star_kmaps", "star_assignment",
           "valence_energies", "populations", "path_indices",
           "boltzmann_weights", "geometry_factor",
           "photoelectron_momentum"]

HBAR2_2M = 3.8099821161      # hbar^2 / 2m_e, eV.Ang^2


def photoelectron_momentum(hnu, work_function=4.5, binding=0.0):
    """|kappa| in 1/Ang for a photoelectron of the given kinetic energy.

    E_kin = hnu - phi - E_B, and |kappa| = sqrt(E_kin / (hbar^2/2m)).
    """
    ek = max(hnu - work_function - binding, 1e-6)
    return float(np.sqrt(ek / HBAR2_2M))


def geometry_factor(kpar, pol, hnu, work_function=4.5, binding=0.0):
    """|eps . kappa|^2 for each in-plane k.

    kpar : (nk, 2) in-plane photoelectron momentum, 1/Ang
    pol  : (3,) photon polarisation, normalised internally

    The out-of-plane component follows from |kappa| and kpar, so the factor
    depends on the emission angle as it does in an experiment. For purely
    in-plane polarisation it vanishes at normal emission, which is the
    qualitative behaviour a constant matrix element throws away.
    """
    kpar = np.atleast_2d(np.asarray(kpar, float))[:, :2]
    kmag = photoelectron_momentum(hnu, work_function, binding)
    kz2 = np.maximum(kmag ** 2 - (kpar ** 2).sum(1), 0.0)
    kap = np.column_stack([kpar, np.sqrt(kz2)])
    e = np.asarray(pol, float)
    e = e / max(np.linalg.norm(e), 1e-30)
    return np.abs(kap @ e) ** 2


def boltzmann_weights(E, T, mu=None):
    """Normalised Boltzmann occupations for a list of exciton energies.

    Used for the quasi-equilibrium spectrum. The chemical potential only
    rescales the whole distribution here, since the map is normalised.
    """
    E = np.asarray(E, float)
    kT = 8.617333262e-5 * float(T)
    w = np.exp(-(E - E.min()) / max(kT, 1e-12))
    return w / max(w.sum(), 1e-300)


def arpes_map(A, E_exc, eps_v, kmap_Q, N, omega, sigma=0.05,
              mat=None, verbose=True):
    """I(k, E) for one set of exciton occupations.

    A       : (nQ, nS, nk, nv, nc) exciton envelopes
    E_exc   : (nQ, nS) exciton energies, eV
    eps_v   : (nk, nv) valence quasiparticle energies, eV
    kmap_Q  : (nQ, nk) index of the hole's momentum for each electron k --
              k+Q for yambo databases (see excsym.shift_map)
    N       : (nQ, nS) occupations; only relative values matter
    omega   : (nE,) energy axis, eV
    sigma   : Gaussian broadening, eV
    mat     : (nk, nc) |M|^2, or None for a constant matrix element

    Returns (nk, nE). The electron momentum is k, and the hole momentum
    k - Q enters only through the valence energy: this is what places the
    exciton BELOW the conduction band by its binding energy, and it is the
    feature that distinguishes an excitonic photoemission signal from a
    free-carrier one.
    """
    A = np.asarray(A)
    nQ, nS, nk, nv, nc = A.shape
    omega = np.asarray(omega, float)
    I = np.zeros((nk, len(omega)))
    W = np.abs(A) ** 2
    if mat is not None:
        W = W * np.asarray(mat)[None, None, :, None, :]
    inv = 1.0 / (2.0 * sigma ** 2)

    tot = float(np.asarray(N).sum())
    if verbose:
        print(f"   {nQ} Q-points, {nS} states, {nk} k, {len(omega)} energies")
        print(f"   total occupation {tot:.4e}"
              + ("" if tot > 0 else "   <-- nothing populated"))

    for iQ in range(nQ):
        nQ_ = np.asarray(N)[iQ]
        if nQ_.sum() <= 0:
            continue
        ev = eps_v[kmap_Q[iQ]]                       # (nk, nv) at k - Q
        for S in range(nS):
            if nQ_[S] <= 0:
                continue
            w = W[iQ, S].sum(axis=2)                 # (nk, nv), over c
            pos = E_exc[iQ, S] + ev                  # (nk, nv): replica
            d = omega[None, None, :] - pos[:, :, None]
            I += nQ_[S] * np.einsum("kv,kve->ke", w, np.exp(-d ** 2 * inv))
    return I / max(I.max(), 1e-300)


# ===========================================================================
#  Driver: `xphd arpes`
# ===========================================================================
#
# Momentum convention. The table index k of A^{SQ}_kvc is the ELECTRON, with
# the hole at k+Q -- the only pairing that gives an exact mirror parity at
# every Q (see excsym.shift_map). Photoemission ejects the electron, so the
# signal appears at photoelectron momentum p = k and energy E_SQ + eps_v(k+Q),
# the hole being left behind at k+Q. exciton_arpes below is the function the
# driver uses.

HA2EV = 27.211386245988


def exciton_arpes(A, E, eps_v, kplusQ, N, omega, sigma):
    """Photoemission from the excitons at ONE Q, indexed by photoelectron p.

    A       (nS, nk, nv, nc)  envelopes; table index k is the electron, i.e.
                              the photoelectron momentum
    E       (nS,)             exciton energies, eV
    eps_v   (nk, nv)          valence energies on the k grid, eV
    kplusQ  (nk,)             index of k+Q -- the hole momentum
    N       (nS,)             occupations
    omega   (nE,)             energy axis, eV -- the band-structure axis, on
                              the same reference as eps_v
    Returns (nk, nE), row p = photoelectron momentum.
    """
    A = np.asarray(A)
    nS, nk = A.shape[0], A.shape[1]
    omega = np.asarray(omega, float)
    inv = 1.0 / (2.0 * sigma ** 2)
    W = np.sum(np.abs(A) ** 2, axis=3)                 # (nS, nk, nv), over c
    out = np.zeros((nk, len(omega)))
    for S in range(nS):
        if N[S] <= 0:
            continue
        pos = E[S] + np.asarray(eps_v)[kplusQ]         # (nk, nv), hole at k+Q
        g = np.exp(-(omega[None, None, :] - pos[:, :, None]) ** 2 * inv)
        out += N[S] * np.einsum("kv,kve->ke", W[S], g)  # at p = k, the electron
    return out


def star_kmaps(save, kf):
    """(matrix, kmap) for every operation, spatial and time-reversed.

    kmap[p] is the index of M^{-1} p. Time reversal combined with a spatial
    R acts on k as -R, so each spatial operation contributes a partner.
    """
    from .symmetry import load_ops
    R_all = np.asarray(load_ops(save))
    ops = []
    for R in R_all[: len(R_all) // 2]:
        for M in (R, -R):
            d = kf[:, None, :] - (kf @ M.T)[None, :, :]
            d -= np.rint(d)
            dist = np.linalg.norm(d, axis=-1)
            km = np.argmin(dist, axis=1)
            if (dist[np.arange(len(kf)), km].max() < 1e-5
                    and len(np.unique(km)) == len(kf)):
                ops.append((M, km))
    return ops


def star_assignment(kf, kidx, first, ops):
    """For each full-zone point i, the kmap of an operation carrying its
    irreducible representative Q_parent onto Q_i.

    The excitons at Q_i are the rotated images of those at the parent, so
    their photoemission at p equals the parent's at M^{-1} p.
    """
    out = []
    for i in range(len(kf)):
        Qj = kf[first[kidx[i]]]
        for M, km in ops:
            dq = Qj @ M.T - kf[i]
            if np.abs(dq - np.rint(dq)).max() < 1e-5:
                out.append(km)
                break
        else:
            raise SystemExit(f"no operation maps the representative of "
                             f"point {i} onto it; the star is incomplete")
    return out


def valence_energies(save, ylat, vbands, shift=0.0):
    """Valence energies on the full zone, eV, for the given 0-based bands.

    Read from EIGENVALUES in ns.db1 (irreducible zone, Hartree) and carried
    to the full zone through kmap. These are Kohn-Sham values; for
    quantitative binding energies add the GW valence correction as `shift`,
    or supply the full quasiparticle array with --valence-npy.
    """
    from netCDF4 import Dataset
    with Dataset(f"{save}/ns.db1") as d:
        ev = np.asarray(d.variables["EIGENVALUES"][:], float)
    if ev.ndim == 3:
        ev = ev[0]                                     # spin-unpolarised
    kibz = np.asarray(ylat.kmap)[:, 0]
    n_ibz = int(kibz.max()) + 1
    if ev.shape[0] != n_ibz and ev.shape[1] == n_ibz:
        ev = ev.T
    if ev.shape[0] != n_ibz:
        raise SystemExit(f"EIGENVALUES {ev.shape} does not match {n_ibz} "
                         f"irreducible k-points")
    return ev[kibz][:, vbands] * HA2EV + shift


def populations(kf, kidx, E_ibz, T=None, snaps=None, time=None):
    """N[i, S] on the full zone: thermal at T, or a transport snapshot.

    Thermal occupations are equal across a star, since the energies are.
    A snapshot is read state by state, each placed at its own Q_red and band.
    """
    nk, nS = len(kf), E_ibz.shape[1]
    N = np.zeros((nk, nS))
    if snaps is None:
        kT = 8.617333262e-5 * float(T)
        Ef = E_ibz[kidx]                               # (nk, nS)
        N = np.exp(-(Ef - Ef.min()) / max(kT, 1e-12))
        return N / N.sum()
    d = np.load(snaps)
    it = int(np.argmin(np.abs(d["t"] - time)))
    F, Qr, band = d["F"][it], np.mod(d["Q_red"], 1.0), np.asarray(d["band"])
    if band.min() == 1:
        band = band - 1
    dd = kf[None, :, :2] - Qr[:, None, :2]
    dd -= np.rint(dd)
    idx = np.argmin(np.linalg.norm(dd, axis=-1), axis=1)
    keep = band < nS
    np.add.at(N, (idx[keep], band[keep]), F[keep])
    print(f"   snapshot t = {d['t'][it]:.1f} fs: "
          f"{keep.sum()} of {len(F)} states within the {nS} kept per Q")
    return N / max(N.sum(), 1e-300)


def path_indices(kf, labels, npts=120):
    """Nearest mesh point along a path of high-symmetry labels."""
    from .excsym import HIGH_SYMMETRY
    pts = [np.array(HIGH_SYMMETRY[l]) for l in labels]
    seg, ticks, x = [], [0], 0
    for a0, a1 in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, npts, endpoint=False):
            seg.append(a0 + t * (a1 - a0))
        x += npts
        ticks.append(x)
    seg.append(pts[-1])
    seg = np.array(seg)
    dd = kf[None, :, :2] - seg[:, None, :]
    dd -= np.rint(dd)
    return np.argmin(np.linalg.norm(dd, axis=-1), axis=1), ticks


def main(argv=None):
    import argparse
    from .excsym import shift_map
    from .yambo import full_zone_kpoints, ibz_parents, lattice, load_excitons

    p = argparse.ArgumentParser(prog="xphd arpes",
                                description="exciton photoemission I(k, E)")
    p.add_argument("--save", default="../phonons/SAVE")
    p.add_argument("--bse-dir", default="../BSE/output_all")
    p.add_argument("--nstates", type=int, default=15,
                   help="exciton states per Q")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--temperature", type=float,
                   help="K; quasi-equilibrium Boltzmann occupations")
    g.add_argument("--snapshots",
                   help="snaps.npz from `xphd bte`; use with --time")
    p.add_argument("--time", type=float, default=0.0,
                   help="fs; the snapshot nearest this time is used")
    p.add_argument("--emin", type=float, default=None,
                   help="eV, on the valence energies' reference (default: automatic)")
    p.add_argument("--emax", type=float, default=None)
    p.add_argument("--ne", type=int, default=400)
    p.add_argument("--sigma", type=float, default=0.05,
                   help="eV, energy resolution")
    p.add_argument("--valence-shift", type=float, default=0.0,
                   help="eV added to the Kohn-Sham valence energies")
    p.add_argument("--valence-npy", default=None,
                   help="full-zone valence energies (nk, nv) in eV, used "
                        "instead of ns.db1")
    p.add_argument("--path", nargs="*", default=["G", "M", "K", "G"])
    p.add_argument("-o", "--out", default="arpes.npz")
    a = p.parse_args(argv)

    ylat = lattice(a.save)
    kf = full_zone_kpoints(ylat)
    kidx, first = ibz_parents(ylat)
    n_ibz = len(first)
    print(f"{len(kf)} k-points, {n_ibz} irreducible")

    ops = star_kmaps(a.save, kf)
    smap = star_assignment(kf, kidx, first, ops)
    print(f"{len(ops)} operations (spatial and time-reversed); every "
          f"full-zone point reached from its representative")

    A_ibz, E_ibz, vb = [], [], None
    for j in range(n_ibz):
        exc, A, E = load_excitons(f"{a.bse_dir}/ndb.BS_diago_Q{j + 1}",
                                  ylat, a.nstates)
        A_ibz.append(A)
        E_ibz.append(E[: a.nstates])
        if vb is None:
            vb = np.array(sorted(set(np.asarray(exc.unique_vbands).ravel())),
                          dtype=int)
    E_ibz = np.array(E_ibz)

    if a.valence_npy:
        eps_v = np.load(a.valence_npy)
    else:
        eps_v = valence_energies(a.save, ylat, vb, a.valence_shift)
    print(f"valence bands (0-based) {list(vb)}: eps_v {eps_v.shape}, "
          f"{eps_v.min():.3f} .. {eps_v.max():.3f} eV")
    if eps_v.shape != (len(kf), A_ibz[0].shape[2]):
        raise SystemExit(f"eps_v {eps_v.shape} does not match "
                         f"({len(kf)}, {A_ibz[0].shape[2]})")

    N = populations(kf, kidx, E_ibz, T=a.temperature, snaps=a.snapshots,
                    time=a.time)

    lo = float(E_ibz.min() + eps_v.min()) - 0.5
    hi = float(E_ibz.max() + eps_v.max()) + 0.5
    omega = np.linspace(a.emin if a.emin is not None else lo,
                        a.emax if a.emax is not None else hi, a.ne)

    I = np.zeros((len(kf), len(omega)))
    for j in range(n_ibz):
        members = np.where(kidx == j)[0]
        wj = N[members]                                 # (m, nS)
        if wj.sum() <= 0:
            continue
        kplusQ, w = shift_map(kf, kf[first[j]])
        if kplusQ is None:
            raise SystemExit(f"IBZ {j}: k+Q leaves the grid ({w:.1e})")
        for S in range(a.nstates):
            if wj[:, S].sum() <= 0:
                continue
            one = np.zeros(a.nstates)
            one[S] = 1.0
            Wp = exciton_arpes(A_ibz[j][: a.nstates], E_ibz[j], eps_v,
                               kplusQ, one, omega, a.sigma)
            for i, n in zip(members, wj[:, S]):
                if n > 0:
                    I += n * Wp[smap[i]]
    I /= max(I.max(), 1e-300)

    pidx, ticks = path_indices(kf, a.path)
    np.savez(a.out, I=I, omega=omega, kf=kf, I_path=I[pidx],
             path=np.array(a.path), ticks=np.array(ticks),
             T=np.array(a.temperature if a.temperature else np.nan),
             time=np.array(a.time))
    ip, ie = np.unravel_index(np.argmax(I), I.shape)
    print(f"\nwrote {a.out}: I {I.shape} (k, E), path {I[pidx].shape}")
    print(f"   brightest feature: E = {omega[ie]:.3f} eV at k = "
          f"{np.round(kf[ip, :2], 4)}")


if __name__ == "__main__":
    main()
