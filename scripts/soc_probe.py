"""
soc_probe.py
============
Read-only diagnostic of a spin-orbit (spinor) calculation, for extending the
mirror-parity analysis to WSe2. It writes nothing; paste its whole output.

Run it by editing the SETTINGS block below and running the file -- in VS
Code, Run Python File. No command line is needed. Every section runs even if
an earlier one fails, and a failure prints its error instead of stopping.

What it checks
--------------
1. The lattice: spinor components, electron count, symmetry list, atomic
   positions, and how sigma_h permutes the atoms (W onto itself, the two Se
   onto each other).
2. The Q = 0 excitons: envelope shape and band window (does it contain both
   spin-split conduction bands?), and the lowest energies.
3. The rotation matrices: shape, and D(sigma_h). For spinors sigma_h squares
   to -1 and its diagonal is +-i; if Dmats were built without the spin
   rotation, it squares to +1 instead, and exciton parities will be wrong.
4. Exciton parity at Q = 0 with the existing formula. With spinor-consistent
   D-matrices every state must come out +-1.
5. Phonon parity from ndb.elph, computed two ways: the new permutation
   formula (correct for a non-planar layer) and the old out-of-plane weight
   (valid only when every atom lies in the mirror plane).
6. The archive's shapes: how many phonon branches it holds.
"""
import numpy as np

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
SAVE = "SAVE"                          # the yambo SAVE used by the BSE
BSE_DIR = "bse"                        # folder holding ndb.BS_diago_Q1
DMATS = "Dmats.npy"                    # rotation matrices used for the archives
ELPH = "ndb.elph"                      # LetzElPhC output
ARCHIVE = "GI_ExcPh_Q0001.npz"         # any exciton-phonon archive
NSTATES = 12                           # Q = 0 states to examine
DEG_TOL = 0.002                        # eV: states closer than this form a manifold

# ==========================================================================
# 1. HELPERS
# ==========================================================================
from xphd.mirror import band_labels, phonon_parity, sigma_h_permutation

RESULT = {}


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def attempt(fn):
    try:
        fn()
    except (Exception, SystemExit) as e:         # xphd helpers may SystemExit
        print(f"   FAILED: {type(e).__name__}: {e}")


# ==========================================================================
# 2. LATTICE AND SYMMETRY
# ==========================================================================
def lattice_section():
    from xphd.yambo import lattice, full_zone_kpoints
    ylat = lattice(SAVE)
    RESULT["ylat"] = ylat
    print(f"   spinor components : {ylat.spinor_components}")
    print(f"   electrons         : {ylat.nelectrons}  -> occupied bands "
          f"{ylat.nbandsv}")
    print(f"   time reversal     : {ylat.time_rev}")
    print(f"   k-points          : {ylat.ibz_nkpoints} irreducible, "
          f"{len(full_zone_kpoints(ylat))} in the full zone")
    print(f"   lattice (rows, bohr):\n{np.round(ylat.lat, 5)}")
    car = np.asarray(ylat.car_atomic_positions, float)
    red = np.asarray(ylat.red_atomic_positions, float)
    Z = np.asarray(ylat.atomic_numbers)
    for n in range(len(Z)):
        print(f"   atom {n}: Z = {Z[n]:3d}  cart {np.round(car[n], 5)}  "
              f"red {np.round(red[n], 5)}")
    perm = sigma_h_permutation(car, Z, ylat.lat, tol=1e-2)
    RESULT["perm"] = perm
    print(f"   sigma_h maps atoms {list(range(len(Z)))} -> {list(perm)}")


def symmetry_section():
    from xphd.symmetry import load_ops
    from xphd.excsym import classify_d3h
    R = np.asarray(load_ops(SAVE))
    labels = [classify_d3h(r) for r in R]
    print(f"   {len(R)} operations; first half: {labels[:len(R) // 2]}")
    print(f"   second half: {labels[len(R) // 2:]}")
    RESULT["R"], RESULT["labels"] = R, labels


section("1. LATTICE, ATOMS AND THE MIRROR")
attempt(lattice_section)
attempt(symmetry_section)


# ==========================================================================
# 3. EXCITONS AT Q = 0
# ==========================================================================
def exciton_section():
    from xphd.yambo import load_excitons
    exc, A, E = load_excitons(f"{BSE_DIR}/ndb.BS_diago_Q1", RESULT["ylat"],
                              NSTATES)
    RESULT["A"], RESULT["E"] = A, E
    print(f"   envelopes A {A.shape}  (states, k, v, c)")
    print(f"   BSE band window {list(np.asarray(exc.bs_bands).astype(int))}, "
          f"valence {A.shape[2]}, conduction {A.shape[3]}")
    print("   lowest energies (eV): " + "  ".join(f"{e:.4f}" for e in E[:NSTATES]))


section("2. EXCITONS AT Q = 0")
attempt(exciton_section)


# ==========================================================================
# 4. ROTATION MATRICES AND D(sigma_h)
# ==========================================================================
def dmats_section():
    from xphd.yambo import load_dmats, full_zone_kpoints
    raw = np.load(DMATS)
    print(f"   Dmats.npy {raw.shape} {raw.dtype}")
    if raw.ndim == 5:
        print(f"   axis 2 (dropped by load_dmats) has length {raw.shape[2]}")
        if raw.shape[2] > 1:
            print(f"   max |component 1 - component 0| = "
                  f"{np.abs(raw[:, :, 1] - raw[:, :, 0]).max():.2e}")
    D = load_dmats(DMATS)
    kf = full_zone_kpoints(RESULT["ylat"])
    labels = RESULT["labels"]
    ish = [i for i in range(len(labels) // 2) if labels[i] == "sh"]
    if not ish:
        raise ValueError("no sigma_h among the operations")
    Dsh = D[ish[0]]
    RESULT["Dsh"] = Dsh
    print(f"   sigma_h is operation {ish[0]}; D(sigma_h) {Dsh.shape}, "
          f"{len(kf)} full-zone k")
    sq = np.einsum("kab,kbc->kac", Dsh, Dsh)
    trace_sq = np.mean(np.real(np.diagonal(sq, axis1=1, axis2=2)))
    print(f"   mean diagonal of D(sigma_h)^2 = {trace_sq:+.4f}   "
          f"(+1: spinless, -1: spinors with the spin rotation included)")
    diag = np.diagonal(Dsh, axis1=1, axis2=2)
    lab, kind, ok = band_labels(diag)
    print(f"   diagonal reads as {kind} labels; {100 * ok.mean():.1f}% of "
          f"(k, band) within 0.1 of a unit value")
    off = np.sqrt(np.maximum(np.sum(np.abs(Dsh) ** 2, axis=2)
                             - np.abs(diag) ** 2, 0))
    print(f"   largest off-diagonal row weight {off.max():.2e}")
    for name, target in (("Gamma", [0, 0]), ("K", [1 / 3, 1 / 3])):
        d = kf[:, :2] - np.array(target)
        d -= np.rint(d)
        k = int(np.argmin(np.linalg.norm(d, axis=1)))
        print(f"   diag at {name:5s}: " + "  ".join(
            f"{z.real:+.2f}{z.imag:+.2f}i" for z in diag[k]))


section("3. ROTATION MATRICES")
attempt(dmats_section)


# ==========================================================================
# 5. EXCITON PARITY AT Q = 0
# ==========================================================================
def exciton_parity_section():
    from xphd.yambo import load_dmats, full_zone_kpoints, spatial_ops, sigma_h_op
    from xphd.excsym import rep_matrix
    A, E = RESULT["A"], RESULT["E"]
    kf = full_zone_kpoints(RESULT["ylat"])
    D = load_dmats(DMATS)
    ops = spatial_ops(SAVE, kf, D, A.shape[2])
    lab, R, Dc, Dv, kmap = sigma_h_op(ops)
    print("   state    E (eV)     chi(sigma_h)")
    for s in range(min(NSTATES, len(E))):
        chi = rep_matrix(A, Dc, Dv, kmap, [s])[0, 0]
        print(f"   {s + 1:5d}  {E[s]:.4f}   {chi.real:+.3f}{chi.imag:+.3f}i")
    # degenerate manifolds: the trace is well defined even when states mix
    groups, cur = [], [0]
    for s in range(1, min(NSTATES, len(E))):
        if E[s] - E[cur[-1]] < DEG_TOL:
            cur.append(s)
        else:
            groups.append(cur)
            cur = [s]
    groups.append(cur)
    print("   manifolds: " + "  ".join(
        f"{[g + 1 for g in grp]}: {np.trace(rep_matrix(A, Dc, Dv, kmap, grp)).real / len(grp):+.3f}"
        for grp in groups))


section("4. EXCITON PARITY AT Q = 0 (existing formula)")
attempt(exciton_parity_section)


# ==========================================================================
# 6. PHONON PARITY FROM ndb.elph
# ==========================================================================
def phonon_section():
    from netCDF4 import Dataset
    with Dataset(ELPH) as db:
        print("   variables: " + ", ".join(
            f"{k}{tuple(v.shape)}" for k, v in db.variables.items()))
        q = np.asarray(db.variables["qpoints"][:], float)
        ev = db.variables["POLARIZATION_VECTORS"][:]
        ev = np.asarray(ev[..., 0]) + 1j * np.asarray(ev[..., 1])
        freq = np.asarray(db.variables["FREQ"][...].data, float) * 13.605693122994
    print(f"   eigenvectors {ev.shape} (q, mode, atom, xyz); {len(q)} q-points")
    perm = RESULT.get("perm")
    if perm is None:
        raise ValueError("no atom permutation (section 1 failed)")
    p_new = phonon_parity(ev, perm)
    wz = np.sum(np.abs(ev[..., 2]) ** 2, -1) / np.sum(np.abs(ev) ** 2, (-2, -1))
    p_old = 1 - 2 * wz
    good = np.abs(np.abs(p_new) - 1) < 0.05
    print(f"   permutation formula: {100 * good.mean():.2f}% of (q, mode) "
          f"within 0.05 of +-1; worst deviation {np.abs(np.abs(p_new) - 1).max():.2e}")
    differ = np.sign(p_new) != np.sign(p_old)
    print(f"   the old out-of-plane-weight labels DISAGREE at {100 * differ.mean():.1f}% "
          f"of (q, mode) -- modes such as A1' (Se breathing, out of plane but even)")
    print(f"   and E'' (Se antiphase, in plane but odd), which the old formula cannot "
          f"label for a non-planar layer")
    for name, target in (("Gamma", [0, 0]), ("K", [1 / 3, 1 / 3]), ("M", [0.5, 0])):
        d = q[:, :2] - np.array(target)
        d -= np.rint(d)
        iq = int(np.argmin(np.linalg.norm(d, axis=1)))
        print(f"   {name:5s} (q = {np.round(q[iq, :2], 4)}): " + "  ".join(
            f"{1e3 * f:6.1f} meV {'even' if p > 0.5 else 'odd ' if p < -0.5 else 'mix '}"
            for f, p in zip(freq[iq], p_new[iq])))


section("5. PHONON PARITY")
attempt(phonon_section)


# ==========================================================================
# 7. ARCHIVE
# ==========================================================================
def archive_section():
    d = np.load(ARCHIVE)
    print("   " + ", ".join(f"{k}{d[k].shape}" for k in d.files))


section("6. ARCHIVE")
attempt(archive_section)
