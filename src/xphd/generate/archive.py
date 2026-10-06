"""
generate_excph_Q.py
===================
Exciton-phonon matrix elements at a single exciton momentum Q, from
LetzElPhC + yambo BSE databases.

Follows yambopy's exciton_phonon_matelem_iQ (MN/FP). Differences from the
reference are only that the WFdb-derived quantities (kBZ, kpts_iBZ, ktree,
Dmats, band range) are reconstructed from the lattice DB instead of a
YamboWFDB, so every one of those is verified against the lattice at startup
rather than assumed.

FIXES relative to the earlier version, each verified:

  1. kpts_iBZ was lat.red_kpoints, which is the FULL BZ (576 rows), not the
     IBZ (61). Indexing it with an IBZ index returned the wrong k-point for
     every q whose IBZ parent index >= 2, corrupting the exciton rotation.
     The IBZ list is now built from the BZ list via kpoints_indexes and
     cross-checked against lat.ibz_kpoints.

  2. The nearest-BZ-point search had no periodic wrapping. A Qpt landing at
     0.99999999 is distance 1.0 from the correct point at 0.0 and argmin
     picks a neighbour. Invisible at Q=Gamma, fires for Q != 0.

  3. bands_range is now derived from the exciton DB instead of hardcoded.
     read_iq's docstring: "start index follows Python indexing (starting
     from 0), and the end index is excluded", so bands 7..12 is [6, 12].

  4. Serial by default. read_iq touches a shared netCDF4 handle and HDF5 is
     not thread-safe in many builds; it corrupts silently rather than
     raising. Use --threads N only after --verify-threads passes.

  5. reshape() replaced by an explicit shape assertion.

  6. Dense E_m_grid / hw_grid / q_red / Q_red written alongside the sparse
     event list, removing all index-ordering guesswork downstream.

  7. G_grid (the COMPLEX matrix elements in eV, same layout as g2_grid) is
     written as well. g2_grid alone discards the phase, which is needed for
     the real-space localisation and gauge diagnostics. |G_grid|**2 must
     equal g2_grid exactly; that identity is the downstream sanity check.

  8. The electron/hole split is REQUIRED by default. Previously a failed
     probe printed a warning and silently wrote the total only, so Ge_grid
     and Gh_grid could be missing from an archive with no error -- found
     only when the interference analysis failed, after every Q had run.
     Pass --allow-no-split to accept a total-only archive deliberately.

  9. Mesh, exciton count and paths are command-line arguments, not edits to
     CONFIG. The mesh is derived from the lattice when omitted, so the same
     script runs unchanged on any system.

 10. elph_convention, bse_nv and bse_nc are written into the archive, so
     downstream code need not infer the band window or the momentum
     convention.

NOT bugs, confirmed and left alone:
  - ph_eig from read_iq is discarded. It is POLARIZATION_VECTORS, shape
    (nmodes, natoms, 3); elph_mat axis 1 is already the mode index.
  - elph_mat.transpose(1,0,2,4,3) maps (nk,nm,ns,init,fin) to
    (nm,nk,ns,fin,init), matching exciton_X_matelem's documented Omn shape.
  - ph_energies is in eV. read_iq divides it by ha2ev/2 to reach Ry.
  - The 0.5 factor is Ry -> Ha, applied before ha2ev.

Usage:
    python generate_excph.py <Q_index 0..nBZ-1> [outdir] [--threads N]
    python generate_excph.py 0 . --verify-threads

The index is 0-based over the FULL zone and the output is named with it
plus one: iQ=0 writes GI_ExcPh_Q0001.npz. The real-time transport needs
every one of the nBZ archives, not only the irreducible set -- a missing
Q becomes a zero-energy sink that silently absorbs population.
"""
import argparse
import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# yambopy and tqdm are bound as module globals at the start of main(), so
# `xphd generate --help` works without them while the helper functions
# below still resolve these names at call time.
tqdm = YamboLatticeDB = LetzElphElectronPhononDB = YamboExcitonDB = None
exciton_X_matelem = rotate_exc_wf = build_ktree = None

# Tolerance for Ge + Gh == G. The electron-phonon elements may be held in
# single precision, and evaluating the electron and hole parts separately
# then summing differs from the combined call at float32 rounding -- about
# 1e-7 relative. A WRONG contribution flag gives an O(1) discrepancy (a flag
# returning the total makes Ge + Gh = 2G, relative error 1), so any value
# between ~1e-6 and ~1e-1 separates the two cleanly. 1e-5 sits well clear of
# both. The earlier 1e-10 assumed double precision and rejected a correct
# split.
SPLIT_TOL = 1e-5

CONFIG = {
    'savepath':   '../ELPH/SAVE',
    'ndb_elph':   '../ELPH/ndb.elph',
    'bse_dir':    '../BSE_EXCPH/output',
    'dmat_file':  'Dmats.npy',
    'n_excitons': 15,
    'mesh':       [6, 6, 1],
    'ha2ev':      27.211386245988,
}


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------
def wrap(d):
    """Wrap reduced-coordinate differences into [-0.5, 0.5)."""
    d = np.asarray(d, dtype=float)
    return d - np.rint(d)


def nearest_kpt(kBZ, Qpt):
    """Index of the BZ point closest to Qpt, WITH periodic wrapping.

    Without the wrap, a Qpt at 0.99999999 sits distance 1.0 from the correct
    point at 0.0 and argmin returns a neighbour instead.
    """
    d = wrap(kBZ - np.asarray(Qpt, dtype=float)[None, :])
    return int(np.argmin(np.linalg.norm(d, axis=1)))


def on_mesh(k, mesh, tol=1e-5):
    f = np.asarray(k, dtype=float) * np.asarray(mesh, dtype=float)[None, :]
    return np.abs(f - np.rint(f)).max() < tol


def as_reduced(v, rlat, mesh, name):
    """Return v in reduced coordinates on the mesh, trying the identity and
    the two plausible cartesian->reduced maps. Raises if none land on grid."""
    v = np.asarray(v, dtype=float)
    inv = np.linalg.inv(rlat)
    for tag, w in (('as-is', v), ('@inv(rlat)', v @ inv), ('@inv(rlat).T', v @ inv.T)):
        r = np.mod(w, 1.0)
        r[np.abs(r - 1.0) < 1e-6] = 0.0
        if on_mesh(r, mesh):
            return r, tag
    raise SystemExit(f"could not put '{name}' on the {mesh} mesh in reduced coords")


def build_ibz_list(kBZ, lat):
    """IBZ representatives, taken as the first BZ image of each IBZ parent.

    lat.red_kpoints is the FULL BZ (576 rows) despite the name; lat.ibz_kpoints
    is the 61-row IBZ. Building from kpoints_indexes avoids depending on which
    of the two is in which unit convention.
    """
    kidx = np.asarray(lat.kpoints_indexes, dtype=int)
    sidx = np.asarray(lat.symmetry_indexes, dtype=int)
    n_ibz = int(kidx.max()) + 1
    first = np.full(n_ibz, -1, dtype=int)
    for i, j in enumerate(kidx):
        if first[j] < 0:
            first[j] = i
    assert (first >= 0).all(), "some IBZ index never appears in kpoints_indexes"

    ident = [s for s in range(len(lat.sym_car))
             if np.allclose(lat.sym_car[s], np.eye(3), atol=1e-8)]
    assert ident, "no identity found in lat.sym_car"
    bad = [j for j in range(n_ibz) if sidx[first[j]] not in ident]
    if bad:
        print(f"  [warn] {len(bad)} IBZ parents whose first BZ image is not the "
              f"identity image; using it anyway (first bad: IBZ {bad[0]})")
    return kBZ[first], first


# ----------------------------------------------------------------------
def rotate_Akcv_Q(lat, kBZ, ktree, kpts_iBZ, exdbs, Qpt, Dmats, neigs=-1):
    """yambopy's rotate_Akcv_Q, with the lattice pieces passed in directly."""
    idx_BZQ = nearest_kpt(kBZ, np.mod(Qpt, 1.0))
    iQ_isymm = int(lat.symmetry_indexes[idx_BZQ])
    iQ_iBZ = int(lat.kpoints_indexes[idx_BZQ])
    trev = (iQ_isymm >= len(lat.sym_car) / (1 + int(np.rint(lat.time_rev))))
    symm_mat_red = lat.lat @ lat.sym_car[iQ_isymm] @ np.linalg.inv(lat.lat)
    exe_iQIBZ = kpts_iBZ[iQ_iBZ]
    AQibz = exdbs[iQ_iBZ].get_Akcv()
    AQ_rot = rotate_exc_wf(AQibz, symm_mat_red, kBZ, exe_iQIBZ,
                           Dmats[iQ_isymm], trev, ktree)
    if neigs != -1:
        AQ_rot = AQ_rot[:neigs]
    return AQ_rot, iQ_iBZ


# ----------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(prog="xphd generate")
    p.add_argument("iQ", type=int, help="exciton momentum index in the full BZ")
    p.add_argument("outdir", nargs="?", default=".")
    p.add_argument("--threads", type=int, default=1,
                   help="1 = serial (safe). >1 shares a netCDF handle across "
                        "threads; run --verify-threads first.")
    p.add_argument("--verify-threads", action="store_true",
                   help="run 8 q-points serially and threaded, assert identical")
    p.add_argument("--mesh", type=int, nargs=2, default=None,
                   help="n1 n2 of the k/q mesh; derived from the lattice "
                        "when omitted")
    p.add_argument("--nexc", type=int, default=CONFIG['n_excitons'],
                   help="exciton states per Q to keep")
    p.add_argument("--savepath", default=CONFIG['savepath'])
    p.add_argument("--ndb-elph", default=CONFIG['ndb_elph'])
    p.add_argument("--bse-dir", default=CONFIG['bse_dir'])
    p.add_argument("--dmats", default=CONFIG['dmat_file'])
    p.add_argument("--allow-no-split", action="store_true",
                   help="accept an archive without Ge_grid/Gh_grid if the "
                        "electron/hole flags cannot be found. Off by default: "
                        "the interference analysis needs them, and a silent "
                        "omission is only discovered after every Q has run.")
    a = p.parse_args(argv)
    global tqdm, YamboLatticeDB, LetzElphElectronPhononDB, YamboExcitonDB
    global exciton_X_matelem, rotate_exc_wf, build_ktree
    from tqdm import tqdm
    from yambopy import YamboLatticeDB, LetzElphElectronPhononDB
    from yambopy.dbs.excitondb import YamboExcitonDB
    from yambopy.bse.exciton_matrix_elements import exciton_X_matelem
    from yambopy.bse.rotate_excitonwf import rotate_exc_wf
    from yambopy.kpoints import build_ktree

    os.makedirs(a.outdir, exist_ok=True)
    nexc = a.nexc
    ha2ev = CONFIG['ha2ev']

    # ---- lattice + elph ------------------------------------------------
    print("loading databases ...")
    lat = YamboLatticeDB.from_db_file(os.path.join(a.savepath, 'ns.db1'))
    elph = LetzElphElectronPhononDB(a.ndb_elph, read_all=True)

    nbz_lat = len(lat.kpoints_indexes)
    if a.mesh is None:
        n = int(round(np.sqrt(nbz_lat)))
        if n * n != nbz_lat:
            raise SystemExit(f"{nbz_lat} BZ points is not a square mesh; "
                             f"pass --mesh n1 n2")
        mesh = [n, n, 1]
        print(f"  mesh derived from the lattice: {mesh}")
    else:
        mesh = [a.mesh[0], a.mesh[1], 1]
    if not (0 <= a.iQ < nbz_lat):
        raise SystemExit(f"iQ={a.iQ} outside 0..{nbz_lat-1}")
    kBZ_lat, tag = as_reduced(lat.car_kpoints, lat.rlat, mesh, 'car_kpoints')
    kBZ = np.mod(np.asarray(elph.kpoints, dtype=float), 1.0)
    kBZ[np.abs(kBZ - 1.0) < 1e-6] = 0.0

    print(f"  lattice BZ : {nbz_lat} points  (from car_kpoints {tag})")
    print(f"  elph k     : {len(kBZ)} points")
    assert len(kBZ) == nbz_lat, f"elph has {len(kBZ)} k-points, lattice has {nbz_lat}"
    assert np.prod(mesh) == nbz_lat, f"mesh {mesh} != {nbz_lat} BZ points"

    # elph ordering must match the lattice's, because idx_BZQ from kBZ is used
    # on lat.symmetry_indexes / lat.kpoints_indexes / Dmats.
    dev = np.abs(wrap(kBZ - kBZ_lat)).max()
    assert dev < 1e-5, (
        f"elph.kpoints and the lattice BZ list differ (max {dev:.2e}).\n"
        f"idx_BZQ would then index symmetry_indexes/Dmats with the wrong "
        f"ordering. Reorder elph_mat's k-axis before proceeding.")
    print(f"  k-ordering agrees with the lattice (max deviation {dev:.2e})")

    # ---- IBZ list (this is the one that was wrong) ----------------------
    kpts_iBZ, first_img = build_ibz_list(kBZ, lat)
    n_ibz = len(kpts_iBZ)
    print(f"  IBZ        : {n_ibz} points")
    try:
        ref_ibz, itag = as_reduced(lat.ibz_kpoints, lat.rlat, mesh, 'ibz_kpoints')
        if len(ref_ibz) == n_ibz:
            dv = np.abs(wrap(kpts_iBZ - ref_ibz)).max()
            print(f"  cross-check vs lat.ibz_kpoints ({itag}): max dev {dv:.2e}"
                  + ("  OK" if dv < 1e-5 else "  <-- DISAGREE, inspect by hand"))
    except SystemExit:
        print("  [warn] could not put lat.ibz_kpoints on the mesh; skipping cross-check")
    assert len(lat.red_kpoints) == nbz_lat, (
        "lat.red_kpoints is not the full BZ here; re-check which array is the IBZ")

    ktree = build_ktree(kBZ)
    Dmats = np.load(a.dmats)
    print(f"  Dmats      : {np.shape(Dmats)}")
    assert len(Dmats) >= len(lat.sym_car), "Dmats has fewer entries than symmetries"

    # ---- exciton databases ---------------------------------------------
    print(f"loading {n_ibz} exciton databases ...")
    exdbs = []
    for j in range(n_ibz):
        f = os.path.join(a.bse_dir, f'ndb.BS_diago_Q{j+1}')
        if not os.path.isfile(f):
            raise SystemExit(f"missing {f} -- need all {n_ibz} IBZ databases, "
                             f"found {j}")
        exdbs.append(YamboExcitonDB.from_db_file(lat, filename=f,
                                                 Load_WF=True, neigs=nexc))
    assert len(exdbs) == n_ibz

    # band range, derived not hardcoded (yambopy's own sanity check)
    b0 = int(np.min(exdbs[0].unique_vbands))
    b1 = int(np.max(exdbs[0].unique_cbands)) + 1
    bse_bnds_range = [b0, b1]
    nv = len(np.unique(exdbs[0].unique_vbands))
    nc = len(np.unique(exdbs[0].unique_cbands))
    print(f"  BSE bands  : {bse_bnds_range} (python slice)  nv={nv} nc={nc} "
          f"-> {nv*nc} transitions/k")
    assert b1 - b0 == nv + nc, (
        f"band window {b1-b0} != nv+nc = {nv+nc}; the BSE space is not a "
        f"simple nv x nc block")
    # Dmats must cover the SAME bands as the BSE. The symmetry count is
    # checked above, but a Dmats built for another band window has the right
    # number of operations and the wrong band dimension, and the exciton
    # rotation then proceeds silently with mismatched blocks.
    nb_dmat = int(np.shape(Dmats)[-1])
    assert nb_dmat == nv + nc, (
        f"Dmats covers {nb_dmat} bands but the BSE window has nv+nc = "
        f"{nv+nc}. Rebuild Dmats.npy with calculate_excph.py for this band "
        f"range ({bse_bnds_range}).")
    print(f"  Dmats band dimension {nb_dmat} matches the BSE window")

    # ---- phonons --------------------------------------------------------
    ph_energies = np.asarray(elph.ph_energies, dtype=float)   # eV
    nq = int(elph.nq) if hasattr(elph, 'nq') else len(elph.qpoints)
    nm = ph_energies.shape[1]
    q_red = np.mod(np.asarray(elph.qpoints, dtype=float), 1.0)
    q_red[np.abs(q_red - 1.0) < 1e-6] = 0.0
    assert nq == nbz_lat, (
        f"{nq} q-points but {nbz_lat} BZ points. The 1/N_q sum below assumes "
        f"elph.qpoints is the FULL BZ, not the IBZ.")
    assert ph_energies.shape[0] == nq
    assert on_mesh(q_red, mesh), "elph.qpoints are not on the mesh in reduced coords"
    print(f"  phonons    : {nq} q x {nm} modes, "
          f"{ph_energies.min()*1e3:.2f} .. {ph_energies.max()*1e3:.2f} meV")
    if not (1e-4 < ph_energies.max() < 1.0):
        print("  [warn] max phonon energy is outside 0.1 meV .. 1 eV; check units")

    Q_in = kBZ[a.iQ]
    print(f"\nQ index {a.iQ}   Q_red = {np.round(Q_in, 8)}")

    Ak, iQ_iBZ = rotate_Akcv_Q(lat, kBZ, ktree, kpts_iBZ, exdbs, Q_in, Dmats,
                               neigs=nexc)
    E_n_Q = exdbs[iQ_iBZ].eigenvalues[:nexc].real.astype(float)
    print(f"  Q maps to IBZ {iQ_iBZ}, E_n = {np.round(E_n_Q, 4)}")

    # ---- electron / hole contribution flags -----------------------------
    # exciton_X_matelem takes contribution='b' for the summed amplitude. The
    # separate electron and hole pieces are what the interference analysis
    # needs, but the flag spelling is not documented, so probe for it and
    # VERIFY Ge + Gh == Gb rather than trusting a guess.
    def probe_contributions():
        _pe, em = elph.read_iq(0, bands_range=bse_bnds_range,
                               convention='standard')
        em = em.transpose(1, 0, 2, 4, 3)
        Akq0, _ = rotate_Akcv_Q(lat, kBZ, ktree, kpts_iBZ, exdbs,
                                Q_in + q_red[0], Dmats, neigs=nexc)
        base = dict(diagonal_only=False, ktree=ktree)
        ref = np.asarray(exciton_X_matelem(Q_in, q_red[0], Akq0, Ak, em, kBZ,
                                           contribution='b', **base))
        for ce, ch in (('e', 'h'), ('el', 'ho'), ('electron', 'hole'), ('c', 'v')):
            try:
                Ge = np.asarray(exciton_X_matelem(Q_in, q_red[0], Akq0, Ak, em,
                                                  kBZ, contribution=ce, **base))
                Gh = np.asarray(exciton_X_matelem(Q_in, q_red[0], Akq0, Ak, em,
                                                  kBZ, contribution=ch, **base))
            except Exception:
                continue
            if Ge.shape != ref.shape or Gh.shape != ref.shape:
                continue
            res = np.abs(Ge + Gh - ref).max() / max(np.abs(ref).max(), 1e-300)
            print(f"  contribution '{ce}'/'{ch}': |Ge+Gh-Gb|/|Gb| = {res:.3e}")
            if res < SPLIT_TOL:
                return ce, ch
        return None, None

    C_E, C_H = probe_contributions()
    if C_E is None:
        msg = ("could not find working electron/hole contribution flags for "
               "exciton_X_matelem, so Ge_grid and Gh_grid cannot be written.")
        if not a.allow_no_split:
            raise SystemExit(
                f"  ABORT: {msg}\n"
                f"  The interference analysis needs them, and running every Q\n"
                f"  without them means regenerating the whole set later.\n"
                f"  Check the contribution argument in your yambopy version,\n"
                f"  or pass --allow-no-split to write the total only on purpose.")
        print(f"  [warn] {msg} Writing the total only (--allow-no-split).")
    else:
        print(f"  electron/hole split via contribution='{C_E}' / '{C_H}'")

    # ---- per-q worker ---------------------------------------------------
    def process_q(iq):
        _ph_eig, elph_mat = elph.read_iq(iq, bands_range=bse_bnds_range,
                                         convention='standard')
        # (nk,nm,ns,init,fin) -> (nm,nk,ns,fin,init) == Omn in exciton_X_matelem
        elph_mat = elph_mat.transpose(1, 0, 2, 4, 3)
        Akq, iK_iBZ = rotate_Akcv_Q(lat, kBZ, ktree, kpts_iBZ, exdbs,
                                    Q_in + q_red[iq], Dmats, neigs=nexc)
        args = (Q_in, q_red[iq], Akq, Ak, elph_mat, kBZ)
        base = dict(diagonal_only=False, ktree=ktree)
        tot = np.asarray(exciton_X_matelem(*args, contribution='b', **base))
        if C_E is None:
            parts = (None, None)
        else:
            parts = (np.asarray(exciton_X_matelem(*args, contribution=C_E, **base)),
                     np.asarray(exciton_X_matelem(*args, contribution=C_H, **base)))
        return (iq, tot, parts,
                exdbs[iK_iBZ].eigenvalues[:nexc].real.astype(float))

    # ---- optional thread-safety verification ----------------------------
    if a.verify_threads:
        probe = list(range(0, nq, max(1, nq // 8)))[:8]
        print(f"\nverifying thread safety on q = {probe}")
        ser = {i: process_q(i)[1] for i in probe}
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
            thr = {i: r[1] for i, r in
                   ((r[0], r) for r in ex.map(process_q, probe))}
        worst = max(np.abs(ser[i] - thr[i]).max() for i in probe)
        print(f"  max |serial - threaded| = {worst:.3e}")
        if worst > 0:
            raise SystemExit("  THREADED READS ARE CORRUPTING DATA. Use --threads 1.")
        print("  identical -- threading is safe on this build.")

    # ---- run ------------------------------------------------------------
    exph_list = [None] * nq
    part_e = [None] * nq
    part_h = [None] * nq
    E_m_all = np.zeros((nq, nexc))
    print(f"\nrunning {nq} q-points ({'serial' if a.threads == 1 else f'{a.threads} threads'}) ...")
    if a.threads == 1:
        it = (process_q(i) for i in range(nq))
    else:
        pool = ThreadPoolExecutor(max_workers=a.threads)
        it = pool.map(process_q, range(nq))
    for iq, tmp, parts, E_m in tqdm(it, total=nq, desc=f"Q={a.iQ}"):
        exph_list[iq] = tmp
        part_e[iq], part_h[iq] = parts
        E_m_all[iq] = E_m
    if a.threads != 1:
        pool.shutdown()
    assert all(x is not None for x in exph_list), "some q-points produced no result"

    # ---- assemble -------------------------------------------------------
    arr = np.array(exph_list)
    assert arr.shape == (nq, nm, nexc, nexc), (
        f"exciton_X_matelem returned {arr.shape}, expected {(nq, nm, nexc, nexc)}. "
        f"Do NOT reshape past this -- the layout changed.")
    # 0.5: Ry -> Ha. transpose: (nq,nm,out,in) -> (nq,nm,in,out)
    exph_eV = 0.5 * arr.transpose(0, 1, 3, 2) * ha2ev
    g2 = np.abs(exph_eV) ** 2                       # (nq, nm, n_in, n_out)

    # electron / hole amplitudes, same units and layout as G_grid.
    # The cross term 2*Re(Ge Gh*) is gauge-INVARIANT: the arbitrary exciton
    # phase exp(-i theta) sits in both factors and cancels in the conjugate
    # pairing. So this decomposition is safe even though the phase of G
    # individually is not.
    if part_e[0] is not None:
        Ge_eV = 0.5 * np.array(part_e).transpose(0, 1, 3, 2) * ha2ev
        Gh_eV = 0.5 * np.array(part_h).transpose(0, 1, 3, 2) * ha2ev
        res = np.abs(Ge_eV + Gh_eV - exph_eV).max() / max(np.abs(exph_eV).max(), 1e-300)
        print(f"electron+hole vs total: max rel. deviation {res:.3e}")
        assert res < SPLIT_TOL, (
            f"Ge + Gh != G over the full mesh ({res:.2e} > {SPLIT_TOL:.0e}) "
            f"-- wrong contribution flags")
        cross = 2.0 * np.real(Ge_eV * np.conj(Gh_eV))
        inc = np.abs(Ge_eV) ** 2 + np.abs(Gh_eV) ** 2
        w = inc > 1e-16
        print(f"interference: <2Re(Ge Gh*)> / <|Ge|^2+|Gh|^2> = "
              f"{cross[w].sum() / inc[w].sum():+.4f}  "
              f"({'destructive' if cross[w].sum() < 0 else 'constructive'} overall)")
    else:
        Ge_eV = Gh_eV = None

    thr = 1e-12
    valid = g2 > thr
    kept = float(g2[valid].sum()) / max(float(g2.sum()), 1e-300)
    qi, vi, ni, mi = np.where(valid)
    print(f"\n{valid.sum()} of {g2.size} events above {thr:g} eV^2 "
          f"({kept*100:.6f}% of total weight retained)")

    out = os.path.join(a.outdir, f'GI_ExcPh_Q{a.iQ+1:04d}.npz')
    np.savez_compressed(
        out,
        # sparse event list
        Q=np.full(len(qi), a.iQ + 1, dtype=np.int32),
        q=(qi + 1).astype(np.int32),
        nu=(vi + 1).astype(np.int32),
        n=(ni + 1).astype(np.int32),
        m=(mi + 1).astype(np.int32),
        E_n=E_n_Q[ni],
        E_m=E_m_all[qi, mi],
        hw=ph_energies[qi, vi],
        g_squared=g2[valid],
        # dense arrays: no index-ordering guesswork downstream
        G_grid=exph_eV,
        g2_grid=g2,
        **({} if Ge_eV is None else
           {'Ge_grid': Ge_eV, 'Gh_grid': Gh_eV}),
        E_n_grid=E_n_Q,
        E_m_grid=E_m_all,
        hw_grid=ph_energies,
        q_red=q_red,
        Q_red=Q_in,
        mesh=np.array(mesh, dtype=np.int32),
        bse_bands=np.array(bse_bnds_range, dtype=np.int32),
        bse_nv=np.array([nv], dtype=np.int32),
        bse_nc=np.array([nc], dtype=np.int32),
        elph_convention=np.array('standard'),
        N_q=np.array([nq], dtype=np.int32),
        N_fine_total=np.array([nq], dtype=np.int32),   # legacy alias
    )
    print(f"saved {out}")


if __name__ == '__main__':
    main()
