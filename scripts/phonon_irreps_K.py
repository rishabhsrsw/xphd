"""
phonon_irreps_K.py
==================
C3 characters and C3h labels of the phonon modes at K, about a chosen C3 axis
-- the phonon half of the emission rule  Gamma_exc in Gamma_nu x Gamma_photon,
whose exciton half `xphd irreps` supplies.

Run it by editing the SETTINGS block below and running the file.

What it computes
----------------
C3 (counter-clockwise by 120 degrees about the axis O) maps atom kappa onto atom
kappa' = F(kappa) of the same species, shifted by a lattice vector, and maps K
onto K + G. Applied to a phonon at K it gives

    e'_{kappa'} = R e_kappa  exp(i [K.(tau_kappa' - tau_kappa) + G.(tau_kappa' - O)])

in the lattice convention u_kappa(L) = e_kappa exp(i K.R_L), or the same
without the K term in the atom convention u_kappa(L) = e_kappa exp(i K.(R_L +
tau_kappa)). For a non-degenerate mode e' = lambda e, with lambda one of
1, w, w^2 (w = exp(2 pi i / 3)); near-degenerate modes are diagonalised
together. sigma_h gives +-1 (planar layer: 1 - 2 w_z).

    lambda = 1    -> A'  (sigma_h even) / A''  (odd)
    lambda = w    -> E'_1             / E''_1
    lambda = w^2  -> E'_2             / E''_2

Which convention matdyn's eigenvectors follow is not assumed. A mode living on
one sublattice gives the same lambda in both; a mode mixing the two sublattices
gives |lambda| = 1 only in the right one. The script evaluates both, keeps the
one in which every mode is a clean eigenvector, and says which it used.

Conventions that change the NAMES (not the physics): E_1 and E_2 swap for the
opposite rotation sense and between K and K'; A and E_k move with the axis.
Use the same axis and sense as for the exciton labels -- `xphd irreps` rotates
about the origin of the cell, so AXIS = "origin" matches it.

Reads
-----
MODES      matdyn's flvec file (matdyn.modes) containing K: a full mesh whose
           size is a multiple of 3, or a single-q run at K.
POSITIONS  the atoms in the order of the QE input: (label, [x, y]), copied from
           ATOMIC_POSITIONS in scf.in. POSITIONS_UNITS says which kind:
           "crystal" for ATOMIC_POSITIONS {crystal}, "alat" for {alat}
           (Cartesian, in units of alat). The crystal coordinates used are
           printed, so a wrong choice is visible at once.
CELL       a1, a2 in units of alat (QE ibrav = 4 by default; or the first two
           rows of CELL_PARAMETERS {alat}).
"""
import numpy as np

from xphd.modes import read_modes

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
MODES = "matdyn.modes"                       # flvec of a mesh containing K, or a run at K
POSITIONS = None                             # e.g. [("N", [0.0, 0.0]), ("Ga", [2/3, 1/3])]
POSITIONS_UNITS = "crystal"                  # "crystal" ({crystal}) or "alat" ({alat}: Cartesian / alat)
CELL = [[1.0, 0.0], [-0.5, 0.8660254037844386]]   # a1, a2 / alat (ibrav = 4)
AXIS = "origin"                              # "origin", an atom label, or [x, y] crystal
K_POINT = [1 / 3, 1 / 3]                     # K in reduced coordinates of b1, b2
DEG_TOL = 0.5                                # cm-1: modes closer than this are diagonalised together
CLEAN = 1e-3                                 # |e' - lambda e| / |e| below this is a clean eigenvector


# ==========================================================================
# 1. GEOMETRY
# ==========================================================================
if POSITIONS is None:
    raise SystemExit("Set POSITIONS: copy ATOMIC_POSITIONS {crystal} from scf.in, "
                     "atoms in the same order, e.g. [('Ga', [0, 0]), ('N', [1/3, 2/3])].")
A = np.array(CELL, float)                      # rows a1, a2 (alat)
B = np.linalg.inv(A).T                         # rows b1, b2 (2 pi / alat)
labels = [p[0] for p in POSITIONS]
given = np.array([p[1][:2] for p in POSITIONS], float)
if POSITIONS_UNITS == "crystal":
    frac = given
elif POSITIONS_UNITS == "alat":
    frac = np.linalg.solve(A.T, given.T).T     # Cartesian (alat) -> crystal
else:
    raise SystemExit(f"POSITIONS_UNITS = {POSITIONS_UNITS!r}: use 'crystal' or 'alat'")
tau = frac @ A                                 # Cartesian, alat
nat = len(labels)
print("atoms (crystal coordinates used):  "
      + "   ".join(f"{labels[k]} ({frac[k, 0]:.4f}, {frac[k, 1]:.4f})" for k in range(nat)))

if isinstance(AXIS, str) and AXIS == "origin":
    O, axis_txt = np.zeros(2), "the origin of the cell"
elif isinstance(AXIS, str):
    if AXIS not in labels:
        raise SystemExit(f"AXIS = {AXIS!r} is not one of the atoms {labels}")
    O, axis_txt = tau[labels.index(AXIS)], f"the {AXIS} site"
else:
    O, axis_txt = np.array(AXIS[:2], float) @ A, f"crystal point {list(AXIS)}"
on_axis = [labels[k] for k in range(nat)
           if np.allclose(((frac[k] - np.linalg.solve(A.T, O)) + 0.5) % 1 - 0.5, 0, atol=1e-6)]

c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
R2 = np.array([[c, -s], [s, c]])               # counter-clockwise 120 degrees
R3 = np.eye(3)
R3[:2, :2] = R2

# the atom map kappa -> kappa' with R(tau_k - O) + O = tau_k' + lattice vector
image = np.full(nat, -1)
for k in range(nat):
    p = R2 @ (tau[k] - O) + O
    for kp in range(nat):
        if labels[kp] != labels[k]:
            continue
        d = np.linalg.solve(A.T, p - tau[kp])  # crystal coordinates of the shift
        if np.allclose(d, np.rint(d), atol=1e-5):
            image[k] = kp
            break
    if image[k] < 0:
        other = "alat" if POSITIONS_UNITS == "crystal" else "crystal"
        alt = (np.linalg.solve(A.T, given.T).T if other == "alat" else given @ A)
        hint = ("   ".join(f"{labels[a]} ({alt[a, 0]:.4f}, {alt[a, 1]:.4f})" for a in range(nat)))
        raise SystemExit(
            f"C3 about {axis_txt} does not map atom {k + 1} ({labels[k]}) onto an atom of the same "
            f"species.\n  The positions were read as {POSITIONS_UNITS!r}. Read as {other!r} instead, "
            f"they would be {'in crystal coordinates' if other == 'alat' else 'in alat units'}: {hint}\n"
            f"  If that looks right (e.g. 1/3, 2/3 values), set POSITIONS_UNITS = {other!r}; "
            f"otherwise check CELL and AXIS.")

Kc = np.array(K_POINT, float) @ B              # Cartesian, 2 pi / alat
Gv = R2 @ Kc - Kc
g_red = np.linalg.solve(B.T, Gv)
if not np.allclose(g_red, np.rint(g_red), atol=1e-6):
    raise SystemExit(f"K_POINT = {K_POINT} is not mapped onto itself by C3 (R K - K = {g_red} "
                     f"in reduced units): is it K?")

# ==========================================================================
# 2. THE MODES AT K
# ==========================================================================
q, freq, ev = read_modes(MODES)
if ev.shape[2] != nat:
    raise SystemExit(f"{MODES} has {ev.shape[2]} atoms, POSITIONS has {nat}")
q_red = np.linalg.solve(B.T, q[:, :2].T).T
dk = (q_red - np.array(K_POINT) + 0.5) % 1.0 - 0.5
iK = int(np.argmin(np.linalg.norm(dk, axis=1)))
if np.linalg.norm(dk[iK]) > 1e-4:
    raise SystemExit(f"K = {K_POINT} is not among the {len(q)} q-points of {MODES} "
                     f"(nearest: {q_red[iK].round(4)}). Use a mesh whose size is a multiple "
                     f"of 3, or a matdyn run at K.")
w = freq[iK]
E = ev[iK].reshape(len(w), -1)                 # (nmod, 3 nat), matdyn's displacement patterns
print(f"\nK = {K_POINT} is q-point {iK + 1} of {MODES}: {len(w)} modes, {nat} atoms")
print(f"C3: counter-clockwise by 120 degrees about {axis_txt}"
      + (f" (atom on the axis: {', '.join(on_axis)})" if on_axis else " (no atom on the axis)"))


def transform(vec, convention):
    """Apply C3 to one mode, (3 nat,) -> (3 nat,)."""
    v = vec.reshape(nat, 3)
    out = np.zeros_like(v)
    for k in range(nat):
        kp = image[k]
        ph = Gv @ (tau[kp] - O)
        if convention == "lattice":
            ph += Kc @ (tau[kp] - tau[k])
        out[kp] = (R3 @ v[k]) * np.exp(2j * np.pi * ph)
    return out.ravel()


def group_modes(freqs, tol):
    groups, cur = [], [0]
    for i in range(1, len(freqs)):
        if abs(freqs[i] - freqs[cur[-1]]) < tol:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return groups


def characters(convention):
    lam = np.zeros(len(w), complex)
    worst = 0.0
    for g in group_modes(w, DEG_TOL):
        V = E[g].T                                         # (3 nat, ng)
        Vp = np.stack([transform(E[i], convention) for i in g], axis=1)
        S = V.conj().T @ V
        M = V.conj().T @ Vp
        C = np.linalg.solve(S, M)                          # C3 in this subspace
        vals = np.linalg.eigvals(C)
        resid = np.linalg.norm(Vp - V @ C) / max(np.linalg.norm(V), 1e-30)
        worst = max(worst, resid, float(np.max(np.abs(np.abs(vals) - 1))))
        order = np.argsort(np.angle(vals))
        for i, v in zip(g, vals[order]):
            lam[i] = v
    return lam, worst


res = {cv: characters(cv) for cv in ("lattice", "atom")}
conv = min(res, key=lambda cv: res[cv][1])
lam, worst = res[conv]
other = res["atom" if conv == "lattice" else "lattice"][1]
if worst > CLEAN:
    print(f"\n  WARNING: no convention gives clean eigenvectors (worst {worst:.1e} lattice / "
          f"{other:.1e} other). Check POSITIONS (order, coordinates) and CELL.")
elif np.all(image == np.arange(nat)):
    print(f"\n  C3 maps every atom onto its own sublattice, so the labels do not depend on "
          f"the phase convention of the eigenvectors (clean to {worst:.0e})")
elif other < CLEAN:
    print(f"\n  both phase conventions give clean eigenvectors (to {worst:.0e}): the labels do not depend on it")
else:
    print(f"\n  eigenvector phase convention: {conv} (clean to {worst:.0e}; the other gives {other:.0e})")

# ==========================================================================
# 3. LABELS
# ==========================================================================
omega = np.exp(2j * np.pi / 3)
wz = np.array([np.sum(np.abs(E[i].reshape(nat, 3)[:, 2]) ** 2) / np.sum(np.abs(E[i]) ** 2)
               for i in range(len(w))])
print(f"\n  mode   freq (cm-1)   freq (meV)    C3 character        sigma_h   C3h label   on atoms")
for i in range(len(w)):
    k = int(np.argmin(np.abs(lam[i] - np.array([1, omega, omega ** 2]))))
    sh = 1 - 2 * wz[i]
    prime = "'" if sh > 0 else "''"
    lab = ("A" + prime) if k == 0 else (f"E{prime}_{k}")
    amp = np.sum(np.abs(E[i].reshape(nat, 3)) ** 2, axis=1)
    who = ", ".join(f"{labels[a]} {100 * amp[a] / amp.sum():.0f}%" for a in range(nat) if amp[a] / amp.sum() > 0.005)
    ctxt = ["1", "w", "w^2"][k]
    print(f"  {i + 1:4d}   {w[i]:10.2f}   {w[i] * 0.123984:9.2f}    "
          f"{lam[i].real:+.4f}{lam[i].imag:+.4f}i (={ctxt:<3})  {sh:+5.2f}    {lab:<9}   {who}")
print("\n  w = exp(2 pi i / 3). E_1 and E_2 swap for the opposite rotation sense and between K and K';")
print("  label the excitons (xphd irreps) with the same axis and sense.")
