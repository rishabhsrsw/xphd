"""
qp_decompose.py
===============
Where does a k-dependent GW correction come from: exchange or correlation?

yambo's Newton solution is  E - Eo = Z [ (Sx - Vxc) + Sc(Eo) ].  The split
into an exchange part Z (Sx - Vxc) and a correlation part is most reliable
when Sx - Vxc comes from a Hartree-Fock run with the SAME exchange settings
(`yambo -x`, runlevel HF_and_locXC: columns Eo, Ehf, Vxc, Vnlxc), so give
HF_FILE whenever you can. The correlation part is then (E - Eo) - Z(Sx - Vxc).

Without an HF file the script falls back on the GW file's own Sc|Eo column.
Columns are found by their HEADER NAMES, never by position: `ExtendOut`
changes the layout of o-*.qp, and reading a column by position once mistook
another quantity for Sc. When both an HF file and an Sc column are present,
the Sc the HF exchange implies is compared with the column, and a mismatch
is reported -- it means the two runs do not share their exchange settings,
or the column is not what its name suggests.

Run it by editing the SETTINGS block below and running the file -- in VS
Code, Run Python File. No command line is needed.
"""
import re

import numpy as np

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
QP_FILE = "o-output.qp"        # GW output: needs Eo and E-Eo (and Sc|Eo if no HF file)
HF_FILE = "o-hf.hf"            # HF output with the same exchange settings; "" for none
QP_DB = "ndb.QP"               # for Z; "" to use Z_DEFAULT
Z_DEFAULT = 0.78
KPOINTS = {"Gamma": 1, "M": 7, "Q": 13, "K": 19}   # 1-based irreducible indices
REFERENCE = "K"
VB, CB = 62, 63
MISMATCH = 0.05                # eV: report Sc disagreements larger than this


# ==========================================================================
# 1. READ COLUMNS BY NAME
# ==========================================================================
def read_named(path):
    """Data of a yambo o-* file and the names of its columns, from the last
    comment line that names as many columns as the data has."""
    header = []
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                names = [n.strip() for n in re.split(r"\s{2,}", line.lstrip("# ").strip()) if n.strip()]
                if names:
                    header.append(names)
    data = np.loadtxt(path, ndmin=2)
    for names in reversed(header):
        if len(names) == data.shape[1]:
            return data, [re.sub(r"\s*\[.*?\]", "", n).strip() for n in names]
    raise SystemExit(f"{path}: no header line names its {data.shape[1]} columns; "
                     f"header lengths found: {[len(h) for h in header]}")


def col(names, *wanted):
    for w in wanted:
        for i, n in enumerate(names):
            if n.replace(" ", "").lower() == w.replace(" ", "").lower():
                return i
    return None


q, qn = read_named(QP_FILE)
ik, ib = col(qn, "K-point"), col(qn, "Band")
iEo, idE, iSc = col(qn, "Eo"), col(qn, "E-Eo"), col(qn, "Sc|Eo", "Sc")
if None in (ik, ib, iEo, idE):
    raise SystemExit(f"{QP_FILE}: columns {qn}; K-point, Band, Eo and E-Eo are needed")
print(f"{QP_FILE}: columns {qn}")


def key(r, a, b):
    return int(r[a]), int(r[b])


Eo = {key(r, ik, ib): r[iEo] for r in q}
dE = {key(r, ik, ib): r[idE] for r in q}
ScCol = {key(r, ik, ib): r[iSc] for r in q} if iSc is not None else {}

X = {}                                              # Sx - Vxc
if HF_FILE:
    h, hn = read_named(HF_FILE)
    jk, jb, jvx, jsx = (col(hn, "K-point"), col(hn, "Band"), col(hn, "Vxc"),
                        col(hn, "Vnlxc", "Sx", "nlXC"))
    if None in (jk, jb, jvx, jsx):
        raise SystemExit(f"{HF_FILE}: columns {hn}; Vxc and Vnlxc are needed")
    X = {key(r, jk, jb): r[jsx] - r[jvx] for r in h}
    print(f"{HF_FILE}: columns {hn}  ->  Sx - Vxc = Vnlxc - Vxc")
elif not ScCol:
    raise SystemExit(f"{QP_FILE} has no Sc column and no HF_FILE is given")

Z = {}
if QP_DB:
    from netCDF4 import Dataset
    db = Dataset(QP_DB)
    tab = np.array(db.variables["QP_table"][:]).astype(int)
    tab = tab.T if tab.shape[0] in (3, 4) else tab
    zz = np.array(db.variables["QP_Z"][:])
    zre = zz[:, 0] if zz.ndim == 2 else np.real(zz)
    nk = max(k for k, _ in dE)
    kcol = [c for c in range(tab.shape[1]) if set(tab[:, c]) == set(range(1, nk + 1))][-1]
    bcol = [c for c in range(tab.shape[1]) if c != kcol][0]
    Z = {(int(r[kcol]), int(r[bcol])): float(z) for r, z in zip(tab, zre)}
    print(f"Z from {QP_DB}: {min(Z.values()):.3f} .. {max(Z.values()):.3f}")


def parts(s):
    z = Z.get(s, Z_DEFAULT)
    if X:
        x = z * X[s]
        return dE[s], x, dE[s] - x, z
    c = z * ScCol[s]
    return dE[s], dE[s] - c, c, z


states = [(n, k, b) for n, k in KPOINTS.items() for b in (VB, CB)
          if (k, b) in dE and (not X or (k, b) in X)]

# ==========================================================================
# 2. THE CORRECTIONS, SPLIT
# ==========================================================================
src = "the HF run" if X else "the Sc|Eo column"
print(f"\nexchange from {src}")
print("   point   band     Eo      E-Eo   = Z(Sx-Vxc)  +  correlation    Z")
for n, k, b in states:
    t, x, c, z = parts((k, b))
    print(f"   {n:<6} {b:4d}  {Eo[(k, b)]:8.4f}  {t:+8.4f}   {x:+9.4f}     {c:+9.4f}    {z:.3f}")

kr = KPOINTS[REFERENCE]
print(f"\ndifferences from {REFERENCE} (eV): how much each part adds to a band's offset")
print("   point   band   d(Eo)    d(E-Eo)   from exchange   from correlation")
for n, k, b in states:
    if k == kr or (kr, b) not in dE:
        continue
    t, x, c, _ = parts((k, b))
    t0, x0, c0, _ = parts((kr, b))
    print(f"   {n:<6} {b:4d}  {Eo[(k, b)] - Eo[(kr, b)]:+7.4f}  {t - t0:+8.4f}      "
          f"{x - x0:+8.4f}         {c - c0:+8.4f}")

if X and ScCol:
    bad = []
    for n, k, b in states:
        z = Z.get((k, b), Z_DEFAULT)
        implied = dE[(k, b)] / z - X[(k, b)]
        if abs(implied - ScCol[(k, b)]) > MISMATCH:
            bad.append((n, b, implied, ScCol[(k, b)]))
    print()
    if bad:
        print(f"WARNING: the Sc the HF exchange implies disagrees with {QP_FILE}'s Sc column:")
        for n, b, imp, cc in bad:
            print(f"   {n:<6} {b}: implied {imp:+.4f}, column {cc:+.4f}")
        print("   Either the HF and GW runs differ in their exchange settings, or that")
        print("   column is not Sc(Eo). The split above uses the HF exchange.")
    else:
        print(f"the HF exchange and {QP_FILE}'s Sc column agree within {MISMATCH} eV")

print("\nCompare the offsets with published G0W0 for the material (for WSe2: whether Q ends")
print("up below or above K). The part that carries an implausible difference is where to")
print("look: correlation -> the screening (RIM-W, bands, block size); exchange -> the")
print("exchange cutoff or the exchange-correlation potential.")
