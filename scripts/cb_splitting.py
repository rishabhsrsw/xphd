"""
cb_splitting.py
===============
Band edges and the conduction-band spin splitting in yambo's quasiparticle
energies, against DFT: where the valence maximum and conduction minimum lie,
whether the gap is direct or indirect, how far the Q valley sits above K, and
whether the two conduction bands at K keep their DFT order.

In WSe2 the lower conduction band at K has the OPPOSITE spin to the top
valence band, so the spin-forbidden dark exciton lies below the bright one.
If the corrections of the two conduction bands differ by more than their
splitting, the order flips. The Q valley (midway between Gamma and K) decides
whether the gap is direct (K-K) or indirect (K-Q).

Columns are read by their HEADER NAMES. With `ExtendOut` the layout of o-*.qp
is  K-point, Band, Eo, E, E-Eo, Vxc, Vnlxc, Sc|Eo, ...  -- the fourth column is
the quasiparticle energy E, not the correction. Reading by position once took
E for E-Eo; this script never does.

Run it by editing the SETTINGS block below and running the file.

Reads
-----
QP_FILE      yambo's GW output (o-*.qp), with or without ExtendOut
BAND_PARITY  optional: band_parity.npz from `xphd band-parity`. At K the band
             edges of WSe2 are mirror-even d states, so their sigma_h label
             (+-1, from the +-i eigenvalue) is their spin.
"""
import re

import numpy as np

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
QP_FILE = "o-GW.qp"                    # yambo's GW output
BAND_PARITY = None                     # "band_parity.npz" or None
VBM_BAND = 62                          # 1-based: the top valence band
WINDOW = [59, 66]                      # the BSE band window (1-based)
Q_POINT = 13                           # irreducible index of the Q valley (Gamma-K midpoint)


# ==========================================================================
# 1. READ THE QUASIPARTICLE ENERGIES, BY COLUMN NAME
# ==========================================================================
def read_named(path):
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
            return data, [re.sub(r"\s*\[.*?\]", "", n).strip().replace(" ", "") for n in names]
    raise SystemExit(f"{path}: no header line names its {data.shape[1]} columns")


d, names = read_named(QP_FILE)
col = {n.lower(): i for i, n in enumerate(names)}
for need in ("k-point", "band", "eo"):
    if need not in col:
        raise SystemExit(f"{QP_FILE}: no '{need}' column in {names}")
if "e-eo" in col:
    dE = d[:, col["e-eo"]]
elif "e" in col:
    dE = d[:, col["e"]] - d[:, col["eo"]]
else:
    raise SystemExit(f"{QP_FILE}: neither 'E-Eo' nor 'E' among {names}")
qp = {(int(r[col["k-point"]]), int(r[col["band"]])): (r[col["eo"]], de) for r, de in zip(d, dE)}
ks = sorted({k for k, _ in qp})
bands = sorted({b for _, b in qp})
print(f"{QP_FILE}: columns {names}")
print(f"   {len(qp)} entries, k-points {ks[0]}-{ks[-1]} ({len(ks)}), bands {bands[0]}-{bands[-1]}")
missing = [(k, b) for k in ks for b in range(WINDOW[0], WINDOW[1] + 1) if (k, b) not in qp]
if missing:
    print(f"   WARNING: {len(missing)} (k, band) of the BSE window have no correction "
          f"(bands {sorted({b for _, b in missing})})")
else:
    print(f"   every band {WINDOW[0]}-{WINDOW[1]} is corrected at every k-point")


def E(k, b, which):
    eo, de = qp[(k, b)]
    return eo if which == "DFT" else eo + de


# ==========================================================================
# 2. BAND EDGES: DIRECT OR INDIRECT?
# ==========================================================================
cb = [b for b in bands if b > VBM_BAND]
print("\n           VBM (k, band, eV)       CBM (k, band, eV)       gap      smallest direct gap")
edges = {}
for which in ("DFT", "GW"):
    kv = max((k for k in ks if (k, VBM_BAND) in qp), key=lambda k: E(k, VBM_BAND, which))
    kc, bc = min(((k, b) for k in ks for b in cb if (k, b) in qp), key=lambda t: E(*t, which))
    ev, ec = E(kv, VBM_BAND, which), E(kc, bc, which)
    direct = min((min(E(k, b, which) for b in cb if (k, b) in qp) - E(k, VBM_BAND, which), k)
                 for k in ks if (k, VBM_BAND) in qp)
    kind = "direct" if kv == kc else "INDIRECT"
    edges[which] = (kv, kc)
    print(f"   {which:<4}   k {kv:2d}, {VBM_BAND}, {ev:8.4f}      k {kc:2d}, {bc}, {ec:8.4f}    "
          f"{ec - ev:6.3f} eV {kind:<8}  {direct[0]:6.3f} eV at k {direct[1]}")

kK = edges["DFT"][0]
if Q_POINT in ks:
    print(f"\nQ valley (k {Q_POINT}) against K (k {kK}), lowest conduction band at each:")
    for which in ("DFT", "GW"):
        eq = min(E(Q_POINT, b, which) for b in cb if (Q_POINT, b) in qp)
        ek = min(E(kK, b, which) for b in cb if (kK, b) in qp)
        print(f"   {which:<4}  Q - K = {1e3 * (eq - ek):+7.1f} meV"
              + ("   (Q below K: the gap is indirect K -> Q)" if eq < ek else ""))

# ==========================================================================
# 3. THE CONDUCTION-BAND SPLITTING AT K
# ==========================================================================
c1, c2 = VBM_BAND + 1, VBM_BAND + 2
print(f"\nat K (k {kK}):")
print("   band      Eo (eV)    E-Eo (eV)     E (eV)")
for b in range(VBM_BAND - 1, VBM_BAND + 5):
    if (kK, b) in qp:
        eo, de = qp[(kK, b)]
        print(f"   {b:4d}   {eo:10.4f}   {de:10.4f}   {eo + de:10.4f}")
if (kK, c1) in qp and (kK, c2) in qp:
    d_dft = E(kK, c2, "DFT") - E(kK, c1, "DFT")
    d_qp = E(kK, c2, "GW") - E(kK, c1, "GW")
    print(f"\n   splitting E({c2}) - E({c1}) at K:  DFT {1e3 * d_dft:+.1f} meV,  "
          f"after GW {1e3 * d_qp:+.1f} meV")
    if np.sign(d_dft) != np.sign(d_qp):
        print("   ==> GW REVERSES the order of the two conduction bands at K")

if BAND_PARITY:
    bp = np.load(BAND_PARITY)
    kf, p = bp["kf"], bp["p"]
    blist = list(np.asarray(bp["bands"]).astype(int))
    iK = int(np.argmin(np.linalg.norm(((kf[:, :2] - [1 / 3, 1 / 3] + 0.5) % 1) - 0.5, axis=1)))
    lab = {b: np.sign(np.real(p[iK, blist.index(b)])) for b in (VBM_BAND, c1, c2) if b in blist}
    if len(lab) == 3:
        low = c1 if E(kK, c1, "GW") < E(kK, c2, "GW") else c2
        same = lab[low] == lab[VBM_BAND]
        print(f"\nsigma_h labels at K: band {VBM_BAND} {lab[VBM_BAND]:+.0f}, band {c1} "
              f"{lab[c1]:+.0f}, band {c2} {lab[c2]:+.0f}")
        print(f"   after GW the lower conduction band at K is {low}: "
              f"{'SAME spin as the valence maximum -> bright exciton lowest (Mo-like)' if same else 'OPPOSITE spin -> dark exciton lowest (W-like)'}")
