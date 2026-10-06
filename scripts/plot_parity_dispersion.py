"""
plot_parity_dispersion.py
=========================
yambo's interpolated band structure, each band coloured by its sigma_h parity.

Works for both kinds of band structure:

  excitons   ypp -e i  ->  o-*.excitons_interpolated,  with parity.npz
             from `xphd parity` (chi of each exciton state across the zone)

  electrons  ypp -s b  ->  o-*.bands_interpolated,     with band_parity.npz
             from `xphd band-parity` (p_n(k) of each band in the BSE window)

The kind is read from the npz: band_parity.npz carries `bands`, parity.npz
does not.

    python plot_parity_dispersion.py o-output.excitons_interpolated_01 \\
           --parity parity.npz --out exciton_parity_dispersion.pdf
    python plot_parity_dispersion.py o-output.bands_interpolated \\
           --parity band_parity.npz --first-band 1 --out band_parity_dispersion.pdf

The interpolated file
---------------------
Column 0 is the distance along the path, the next NB columns the band
energies, and the last three the path coordinates in reduced units (rlu) --
the layout fatbands_pl.py and bte_figure.py already read. NB is inferred as
(columns - 4) and confirmed by checking that the trailing columns really are
reduced coordinates; --nbands overrides it. For the electronic file, ypp must
write rlu: set cooOut = "rlu" in its input.

Those reduced coordinates are the same as Q_red in the npz, which is what
lets every path point be located on the parity mesh.

What the colour means
---------------------
Parity is COMPUTED only on the BSE mesh (24x24). Between mesh points the colour
is interpolated -- linearly by default, which is what gives smooth transitions;
`--method nearest` shows parity as the step it really is. A colour between the
two extremes therefore means one of two things: a state that genuinely mixes
parities (|chi| < 1 at the mesh points themselves, like the annulus in GaN),
or the interpolation crossing a boundary between mesh points. `--mesh-points`
marks the computed values so the two can be told apart.

Bands are energy-ordered at every point, in the file and in the npz alike, so
where two bands of opposite parity cross, the colour of "band n" changes
although no single state did.

Checks
------
Where the path passes through a mesh point -- Gamma, M and K at least -- the
interpolated energy is compared with the npz's. For excitons the two must
agree directly. For electrons ypp usually shifts the energies, so they are
compared after removing one common offset -- which works for a Kohn-Sham file.
A GW file moves each band by its own correction, the conduction bands more
than the valence bands, and no common offset fits; the check then reports the
per-band offsets and confirms each band's identity from the SHAPE of its
dispersion instead. A wrong --first-band fails either way.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
# yambo o-*.excitons_interpolated or o-*.bands_interpolated
INTERP = 'o-output.excitons_interpolated_01'

# parity.npz (excitons) or band_parity.npz (electrons)
PARITY = "parity.npz"

# energy columns in the interpolated file; inferred if omitted
NBANDS = None

# electrons only: 1-based absolute index of the first band in the
# interpolated file -- the first of ypp's BANDS_bands
FIRST_BAND = None

# electrons: how to check the file against band_parity.npz, which holds
# Kohn-Sham energies. ks: one common offset. qp: a quasiparticle file --
# per-band offsets, and band identity from the shape of each dispersion.
# auto: ks, falling back to qp when a common offset does not fit  One of
# ("auto", "ks", "qp").
ENERGIES = "auto"

# how parity is carried between mesh points  One of ("linear", "nearest",
# "cubic").
METHOD = "linear"

# mark where the path meets the mesh, i.e. computed values
MESH_POINTS = False
EMIN = None
EMAX = None

# line width
LW = 3.0
CMAP = "coolwarm"
TITLE = None
FIGSIZE = [6.2, 5.2]
DPI = 400
OUT = "parity_dispersion.pdf"

# also open the figure in a window after saving it; False only saves it
SHOW = True




def to_cart(q):
    """Reduced -> Cartesian with b1 = (1, 0), b2 = (1/2, sqrt3/2)."""
    q = np.asarray(q, float)
    return np.stack([q[:, 0] + 0.5 * q[:, 1], np.sqrt(3.0) / 2 * q[:, 1]], 1)


def wrap(d):
    return d - np.rint(d)


# ---- the interpolated file ----------------------------------------------------
raw = np.loadtxt(INTERP, comments="#")
ncol = raw.shape[1]


def coords_ok(nb):
    """Do the columns after nb energies look like reduced coordinates?"""
    if nb < 1 or 1 + nb + 2 > ncol:
        return False
    c = raw[:, 1 + nb:]
    return c.shape[1] in (2, 3) and np.abs(c[:, :2]).max() <= 1.01 and (
        c.shape[1] == 2 or np.abs(c[:, 2]).max() < 1e-3)


if NBANDS is not None:
    NB = NBANDS
    if not coords_ok(NB):
        raise SystemExit(f"--nbands {NB}: columns {1 + NB}.. are not reduced "
                         f"coordinates; check the value, or cooOut = \"rlu\"")
else:
    NB = next((nb for nb in (ncol - 4, ncol - 3) if coords_ok(nb)), None)
    if NB is None:
        raise SystemExit(f"{INTERP}: {ncol} columns, and the trailing ones "
                         f"are not reduced coordinates. Set cooOut = \"rlu\" "
                         f"in the ypp input, or pass --nbands")
x = raw[:, 0]
Ei = raw[:, 1:1 + NB]
qp = raw[:, 1 + NB:3 + NB]
print(f"{INTERP}: {len(x)} path points, {NB} bands, reduced coordinates "
      f"{qp.min():.3f} .. {qp.max():.3f}")

# ---- the parity ---------------------------------------------------------------
d = np.load(PARITY)
Q = d["Q_red"] if "Q_red" in d.files else d["kf"]
chi = d["chi"] if "chi" in d.files else d["p"]
Enpz = d["E"] if "E" in d.files else None
electrons = "bands" in d.files
if electrons:
    bands = np.asarray(d["bands"], int)              # 1-based absolute
    if FIRST_BAND is None:
        raise SystemExit("band_parity.npz given: pass --first-band, the "
                         "1-based index of the first band in the interpolated "
                         "file (the first of ypp's BANDS_bands)")
    col_of = {}
    for j in range(NB):
        b = FIRST_BAND + j
        if b in bands:
            col_of[j] = int(np.where(bands == b)[0][0])
    kind = "band"
    print(f"electrons: file bands {FIRST_BAND}..{FIRST_BAND + NB - 1}; "
          f"parity for {bands[0]}..{bands[-1]}; coloured: "
          f"{[FIRST_BAND + j for j in col_of]}")
else:
    col_of = {j: j for j in range(min(NB, chi.shape[1]))}
    kind = "state"
    print(f"excitons: {NB} states in the file, {chi.shape[1]} in "
          f"{PARITY}; coloured: 1..{len(col_of)}")
    if NB > chi.shape[1]:
        print(f"   states {chi.shape[1] + 1}..{NB} have no parity (xphd parity "
              f"--nstates {chi.shape[1]}) and are drawn grey")

# ---- where the path meets the mesh: the energy check ---------------------------
dd = wrap(qp[:, None, :] - np.asarray(Q)[None, :, :2])
dist = np.linalg.norm(to_cart(dd.reshape(-1, 2)).reshape(dd.shape), axis=2)
nearest = np.argmin(dist, axis=1)
on_mesh = dist[np.arange(len(qp)), nearest] < 1e-4
print(f"path points on the mesh: {on_mesh.sum()} (these carry computed values)")
def _corr(u, v):
    """Correlation of two dispersions along the path, means removed."""
    u = u - u.mean()
    v = v - v.mean()
    d = float(np.sqrt((u @ u) * (v @ v)))
    return float(u @ v) / d if d > 1e-12 else float("nan")


if Enpz is not None and on_mesh.any() and col_of:
    js = sorted(col_of)
    ei = Ei[on_mesh][:, js]                            # file, (points, bands)
    En = np.asarray(Enpz)[nearest[on_mesh]]            # every npz band
    en = En[:, [col_of[j] for j in js]]
    diff = ei - en
    if not electrons:
        res = float(np.abs(diff).max())
        print(f"energy check at those points: largest mismatch "
              f"{res * 1e3:.1f} meV")
        if res > 0.05:
            print("   WARNING: more than 50 meV -- the file and the npz may be "
                  "from different calculations")
    else:
        shift = float(np.median(diff))
        res = float(np.abs(diff - shift).max())
        ks_ok = res <= 0.05
        if ENERGIES == "ks" or (ENERGIES == "auto" and ks_ok):
            print(f"energy check after a common offset of {shift:+.3f} eV at "
                  f"those points: largest mismatch {res * 1e3:.1f} meV")
            if not ks_ok:
                print("   WARNING: more than 50 meV -- the file and the npz may "
                      "be from different calculations, or --first-band is wrong")
        else:
            # A quasiparticle file: band_parity.npz holds Kohn-Sham energies,
            # and GW moves every band by its own amount. Compare each band's
            # offset instead, and judge identity by the SHAPE of its dispersion,
            # which a wrong --first-band would still get wrong.
            off = diff.mean(axis=0)
            nvw = int(d["nv"]) if "nv" in d.files else None
            print("energies are not one common shift of the npz's -- per-band "
                  "offsets " + ", ".join(f"{FIRST_BAND + j}: {o:+.2f}"
                                          for j, o in zip(js, off)) + " eV")
            if nvw:
                val = [o for j, o in zip(js, off) if col_of[j] < nvw]
                con = [o for j, o in zip(js, off) if col_of[j] >= nvw]
                if val and con:
                    print(f"   conduction shifted {np.mean(con) - np.mean(val):+.2f} "
                          f"eV relative to valence: a quasiparticle-corrected "
                          f"file against Kohn-Sham parities")
            print(f"   k-dependence of each band's correction: up to "
                  f"{np.abs(diff - off).max() * 1e3:.0f} meV along the path")
            bad, worst = [], 1.0
            bands_abs = np.asarray(d["bands"], int)
            for i, j in enumerate(js):
                c = np.array([_corr(ei[:, i], En[:, k])
                              for k in range(En.shape[1])])
                mine = c[col_of[j]]
                if not np.isfinite(mine):
                    continue                           # flat here: no verdict
                worst = min(worst, mine)
                # ties within 0.02 are degenerate partners, not a mismatch
                if mine < np.nanmax(c) - 0.02 or mine < 0.8:
                    bad.append((FIRST_BAND + j,
                                int(bands_abs[int(np.nanargmax(c))])))
            if bad:
                print("   WARNING: by the shape of the dispersion, "
                      + ", ".join(f"file band {f} looks like npz band {g}"
                                  for f, g in bad)
                      + " -- --first-band is probably wrong")
            else:
                print(f"   band identity confirmed by the shape of each band's "
                      f"dispersion (worst correlation {worst:.3f})")

# ---- parity on the path ---------------------------------------------------------
img = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], float)
Qt = np.concatenate([np.asarray(Q)[:, :2] + s for s in img])
Pc = to_cart(qp)
Qc = to_cart(Qt)
chi_path = np.full((len(x), NB), np.nan)
for j, c in col_of.items():
    v = np.tile(np.asarray(chi)[:, c], len(img))
    z = griddata(Qc, v, Pc, method=METHOD)
    if np.isnan(z).any():
        z = np.where(np.isnan(z), griddata(Qc, v, Pc, method="nearest"), z)
    chi_path[:, j] = np.clip(z, -1, 1)

# ---- high-symmetry ticks along the path -------------------------------------------
HS = {r"$\Gamma$": [(0, 0)], "M": [(0.5, 0), (0, 0.5), (0.5, 0.5)],
      "K": [(1 / 3, 1 / 3), (2 / 3, 2 / 3), (1 / 3, 2 / 3), (2 / 3, 1 / 3)]}
# A dense path puts several consecutive points within tolerance of each
# corner, and ypp may repeat a corner point. Take ONE tick per run of
# consecutive matches, at the point closest to the corner, then merge any
# same-label ticks closer than 1% of the path length.
ticks = []
for lab, pts in HS.items():
    dl = np.array([min(np.linalg.norm(to_cart(wrap(q - np.array(pt))[None, :]))
                       for pt in pts) for q in qp])
    hit = dl < 1e-3
    i = 0
    while i < len(hit):
        if not hit[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(hit) and hit[j + 1]:
            j += 1
        k = i + int(np.argmin(dl[i:j + 1]))
        ticks.append((x[k], lab))
        i = j + 1
ticks.sort()
merged = []
for v, lab in ticks:
    if merged and merged[-1][1] == lab and v - merged[-1][0] < 0.01 * (x[-1] - x[0]):
        continue
    merged.append((v, lab))
ticks = merged
print("ticks: " + ", ".join(f"{l} at {v:.3f}" for v, l in ticks))

# ---- figure -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=tuple(FIGSIZE))
norm = Normalize(-1, 1)
for j in range(NB):
    y = Ei[:, j]
    if j not in col_of:
        ax.plot(x, y, color="0.7", lw=LW * 0.6, zorder=1)
        continue
    pts = np.stack([x, y], 1)[:, None, :]
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cval = 0.5 * (chi_path[:-1, j] + chi_path[1:, j])
    lc = LineCollection(seg, cmap=CMAP, norm=norm, linewidths=LW,
                        capstyle="round", zorder=2)
    lc.set_array(cval)
    ax.add_collection(lc)
    if MESH_POINTS:
        ax.scatter(x[on_mesh], y[on_mesh], s=10, c="black", zorder=3,
                   linewidths=0)

for v, _ in ticks[1:-1]:
    ax.axvline(v, color="0.6", ls="--", lw=0.9, zorder=0)
ax.set_xticks([v for v, _ in ticks])
ax.set_xticklabels([l for _, l in ticks], fontsize=15)
ax.set_xlim(x[0], x[-1])
lo = EMIN if EMIN is not None else float(np.nanmin(Ei)) - 0.05
hi = EMAX if EMAX is not None else float(np.nanmax(Ei)) + 0.05
ax.set_ylim(lo, hi)
ax.set_ylabel("Energy (eV)", fontsize=14)
ax.tick_params(direction="in", labelsize=12, right=True, top=False)

sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
cb = fig.colorbar(sm, ax=ax, pad=0.04, fraction=0.05)
cb.set_ticks([-1, 0, 1])
cb.set_ticklabels(["-1 (Odd)", "0 (Mixed)", "+1 (Even)"])
cb.set_label(r"$\sigma_h$ parity character", fontsize=12)
default = ("Exciton parity dispersion ($\\sigma_h$)" if not electrons
           else "Band parity ($\\sigma_h$)")
ax.set_title(TITLE or default, fontsize=14)
fig.tight_layout()
fig.savefig(OUT, dpi=DPI)
print(f"saved {OUT}  ({kind}s coloured: {len(col_of)}, method {METHOD})")

# ==========================================================================
# show the figure -- set SHOW = False in SETTINGS to only save it
# ==========================================================================
if SHOW:
    plt.show()
