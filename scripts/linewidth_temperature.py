"""
SI_K.py -- linewidth vs temperature and the emission/absorption decomposition,
with broken y-axes so the low-lying states stay readable alongside a state an
order of magnitude wider.

A broken axis is two stacked axes sharing an x-axis, each showing one range.
Bars and lines are drawn identically on both, so nothing is rescaled and the
bar lengths remain linear within each segment -- unlike a log axis, which
would compress the differences that matter at the bottom.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ----------------------------------------------------------------- inputs
FILE = "K.txt"
OUT = "SI_K.pdf"
STATES = np.arange(1, 6)          # which exciton states to show in the bars
LINES = [1, 2, 4]                 # which states in the temperature panel
T_SHOW = (75.0, 300.0)            # the two histogram temperatures

# Break limits. Set to None to detect the widest gap in the data instead.
TOP_BREAK = None                  # e.g. ((0, 100), (400, 560))
BAR_BREAK = None                  # e.g. ((0, 100), (500, 560))
HEIGHT_RATIO = (1, 2)             # upper segment : lower segment


# ------------------------------------------------------------- utilities
def auto_break(vals, pad=0.08, min_frac=0.25):
    """Find the widest gap in `vals` and return ((lo0, lo1), (hi0, hi1)).

    Returns None if no gap spans at least `min_frac` of the range, in which
    case a break would be gratuitous and a single axis is better.
    """
    v = np.sort(np.asarray(vals, float).ravel())
    v = v[np.isfinite(v)]
    if v.size < 2:
        return None
    d = np.diff(v)
    k = int(np.argmax(d))
    span = v[-1] - v[0]
    if span <= 0 or d[k] < min_frac * span:
        return None
    m = pad * span
    return ((max(0.0, v[0] - m), v[k] + m), (v[k + 1] - m, v[-1] + m))


def broken(fig, spec, lo, hi, ratio=HEIGHT_RATIO, hspace=0.06):
    """Two stacked axes with a break between them. Returns (ax_hi, ax_lo)."""
    g = spec.subgridspec(2, 1, height_ratios=ratio, hspace=hspace)
    ax_hi = fig.add_subplot(g[0])
    ax_lo = fig.add_subplot(g[1], sharex=ax_hi)
    ax_hi.set_ylim(*hi)
    ax_lo.set_ylim(*lo)
    ax_hi.spines["bottom"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_hi.tick_params(axis="x", bottom=False, labelbottom=False)
    return ax_hi, ax_lo


def draw_break(ax_hi, ax_lo, d=0.014):
    """Slanted marks on the facing spines, marking the discontinuity."""
    kw = dict(marker=[(-1, -d), (1, d)], markersize=11, linestyle="none",
              color="k", mec="k", mew=1.1, clip_on=False)
    ax_hi.plot([0, 1], [0, 0], transform=ax_hi.transAxes, **kw)
    ax_lo.plot([0, 1], [1, 1], transform=ax_lo.transAxes, **kw)


def ylabel_between(ax_lo, text, x=-0.09, y=0.72, size=12):
    """One y-label for the pair, nudged up toward the break."""
    ax_lo.set_ylabel(text, fontsize=size)
    ax_lo.yaxis.set_label_coords(x, y)


# ----------------------------------------------------------------- data
data = np.loadtxt(FILE, comments="#")
data = data[np.argsort(data[:, 0])]
T = data[:, 0]

# layout is T, then Gamma_n, em_n, ab_n -- so ncols = 1 + 3*ne
ne = (data.shape[1] - 1) // 3
if 1 + 3 * ne != data.shape[1]:
    raise SystemExit(f"{FILE} has {data.shape[1]} columns, not 1 + 3*ne")
print(f"{FILE}: {len(T)} temperatures, {ne} states")
STATES = STATES[STATES <= ne]
LINES = [s for s in LINES if s <= ne]

it = [int(np.argmin(np.abs(T - t))) for t in T_SHOW]
em = [data[k, ne + STATES] for k in it]
ab = [data[k, 2 * ne + STATES] for k in it]

top_vals = np.concatenate([data[:, s] for s in LINES])
bar_vals = np.concatenate(em + ab)
tb = TOP_BREAK or auto_break(top_vals)
bb = BAR_BREAK or auto_break(bar_vals)
print(f"  temperature panel break: {tb}")
print(f"  histogram break        : {bb}")

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(10, 9))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1], hspace=0.32,
              wspace=0.22)

# ---- temperature panel -------------------------------------------------
mk = ["o-", "s-", "^-"]
col = [None, None, "crimson"]


def plot_lines(ax):
    for j, s in enumerate(LINES):
        ax.plot(T, data[:, s], mk[j % len(mk)], ms=5,
                color=col[j % len(col)], label=f"State {s} (K$_{s}$)")


if tb:
    a_hi, a_lo = broken(fig, gs[0, :], *tb)
    plot_lines(a_hi)
    plot_lines(a_lo)
    draw_break(a_hi, a_lo)
    a_lo.set_xlabel("Temperature (K)", fontsize=12)
    ylabel_between(a_lo, "Linewidth FWHM (meV)")
    a_hi.legend(loc="center left", frameon=False)
else:
    a_lo = fig.add_subplot(gs[0, :])
    plot_lines(a_lo)
    a_lo.set_xlabel("Temperature (K)", fontsize=12)
    a_lo.set_ylabel("Linewidth FWHM (meV)", fontsize=12)
    a_lo.legend(loc="center left", frameon=False)

# ---- histograms --------------------------------------------------------
w = 0.35
pairs = []
for c in (0, 1):
    if bb:
        h, l = broken(fig, gs[1, c], *bb)
    else:
        h, l = None, fig.add_subplot(gs[1, c])
    pairs.append((h, l))

for c, (h, l) in enumerate(pairs):
    for ax in (h, l):
        if ax is None:
            continue
        ax.bar(STATES - w / 2, em[c], w, label="Emission",
               color="steelblue", edgecolor="black")
        ax.bar(STATES + w / 2, ab[c], w, label="Absorption",
               color="firebrick", edgecolor="black")
        ax.set_xticks(STATES)
    if h is not None:
        draw_break(h, l)
        h.set_title(f"{T[it[c]]:.0f} K", fontsize=14)
    else:
        l.set_title(f"{T[it[c]]:.0f} K", fontsize=14)
    l.set_xlabel("Exciton State Index", fontsize=12)
    if c == 0:
        ylabel_between(l, "Scattering Rate (meV)")
        l.legend(loc="upper left", frameon=False)
    else:
        for ax in (h, l):
            if ax is not None:
                ax.tick_params(labelleft=False)

fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"wrote {OUT}")
print("\n  Bars are linear WITHIN each segment; the break marks the omitted")
print("  range. State the break in the caption -- an unremarked broken axis")
print("  is easy to misread as a continuous one.")
