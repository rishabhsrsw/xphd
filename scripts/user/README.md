# Your figure scripts

The plotting scripts you wrote or shared during the GaN / hBN / WSe2 work, kept
**exactly as you provided them** (not restyled). The package scripts in
`scripts/` are the maintained counterparts; where one supersedes a script here,
it is named below.

| Script | Origin | What it plots | Notes |
|---|---|---|---|
| `Fig_2.py` | your Fig. 2 script, as last revised in our sessions | Fig. 2: GW band structure coloured by sigma_h parity, with sigma / pi orbital labels | needs `band_parity.npz`; GW_FIRST_BAND = 6; the ypp file in **reduced** k-coordinates |
| `bte_final.py` | uploaded | the transport figure: populations on the exciton dispersion, scattering events (emission / absorption by branch), F(E) | argparse: `python bte_final.py snaps.npz --times 0 500 10000 --panel-times 10 100` |
| `bte_snapshots.py` | uploaded | exciton populations F_nQ(t) on the exciton band structure | reads `xphd bte --snapshots` output |
| `bte_3d.py` | uploaded | stacked 3D/2D snapshots of F(E) at chosen times | argparse: `--times 30 200 400 --gamma 0.35` |
| `bte_animate.py` | uploaded (your original) | animation of the populations on the band structure | the package `scripts/bte_animate.py` is a later variant |
| `pa_plot.py` | uploaded | phonon-assisted radiative rate against temperature | |
| `Paleari.py` | uploaded | exciton weights over the zone (Paleari-style weight extraction), RBF-interpolated and tiled | uses yambopy |
| `Q_point_heatmap.py` | uploaded | exciton-phonon coupling over q, tiled extended zone | uses yambopy; superseded by `scripts/plot_coupling_maps.py` |
| `exph_heatmap.py` | uploaded | exciton-phonon coupling heatmap, extended-zone scheme | uses yambopy |
| `mode_parity_plot.py` | project file | coupling at Q = 0 split by the parity of the phonon | uses the planar ZA..LO labels; superseded by `scripts/plot_coupling_maps.py` with `OVERLAY = "both"` |
| `parity_plot.py` | project file | sigma_h parity across the zone, excitons and phonons | planar labels; superseded by `scripts/plot_parity_bz.py` (which takes `PERM`) |
| `selection_rule_test.py` | project file | the selection rule tested directly, with maps | earlier form of `xphd selection-rule` |
| `mode_parity_map.py`, `plot_test.py` | project files | helpers: build `coupling_by_parity.npz` and print its statistics | inputs to `mode_parity_plot.py`; planar labels |

Scripts that read phonon labels use the old `mode_labels.npy` (ZA TA LA ZO TO
LO, planar two-atom layers only). For current work take the labels from
`scripts/labels_from_elph.py`, which is exact for planar and non-planar layers.
