# Where each old script went

## Production path — five steps, nothing else

| step | run | where |
|---|---|---|
| 1 | `pw.x`, `ph.x`, `q2r.x`, `matdyn.x` | QE, your inputs |
| 2 | `calculate_excph.py` -> `Dmats.npy` | cluster, yambopy |
| 3 | `generate_excph.py --iQ n` -> `GI_ExcPh_Q*.npz` | cluster, yambopy |
| 4 | `xphd matdyn` -> `hw_fine.npy` | xphd |
| 5 | `xphd sweep ... --unfold SAVE` -> `lw_*.npz` | xphd |

`xphd check`, `channels`, `frohlich`, `gauge`, `localization` are diagnostics
you run when a number looks wrong, not part of the path.

## Superseded — delete these

| old | replaced by |
|---|---|
| `tetra2d.py` | `xphd.tetra` (same kernel, plus an explicit-vertex entry point) |
| `fourier_refine.py` | `xphd.core.interp` (no scipy.signal) |
| `run_lw.py` | `xphd.linewidth.compute` |
| `aggregate.py` | `xphd.sweep` |
| `extract_lifetimes.py` | `xphd.sweep` (it was a one-state special case) |
| `unfold_bz.py` | `xphd.symmetry.unfold` / `xphd unfold` |
| `symmetry_test.py` | `xphd.symmetry.violation` |
| `rt-BTE.py` | `xphd.bte` |
| `lw_calc.py` | `xphd.diagnostics.channels` |

**You have two conflicting `aggregate.py` versions.** One defaults to
`--g2-mode const`, `imap_unordered`, unscaled `flat_width`; the other to
`--g2-mode complexfft`, `imap`, `flat_width/refine`. The `const` default is
the correct one — `complexfft` interpolates a randomly-phased amplitude. Any
result produced by the `complexfft` version needs regenerating.

## Closed questions — archive, do not port

These answered "where does the symmetry violation come from", and the answer
was: unfold the final field. `xphd.symmetry` does that.

`debug.py`, `decisive_test.py`, `scalar_symmetry_test.py`, `star_test.py`,
`symmetrize_fields.py`

`star_test.py` in particular concluded the star filter cannot fix `g2` and
pointed at `unfold_bz.py` — which is now the built-in behaviour.

## Cluster-only, kept standalone

`calculate_excph.py` and `generate_excph.py` need yambopy, netCDF4 and tqdm
and run where the databases are. They stay outside the package so `xphd`
itself installs with numpy alone.

## yambo cross-checks — keep, not part of the path

`compare.py`, `star_check.py`, `total_g2_check.py` (== `debug.py` in one
upload), `side.py`, `test.py` compare against yambo's own `ndb.excph_gkkp`.
Fold into `diagnostics/` if they get reused.
