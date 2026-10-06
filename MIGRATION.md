# Migration from the standalone scripts

| old script | now |
|---|---|
| `wr_decay.py` | `xphd localization` / `diagnostics.localization` |
| `gauge_probe.py` | `xphd gauge` / `diagnostics.gauge` |
| `diag_lw.py` | `xphd channels` / `diagnostics.channels` |
| `frohlich_check2.py` | `xphd frohlich` / `diagnostics.frohlich` |
| `parse_matdyn.py` | `xphd matdyn` / `io.matdyn` |
| `lw_tetra.py`, `run_lw.py` | `xphd linewidth` / `linewidth.compute` |
| `rt-BTE.py` | `xphd.bte` |
| `tetra2d.py` | `xphd.tetra` (weight kernel kept verbatim) |
| `fourier_refine.py` | `core.interp` (no scipy dependency) |
| `to_grid` (x4 copies) | `core.mesh.to_grid` |
| `mesh_indices`, `shift_grid` | `core.mesh` |
| `bose`, `degeneracy_groups` | `core.stats` |

## Behaviour changes

- **`blockwise` uses nearest, not floor.** Fixes spurious low-temperature
  absorption. Linewidths below ~50 K will change.
- **`flat_width` defaults to `2.4e-3 / N_fine`**, tied to the fine mesh rather
  than the refinement ratio, so a convergence series varies only one thing.
- **The DC coefficient is dropped** in localisation transforms, so complex and
  positive-definite fields are compared on the same footing.
- **`matdyn` validation separates the Gamma acoustic deviation** from the rest.
  That deviation is the ASR correction and matdyn's zeros are the correct
  values, so it is no longer reported as a failure.

### rt-BTE specifically

- **`tetra_2d_weights` is gone.** It was a second implementation of the
  triangle weights. The ordinary branches agreed with `tetra2d` to 2.7e-16,
  but the flat branch was a factor of **two low** -- dormant at
  `flat_width=0.0`, live the moment it is switched on. Both entry points now
  share `_tri_weights`.
- **`_P_CHAN` module global removed**; channel matrices are returned on
  `RateMatrix.channels`.
- **`inject_optical` weights by oscillator strength.** The old `--inject
  gamma` gave every selected state the same occupation.
- **`refined_out_rates` uses `core.interp`**, dropping the scipy.signal
  dependency and the `fourier_refine.py` import.

## Not migrated

`generate_excph.py` stays a standalone driver — it needs yambopy, netCDF4 and
tqdm, and runs on the cluster rather than as a library call. `compare.py`,
`star_check.py`, `debug.py` and `side.py` are one-off yambo cross-checks;
fold them into `diagnostics/` if they get reused.
