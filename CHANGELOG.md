# Changelog

## 1.0.0 -- 4 October 2026

### Added
- Start-up banner for every command (`xphd.banner`), with the authors and affiliation;
  `XPHD_NO_BANNER=1` disables it.
- Spin-orbit coupling: spinor band labels from the +-i eigenvalues of D(sigma_h)
  (`xphd band-parity` reports spinor/spinless); `xphd.mirror` (sigma_h atom
  permutation, exact phonon parity for planar and non-planar layers).
- `xphd parity --manifold-tol` and `xphd selection-rule --rotate`
  (`xphd.manifold`): selection rule split by the parity eigenstates of
  near-degenerate manifolds, with time-reversed points conjugated.
- `xphd check-archive`: SYMMETRY section (time reversal and C3, manifold sums;
  OK < 1e-2, FAIL > 5e-2), `--detail`, `--deg-tol` (default 15 meV).
- Scripts: `plot_coupling_maps.py`, `where_forbidden.py`, `manifold_closure.py`,
  `window_closure.py`, `add_osc.py`, `exc_transitions.py`, `qp_decompose.py`,
  `soc_probe.py`, `helicity_pl.py`, `crossing_radii.py`, `labels_from_elph.py`,
  `phonon_irreps_K.py` (phonon C3h labels at K), `slice_archive.py` (final-state convergence), `star_uniformity.py` (T_max/T_min per star),
  `irreps_check.py` (diagnoses an irreps unitarity warning).
- `docs/`: the handbook (Parts I and II, references, errata) and the LaTeX
  sources of Part II. `scripts/user/`: the user's own figure scripts.

### Changed
- `cb_splitting.py` rewritten: columns by name; band edges; direct/indirect gap;
  Q-K offset; conduction splitting and its spin.
- `qp_decompose.py` rewritten: columns by name; exchange from an HF run; checks
  the GW file's Sc column against it.
- `check_bte.py`, `valley_times.py`: `CENTERS` (Gamma, K, K'); drift and
  per-valley comparison with Boltzmann.
- `plot_phonon_parity.py`, `plot_parity_bz.py`: `PERM` for non-planar layers.
- `run_generate.sh` / `.ps1`: call `xphd generate` with explicit paths; refuse to
  start without a READY Gamma archive; resumable.
- `xphd selection-rule`: refuses a parity file with too few states; branch names
  from parity-only labels; NumPy 2 fix in `--by-final`.
- `xphd dipoles`: reads the full ndb.dipoles and selects the window (yambopy
  mis-counts DipBandsAll).

### Fixed / clarified
- `xphd mode-labels` refuses non-planar layers (its out-of-plane parity is
  wrong there) and points to `labels_from_elph.py`.
- Documentation: no `KfnQPdb` fault exists (the earlier diagnosis came from
  reading `o-*.qp` by position). RIM-W broke the GW and the BSE for WSe2 only;
  it worked for GaN and hBN (cause not isolated).
