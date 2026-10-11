# Running xphd on a cluster

xphd 1.1 runs unchanged on a Windows workstation, a Linux node, or several
hundred cores under MPI. Nothing about the numbers depends on which: every
parallel path returns its results in task order, and the test suite checks
that serial, process-pool and MPI runs write identical arrays.

## The parallel layout, stage by stage

| stage | what is independent | how it is spread | per-process memory |
|---|---|---|---|
| `generate` | every Q | MPI ranks over nodes; BLAS threads inside each rank | one copy of the elph + BSE databases |
| `sweep` | every Q | MPI ranks (one per core) or `--workers` on one node | one archive + the fine mesh (small) |
| `bte` step 1: out-rate refinement | every Q | MPI ranks or `--workers` | chunked to `XPHD_TASK_MEM_GB` |
| `bte` step 2: assembly + propagation | nothing — one dense ODE | one process, all cores as BLAS/OpenMP threads | (states)² × ~10 doubles |
| everything else | — | serial; under `mpirun` it runs on rank 0 only | — |

Two kinds of parallelism, used where each fits:

- **MPI ranks / processes** for the Q loops. A Q-point needs nothing from
  any other Q until the end, so there is no communication and the scaling
  is close to ideal until there are fewer Q per rank than ~2.
- **BLAS/OpenMP threads** for dense linear algebra: the matrix–vector
  products and LU factorisations of the BTE integrator, and the tensor
  contractions inside yambopy during `generate`.

`xphd` never starts MPI on its own. It looks for the variables `mpirun` and
`srun` set; without them nothing MPI-related is imported. If the job *was*
launched with several ranks but `mpi4py` is missing, it stops with an error
rather than letting every rank run the whole job.

## What else changed for speed (same numbers)

| change | effect | check |
|---|---|---|
| all temperatures in one pass (`sweep`, `linewidth`) | the delta-function weights do not depend on T, so they are computed once; cost no longer grows with the number of temperatures | bit-identical, `test_all_temperatures_in_one_pass_is_bit_identical` |
| `build_rates`: `np.bincount` in place of `np.add.at` | 2.2× faster assembly at 864 states; little at 3456, where the triangle weights dominate | bit-identical, `test_bincount_assembly_is_bit_identical_to_add_at` |
| analytic Jacobian for the BTE (`--jac analytic`, default) | O(n²) instead of O(n³) per Jacobian | same trajectory within `--rtol`; `--jac fd` restores the old solver exactly |
| out-rate refinement in chunks (`--mem-per-task`) | bounded memory per task, so more tasks fit on a node | relative differences ~1e-15 |
| `generate` loads the databases once per process | no reload per Q | archives byte-identical, `test_generate_many_q_equals_one_at_a_time` |

Measured on synthetic data in a 2-core container, one core unless stated (12×12 source mesh,
6 modes, 6 states, refinement ×5, 4 temperatures; 24×24 for the propagation):

| step | 1.0.0 | 1.1.0 | 1.1.0, 2 cores |
|---|---|---|---|
| sweep, 144 Q × 4 T | 330 s | 89 s | 41 s |
| out-rate refinement, 144 Q | 89 s | 90 s | 44 s |
| `build_rates`, 3456 states | 47 s | 44 s | — |
| propagation, 3456 states (2 BLAS threads) | 97 s | 29 s | — |

The Q loops then scale with the number of ranks: 576 Q on 96 ranks is six
per rank.

## Install, once per cluster

`scripts/hpc/install_cluster.sh` does this; the essential lines are

```bash
# the MPI that srun/mpirun will launch -- same toolchain as QE/yambo
spack load ...            # or: module load gcc openmpi
python3 -m venv ~/xphd-env && source ~/xphd-env/bin/activate
MPICC=$(which mpicc) pip install --no-binary=mpi4py mpi4py    # FROM SOURCE
pip install -e "./xphd[hpc,generate]"
mpirun -np 4 xphd hpc-info
```

Build `mpi4py` from source against the cluster's MPI. The PyPI wheel brings
its own MPI and on a cluster either fails at start-up or runs N unconnected
copies. With Intel MPI use `MPICC=mpiicc` (or `mpiicx`).

`xphd hpc-info` prints what xphd sees: the number of ranks, the nodes and
ranks per node, cores per rank, the BLAS library and its thread count, and
the node memory. Run `scripts/hpc/slurm_hpc_info.sh` once before the first
production job.

## The production path on SLURM, in order

```bash
# 0. once
bash scripts/hpc/install_cluster.sh
sbatch scripts/hpc/slurm_hpc_info.sh            # read hpc_info_<id>.out

# 1. phonons on the fine mesh (unchanged)
xphd matdyn-split --n 360 --chunks 48
sbatch scripts/run_matdyn.sh                    # runs and merges the chunks
xphd matdyn bn.freq --fine 360 --nbnd 6 --npz archives/GI_ExcPh_Q0001.npz -o hw_fine.npy

# 2. archives: Gamma by hand, then the zone over MPI
xphd dmats --nexc 15
xphd generate 0 archives --nexc 15 --savepath ../ELPH/SAVE \
     --ndb-elph ../ELPH/ndb.elph --bse-dir ../BSE_EXCPH/output --dmats Dmats.npy
xphd check-archive archives/GI_ExcPh_Q0001.npz            # must print READY
/usr/bin/time -v xphd generate 1 test ...                 # memory of one rank
sbatch scripts/hpc/slurm_generate.sh                      # resubmit until complete
xphd verify-archives --dir archives --n 576

# 3. linewidths at every Q, all temperatures in one pass
sbatch scripts/hpc/slurm_sweep.sh                         # -> lw_field.npz

# 4. transport: MPI refinement, then one threaded node
sbatch scripts/hpc/slurm_bte.sh                           # -> snaps_300K.npz

# 5. figures, on the workstation (unchanged)
```

Every step from 2 on is resumable: complete archives are skipped, and
`--checkpoint DIR` keeps each finished Q of the sweep and the refinement.
After a wall-clock stop, resubmit the same script.

## Sizing a job

**generate.** Memory sets the ranks per node, because each rank holds the
databases. Measure one Q (`/usr/bin/time -v`, "Maximum resident set size"),
then

    ranks per node = floor(0.85 × node memory / that size)
    --cpus-per-task = cores per node / ranks per node     (BLAS threads)

**sweep.** One rank per core, `OMP_NUM_THREADS=1`. More ranks than Q-points
leaves ranks idle; there is no gain beyond one Q per rank.

**bte, step 1.** Ranks as for the sweep; `XPHD_TASK_MEM_GB` (or
`--mem-per-task`) bounds each task. `slurm_bte.sh` sets it to 80% of node
memory divided by the ranks per node.

**bte, step 2.** One process with every core of a node as threads. The
dense rate matrices take about 10 × (states)² × 8 bytes: 3456 states is
~1 GB, 7776 states (36×36 × 6) is ~5 GB; `--channels` adds 2 × modes ×
(states)² × 4 bytes. `--matrix sparse` helps only when most state pairs are
uncoupled — the run prints the fraction of non-zero entries (32% for the
synthetic 24×24 set, where dense is right).

## Without MPI

Everything also runs on one machine with processes:

```bash
xphd sweep 'archives/GI_ExcPh_Q*.npz' ... --workers 16
xphd bte   'archives/GI_ExcPh_Q*.npz' ... --workers 16 --threads 16
xphd generate 0-575 archives --resume ...     # serial, databases loaded once
```

On Windows this is the only mode, exactly as in 1.0. `--workers` and MPI
can be combined (each rank starts its own processes), but on InfiniBand
clusters forking inside an MPI rank can upset the interconnect — prefer one
rank per core.

## Environment variables

| variable | meaning |
|---|---|
| `OMP_NUM_THREADS` (and `MKL_`, `OPENBLAS_`) | BLAS threads per process; `--threads`/`--blas-threads` override at run time |
| `XPHD_TASK_MEM_GB` | memory bound per refinement task (default 2) |
| `XPHD_MPI=0` / `1` | disable / force MPI detection |
| `XPHD_ALL_RANKS=1` | let every rank print, for debugging |
| `XPHD_MP_START` | start method for `--workers` (default: the platform's) |
| `XPHD_NO_BANNER=1` | no start-up banner |

## When something looks wrong

| symptom | cause |
|---|---|
| `hpc-info` says "MPI not active" under srun | mpi4py not built against this MPI, or srun without `--mpi=pmix`/`pmi2` |
| "launched with N MPI ranks but mpi4py cannot be imported" | the environment was not activated inside the job script |
| every rank prints the banner | stdout of ranks > 0 is silenced only when MPI is active — same cause as above |
| step 2 of the BTE is slow, `hpc-info` shows 1 thread | the step was bound to one core: use `--cpu-bind=none` (srun) or `--bind-to none` (mpirun) |
| `outrates.npz was computed with different settings` | step 2 got another `--T`, `--hw-fine`, scheme or archive set than step 1 — as intended |
| a sweep or refinement task fails | its traceback is printed once by rank 0 and every rank stops; fix and resubmit — finished Q come back from the checkpoint |
