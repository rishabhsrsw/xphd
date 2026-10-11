#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --job-name=xphd-info
#SBATCH --output=hpc_info_%j.out
##SBATCH --partition=debug

# ---------------------------------------------------------------------------
# slurm_hpc_info.sh -- a five-minute check before the first production job.
#
# Expected output: "MPI  8 ranks", two node lines with 4 rank(s) each, cores
# per rank [2], and the BLAS library with 2 threads. If it says
# "MPI not active", mpi4py was not built against this MPI (install_cluster.sh)
# or srun is missing --mpi=pmix / pmi2.
# ---------------------------------------------------------------------------
# load the compilers and MPI your QE/yambo were built with, e.g.
#   module load gcc openmpi        (or: spack load ...)
source ~/xphd-env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --mpi=pmix xphd hpc-info --threads "$SLURM_CPUS_PER_TASK"
# Intel MPI alternative:  mpirun -np $SLURM_NTASKS xphd hpc-info
