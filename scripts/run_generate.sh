#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=excph
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=gen_%j.out
#SBATCH --error=gen_%j.err
#SBATCH --exclusive
##SBATCH --partition=standard

# ---------------------------------------------------------------------------
# run_generate.sh -- every exciton-phonon archive of the full zone, several at
# a time on one node, with `xphd generate`.
#
# Before submitting: generate the Gamma archive by hand and make sure
#     xphd check-archive <OUT>/GI_ExcPh_Q0001.npz
# prints READY. This script refuses to start otherwise.
#
# generate is serial per Q, so the parallelism is several Q at once. The limit
# is MEMORY: each process loads the whole electron-phonon database, so NPAR
# processes hold NPAR copies. Measure one first:
#     /usr/bin/time -v xphd generate 0 <OUT> ...      # "Maximum resident set size"
#
# Resumable: an archive that exists and holds every required field is skipped,
# so resubmitting after a time limit continues where it stopped.
#
# Set the paths and sizes below for the material (examples):
#   GaN : N=24 NEXC=15 SAVE=../ELPH/SAVE    ELPH=../ELPH/ndb.elph   BSE=../BSE_EXCPH/output
#   hBN : N=24 NEXC=15 SAVE=../phonons/SAVE ELPH=../phonons/ndb.elph BSE=../BSE/output_all
#   WSe2: N=12 NEXC=10 SAVE=../LELPH/SAVE   ELPH=<path>/ndb.elph     BSE=../BSE/output
# ---------------------------------------------------------------------------

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

. /home/apps/spack/share/spack/setup-env.sh
spack load /6asbh6t
spack load /r7xt55l
spack load /pmjv32t
# activate the python environment that has yambopy and xphd
# source ~/miniconda3/bin/activate yambopy

N=${N:-24}                         # source mesh N x N; the zone has N*N points
NQ=$((N * N))                      # the transport needs every one of them
NPAR=${NPAR:-8}                    # concurrent processes; bounded by memory
NEXC=${NEXC:-15}
OUT=${OUT:-archives}
SAVE=${SAVE:-../ELPH/SAVE}
ELPH=${ELPH:-../ELPH/ndb.elph}
BSE=${BSE:-../BSE_EXCPH/output}
DMATS=${DMATS:-Dmats.npy}
mkdir -p "$OUT" logs

G0="$OUT/GI_ExcPh_Q0001.npz"
if [ ! -s "$G0" ] || ! xphd check-archive "$G0" | grep -q "READY: launch the rest"; then
    echo "CRITICAL: $G0 missing or not READY. Generate it by hand first:"
    echo "  xphd generate 0 $OUT --nexc $NEXC --savepath $SAVE --ndb-elph $ELPH \\"
    echo "        --bse-dir $BSE --dmats $DMATS --mesh $N $N"
    echo "  xphd check-archive $G0"
    exit 1
fi

run_one() {
    iQ=$1
    f=$(printf "%s/GI_ExcPh_Q%04d.npz" "$OUT" $((iQ + 1)))
    if [ -s "$f" ] && python -c "import numpy as np; d=np.load('$f'); \
        assert {'G_grid','Ge_grid','Gh_grid','g2_grid','mesh'} <= set(d.files)" \
        2>/dev/null; then
        echo "skip $iQ (complete)"
        return 0
    fi
    xphd generate "$iQ" "$OUT" --nexc "$NEXC" --savepath "$SAVE" --ndb-elph "$ELPH" \
        --bse-dir "$BSE" --dmats "$DMATS" --mesh "$N" "$N" \
        > "logs/gen_$(printf %04d $((iQ+1))).log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && echo "done $iQ" || echo "FAILED $iQ (rc=$rc) -- see logs/"
}
export -f run_one
export OUT NEXC SAVE ELPH BSE DMATS N

echo "generating $NQ archives ($N x $N), $NPAR at a time, starting $(date)"
seq 1 $((NQ - 1)) | xargs -P "$NPAR" -I{} bash -c 'run_one {}'
echo "finished $(date)"

xphd verify-archives --dir "$OUT" --n "$NQ"
