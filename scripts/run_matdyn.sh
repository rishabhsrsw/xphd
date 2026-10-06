#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=rekhav
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --time=1:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --partition=debug
#SBATCH --exclusive

# ---------------------------------------------------------------------------
# run.sh
#
# Runs matdyn.x over a long q-list in chunks and merges the outputs.
#
# None of matdyn's three output files survives a plain `cat`:
#
#   matdyn.modes  writes "diagonalizing the dynamical matrix" before EVERY
#                 q-block, not once per file, so there is no file header to
#                 deduplicate and plain concatenation is already correct.
#                 (An earlier version stripped the first one from each chunk,
#                 which removed 53 legitimate block headers.)
#
#   bn.freq       begins with "&plot nbnd=..., nks=..." giving the TOTAL
#                 number of q-points. Concatenating leaves one namelist per
#                 chunk, each claiming only that chunk's count.
#
#   bn.freq.gp    has no header, but its first column is the CUMULATIVE path
#                 distance, which restarts at zero in every chunk. Concatenated
#                 raw it saws back to zero 54 times.
#
# Here the .modes blocks are concatenated as written, the .freq namelist is
# written once with nks summed, and the .gp distances are offset so the column
# runs monotonically through the merged file.
#
# BASE_IN must hold the namelist ONLY, ending in "/". The q-count and the
# coordinates are appended per chunk. Its flfrq and flvec must match the
# FREQ and MODES names below.
# ---------------------------------------------------------------------------

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

export I_MPI_SHM_COLL=0
export I_MPI_ADJUST_BARRIER=1

# 1. Initialize Spack
. /home/apps/spack/share/spack/setup-env.sh

# 2. Load the exact Intel toolchain
spack load /6asbh6t
spack load /r7xt55l
spack load /pmjv32t

# 3. Setup paths
MATDYN_BIN="/home/nsmext/rekhav.iiita/ondemand/data/software/q-e-qe-7.2/bin/matdyn.x"
MASTER_Q="qlist.txt"
BASE_IN="matdyn.in"
CHUNK=2400
EXPECT=129600

MODES="matdyn.modes"          # must equal flvec in BASE_IN
FREQ="bn.freq"                # must equal flfrq in BASE_IN
GP="${FREQ}.gp"               # matdyn writes this alongside FREQ

# 3a. Sanity checks on the base input, before spending any time
if ! grep -q "^[[:space:]]*/" "$BASE_IN"; then
    echo "CRITICAL: $BASE_IN has no terminating '/'. matdyn will block"
    echo "reading stdin and never finish -- this is what hung the 7 h run."
    exit 1
fi
if [ "$(awk '/^[[:space:]]*\//{f=1;next} f&&NF' "$BASE_IN" | wc -l)" -ne 0 ]; then
    echo "CRITICAL: $BASE_IN has lines after the '/'. It must hold the"
    echo "namelist only; the q-list is appended per chunk by this script."
    exit 1
fi
for key in flvec flfrq nosym q_in_band_form; do
    grep -qi "$key" "$BASE_IN" || echo "WARNING: $key not set in $BASE_IN"
done
grep -qi "flvec[[:space:]]*=[[:space:]]*'$MODES'" "$BASE_IN" || \
    echo "WARNING: flvec in $BASE_IN is not '$MODES'"
grep -qi "flfrq[[:space:]]*=[[:space:]]*'$FREQ'" "$BASE_IN" || \
    echo "WARNING: flfrq in $BASE_IN is not '$FREQ'"

# 3b. Clean up any leftover files
rm -f "$MODES" "$FREQ" "$GP" chunk_* matdyn_chunk.* clean_qlist.txt
rm -f modes.body freq.body gp.body

# 4. SANITIZE: drop blank lines, and a leading count line if qlist.txt has one
sed '/^[[:space:]]*$/d' "$MASTER_Q" > clean_qlist.txt
if [ "$(head -1 clean_qlist.txt | awk '{print NF}')" -eq 1 ]; then
    echo "dropping the count line at the top of $MASTER_Q"
    sed -i '1d' clean_qlist.txt
fi
NTOT=$(wc -l < clean_qlist.txt | xargs)
echo "$NTOT q-points in $MASTER_Q"
[ "$NTOT" -eq "$EXPECT" ] || echo "WARNING: expected $EXPECT"

# 5. Split into chunks
split -l "$CHUNK" -d -a 3 clean_qlist.txt chunk_
NCH=$(ls chunk_* | wc -l)
echo "split into $NCH chunks of $CHUNK"

# 6. Process each chunk, accumulating header-free bodies
TOTAL=0
NBND=""
GP_OFFSET=0
HAVE_GP=1
: > modes.body
: > freq.body
: > gp.body

for chunk in chunk_*; do
    NPTS=$(wc -l < "$chunk" | xargs)

    cat "$BASE_IN" > matdyn_chunk.in
    echo "" >> matdyn_chunk.in
    echo "$NPTS" >> matdyn_chunk.in
    cat "$chunk" >> matdyn_chunk.in

    "$MATDYN_BIN" < matdyn_chunk.in > matdyn_chunk.out

    if [ ! -s "$MODES" ] || [ ! -s "$FREQ" ]; then
        echo "CRITICAL: matdyn produced no output for $chunk."
        echo "Check matdyn_chunk.out for the Fortran error."
        tail -20 matdyn_chunk.out
        exit 1
    fi

    # --- matdyn.modes: every block carries its own header, so concatenate
    #     as written
    cat "$MODES" >> modes.body
    NQ_CHUNK=$(grep -c 'q *=' "$MODES")
    if [ "$NQ_CHUNK" -ne "$NPTS" ]; then
        echo "CRITICAL: $chunk gave $NQ_CHUNK q-blocks for $NPTS points"
        exit 1
    fi

    # --- bn.freq: record nbnd once, drop the &plot namelist
    if [ -z "$NBND" ]; then
        NBND=$(sed -n 's/.*nbnd *= *\([0-9]*\).*/\1/p' "$FREQ" | head -1)
    fi
    sed '1{/&plot/d}' "$FREQ" >> freq.body

    # --- bn.freq.gp: shift the cumulative distance by where the previous
    #     chunk ended, so the column is monotonic across the merged file
    if [ -s "$GP" ]; then
        awk -v off="$GP_OFFSET" '
            NF > 0 {
                printf "%14.6f", $1 + off
                for (i = 2; i <= NF; i++) printf "%12.4f", $i
                printf "\n"
            }' "$GP" >> gp.body
        GP_OFFSET=$(tail -1 gp.body | awk '{print $1}')
    else
        HAVE_GP=0
    fi

    TOTAL=$((TOTAL + NPTS))
    echo "processed $chunk ($NPTS points, running total $TOTAL)"
    rm -f "$MODES" "$FREQ" "$GP"
done

# 7. Assemble each file with a single header
mv modes.body "$MODES"

{ printf ' &plot nbnd= %4d, nks= %7d /\n' "$NBND" "$TOTAL"
  cat freq.body; } > "$FREQ"

if [ "$HAVE_GP" -eq 1 ]; then
    mv gp.body "$GP"
else
    echo "NOTE: matdyn did not write $GP; skipped"
fi

# 8. Verify
NQ_MODES=$(grep -c 'q *=' "$MODES")
NHEAD_MODES=$(grep -c 'diagonalizing' "$MODES")
NHEAD_FREQ=$(grep -c '&plot' "$FREQ")
echo ""
echo "merged $MODES : $NQ_MODES q-blocks, $NHEAD_MODES block headers"
echo "merged $FREQ  : nks=$TOTAL, nbnd=$NBND, $NHEAD_FREQ namelist"
if [ "$HAVE_GP" -eq 1 ]; then
    NGP=$(wc -l < "$GP" | xargs)
    MONO=$(awk 'NR>1 && $1 < prev {bad=1} {prev=$1} END {print bad ? "NO" : "yes"}' "$GP")
    echo "merged $GP : $NGP rows, distance monotonic: $MONO"
fi

FAIL=0
[ "$NQ_MODES" -eq "$TOTAL" ]  || { echo "MISMATCH between $MODES and $FREQ"; FAIL=1; }
[ "$NHEAD_MODES" -eq "$NQ_MODES" ] || \
    { echo "$MODES has $NHEAD_MODES block headers for $NQ_MODES q-blocks"; FAIL=1; }
[ "$NHEAD_FREQ" -eq 1 ]       || { echo "$FREQ has $NHEAD_FREQ namelists"; FAIL=1; }
[ "$TOTAL" -eq "$EXPECT" ]    || { echo "expected $EXPECT, got $TOTAL"; FAIL=1; }
if [ "$HAVE_GP" -eq 1 ]; then
    [ "$NGP" -eq "$TOTAL" ]   || { echo "$GP has $NGP rows, expected $TOTAL"; FAIL=1; }
fi

# 9. Clean up temporary files
rm -f chunk_* matdyn_chunk.* clean_qlist.txt modes.body freq.body gp.body

if [ "$FAIL" -eq 0 ]; then
    echo ""
    echo "Success: $NCH chunks merged into $MODES, $FREQ and $GP"
    echo "size of $MODES: $(du -h "$MODES" | cut -f1)"
else
    echo ""
    echo "Merged with errors -- do not use these files until resolved"
    exit 1
fi
