#!/usr/bin/env bash
# Everything AFTER Quantum ESPRESSO and the generator, for one source mesh.
#   ./run_one_mesh.sh 12 gan_12x12_360.freq
set -euo pipefail
N=${1:?usage: run_one_mesh.sh N <freq-file>}
FREQ=${2:?}
FINE=360; NBND=6; SAVE=../SAVE

xphd matdyn "$FREQ" --fine $FINE --nbnd $NBND \
     --npz GI_ExcPh_Q0001.npz -o hw_fine_${N}.npy

xphd check GI_ExcPh_Q0001.npz

xphd linewidth GI_ExcPh_Q0001.npz --hw-fine hw_fine_${N}.npy \
     --T 10 77 150 300 --acoustic-cut 5e-4 -o lw_gamma_${N}.npz

xphd sweep 'GI_ExcPh_Q*.npz' --hw-fine hw_fine_${N}.npy \
     --T 77 --workers 8 --unfold $SAVE -o lw_77K_${N}.npz

xphd channels GI_ExcPh_Q0001.npz --state 1 --T 10
