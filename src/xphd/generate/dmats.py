#!/usr/bin/env python3
"""
calculate_excph.py
==================
Build Dmats.npy -- the electronic rotation matrices -- and a reference set of
exciton-phonon matrix elements from yambopy's own driver.

The band window is taken from the BSE database rather than hardcoded. The
earlier version fixed bands_range = [6, 12], which is GaN's window; run
unchanged on another material it builds rotation matrices over the wrong
bands. generate_excph.py now refuses a Dmats whose band dimension disagrees
with the BSE, but deriving the window here means the mismatch never arises.

bands_range follows yambopy's convention: 0-based, end excluded, so bands
7..12 is [6, 12].

    python calculate_excph.py --nexc 15
"""
import argparse

import numpy as np
# yambopy is imported inside main(), so `xphd dmats --help` works without it


def main(argv=None):
    p = argparse.ArgumentParser(prog="xphd dmats")
    p.add_argument("--bse-dir", default="../BSE_EXCPH/output")
    p.add_argument("--savepath", default="../ELPH/SAVE")
    p.add_argument("--ndb-elph", default="../ELPH/ndb.elph")
    p.add_argument("--nexc", type=int, default=15)
    p.add_argument("--tag", default="hBN", help="prefix for the reference file")
    a = p.parse_args(argv)
    from yambopy import YamboLatticeDB, YamboWFDB, LetzElphElectronPhononDB
    from yambopy.dbs.excitondb import YamboExcitonDB
    from yambopy.exciton_phonon.excph_matrix_elements import (
        exciton_phonon_matelem)

    lattice = YamboLatticeDB.from_db_file(filename=f"{a.savepath}/ns.db1")

    # band window from the BSE itself, not a hardcoded guess
    exc = YamboExcitonDB.from_db_file(
        lattice, filename=f"{a.bse_dir}/ndb.BS_diago_Q1", neigs=a.nexc)
    b0 = int(np.min(exc.unique_vbands))
    b1 = int(np.max(exc.unique_cbands)) + 1
    nv = len(np.unique(exc.unique_vbands))
    nc = len(np.unique(exc.unique_cbands))
    bands_range = [b0, b1]
    print(f"BSE window {bands_range} (python slice): nv={nv}, nc={nc}")
    assert b1 - b0 == nv + nc, "BSE space is not a simple nv x nc block"

    elph = LetzElphElectronPhononDB(a.ndb_elph, read_all=False)
    wfcs = YamboWFDB(filename="ns.wf", save=a.savepath, latdb=lattice,
                     bands_range=bands_range)

    exciton_phonon_matelem(lattice, elph, wfcs, BSE_dir=a.bse_dir,
                           nexc_in=a.nexc, nexc_out=a.nexc,
                           dmat_mode="save", exph_file=f"{a.tag}-ph.npy")

    D = np.load("Dmats.npy")
    print(f"Dmats.npy {D.shape}; band dimension {D.shape[-1]} "
          f"{'matches' if D.shape[-1] == nv + nc else 'DOES NOT MATCH'} "
          f"nv+nc = {nv + nc}")


if __name__ == "__main__":
    main()
