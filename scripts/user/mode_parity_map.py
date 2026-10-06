import numpy as np, xphd
arc = xphd.ExcPhArchive("GI_ExcPh_Q0001.npz")
g2 = arc.grid("g2").sum(axis=(3, 4))            # (n1, n2, nmod)
lab = np.load("mode_labels.npy")
labg = np.zeros((arc.n1, arc.n2, arc.nmod), int)
labg[arc._i, arc._j] = lab
odd = np.isin(labg, [0, 3]) & (labg >= 0)
G_even = np.where(~odd, g2, 0).sum(axis=2)
G_odd  = np.where( odd, g2, 0).sum(axis=2)
np.savez("coupling_by_parity.npz", G_even=G_even, G_odd=G_odd,
         Q_red=arc.q_red)