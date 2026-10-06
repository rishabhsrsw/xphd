"""
Mirror (sigma_h) parity for layers that are not planar, and for spinors.

For a planar layer every atom lies in the mirror plane, and the parity of a
phonon is 1 - 2 w_z, with w_z its out-of-plane weight. In a layer such as
1H-WSe2 the mirror plane passes through W and exchanges the two Se atoms, so
parity must come from how sigma_h permutes the atoms:

    p(nu, q) = sum_kappa < e_{pi(kappa)} | M e_kappa >,   M = diag(1, 1, -1),

for normalized eigenvectors. The mirror maps each atom onto another at the
same in-plane position, so no Bloch phase enters. For a planar layer pi is the
identity and p reduces exactly to 1 - 2 w_z.

For spinors sigma_h squares to -1 (a C2 rotation times inversion acting on
spin as well), so a band's sigma_h eigenvalue is +i or -i rather than +1 or
-1. band_labels returns +-1 labels in both cases: the real part of the
eigenvalue for spinless bands, the imaginary part for spinors. With those
labels an exciton's parity is label_c * label_v in both cases, because
lambda_c * conj(lambda_v) = (i l_c)(-i l_v) = l_c * l_v.
"""
import numpy as np

__all__ = ["mirror_plane_z", "sigma_h_permutation", "phonon_parity",
           "band_labels"]


def mirror_plane_z(pos_car):
    """z of the mirror plane: halfway between the highest and lowest atom."""
    z = np.asarray(pos_car, float)[:, 2]
    return 0.5 * (z.max() + z.min())


def sigma_h_permutation(pos_car, species, cell, tol=1e-3):
    """pi[kappa] = the atom that sigma_h carries atom kappa onto.

    pos_car : (nat, 3) Cartesian positions
    species : (nat,) anything comparable (atomic numbers, symbols)
    cell    : (3, 3) lattice vectors as rows, same units as pos_car
    Raises if some atom has no image: then the layer has no sigma_h.
    """
    pos = np.asarray(pos_car, float)
    cell = np.asarray(cell, float)
    z0 = mirror_plane_z(pos)
    img = pos.copy()
    img[:, 2] = 2 * z0 - pos[:, 2]
    inv = np.linalg.inv(cell)
    perm = np.full(len(pos), -1)
    for k in range(len(pos)):
        for m in range(len(pos)):
            if species[m] != species[k]:
                continue
            d = (img[k] - pos[m]) @ inv          # difference in lattice units
            d[:2] -= np.rint(d[:2])              # in-plane lattice translations
            if np.linalg.norm(d @ cell) < tol:
                perm[k] = m
                break
        if perm[k] < 0:
            raise ValueError(f"atom {k} has no mirror image: the layer has no "
                             "horizontal mirror plane")
    if len(np.unique(perm)) != len(perm):
        raise ValueError("sigma_h does not map the atoms one to one")
    return perm


def phonon_parity(evec, perm):
    """sigma_h parity of phonon eigenvectors.

    evec : (..., nat, 3) complex eigenvectors (any normalization; mass
           scaling does not matter, since sigma_h maps each atom onto one of
           the same species)
    perm : (nat,) from sigma_h_permutation
    Returns (...,) real values: +1 even, -1 odd; anything else means the mode
    is a mixture (a degenerate pair of opposite parity, or a bad input).
    """
    e = np.asarray(evec)
    Me = e * np.array([1.0, 1.0, -1.0])
    num = np.sum(np.conj(e[..., perm, :]) * Me, axis=(-2, -1))
    den = np.sum(np.abs(e) ** 2, axis=(-2, -1))
    return np.real(num / den)


def band_labels(diag, tol=0.1):
    """+-1 labels from the diagonal of D(sigma_h), and which kind of bands.

    Returns (labels, kind, defined): kind is "spinless" when the diagonal
    elements lie near +-1, "spinor" when they lie near +-i. defined marks
    elements within tol of the expected unit value.
    """
    d = np.asarray(diag)
    near_real = np.abs(np.abs(d.real) - 1) < tol
    near_imag = np.abs(np.abs(d.imag) - 1) < tol
    if near_imag.sum() > near_real.sum():
        return np.sign(d.imag), "spinor", near_imag
    return np.sign(d.real), "spinless", near_real
