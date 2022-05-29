import numpy as np


def retained_bond_indices(s, tol):
    """
    Indices of retained singular values based on given tolerance.
    """
    w = np.linalg.norm(s)
    if w == 0:
        return np.array([], dtype=int)
    # normalized squares
    s = (s / w)**2
    # accumulate values from smallest to largest
    sort_idx = np.argsort(s)
    s[sort_idx] = np.cumsum(s[sort_idx])
    return np.where(s > tol)[0]


def split_matrix_svd(a, tol):
    """
    Split a matrix by singular value decomposition,
    and truncate small singular values based on tolerance.
    """
    assert a.ndim == 2
    u, s, v = np.linalg.svd(a, full_matrices=False)
    # truncate small singular values
    idx = retained_bond_indices(s, tol)
    u = u[:, idx]
    v = v[idx, :]
    s = s[idx]
    return u, s, v


def dual_transpose(u, d=2):
    """
    Dual transposition of a quantum gate.
    """
    u = np.reshape(u, (d, d, d, d))
    u = np.transpose(u, (0, 2, 1, 3))
    u = np.reshape(u, (d**2, d**2))
    return u


def decompose_two_particle_gate(u, d=2):
    """
    Decompose a two-particle quantum gate using the singular value decomposition.
    """
    u = dual_transpose(u, d)
    v, s, w = split_matrix_svd(u, 1e-14)    # select the analytically non-zero singular values
    w = w.T
    v = np.reshape(v, (d, d, -1))
    w = np.reshape(w, (d, d, -1))
    return v, s, w


def reassemble_two_particle_gate(v, s, w):
    """
    Reassemble a two-particle quantum gate from its decomposed version.
    """
    return np.sum([s[j] * np.kron(v[:, :, j], w[:, :, j]) for j in range(len(s))], axis=0)
