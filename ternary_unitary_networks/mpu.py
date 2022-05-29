"""
Matrix product unitary (MPU) operators.

References:
  - J. Ignacio Cirac, David Perez-Garcia, Norbert Schuch, Frank Verstraete
    Matrix product unitaries: structure, symmetries, and topological invariants
    J. Stat. Mech. (2017) 083105, https://doi.org/10.1088/1742-5468/aa7e55
    (arXiv:1703.09188)
  - M. Burak Şahinoğlu, Sujeet K. Shukla, Feng Bi, Xie Chen
    Matrix product representation of locality preserving unitaries
    Phys. Rev. B 98, 245122 (2018), https://doi.org/10.1103/PhysRevB.98.245122
    (arXiv:1704.01943)
"""

import numpy as np
from scipy.stats import unitary_group
from .impo import iMPO


class MPU(iMPO):
    """
    Matrix product unitary (MPU) operator class.
    """
    def __init__(self, U):
        """
        Create a matrix product unitary operator.
        """
        super().__init__(U)
        # test whether criteria for a valid MPU are satisfied,
        # and find left and right transfer states
        E = self.transfer_operator()
        # right spectrum of E
        w, v = np.linalg.eig(E)
        idx = np.argsort(w.real)
        w = w[idx]
        v = v[:, idx]
        # transfer operator must have a single eigenvalue 1, and all others must be zero
        e1 = np.zeros(self.D**2)
        e1[-1] = 1
        if not np.allclose(w, e1):
            raise RuntimeError("transfer operator must have a single eigenvalue 1, and all others must be zero")
        # right eigenvector corresponding to eigenvalue 1
        self.right_transfer_state = np.reshape(v[:, -1], (self.D, self.D))
        # left spectrum of E
        w, v = np.linalg.eig(E.T)
        idx = np.argsort(w.real)
        w = w[idx]
        v = v[:, idx]
        # left eigenvector corresponding to eigenvalue 1
        self.left_transfer_state = np.reshape(v[:, -1], (self.D, self.D))
        # normalize such that inner product (without conjugation) is 1
        self.right_transfer_state /= np.trace(self.left_transfer_state.T @ self.right_transfer_state)
        # unitary property when sandwiched between eigenstates of transfer operator,
        # see Eq. (14a) of J. Stat. Mech. (2017) 083105
        T = np.einsum(self.A,        (0, 1, 2, 3),
                      self.A.conj(), (0, 4, 5, 6),
                      self.left_transfer_state, (2, 5), self.right_transfer_state, (3, 6),
                      (1, 4))
        if not np.allclose(T, np.identity(self.d)):
            raise RuntimeError("unitary property violated")

    def is_simple(self):
        """
        Use the equivalence in Theorem 3.8 of J. Stat. Mech. (2017) 083105
        to test whether the MPU is simple.
        """
        M = [self.A.transpose((0, 2, 1, 3)).reshape((self.d*self.D, self.d*self.D)),
             self.A.transpose((0, 3, 1, 2)).reshape((self.d*self.D, self.d*self.D))]
        return np.linalg.matrix_rank(M[0]) * np.linalg.matrix_rank(M[1]) == self.d**2

    def index(self):
        """
        Index of a MPU, assuming that it is simple (can always be achieved by blocking),
        see Definition 4.1 in J. Stat. Mech. (2017) 083105.
        """
        M = [self.A.transpose((0, 2, 1, 3)).reshape((self.d*self.D, self.d*self.D)),
             self.A.transpose((0, 3, 1, 2)).reshape((self.d*self.D, self.d*self.D))]
        r = [np.linalg.matrix_rank(M[0]),
             np.linalg.matrix_rank(M[1])]
        if r[0]*r[1] != self.d**2:
            raise RuntimeError("require simple MPU for computing its index")
        return 0.5*np.log2(r[1]/r[0])


def random_simple_mpu(d: int, D0: int, D1: int):
    """
    Construct a random simple MPU with physical dimension `d` and
    virtual bond dimension `D0 * D1` using two internal unitary maps.
    The construction requires that `d` is divisible by the virtual bond dimension.

           |
         __|__
        |     |
        |  A  |
        |_____|
     \  /  |  \  /
      \/   |   \/
      /\   |   /\
     /  \__|__/  \
        |     |
        |  B  |
        |_____|
           |
           |
    """
    assert d >= 1 and D0 >= 1 and D1 >= 1
    D = D0 * D1
    assert d % D == 0, "physical dimension must be divisible by virtual bond dimension"
    c = d // D
    A = unitary_group.rvs(d).reshape((d, c, D0, D1))
    B = unitary_group.rvs(d).reshape((d, c, D1, D0))
    U = np.einsum(A, (0, 2, 3, 4), B, (1, 2, 5, 6), (0, 1, 3, 5, 6, 4)).reshape((d, d, D, D))
    return MPU(U)


def czx_mpu():
    """
    Construct the CZX MPU, see Example 7.2 in J. Stat. Mech. (2017) 083105
    """
    I = np.identity(2)
    X = np.array([[0., 1.], [1., 0.]])
    H = np.array([[1., 1.], [1.,-1.]])  # Hadamard gate without sqrt(2) factor
    U = np.array([[[[X[u, v] * I[u, i] * H[i, j] for i in range(2)]
                                                 for j in range(2)]
                                                 for u in range(2)]
                                                 for v in range(2)])
    return MPU(U)
