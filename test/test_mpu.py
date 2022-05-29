import unittest
import numpy as np
import ternary_unitary_networks as tun


class TestMPU(unittest.TestCase):

    def test_identity(self):
        """
        Test MPU representation of the identity map.
        """
        # dimensions
        d = 3
        L = 6
        # construct MPU representation of the identity map
        idop = tun.MPU.identity(d)
        self.assertEqual(np.linalg.norm(idop.as_matrix(L) - np.identity(d**L)), 0.,
            msg="MPU representation of identity")

    def test_random_simple_mpu(self):
        """
        Test creation of a random simple MPU.
        """
        d = 30
        D0 = 3
        D1 = 5
        L = 2
        mpu = tun.random_simple_mpu(d, D0, D1)
        U = mpu.as_matrix(L)
        self.assertTrue(np.allclose(U.conj().T @ U, np.identity(d**L)))
        self.assertTrue(mpu.is_simple())
        # without factor 1/2 since the bonds D0 and D1 are cut twice
        self.assertTrue(np.allclose(mpu.index(), np.log2(D0 / D1)))

    def test_czx_mpu(self):
        """
        Test construction of the CZX MPU.
        """
        mpu = tun.czx_mpu()
        self.assertTrue(mpu.is_simple())
        L = 5
        U = mpu.as_matrix(L)
        self.assertTrue(np.allclose(U.conj().T @ U, np.identity(mpu.d**L)))
        self.assertTrue(np.allclose(mpu.index(), 0))

    def test_time_evolution(self):
        """
        Test properties of a unitary MPO representing the time evolution
        governed by an Ising-type Hamiltonian.
        """
        # Hamiltonian parameters
        J = 1.2
        h = 0.3
        L = 7
        # reference Hamiltonian
        Href = construct_ising_diagonal(J, h, L)
        expiHt = np.exp(-1j*Href)
        # construct MPU
        Z = np.diag([1., -1.])
        Rzh = np.diag([np.exp(1j*h), np.exp(-1j*h)])
        mpu = tun.MPU(np.array([[np.cos(J)*Rzh, 1j*np.sin(J)*(Z @ Rzh)],
                                [np.cos(J)*(Z @ Rzh), 1j*np.sin(J)*Rzh]]).transpose((2, 3, 0, 1)))
        self.assertTrue(np.allclose(mpu.index(), 0))
        self.assertTrue(np.allclose(mpu.as_matrix(L), np.diag(expiHt)))


def construct_ising_diagonal(J, h, L: int):
    """
    Construct diagonal matrix entries of the Ising-type Hamiltonian -J Z Z - h Z
    on a one-dimensional lattice of length `L` with periodic boundary conditions.
    """
    H = np.zeros(2**L, dtype=float)
    for i in range(L - 1):
        H -= J * np.kron(np.ones(2**i),
                 np.kron([1., -1.],
                 np.kron([1., -1.],
                         np.ones(2**(L-i-2)))))
    H -= J *np.kron([1., -1.],
            np.kron(np.ones(2**(L-2)),
                    [1., -1.]))
    # external field
    for i in range(L):
        H -= h * np.kron(np.ones(2**i),
                 np.kron([1., -1.],
                         np.ones(2**(L-i-1))))
    return H


if __name__ == "__main__":
    unittest.main()
