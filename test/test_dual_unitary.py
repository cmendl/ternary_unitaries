import unittest
import numpy as np
import ternary_unitary_networks as tun


class TestDualUnitary(unittest.TestCase):

    def test_dual_unitary(self):
        """
        Test dual unitary gate functions.
        """
        u = tun.random_dual_gate()
        v = tun.dual_transpose(u)
        self.assertTrue(np.allclose(u @ u.conj().T, np.identity(4)))
        self.assertTrue(np.allclose(v @ v.conj().T, np.identity(4)))

        d = 3
        u = tun.swap_gate(d)
        v = tun.dual_transpose(u, d)
        self.assertTrue(np.allclose(u @ u.conj().T, np.identity(d**2)))
        self.assertTrue(np.allclose(v @ v.conj().T, np.identity(d**2)))


if __name__ == "__main__":
    unittest.main()
