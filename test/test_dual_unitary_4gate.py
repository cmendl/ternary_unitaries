import unittest
import numpy as np
import ternary_unitary_networks as tun


class TestDualUnitary4Gate(unittest.TestCase):

    def test_dual_unitary_4gate(self):
        """
        Test required dual unitary gate properties.
        """
        d = 2
        # 4-gate
        r = tun.random_dual_4gate()
        s = tun.four_gate_transpose(r, 0, d)
        t = tun.four_gate_transpose(r, 1, d)
        # test whether 'r' is indeed a dual unitary
        self.assertTrue(np.allclose(r @ r.conj().T, np.identity(d**4)))
        self.assertTrue(np.allclose(s @ s.conj().T, np.identity(d**4)))
        # not unitary in y-direction
        self.assertFalse(np.allclose(t @ t.conj().T, np.identity(d**4)))


if __name__ == "__main__":
    unittest.main()
