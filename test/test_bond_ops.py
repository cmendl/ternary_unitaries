import unittest
import numpy as np
from scipy.stats import unitary_group
import ternary_unitary_networks as tun


class TestBondOps(unittest.TestCase):

    def test_bond_ops(self):
        """
        Test bond operations.
        """
        d = 3
        u = unitary_group.rvs(d**2)
        v, s, w = tun.decompose_two_particle_gate(u, d)
        # compare
        self.assertTrue(np.allclose(tun.reassemble_two_particle_gate(v, s, w), u))


if __name__ == "__main__":
    unittest.main()
