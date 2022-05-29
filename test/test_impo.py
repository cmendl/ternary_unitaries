import unittest
import numpy as np
import ternary_unitary_networks as tun


class TestiMPO(unittest.TestCase):

    def test_identity(self):
        """
        Test iMPO representation of the identity map.
        """
        # dimensions
        d = 3
        L = 6
        # construct iMPO representation of the identity map
        idop = tun.iMPO.identity(d)
        self.assertEqual(np.linalg.norm(idop.as_matrix(L) - np.identity(d**L)), 0.,
            msg="iMPO representation of identity")


if __name__ == "__main__":
    unittest.main()
