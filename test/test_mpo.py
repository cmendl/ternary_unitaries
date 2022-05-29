import unittest
import numpy as np
import ternary_unitary_networks as tun


class TestMPO(unittest.TestCase):

    def test_split_merge_tensor(self):
        """
        Test splitting and merging of two MPO tensors.
        """
        # physical local dimensions
        d = [4, 5]
        # virtual bond dimensions
        D = [7, 11]
        # create a random MPO tensor
        A = tun.crandn([d[0]*d[1], d[0]*d[1]] + D)
        for svd_distr in ["left", "right", "sqrt"]:
            A0, A1 = tun.split_mpo_tensor(A, d, svd_distr)
            self.assertTrue(np.allclose(tun.merge_mpo_tensor_pair(A0, A1), A))

    def test_identity(self):
        """
        Test MPO representation of the identity map.
        """
        # dimensions
        d = 3
        L = 6
        # construct MPO representation of the identity map
        idop = tun.MPO.identity(d, L)
        self.assertEqual(np.linalg.norm(idop.as_matrix() - np.identity(d**L)), 0.,
            msg="MPO representation of identity")

    def test_canonicalize(self):
        # physical (non-uniform) dimensions
        d = [2, 5, 4, 2, 2, 3]
        D = [2, 6, 5, 3, 7, 4, 2]
        # number of sites
        L = len(d)
        for mode in ["left", "right"]:
            # MPO with random tensors
            mpo = tun.MPO([0.5*tun.crandn((d[i], d[i], D[i], D[i+1])) for i in range(L)])
            Aref = mpo.as_matrix()
            mpo.canonicalize(mode)
            # compare
            self.assertTrue(np.allclose(mpo.as_matrix(), Aref))

    def test_apply_operator(self):
        """
        Test application of a two-site operator.
        """
        # physical (non-uniform) dimensions
        d = [2, 5, 3, 2, 2, 3]
        D = [2, 6, 5, 7, 3, 4, 2]
        # number of sites
        L = len(d)
        # MPO with random tensors
        mpo = tun.MPO([0.5*tun.crandn((d[i], d[i], D[i], D[i+1])) for i in range(L)])
        A = mpo.as_matrix()
        # operator is applied to sites `iloc` and `iloc + 1`
        iloc = 2
        # combined local dimensions
        dpair = d[iloc] * d[iloc+1]
        # fictitious operator
        op = tun.crandn((dpair**2, dpair**2))
        mpo.apply_two_site_operator(iloc, op)
        # reference calculation
        idx_op  = list(range(8))
        idx_mpo = list(range(8, 8 + 2*L))
        idx_mpo[iloc]     = idx_op[4]
        idx_mpo[iloc+1]   = idx_op[5]
        idx_mpo[L+iloc]   = idx_op[6]
        idx_mpo[L+iloc+1] = idx_op[7]
        idx_out = list(range(8, 8 + 2*L))   # keep most of `A`
        idx_out[iloc]     = idx_op[0]
        idx_out[iloc+1]   = idx_op[1]
        idx_out[L+iloc]   = idx_op[2]
        idx_out[L+iloc+1] = idx_op[3]
        opAref = np.einsum(op.reshape(4 * d[iloc:iloc+2]), idx_op,
                            A.reshape(d + d), idx_mpo,
                            idx_out)
        # compare
        self.assertTrue(np.allclose(mpo.as_matrix(), opAref.reshape(2*[np.prod(d)])))

    def test_matmul(self):
        """
        Test multiplication of two MPOs.
        """
        # physical (non-uniform) dimensions
        d = [[4, 2, 5, 1, 2, 3],
             [5, 2, 4, 3, 1, 2],
             [2, 7, 1, 3, 2, 4]]
        D = [[2, 6, 5, 7, 3, 4, 2],
             [4, 7, 6, 5, 2, 3, 4]]
        # number of sites
        L = len(d[0])
        # MPOs with random tensors
        mpo = [tun.MPO([0.5*tun.crandn((d[j][i], d[j+1][i], D[j][i], D[j][i+1])) for i in range(L)]) for j in range(2)]
        self.assertTrue(np.allclose((mpo[0] @ mpo[1]).canonicalize("left").as_matrix(),
                                    mpo[0].as_matrix() @ mpo[1].as_matrix()))


if __name__ == "__main__":
    unittest.main()
