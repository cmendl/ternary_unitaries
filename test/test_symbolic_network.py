import unittest
import numpy as np
import ternary_unitary_networks as tun


class TestSymbolicNetwork(unittest.TestCase):

    def test_gate_cancellation_loop(self):
        """
        Test ternary unitary gate cancellation along time direction.
        """
        net = tun.SymbolicNetwork()
        net.add_tensor(tun.SymbolicTernaryUnitaryGate(17, 42, [0, 0, 0]))
        net.add_tensor(tun.SymbolicTernaryUnitaryGate(42, 17, [0, 0, 1]))
        for i in range(8):
            net.add_bond(tun.SymbolicBond(17, 42, i, i))
        self.assertTrue(net.is_consistent(verbose=True))
        self.assertListEqual(net.open_axes(), [])
        net.simplify()
        # network must be empty now
        self.assertListEqual(net.tensors, [])
        self.assertListEqual(net.bonds, [])
        self.assertEqual(net.identity_loops, 4)

    def test_gate_cancellation_connector(self):
        """
        Test ternary unitary gate cancellation along y-direction with connectors.
        """
        net = tun.SymbolicNetwork()
        net.add_tensor(tun.SymbolicTernaryUnitaryGate(17, 42, [0, 0, 0]))
        net.add_tensor(tun.SymbolicTernaryUnitaryGate(42, 17, [0, 0, 1]))
        for i in range(4):
            net.add_tensor(tun.SymbolicConnector(50 + i, [0.5, 0.5, 0]))
        for i in range(4):
            net.add_tensor(tun.SymbolicConnector(60 + i, [0.5, 0.5, 1]))
        # bonds between tensors
        net.add_bond(tun.SymbolicBond(17, 42, 2, 2))
        net.add_bond(tun.SymbolicBond(17, 42, 3, 3))
        net.add_bond(tun.SymbolicBond(17, 42, 6, 6))
        net.add_bond(tun.SymbolicBond(17, 42, 7, 7))
        # connectors to first gate
        net.add_bond(tun.SymbolicBond(17, 50, 0, 0))
        net.add_bond(tun.SymbolicBond(17, 51, 1, 0))
        net.add_bond(tun.SymbolicBond(17, 52, 4, 0))
        net.add_bond(tun.SymbolicBond(17, 53, 5, 0))
        # connectors to second gate
        net.add_bond(tun.SymbolicBond(42, 60, 0, 0))
        net.add_bond(tun.SymbolicBond(42, 61, 1, 0))
        net.add_bond(tun.SymbolicBond(42, 62, 4, 0))
        net.add_bond(tun.SymbolicBond(42, 63, 5, 0))
        self.assertTrue(net.is_consistent(verbose=True))
        self.assertListEqual(net.open_axes(), [])
        # simplify network
        net.simplify()
        # ternary unitary gates must have cancelled,
        # so the only remaining tensors are the "connectors"
        self.assertListEqual([t.deg for t in net.tensors], 8*[1])
        # still no open axes allowed
        self.assertListEqual(net.open_axes(), [])
        self.assertEqual(len(net.bonds), 4)
        self.assertEqual(net.identity_loops, 0)

    def test_contractions(self):
        """
        Test contractions, including the contraction tree.
        """
        # create some tensors
        t17 = tun.crandn((3, 4))
        t05 = tun.crandn((1, 7, 4, 6))
        t42 = tun.crandn((7,))
        t60 = tun.crandn((5, 1, 8, 3))
        # setup network
        net = tun.SymbolicNetwork()
        net.add_tensor(tun.SymbolicTensor(17, t17.ndim))
        net.add_tensor(tun.SymbolicTensor( 5, t05.ndim))
        net.add_tensor(tun.SymbolicTensor(42, t42.ndim))
        net.add_tensor(tun.SymbolicTensor(60, t60.ndim))
        net.add_bond(tun.SymbolicBond(17,  5, 1, 2))
        net.add_bond(tun.SymbolicBond( 5, 42, 1, 0))
        net.add_bond(tun.SymbolicBond(17, 60, 0, 3))
        net.add_bond(tun.SymbolicBond( 5, 60, 0, 1))
        self.assertTrue(net.is_consistent(verbose=True))
        # full contraction based on a single call of `np.einsum`
        outlist = [(5, 3), (60, 0), (60, 2)]
        tidx, idxout = net.contractions_to_einsum(outlist)
        tnet_ref = np.einsum(t17, tidx[0], t05, tidx[1], t42, tidx[2], t60, tidx[3], idxout)
        self.assertEqual(tnet_ref.shape, (6, 5, 8))
        # alternative approach based on contraction trees
        tree_a = net.build_contraction_tree([[17, 5], [42, 60]])
        tree_b = net.build_contraction_tree([17, [5, [42, 60]]])
        # axis permutation should not affect overall tensor represented by tree
        tree_b.children[1].permute_axes([3, 4, 2, 0, 1])
        tensor_dict = {17: t17, 5: t05, 42: t42, 60: t60}
        tnet_a = tun.perform_contraction(tree_a, tensor_dict)
        tnet_b = tun.perform_contraction(tree_b, tensor_dict)
        # compare tensors
        self.assertAlmostEqual(np.linalg.norm(tnet_a - tnet_ref), 0, delta=1e-12)
        self.assertAlmostEqual(np.linalg.norm(tnet_b - tnet_ref), 0, delta=1e-12)

    def test_rewiring(self):
        """
        Test application of rewiring rule.
        """
        net = tun.SymbolicNetwork()
        net.add_tensor(tun.SymbolicIsometricTensor(17, 4, 42, [[0, 2, 3]]))
        net.add_tensor(tun.SymbolicIsometricTensor(42, 4, 17, [[0, 2, 3]]))
        net.add_tensor(tun.SymbolicTensor(60, 3))
        net.add_bond(tun.SymbolicBond(17, 42, 2, 2))
        net.add_bond(tun.SymbolicBond(17, 60, 0, 1))
        net.add_bond(tun.SymbolicBond(17, 60, 1, 2))
        net.add_bond(tun.SymbolicBond(42, 60, 3, 0))
        net.add_rewiring_rule(tun.SymbolicRewiringRule(
            [tun.SymbolicBond(17, 42, 2, 2), tun.SymbolicBond(60, 17, 2, 1)],
            # to-be deleted bonds
            [tun.SymbolicBond(17, 60, 0, 1), tun.SymbolicBond(60, 42, 0, 3)],
            # to-be added bonds
            [tun.SymbolicBond(42, 17, 0, 0), tun.SymbolicBond(42, 17, 3, 3), tun.SymbolicBond(42, 60, 1, 1)]))
        self.assertTrue(net.is_consistent(verbose=True))
        # simplify network
        net.simplify()
        self.assertTrue(net.is_consistent(verbose=True))
        # isometric tensors cancel, such that a single tensor remains
        self.assertEqual(len(net.tensors), 1)


if __name__ == "__main__":
    unittest.main()
