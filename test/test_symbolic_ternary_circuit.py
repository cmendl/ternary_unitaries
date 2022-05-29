import unittest
import ternary_unitary_networks as tun


class TestSymbolicTernaryCircuit(unittest.TestCase):

    def test_time_evolution_circuit(self):
        """
        Test time evolution circuit construction.
        """
        Lx = 4
        Ly = 6
        T  = 8
        for boundary in tun.TernaryCircuitBoundary:
            net = tun.construct_time_evolution_circuit(Lx, Ly, T, boundary=boundary)
            self.assertTrue(net.is_consistent(verbose=True))
            self.assertListEqual(net.open_axes(), [])
            # simplify network
            net.simplify()
            self.assertListEqual(net.tensors, [])
            self.assertListEqual(net.open_axes(), [])
            # Lx*Ly//2 term from connections of forward and backward time evolution
            self.assertEqual(net.identity_loops, [Lx*Ly, Ly*(T + 1) + Lx*Ly//2, Ly*(T + 1) + Lx*Ly//2][boundary.value])

    def test_dynamical_correlation(self):
        """
        Test correlation function network construction and simplification.
        """
        Lx = 8
        Ly = 8
        T  = 4
        for boundary in tun.TernaryCircuitBoundary:
            x_start = 1
            y_start = 3
            net = tun.construct_time_evolution_circuit(Lx, Ly, T, boundary=boundary)
            # insert local operators
            tug1 = tun.find_ternary_unitary_gate_at(net, (x_start, y_start, T))
            tug2 = tun.find_ternary_unitary_gate_at(net, (x_start + T - 1, y_start + T - 1, 2*T - 1))
            self.assertNotEqual(tug1, None)
            self.assertNotEqual(tug2, None)
            net.thread_tensor(net.max_tensor_id + 1, net.find_bond(tug1.tid, 0))
            net.thread_tensor(net.max_tensor_id + 1, net.find_bond(tug2.tid, 7))
            self.assertTrue(net.is_consistent(verbose=True))
            self.assertListEqual(net.open_axes(), [])
            # solvable PEPS tensors cancel even after inserting local operators
            net.simplify()
            # remaining gates must be arranged along a diagonal
            self.assertEqual(len(net.tensors), 2*T + 2)
            sitelist = []
            for t in net.tensors:
                if isinstance(t, tun.SymbolicTernaryUnitaryGate):
                    sitelist.append(t.site)
            sitelist.sort(key=lambda s: s[2])
            sitelist_ref = (
                [[x_start + T - 1 - t, y_start + T - 1 - t, t] for t in range(T)] +
                [[x_start + t, y_start + t, T + t] for t in range(T)])
            self.assertListEqual(net.open_axes(), [])
            self.assertListEqual(sitelist, sitelist_ref)


if __name__ == "__main__":
    unittest.main()
