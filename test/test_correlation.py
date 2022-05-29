import numpy as np
import unittest
import ternary_unitary_networks as tun


class TestCorrelation(unittest.TestCase):

    def test_equal_time_correlation(self):
        """
        Test computation of equal-time correlation function.
        """
        # local physical dimension
        d = 2
        # number of time steps
        T = 2
        # offset (in terms of 2x2 ternary gate plaquettes)
        Δx = 6
        Δy = 4
        # which respective axes to apply the local operators to
        axes = [0, 3]

        # virtual bond dimension of PEPS tensor in x-direction
        Dx = 4
        ring = tun.random_simple_mpu(d*Dx, 2, 2)
        # transfer states of a simple MPU should be proportional to the identity
        self.assertTrue(np.allclose(ring.left_transfer_state,  np.trace(ring.left_transfer_state) *np.identity(ring.D)/ring.D))
        self.assertTrue(np.allclose(ring.right_transfer_state, np.trace(ring.right_transfer_state)*np.identity(ring.D)/ring.D))
        self.assertTrue(np.allclose(np.dot(ring.left_transfer_state.reshape(-1),
                                           ring.right_transfer_state.reshape(-1)), 1))
        # set to exact values
        ring.left_transfer_state  = np.identity(ring.D)
        ring.right_transfer_state = np.identity(ring.D) / ring.D

        # uniform ternary unitary gate
        G = tun.construct_random_ternary_gate()

        # Pauli matrices
        σ = [np.array([[1,  0 ], [0,  1]]),
             np.array([[0,  1 ], [1,  0]]),
             np.array([[0, -1j], [1j, 0]]),
             np.array([[1,  0 ], [0, -1]])]
        # local operators
        ax = sum(np.random.randn()*σi for σi in σ[1:])
        ay = sum(np.random.randn()*σi for σi in σ[1:])

        # effect of conjugation by ternary gate and partial trace
        M = [tun.conjugation_trace_map_4sites(G, (0, 2), d=d),
             tun.conjugation_trace_map_4sites(G, (1, 3), d=d)]
        # time-evolved operators
        mpo = [tun.MPO.identity(d, 2*T) for _ in range(2)]
        mpo[0].A[T-1] = np.reshape(ax, (d, d, 1, 1))
        mpo[1].A[T]   = np.reshape(ay, (d, d, 1, 1))
        for i in range(2):
            tun.apply_brickwall_cone(mpo[i], M[i], T - 1, T)

        # Δx + 1 since local operators are on opposite sides of plaquette
        nrings = ((Δx + 1) - 2*T + 1) // 2

        # compute correlation
        c = tun.mpo_peps_overlap(mpo[0], mpo[1], ring, nrings, Δy)

        # use symbolic network for reference calculation
        # coordinates of first operator
        x0 = 3
        y0 = 3
        Lx = 12
        Ly = 12
        net = tun.construct_time_evolution_circuit(Lx, Ly, T,
                boundary=tun.TernaryCircuitBoundary.PEPS_PLAQUETTE)
        # insert local operators
        gate_loc = [[x0,      y0,      T - 1],
                    [x0 + Δx, y0 + Δy, T - 1]]
        tug = [tun.find_ternary_unitary_gate_at(net, gate_loc[i]) for i in range(2)]
        self.assertTrue(all(tug))
        op_tid = [-1, -1]
        for i in range(2):
            op_tid[i] = net.max_tensor_id + 1
            net.thread_tensor(op_tid[i], net.find_bond(tug[i].tid, axes[i]))
        self.assertTrue(net.is_consistent(verbose=True))
        self.assertTrue(not net.open_axes())
        net.simplify()
        self.assertTrue(net.is_consistent(verbose=True))
        self.assertTrue(not net.open_axes())
        # build contraction tree
        gate_pairs_L = []
        gate_pairs_R = []
        # count loops appearing in gate conjugations, for correct normalization
        nloops_gate = 0
        for ta in net.tensors:
            for tb in net.tensors:
                if isinstance(ta, tun.SymbolicTernaryUnitaryGate) and isinstance(tb, tun.SymbolicTernaryUnitaryGate):
                    # conjugate-transposed copy
                    if ta.tid < tb.tid and ta.ct_id_ref == tb.tid:
                        self.assertTrue(ta.site[0] == tb.site[0], msg="x-coordinates of gate pairs must agree")
                        for i in range(4, 8):
                            bond = net.find_bond(ta.tid, i)
                            self.assertTrue(bond is not None)
                            if tb.tid in bond.tids:
                                nloops_gate += 1
                        if ta.site[0] <= x0 + Δx//2:
                            gate_pairs_L.append((ta, tb))
                        else:
                            gate_pairs_R.append((ta, tb))
        self.assertTrue(len(gate_pairs_L) == len(gate_pairs_R))
        # sort by x-coordinate
        gate_pairs_L = sorted(gate_pairs_L, key=lambda t: t[0].site[0])
        gate_pairs_R = sorted(gate_pairs_R, key=lambda t: t[0].site[0], reverse=True)
        scaffold_L = op_tid[0]
        for g in gate_pairs_L:
            scaffold_L = [scaffold_L, [g[0].tid, g[1].tid]]
        scaffold_R = op_tid[1]
        for g in gate_pairs_R:
            scaffold_R = [scaffold_R, [g[0].tid, g[1].tid]]
        peps_pairs = []
        # keep track of normalization of identity transfer states in x-direction
        ymin_symbolic_peps = Ly
        ymax_symbolic_peps = 0
        for ta in net.tensors:
            for tb in net.tensors:
                if isinstance(ta, tun.SymbolicSolvablePEPSPlaquette) and isinstance(tb, tun.SymbolicSolvablePEPSPlaquette):
                    # conjugate-transposed copy
                    if ta.tid < tb.tid and ta.ct_id_ref == tb.tid:
                        self.assertTrue(ta.xcoord[0] == tb.xcoord[0] and ta.ycoord == tb.ycoord,
                                        msg="coordinates of PEPS pairs must agree")
                        peps_pairs.append((ta, tb))
                        if ta.ycoord < ymin_symbolic_peps:
                            ymin_symbolic_peps = ta.ycoord
                        if ta.ycoord > ymax_symbolic_peps:
                            ymax_symbolic_peps = ta.ycoord
        # sort by y-coordinate
        peps_pairs = sorted(peps_pairs, key=lambda t: t[0].ycoord)
        scaffold_peps = None
        for p in peps_pairs:
            if scaffold_peps is not None:
                scaffold_peps = [scaffold_peps, [p[0].tid, p[1].tid]]
            else:
                scaffold_peps = [p[0].tid, p[1].tid]
        tree = net.build_contraction_tree([[scaffold_L, scaffold_peps], scaffold_R])
        # actual PEPS tensor
        P = ring.A.reshape((d, Dx, d, Dx, ring.D, ring.D)).transpose((0, 2, 1, 3, 4, 5)) / np.sqrt(d)
        # assemble tensor dictionary and perform contraction
        tensor_dict = { op_tid[0]: ax, op_tid[1]: ay }
        for t in net.tensors:
            if isinstance(t, tun.SymbolicTernaryUnitaryGate):
                tensor_dict[t.tid] = G.reshape(8*(d,)) if t.site[2] < T else G.conj().reshape(8*(d,))
            elif isinstance(t, tun.SymbolicSolvablePEPSPlaquette):
                tensor_dict[t.tid] = P if t.tid % 2 == 0 else P.conj()
        # account for normalization of gate loops and transfer states
        c_ref = tun.perform_contraction(tree, tensor_dict) / (d**nloops_gate * ring.D**nrings * Dx**(ymax_symbolic_peps - ymin_symbolic_peps + 1))

        self.assertTrue(np.allclose(c, c_ref), msg="computed correlation: {}, reference: {}".format(c, c_ref))


if __name__ == "__main__":
    unittest.main()
