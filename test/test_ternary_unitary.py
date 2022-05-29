import unittest
import numpy as np
import ternary_unitary_networks as tun


class TestTernaryUnitary(unittest.TestCase):

    def test_ternary_unitary_gate(self):
        """
        Test required ternary unitary gate properties.
        """
        d = 2
        r = tun.construct_random_ternary_gate()
        s = tun.four_gate_transpose(r, 0, d)
        t = tun.four_gate_transpose(r, 1, d)
        # test whether 'r' is indeed a ternary unitary
        self.assertTrue(np.allclose(r @ r.conj().T, np.identity(d**4)))
        self.assertTrue(np.allclose(s @ s.conj().T, np.identity(d**4)))
        self.assertTrue(np.allclose(t @ t.conj().T, np.identity(d**4)))
        # ternary-SWAP gate
        t = tun.ternary_swap_gate()
        self.assertTrue(np.allclose(t @ t.conj().T, np.identity(d**4)))
        # self-dual
        self.assertTrue(np.allclose(t, tun.four_gate_transpose(t, 0, d)))
        self.assertTrue(np.allclose(t, tun.four_gate_transpose(t, 1, d)))

    def test_conjugation_trace_map(self):
        """
        Test invariances of conjugation-trace maps under axes reorderings.
        """
        d = 2
        u = tun.construct_random_ternary_gate()
        dswap = tun.swap_gate(d**2)
        tswap = tun.ternary_swap_gate()
        u2 = dswap @ u @ dswap
        u3 = tswap @ u @ tswap
        self.assertTrue(np.allclose(tun.conjugation_trace_map_4sites(u,  (0, 1)),
                                    tun.conjugation_trace_map_4sites(u2, (2, 3))))
        self.assertTrue(np.allclose(tun.conjugation_trace_map_4sites(u,  (2, 3)),
                                    tun.conjugation_trace_map_4sites(u2, (0, 1))))
        for axis in range(4):
            self.assertTrue(np.allclose(tun.conjugation_trace_map_4sites(u,  axis),
                                        tun.conjugation_trace_map_4sites(u3, 3 - axis)))
        self.assertTrue(np.allclose(tun.conjugation_trace_map_4sites(u,  (0, 2)),
                                    tun.conjugation_trace_map_4sites(u3, (1, 3)).
                                        reshape((d, d, d, d, d, d, d, d)).
                                        transpose((1, 0, 3, 2, 5, 4, 7, 6)).
                                        reshape((d**4, d**4))))
        self.assertTrue(np.allclose(tun.conjugation_trace_map_4sites(tswap, (0, 1)),
                                    np.kron(tun.swap_gate(d), tun.swap_gate(d))))


if __name__ == "__main__":
    unittest.main()
