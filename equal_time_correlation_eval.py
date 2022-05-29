"""
Evaluate equal-time correlation functions for various operator distances.
"""

import numpy as np
import h5py
import ternary_unitary_networks as tun


def main():

    # local physical dimension
    d = 2

    seed = 142
    np.random.seed(seed)

    # uniform ternary unitary gate
    G = tun.construct_random_ternary_gate()

    # Pauli matrices
    σ = [np.array([[1,  0 ], [0,  1]]),
         np.array([[0,  1 ], [1,  0]]),
         np.array([[0, -1j], [1j, 0]]),
         np.array([[1,  0 ], [0, -1]])]
    # local operators
    rx = np.random.randn(3); rx /= np.linalg.norm(rx)
    ry = np.random.randn(3); ry /= np.linalg.norm(ry)
    ax = sum(rx[i]*σ[i+1] for i in range(3))
    ay = sum(ry[i]*σ[i+1] for i in range(3))

    # virtual bond dimension of PEPS tensor in x-direction
    Dx = 4
    ring = tun.random_simple_mpu(d*Dx, 2, 2)
    # transfer states of a simple MPU should be proportional to the identity
    assert np.allclose(ring.left_transfer_state,  np.trace(ring.left_transfer_state) *np.identity(ring.D)/ring.D)
    assert np.allclose(ring.right_transfer_state, np.trace(ring.right_transfer_state)*np.identity(ring.D)/ring.D)
    assert np.allclose(np.dot(ring.left_transfer_state.reshape(-1),
                              ring.right_transfer_state.reshape(-1)), 1)
    # set to exact values
    ring.left_transfer_state  = np.identity(ring.D)
    ring.right_transfer_state = np.identity(ring.D) / ring.D

    # effect of conjugation by ternary gate and partial trace
    M = [tun.conjugation_trace_map_4sites(G, (0, 2), d=d),
         tun.conjugation_trace_map_4sites(G, (1, 3), d=d)]

    # maximum number of time steps
    Tmax = 4
    # largest spatial offset
    Δxmax = 12
    Δymax = 10

    # logical time step is T//2
    corr = np.zeros((Tmax//2 + 1, Δxmax + 1, 2*Δymax + 1), dtype=complex)

    for T in range(2, Tmax+1, 2):
        print("T:", T)

        # time-evolved operators
        mpo = [[tun.MPO.identity(d, 2*T) for j in range(2)] for i in range(2)]
        mpo[0][0].A[T-1] = np.reshape(ax, (d, d, 1, 1))
        mpo[0][1].A[T]   = np.reshape(ax, (d, d, 1, 1))
        mpo[1][0].A[T-1] = np.reshape(ay, (d, d, 1, 1))
        mpo[1][1].A[T]   = np.reshape(ay, (d, d, 1, 1))
        for i in range(2):
            for j in range(2):
                tun.apply_brickwall_cone(mpo[i][j], M[i], T - 1, T)

        for Δx in range(Δxmax + 1):
            # local operators must be located on opposite (in x-direction) edges of plaquette
            if Δx % 2 == 0: continue
            # need at least one ring for the correlation to be non-zero
            # (otherwise equivalent to case of periodic boundary conditions)
            if Δx <= 2*T: continue
            nrings = (Δx - 2*T + 1) // 2
            assert nrings >= 1
            print("Δx:", Δx)
            print("number of rings:", nrings)
            for Δy in range(-Δymax, Δymax + 1):
                # compute correlation
                if Δy % 2 == 0:
                    c1 = tun.mpo_peps_overlap(mpo[0][0], mpo[1][0], ring, nrings, Δy)
                    c2 = tun.mpo_peps_overlap(mpo[0][1], mpo[1][1], ring, nrings, Δy)
                else:
                    c1 = tun.mpo_peps_overlap(mpo[0][0], mpo[1][1], ring, nrings, Δy - 1)
                    c2 = tun.mpo_peps_overlap(mpo[0][1], mpo[1][0], ring, nrings, Δy + 1)
                # average to account for odd-even-effects
                corr[T//2, Δx, Δymax + Δy] = (c1 + c2) /2

    # save data to disk
    with h5py.File(f"data/equal_time_correlation_{seed}.hdf5", "w") as f:
        # MPU tensor of ring
        dset_ring = f.create_dataset("ring", ring.A.shape, dtype=ring.A.dtype)
        dset_ring[...] = ring.A
        # ternary unitary gate
        dset_G = f.create_dataset("G", G.shape, dtype=G.dtype)
        dset_G[...] = G
        # local operators
        dset_rx = f.create_dataset("rx", rx.shape, dtype=rx.dtype)
        dset_rx[...] = rx
        dset_ry = f.create_dataset("ry", ry.shape, dtype=ry.dtype)
        dset_ry[...] = ry
        dset_ax = f.create_dataset("ax", ax.shape, dtype=ax.dtype)
        dset_ax[...] = ax
        dset_ay = f.create_dataset("ay", ay.shape, dtype=ay.dtype)
        dset_ay[...] = ay
        # correlations
        dset_corr = f.create_dataset("corr", corr.shape, dtype=corr.dtype)
        dset_corr[...] = corr
        dset_corr.attrs["Δxmax"] = Δxmax
        dset_corr.attrs["Δymax"] = Δymax


if __name__ == "__main__":
    main()
