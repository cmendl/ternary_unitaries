"""
Evaluate equal-time correlation function, and
compute and visualize corresponding symbolic tensor network.
"""

import numpy as np
import matplotlib.pyplot as plt
import ternary_unitary_networks as tun


def main():

    # local physical dimension
    d = 2
    # number of time steps
    T = 4
    # offset (in terms of 2x2 ternary gate plaquettes)
    Δx = 10
    Δy = 6
    # which respective axes to apply the local operators to
    axes = [0, 3]

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
    print("number of rings:", nrings)

    # compute correlation
    c = tun.mpo_peps_overlap(mpo[0], mpo[1], ring, nrings, Δy)
    print("computed correlation:", c)

    # symbolic network representation
    # system size
    Lx = 14
    Ly = 16
    # coordinates of first operator
    x0 = 3
    y0 = 3
    net = tun.construct_time_evolution_circuit(Lx, Ly, T,
            boundary=tun.TernaryCircuitBoundary.PEPS_PLAQUETTE)
    # insert local operators
    gate_loc = [[x0,      y0,      T - 1],
                [x0 + Δx, y0 + Δy, T - 1]]
    tug = [tun.find_ternary_unitary_gate_at(net, gate_loc[i]) for i in range(2)]
    assert all(tug)
    for i in range(2):
        net.thread_tensor(net.max_tensor_id + 1, net.find_bond(tug[i].tid, axes[i]))
    assert net.is_consistent(verbose=True)
    assert not net.open_axes()
    net.simplify()
    assert net.is_consistent(verbose=True)
    assert not net.open_axes()
    print("remaining tensor degrees:", [t.deg for t in net.tensors])
    # locations of local operators
    op_loc = [np.array(gate_loc[i]) + ternary_gate_corner(axes[i]) for i in range(2)]
    # visualize network
    xgrid, ygrid, tgrid = np.meshgrid(
        np.arange(Lx + 1),
        np.arange(Ly + 1),
        np.array([-0.5] + list(range(2*T+1)) + [2*T + 0.5]),
        indexing='ij')
    voxels = np.zeros((Lx, Ly, 2*T + 2), dtype=bool)
    vcolor = np.zeros(voxels.shape + (3,))
    for t in net.tensors:
        if isinstance(t, tun.SymbolicTernaryUnitaryGate):
            voxels[t.site[0], t.site[1], t.site[2] + 1] = True
            if t.site[2] < T:
                vcolor[t.site[0], t.site[1], t.site[2] + 1, :] = [1, 0.75, 0.75]
            else:
                vcolor[t.site[0], t.site[1], t.site[2] + 1, :] = [0.75, 0.75, 1]
        elif isinstance(t, tun.SymbolicSolvablePEPSRing):
            z = 0 if t.tid % 2 == 0 else 2*T + 1
            for y in range(Ly):
                voxels[t.xcoord[0], y, z] = True
                vcolor[t.xcoord[0], y, z, :] = [0, 1, 0]
        elif isinstance(t, tun.SymbolicSolvablePEPSPlaquette):
            z = 0 if t.tid % 2 == 0 else 2*T + 1
            voxels[t.xcoord[0], t.ycoord, z] = True
            vcolor[t.xcoord[0], t.ycoord, z, :] = [0, 1, 0]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(xgrid, ygrid, tgrid, voxels, facecolors=vcolor, edgecolor='k')
    for loc in op_loc:
        ax.scatter([loc[0]], [loc[1]], [loc[2]], color=[1, 1, 0], s=100)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('t')
    plt.show()


def ternary_gate_corner(axis):
    """
    Corner coordinate offset of `axis` of a ternary unitary gate.
    """
    return np.array([
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]][axis])


if __name__ == "__main__":
    main()
