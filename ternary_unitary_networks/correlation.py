import numpy as np
from .mpo import MPO
from .mpu import MPU
from .four_gate import conjugation_trace_map_4sites
from .symbolic_network import SymbolicNetwork, SymbolicTensor, SymbolicBond


def compute_dynamic_correlations(G, ax, ay, Tmax, d=2):
    """
    Compute dynamic correlation functions along the light ray of a ternary unitary network.
    """
    M = conjugation_trace_map_4sites(G, 0, d=d)
    ayt = ay.flatten()
    corr = np.zeros(Tmax + 1, dtype=complex)
    for t in range(Tmax+1):
        corr[t] = np.trace(ax @ np.reshape(ayt, ay.shape)) / d
        # two gate layers form a logical time step
        ayt = M @ (M @ ayt)
    return corr


def apply_brickwall_cone(mpo: MPO, M, i_start: int, nsteps: int):
    """
    Apply the two-site operator `M` to the matrix product operator `mpo`
    in a brickwall pattern forming a light cone.

    `mpo` is updated in-place.
    """
    # apply two-site operators in a brickwall pattern, similar to TEBD algorithm
    for n in range(nsteps):
        for i in range(i_start - n, i_start + n + 2, 2):
            if i < 0 or i >= mpo.nsites - 1:
                raise RuntimeError("extent of MPO ({} sites) not sufficient to apply operators".format(mpo.nsites))
            mpo.apply_two_site_operator(i, M)


def mpo_peps_overlap(mpoL: MPO, mpoR: MPO, ring: MPU, nrings: int, y_shift: int):
    """
    Compute the overlap of two MPOs on the left and right boundary
    sandwiched between solvable PEPS rings represented as MPUs.
    """
    # require (for simplicity) that both MPOs have the same length
    assert mpoL.nsites == mpoR.nsites
    assert mpoL.nsites > 0
    # local physical dimension
    d = mpoL.A[0].shape[0]
    assert d == mpoR.A[0].shape[0]
    assert mpoL.A[0].shape[2] == 1 and mpoR.A[0].shape[2] == 1, "require dummy virtual bond dimensions at boundary"
    # local dimension of MPU consists of actual physical dimension
    # and virtual bond dimension in x-direction
    assert ring.d % d == 0, "local dimension of MPU must be a multiple of physical dimension"
    # virtual bond dimension in x-direction
    Dx = ring.d // d
    # normalized PEPS tensor
    P = ring.A.reshape((d, Dx, d, Dx, ring.D, ring.D)).transpose((0, 2, 1, 3, 4, 5)) / np.sqrt(d)
    # dummy identity map (for creating dummy bonds)
    id1 = np.identity(1)
    # require additional PEPS tensors in y-direction, since "simple" MPU property can only be applied along cone
    y_pad = (nrings + 1) // 2
    y_len = mpoL.nsites + abs(y_shift) + 2*y_pad
    # 2D grid of plaquettes, formed by contracting the PEPS tensors with their conjugated copies,
    # and sandwiched MPO tensors at the boundary;
    # each plaquette retains the virtual bonds connecting its neighbors,
    # which are then contracted at the end;
    # for consistency, all plaquettes have tensor degree 4,
    # using dummy bonds with dimension 1 if necessary
    grid = np.empty((nrings, y_len), dtype=object)
    for y in range(y_len):
        for x in range(nrings):
            # actual to-be contracted tensors:
            #   0, 1: PEPS tensor and its conjugated copy,
            #   2, 3: left and right MPO tensor (only at left and right boundary, or dummy identity for virtual bonds)
            #   4, 5: left and right MPU transfer state (only at top and bottom boundary, or unused identity)
            #   6, 7: dummy virtual PEPS bonds in x-direction (if required, only at left and right boundary)
            #   8, 9: dummy virtual PEPS bonds in y-direction (if required, only at top and bottom boundary)
            tensors = [P, P.conj(), id1, id1, id1, id1, id1, id1, id1, id1]
            # collect output axes (None entries will be filled)
            out_xL = [(0, 2), (1, 2)]               # virtual x-bonds of PEPS tensor and its conjugated copy
            out_xR = [(0, 3), (1, 3)]
            out_yL = [(0, 4), (1, 4), None, None]   # first two entries: virtual y-bonds of PEPS tensor and its conjugated copy
            out_yR = [(0, 5), (1, 5), None, None]
            # symbolic network for local two-site plaquette
            net = SymbolicNetwork()
            net.add_tensor(SymbolicTensor(0, 6))    # PEPS tensor
            net.add_tensor(SymbolicTensor(1, 6))    # conjugated PEPS tensor
            # MPO tensors
            AL = None
            if x == 0:
                if y_shift >= 0:
                    y_eff = y - y_pad
                    if 0 <= y_eff < mpoL.nsites:
                        AL = mpoL.A[y_eff]
                else:
                    y_eff = y - y_pad + y_shift
                    if 0 <= y_eff < mpoL.nsites:
                        AL = mpoL.A[y_eff]
            AR = None
            if x == nrings - 1:
                if y_shift >= 0:
                    y_eff = y - y_pad - y_shift
                    if 0 <= y_eff < mpoR.nsites:
                        AR = mpoR.A[y_eff]
                else:
                    y_eff = y - y_pad
                    if 0 <= y_eff < mpoR.nsites:
                        AR = mpoR.A[y_eff]
            if AL is not None:
                net.add_tensor(SymbolicTensor(2, 4))
                tensors[2] = AL
                # contractions with AL along physical axes
                net.add_bond(SymbolicBond(0, 2, 0, 1))
                net.add_bond(SymbolicBond(1, 2, 0, 0))
                out_yL[2] = (2, 2)
                out_yR[2] = (2, 3)
            else:
                # connect left physical axes
                net.add_bond(SymbolicBond(0, 1, 0, 0))
                # dummy virtual bonds in y-direction
                net.add_tensor(SymbolicTensor(2, 2))
                out_yL[2] = (2, 0)
                out_yR[2] = (2, 1)
            if AR is not None:
                net.add_tensor(SymbolicTensor(3, 4))
                tensors[3] = AR
                # contractions with AR along physical axes
                net.add_bond(SymbolicBond(0, 3, 1, 1))
                net.add_bond(SymbolicBond(1, 3, 1, 0))
                out_yL[3] = (3, 2)
                out_yR[3] = (3, 3)
            else:
                # connect right physical axes
                net.add_bond(SymbolicBond(0, 1, 1, 1))
                # dummy virtual bonds in y-direction
                net.add_tensor(SymbolicTensor(3, 2))
                out_yL[3] = (3, 0)
                out_yR[3] = (3, 1)
            # remaining symbolic tensors, some of which might be unused
            for tid in range(4, 10):
                net.add_tensor(SymbolicTensor(tid, 2))
            # left boundary
            if x == 0:
                # connect virtual bonds in x-direction on the left
                net.add_bond(SymbolicBond(0, 1, 2, 2))
                # use dummy virtual bonds
                out_xL[0] = (6, 0)
                out_xL[1] = (6, 1)
            # right boundary
            if x == nrings - 1:
                # connect virtual bonds in x-direction on the right
                net.add_bond(SymbolicBond(0, 1, 3, 3))
                # use dummy virtual bonds
                out_xR[0] = (7, 0)
                out_xR[1] = (7, 1)
            # bottom boundary
            if y == 0:
                # contract with left transfer state
                tensors[4] = ring.left_transfer_state
                net.add_bond(SymbolicBond(0, 4, 4, 0))
                net.add_bond(SymbolicBond(1, 4, 4, 1))
                # dummy virtual bonds in y-direction
                out_yL[0] = (8, 0)
                out_yL[1] = (8, 1)
            # top boundary
            if y == y_len - 1:
                # contract with right transfer state
                tensors[5] = ring.right_transfer_state
                net.add_bond(SymbolicBond(0, 5, 5, 0))
                net.add_bond(SymbolicBond(1, 5, 5, 1))
                # dummy virtual bonds in y-direction
                out_yR[0] = (9, 0)
                out_yR[1] = (9, 1)
            # perform contraction
            tidx, idxout = net.contractions_to_einsum(out_xL + out_xR + out_yL + out_yR)
            T = np.einsum(*(sum(zip(tensors, tidx), ()) + (idxout,)))
            assert T.ndim == 12
            # group dimensions
            s = T.shape
            T = T.reshape((s[0]*s[1], s[2]*s[3], s[4]*s[5]*s[6]*s[7], s[8]*s[9]*s[10]*s[11]))
            # store plaquette tensor, with y-bonds as leading dimension for later overall contraction
            grid[x, y] = T.transpose((2, 3, 0, 1))

    # contract overall grid
    # TODO: select more efficient contraction order (row-wise or column-wise)
    row = MPO(list(grid[:, 0])).canonicalize("left").canonicalize("right")
    for y in range(1, y_len):
        row = row @ MPO(list(grid[:, y]))
        row.canonicalize("left").canonicalize("right")
    c = row.as_matrix()
    assert c.shape == (1, 1)
    return c[0, 0] / Dx**y_len
