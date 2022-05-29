from enum import Enum
from .symbolic_network import SymbolicIsometricTensor, SymbolicBond, SymbolicRewiringRule, SymbolicNetwork


class SymbolicTernaryUnitaryGate(SymbolicIsometricTensor):
    """
    Symbolic representation of a ternary unitary gate, extending
    a symbolic isometric tensor, and additionally storing
    the site with smallest coordinate (x, y, t) which this gate acts on.
    """
    def __init__(self, tid: int, ct_id_ref: int, site):
        super().__init__(tid, 8, ct_id_ref, [
            [0, 2, 4, 6], [1, 3, 5, 7],     # along x-direction
            [0, 1, 4, 5], [2, 3, 6, 7],     # along y-direction
            [0, 1, 2, 3], [4, 5, 6, 7]])    # along t-direction
        self.site = list(site)


class SymbolicSolvablePEPSRing(SymbolicIsometricTensor):
    """
    Symbolic "solvable" PEPS ring, equivalent to a unitary MPO,
    with physical legs (marked by "x") pointing out of the plane in the drawing,
    and virtual bonds on the left and right.

           :
    ---| x   x |---
       |       |
    ---| x   x |---
       |       |
    ---| x   x |---
           :

    Enumeration of axes:
           :
    10-| 8   9 |11-
       |       |
    -6-| 4   5 |-7-
       |       |
    -2-| 0   1 |-3-
           :
    """
    def __init__(self, tid: int, ct_id_ref: int, length: int, x: int):
        super().__init__(tid, 4*length, ct_id_ref, [
            list(range(0, 4*length, 2)),
            list(range(1, 4*length, 2))])
        self.length = length
        self.xcoord = [x, x + 1]


class SymbolicSolvablePEPSPlaquette(SymbolicIsometricTensor):
    """
    Symbolic plaquette of a "solvable" PEPS state,
    modelled as concatenation of simple MPUs (matrix product unitaries),
    with physical legs (marked by "x") pointing out of the plane in the drawing.

        ___|___
       |       |
    ---| x   x |---
       |_______|
           |

    Enumeration of axes:
           5
        ___|___
       |       |
    -2-| 0   1 |-3-
       |_______|
           |
           4
    """
    def __init__(self, tid: int, ct_id_ref: int, x: int, y: int):
        # isometries realize "simple" MPU tensor property
        super().__init__(tid, 6, ct_id_ref, [[1, 3, 4, 5], [0, 2, 4, 5]])
        self.xcoord = [x, x + 1]
        self.ycoord = y


class TernaryCircuitBoundary(Enum):
    """
    Boundary of ternary unitary time evolution circuit,
    allowing to sandwich the circuit between a "solvable" PEPS.
    """
    PERIODIC       = 0
    PEPS_RING      = 1
    PEPS_PLAQUETTE = 2


def construct_time_evolution_circuit(Lx, Ly, T, boundary=TernaryCircuitBoundary.PERIODIC):
    """
    Construct a forward and backward time-evolution circuit,
    either assuming periodic boundary conditions along each direction (use_peps=False),
    or sandwiching gates between solvable PEPS tensors.

    PEPS rings act on sites (1, 2), (3, 4), ... in x-direction (omitting site 0).
    """
    # dimensions must be even
    assert Lx % 2 == 0 and Ly % 2 == 0 and T % 2 == 0
    Lxh = Lx // 2
    Lyh = Ly // 2
    net = SymbolicNetwork()
    # add ternary unitary gates
    for t in range(T):
        shift = 1 if t % 2 == 1 else 0
        for y in range(Lyh):
            for x in range(Lxh):
                tid = x + Lxh*(y + Lyh*t)
                # forward time evolution
                net.add_tensor(SymbolicTernaryUnitaryGate(tid, Lxh*Lyh*T + tid, [2*x + shift, 2*y + shift, t]))
                # backward time evolution
                net.add_tensor(SymbolicTernaryUnitaryGate(Lxh*Lyh*T + tid, tid, [2*x + shift, 2*y + shift, 2*T - t - 1]))
    if boundary == TernaryCircuitBoundary.PEPS_RING:
        # index offset to start of PEPS tensor IDs
        peps_offset = 2*Lxh*Lyh*T
        for x in range(Lxh):
            net.add_tensor(SymbolicSolvablePEPSRing(peps_offset + 2*x,     peps_offset + 2*x + 1, Ly, 2*x + 1))
            # conjugated copy
            net.add_tensor(SymbolicSolvablePEPSRing(peps_offset + 2*x + 1, peps_offset + 2*x,     Ly, 2*x + 1))
    elif boundary == TernaryCircuitBoundary.PEPS_PLAQUETTE:
        # index offset to start of PEPS tensor IDs
        peps_offset = 2*Lxh*Lyh*T
        for y in range(Ly):
            for x in range(Lxh):
                net.add_tensor(SymbolicSolvablePEPSPlaquette(peps_offset + 2*(x + Lxh*y),     peps_offset + 2*(x + Lxh*y) + 1, 2*x + 1, y))
                # conjugated copy
                net.add_tensor(SymbolicSolvablePEPSPlaquette(peps_offset + 2*(x + Lxh*y) + 1, peps_offset + 2*(x + Lxh*y),     2*x + 1, y))
    # add connecting bonds
    offset = Lxh*Lyh*T  # index offset between tensor of forward and backward time evolution
    # bonds between gates
    for t in range(T - 1):
        for y in range(Lyh):
            for x in range(Lxh):
                tid = x + Lxh*(y + Lyh*t)
                if t % 2 == 0:
                    x_prev = (x - 1) % Lxh
                    y_prev = (y - 1) % Lyh
                    for m in range(2):
                        net.add_bond(SymbolicBond(m*offset + tid, m*offset + x          + Lxh*(y      + Lyh*(t + 1)), 3, 4))
                        net.add_bond(SymbolicBond(m*offset + tid, m*offset + x          + Lxh*(y_prev + Lyh*(t + 1)), 1, 6))
                        if x > 0 or boundary == TernaryCircuitBoundary.PERIODIC:
                            net.add_bond(SymbolicBond(m*offset + tid, m*offset + x_prev + Lxh*(y      + Lyh*(t + 1)), 2, 5))
                            net.add_bond(SymbolicBond(m*offset + tid, m*offset + x_prev + Lxh*(y_prev + Lyh*(t + 1)), 0, 7))
                else:   # t odd
                    x_next = (x + 1) % Lxh
                    y_next = (y + 1) % Lyh
                    for m in range(2):
                        net.add_bond(SymbolicBond(m*offset + tid, m*offset + x          + Lxh*(y      + Lyh*(t + 1)), 0, 7))
                        net.add_bond(SymbolicBond(m*offset + tid, m*offset + x          + Lxh*(y_next + Lyh*(t + 1)), 2, 5))
                        if x < Lxh - 1 or boundary == TernaryCircuitBoundary.PERIODIC:
                            net.add_bond(SymbolicBond(m*offset + tid, m*offset + x_next + Lxh*(y      + Lyh*(t + 1)), 1, 6))
                            net.add_bond(SymbolicBond(m*offset + tid, m*offset + x_next + Lxh*(y_next + Lyh*(t + 1)), 3, 4))
    # connect forward and backward time evolution
    for y in range(Lyh):
        for x in range(Lxh):
            tid = x + Lxh*(y + Lyh*(T - 1))
            for i in range(4):
                net.add_bond(SymbolicBond(tid, tid + offset, i, i))
    if boundary == TernaryCircuitBoundary.PERIODIC:
        # periodic in time
        for y in range(Lyh):
            for x in range(Lxh):
                tid = x + Lxh*y
                for i in range(4, 8):
                    net.add_bond(SymbolicBond(tid, tid + offset, i, i))
    elif boundary == TernaryCircuitBoundary.PEPS_RING:
        # direct connections between gate tensors on the left boundary
        for t in range(0, T, 2):
            for y in range(Lyh):
                tid = Lxh*(y + Lyh*t)
                for i in range(4):
                    net.add_bond(SymbolicBond(tid, tid + offset, 2*i, 2*i))
        # direct connections between gate tensors on the right boundary
        for t in range(1, T, 2):
            for y in range(Lyh):
                tid = (Lxh - 1) + Lxh*(y + Lyh*t)
                if t < T - 1:   # for t == T - 1 there is already a direct connection
                    net.add_bond(SymbolicBond(tid, tid + offset, 1, 1))
                    net.add_bond(SymbolicBond(tid, tid + offset, 3, 3))
                net.add_bond(SymbolicBond(tid, tid + offset, 5, 5))
                net.add_bond(SymbolicBond(tid, tid + offset, 7, 7))
        # virtual PEPS bonds
        for x in range(Lxh - 1):
            for y in range(Ly):
                for m in range(2):
                    net.add_bond(SymbolicBond(peps_offset + 2*x + m, peps_offset + 2*(x + 1) + m, 4*y + 3, 4*y + 2))
        for y in range(Ly):
            # connect leftmost PEPS ring along left virtual axes
            net.add_bond(SymbolicBond(peps_offset, peps_offset + 1, 4*y + 2, 4*y + 2))
            # connect rightmost PEPS ring along right virtual axes
            net.add_bond(SymbolicBond(peps_offset + 2*(Lxh - 1), peps_offset + 2*(Lxh - 1) + 1, 4*y + 3, 4*y + 3))
            # connect rightmost PEPS ring along right physical axes
            net.add_bond(SymbolicBond(peps_offset + 2*(Lxh - 1), peps_offset + 2*(Lxh - 1) + 1, 4*y + 1, 4*y + 1))
        # connect gates with PEPS tensors
        for x in range(Lxh):
            for y in range(Lyh):
                for m in range(2):
                    net.add_bond(SymbolicBond(    m*offset + x     + Lxh*y, peps_offset + 2*x + m, 5, 8*y    ))
                    net.add_bond(SymbolicBond(    m*offset + x     + Lxh*y, peps_offset + 2*x + m, 7, 8*y + 4))
                    if x < Lxh - 1:
                        net.add_bond(SymbolicBond(m*offset + x + 1 + Lxh*y, peps_offset + 2*x + m, 4, 8*y + 1))
                        net.add_bond(SymbolicBond(m*offset + x + 1 + Lxh*y, peps_offset + 2*x + m, 6, 8*y + 5))
    elif boundary == TernaryCircuitBoundary.PEPS_PLAQUETTE:
        # direct connections between gate tensors on the left boundary
        for t in range(0, T, 2):
            for y in range(Lyh):
                tid = Lxh*(y + Lyh*t)
                for i in range(4):
                    net.add_bond(SymbolicBond(tid, tid + offset, 2*i, 2*i))
        # direct connections between gate tensors on the right boundary
        for t in range(1, T, 2):
            for y in range(Lyh):
                tid = (Lxh - 1) + Lxh*(y + Lyh*t)
                if t < T - 1:   # for t == T - 1 there is already a direct connection
                    net.add_bond(SymbolicBond(tid, tid + offset, 1, 1))
                    net.add_bond(SymbolicBond(tid, tid + offset, 3, 3))
                net.add_bond(SymbolicBond(tid, tid + offset, 5, 5))
                net.add_bond(SymbolicBond(tid, tid + offset, 7, 7))
        # virtual PEPS bonds in x-direction
        for x in range(Lxh - 1):
            for y in range(Ly):
                for m in range(2):
                    net.add_bond(SymbolicBond(peps_offset + 2*(x + Lxh*y) + m, peps_offset + 2*((x + 1) + Lxh*y) + m, 3, 2))
        # virtual PEPS bonds in y-direction
        for x in range(Lxh):
            for y in range(Ly):
                y_next = (y + 1) % Ly
                for m in range(2):
                    net.add_bond(SymbolicBond(peps_offset + 2*(x + Lxh*y) + m, peps_offset + 2*(x + Lxh*y_next) + m, 5, 4))
        for y in range(Ly):
            # connect leftmost PEPS plaquettes along left virtual axes
            net.add_bond(SymbolicBond(peps_offset + 2*Lxh*y,
                                      peps_offset + 2*Lxh*y + 1, 2, 2))
            # connect rightmost PEPS plaquettes along right physical and virtual axes
            for j in [1, 3]:
                net.add_bond(SymbolicBond(peps_offset + 2*((Lxh - 1) + Lxh*y),
                                          peps_offset + 2*((Lxh - 1) + Lxh*y) + 1, j, j))
        # connect gates with PEPS plaquettes
        for x in range(Lxh):
            for y in range(Lyh):
                for m in range(2):
                    net.add_bond(SymbolicBond(    m*offset + x     + Lxh*y, peps_offset + 2*(x + Lxh*(2*y  )) + m, 5, 0))
                    net.add_bond(SymbolicBond(    m*offset + x     + Lxh*y, peps_offset + 2*(x + Lxh*(2*y+1)) + m, 7, 0))
                    if x < Lxh - 1:
                        net.add_bond(SymbolicBond(m*offset + x + 1 + Lxh*y, peps_offset + 2*(x + Lxh*(2*y  )) + m, 4, 1))
                        net.add_bond(SymbolicBond(m*offset + x + 1 + Lxh*y, peps_offset + 2*(x + Lxh*(2*y+1)) + m, 6, 1))
        # bond rewiring rules, for "simple" MPU property
        for x in range(Lxh):
            for y in range(Ly):
                y_next = (y + 1) % Ly
                vbonds_y = [SymbolicBond(peps_offset + 2*(x + Lxh*y) + m,
                                         peps_offset + 2*(x + Lxh*y_next) + m, 5, 4) for m in range(2)]
                for xaxes in [[0, 2], [1, 3]]:
                    net.add_rewiring_rule(SymbolicRewiringRule(
                        [SymbolicBond(peps_offset + 2*(x + Lxh*v),
                                      peps_offset + 2*(x + Lxh*v) + 1, j, j) for j in xaxes for v in [y, y_next]] + vbonds_y,
                        vbonds_y,   # delete virtual bonds in y-direction
                        # connect conjugated copy along virtual y-axis
                        [SymbolicBond(peps_offset + 2*(x + Lxh*y),
                                      peps_offset + 2*(x + Lxh*y) + 1, 5, 5),
                         SymbolicBond(peps_offset + 2*(x + Lxh*y_next),
                                      peps_offset + 2*(x + Lxh*y_next) + 1, 4, 4)]))

    else:
        assert False, "unsupported boundary '{}'".format(boundary)
    return net


def find_ternary_unitary_gate_at(net: SymbolicNetwork, site):
    """
    Search for a ternary unitary gate anchored at `site`.
    """
    site = list(site)
    for t in net.tensors:
        if isinstance(t, SymbolicTernaryUnitaryGate):
            if t.site == site:
                return t
    # not found
    return None
