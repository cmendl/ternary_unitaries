import itertools
import networkx as nx
from .contraction_tree import ContractionNode, find_contraction_node, track_open_axis


class SymbolicTensor(object):
    """
    Symbolic tensor class, storing a unique ID and the tensor degree.
    """
    def __init__(self, tid: int, deg: int):
        assert tid >= 0
        assert deg >= 0  # a tensor of degree 0 is a scalar
        self.tid = tid
        self.deg = deg


class SymbolicIsometricTensor(SymbolicTensor):
    """
    Symbolic representation of an isometric tensor (isometry after grouping
    certain axes together).
    Additional member variables:
        ct_id_ref:  ID of conjugate-transposed copy of tensor, if any
        isoaxes:    list of lists: [[i_1, i_2, ...], [j_1, j_2, ...], ...]:
                    simultaneous contraction along axes [i_1, i_2, ...] with
                    conjugated copy gives identity map along the other axes.
    """
    def __init__(self, tid: int, deg: int, ct_id_ref: int, isoaxes: list):
        super().__init__(tid, deg)
        assert ct_id_ref != tid
        self.ct_id_ref = ct_id_ref
        self.isoaxes = isoaxes


class SymbolicConnector(SymbolicTensor):
    """
    Symbolic "connector" to the dangling end of an open bond.
    """
    def __init__(self, tid: int, site):
        super().__init__(tid, 1)
        self.site = list(site)


class SymbolicBond(object):
    """
    Symbolic bond between two tensors.

    Member variables:
        tid0:   index of first tensor
        tid1:   index of second tensor
        i, j:   which axis of each respective tensor to contract
    """
    def __init__(self, tid0, tid1, i, j):
        assert tid0 >= 0 and tid1 >= 0
        assert tid0 != tid1 or i != j
        self.tids = [tid0, tid1]
        self.axes = [i, j]

    def flip(self):
        """
        Flip tensor and axes indices. (Logically the same bond.)
        """
        self.tids = list(reversed(self.tids))
        self.axes = list(reversed(self.axes))

    def __eq__(self, other):
        """
        Equality test of two bonds.
        """
        if isinstance(other, SymbolicBond):
            return ((self.tids == other.tids and self.axes == other.axes) or
                    (self.tids == list(reversed(other.tids)) and self.axes == list(reversed(other.axes))))
        return False

    def __ne__(self, other):
        """
        Inequality test of two bonds.
        """
        return not self.__eq__(other)


class SymbolicRewiringRule(object):
    """
    Bond rewiring rule: if all required bonds `req_bonds` exist in the network,
    `del_bonds` are removed and `add_bonds` are added.
    """
    def __init__(self, req_bonds, del_bonds, add_bonds):
        self.req_bonds = list(req_bonds)
        self.del_bonds = list(del_bonds)
        self.add_bonds = list(add_bonds)

    def matches(self, bondlist):
        """
        Test whether rule matches list of bonds in `bondlist`.
        """
        for b in self.req_bonds:
            if not b in bondlist:
                return False
        return True


class SymbolicNetwork(object):
    """
    Symbolic tensor network, storing a list of tensors and bond contractions.

    Symbolic simplification of the network by cancelling isometries, and
    facility for constructing contraction trees.
    """
    def __init__(self):
        self.tensors = []
        self.bonds = []
        self.rewirings = []
        # count the number of identity loops
        self.identity_loops = 0

    def add_tensor(self, tensor : SymbolicTensor):
        self.tensors.append(tensor)

    def add_bond(self, bond : SymbolicBond):
        self.bonds.append(bond)

    def add_rewiring_rule(self, rule: SymbolicRewiringRule):
        self.rewirings.append(rule)

    @property
    def max_tensor_id(self):
        """
        Largest tensor ID appearing in the network.
        """
        return max(t.tid for t in self.tensors)

    def find_tensor(self, tid):
        """
        Search for tensor with ID `tid`.
        """
        for t in self.tensors:
            if t.tid == tid:
                return t
        # not found
        return None

    def find_bond(self, tid, i):
        """
        Search for a bond connecting tensor `tid` along its i-th axis.
        """
        for b in self.bonds:
            if (b.tids[0] == tid and b.axes[0] == i) or (b.tids[1] == tid and b.axes[1] == i):
                return b
        # not found
        return None

    def thread_tensor(self, tid, bond : SymbolicBond):
        """
        Thread a tensor between a bond connection.
        """
        assert bond in self.bonds
        assert tid not in [t.tid for t in self.tensors]
        # add tensor of degree 2
        self.add_tensor(SymbolicTensor(tid, 2))
        self.add_bond(SymbolicBond(tid, bond.tids[1], 0, bond.axes[1]))
        # re-route existing bond to new tensor
        bond.tids[1] = tid
        bond.axes[1] = 1

    def as_graph(self) -> nx.Graph:
        """
        Graph representation of the network: tensor -> node, bond -> edge
        (Does not contain axes connectivity information.)
        """
        graph = nx.Graph()
        graph.add_nodes_from([(t.tid, {"degree": t.deg}) for t in self.tensors])
        graph.add_edges_from([b.tids for b in self.bonds])
        return graph

    def open_axes(self) -> list:
        """
        Return a list of open (uncontracted) axes in the form [(tid, i), ...]
        """
        od = []
        for t in self.tensors:
            od += [(t.tid, i) for i in range(t.deg)]
        for b in self.bonds:
            od.remove((b.tids[0], b.axes[0]))
            od.remove((b.tids[1], b.axes[1]))
        return od

    def is_consistent(self, verbose=False):
        """
        Consistency checks of the network contractions.
        """
        # range checks
        tids = [t.tid for t in self.tensors]
        for b in self.bonds:
            for j in range(2):
                if not b.tids[j] in tids:
                    if verbose: print("invalid bond tensor index {}".format(b.tids[j]))
                    return False
                i = tids.index(b.tids[j])
                if not (0 <= b.axes[j] < self.tensors[i].deg):
                    if verbose: print("bond axis for tensor {} out of range".format(b.tids[j]))
                    return False
        # no axis can appear twice in a contraction
        caxes = sorted([(b.tids[j], b.axes[j]) for j in range(2) for b in self.bonds])
        for i in range(len(caxes) - 1):
            if caxes[i] == caxes[i + 1]:
                if verbose: print("tensor-axis tuple ({}, {}) appears twice".format(caxes[i][0], caxes[i][1]))
                return False
        return True

    def simplify(self):
        """
        Drop all tensors which cancel, usually by a contraction of
        a ternay unitary gate with its conjugate-transposed version.
        """
        progressing = True
        while progressing:
            progressing = False
            for ta, tb in itertools.product(self.tensors, self.tensors):
                if ta != tb and isinstance(ta, SymbolicIsometricTensor) and isinstance(tb, SymbolicIsometricTensor):
                    # require conjugate-transposed copy
                    if ta.ct_id_ref == tb.tid or tb.ct_id_ref == ta.tid:
                        if self._cancel_isometric_tensor(ta, tb):
                            # need to start a new loop iteration after each cancellation
                            progressing = True
                            break
            if progressing:     # try isometry cancellations first
                continue
            for rule in self.rewirings:
                if rule.matches(self.bonds):
                    self._rewire(rule)
                    progressing = True

    def _cancel_isometric_tensor(self, t: SymbolicIsometricTensor, tdag: SymbolicIsometricTensor):
        """
        Try to cancel an isometric tensor with its conjugate-transposed copy,
        and update the connected bonds accordingly,
        assuming that all there are no open (uncontracted) axes.
        """
        # require mutual reference
        assert tdag.ct_id_ref == t.tid
        assert t.ct_id_ref == tdag.tid
        assert t.deg == tdag.deg
        assert t.isoaxes == tdag.isoaxes
        # search for matching contractions between the two tensors
        bondlist = []
        for b in self.bonds:
            # test whether bond connects the two tensors
            if sorted([t.tid, tdag.tid]) == sorted(b.tids):
                bondlist.append(b)
        for isoaxes in t.isoaxes:
            # search for matching axes contractions
            axbonds = len(isoaxes) * [None]
            for i, ax in enumerate(isoaxes):
                for b in bondlist:
                    if b.axes == [ax, ax]:
                        axbonds[i] = b
            if all(axbonds):
                # complementary axes
                complaxes = list(range(t.deg))
                # could also use set operations here
                for ax in isoaxes: complaxes.remove(ax)
                # shortcut bonds
                for i, ax in enumerate(complaxes):
                    bond_in  = self.find_bond(t.tid,    ax)
                    bond_out = self.find_bond(tdag.tid, ax)
                    # assume that all axes are connected by a bond
                    assert bond_in and bond_out, "encountered unexpected open (uncontracted) axis"
                    # ensure canonical ordering
                    if bond_in.tids[0]  == t.tid:    bond_in.flip()
                    if bond_out.tids[1] == tdag.tid: bond_out.flip()
                    assert bond_in.tids[1]  == t.tid
                    assert bond_out.tids[0] == tdag.tid
                    self._shortcut_bonds(bond_in, bond_out)
                # remove connecting bonds
                for b in axbonds:
                    self.bonds.remove(b)
                # remove tensors
                self.tensors.remove(t)
                self.tensors.remove(tdag)
                return True
        # cancellation not possible
        return False

    def _rewire(self, rule: SymbolicRewiringRule):
        """
        Perform a bond rewiring according to `rule`.
        """
        assert rule.matches(self.bonds)
        for b in rule.del_bonds:
            self.bonds.remove(b)
        for b in rule.add_bonds:
            self.bonds.append(b)

    def _shortcut_bonds(self, bond_in: SymbolicBond, bond_out: SymbolicBond):
        """
        "Shortcut" a bond sequence (ta, i) -- (tb, j); (tc, m) -- (td, n)
        by replacing it by (ta, i) -- (td, n),
        in the situation where the intermediate tensors `tb` and `tc` cancel.

        The bonds in the network are updated accordingly.
        """
        if bond_in == bond_out:
            # this is actually a loop ->
            # remove all bonds and increase loop counter
            self.bonds.remove(bond_in)
            self.identity_loops += 1
            return
        # re-route "in" bond
        bond_in.tids[1] = bond_out.tids[1]
        bond_in.axes[1] = bond_out.axes[1]
        # remove "out" bond
        self.bonds.remove(bond_out)

    def build_contraction_tree(self, scaffold) -> ContractionNode:
        """
        Build the contraction tree based on the contraction ordering in `scaffold`,
        which is a recursively nested list of IDs to specify the tree, e.g.,
        scaffold = [[ta, tb], [[tc, td], te]].
        """
        # search for next available tensor ID
        max_tid = 0
        for t in self.tensors:
            max_tid = max(max_tid, t.tid)
        return self._build_contraction_tree(scaffold, max_tid + 1)[0]

    def _build_contraction_tree(self, scaffold, next_tid) -> tuple:
        """
        Recursively build the contraction tree,
        starting from `next_tid` for generating intermediate tensor IDs.
        Returns a ContractionNode as root of the tree, and
        a list of open axes of the leaf tensors.
        """
        if isinstance(scaffold, int):   # leaf node
            t = self.find_tensor(scaffold)
            assert t, "tensor ID {} not found in network".format(scaffold)
            return ContractionNode(scaffold, None, [], None, [], list(range(t.deg))), [(scaffold, i) for i in range(t.deg)]
        assert isinstance(scaffold, list), "invalid `scaffold` argument"
        assert len(scaffold) == 2, "`scaffold` must specify pairwise contractions"
        # generate child nodes
        nL, openaxesL = self._build_contraction_tree(scaffold[0], next_tid)
        if nL.tid >= next_tid: next_tid = nL.tid + 1
        nR, openaxesR = self._build_contraction_tree(scaffold[1], next_tid)
        if nR.tid >= next_tid: next_tid = nR.tid + 1
        # search for bonds between leaf tensors of the left and right subtrees
        bondlist = []
        for odL in openaxesL:
            bond = self.find_bond(odL[0], odL[1])
            if bond:
                if (bond.tids[1], bond.axes[1]) == odL:
                    bond.flip()
                assert (bond.tids[0], bond.axes[0]) == odL
                if (bond.tids[1], bond.axes[1]) in openaxesR:
                    # found bond connecting subtrees
                    bondlist.append(bond)
        # contraction axes between left and right root tensors
        contraction_axes = []
        for bond in bondlist:
            openaxesL.remove((bond.tids[0], bond.axes[0]))
            openaxesR.remove((bond.tids[1], bond.axes[1]))
            contraction_axes.append([
                track_open_axis(find_contraction_node(nL, bond.tids[0]), bond.axes[0]),
                track_open_axis(find_contraction_node(nR, bond.tids[1]), bond.axes[1])])
        # tensor degrees
        deg = [len(nL.idxout), len(nR.idxout)]
        # contraction indices
        idxL = list(range(deg[0]))
        idxR = list(range(deg[0], deg[0] + deg[1]))
        idxout = idxL + idxR
        for a in contraction_axes:
            idxout.remove(idxL[a[0]])
            idxout.remove(idxR[a[1]])
            idxR[a[1]] = idxL[a[0]]
        return ContractionNode(next_tid, nL, idxL, nR, idxR, idxout), openaxesL + openaxesR

    def contractions_to_einsum(self, outlist):
        """
        Convert the contractions in the network to an `numpy.einsum` argument list,
        for a single call of `numpy.einsum`.

        Args:
            outlist:        ordered output dimensions, of the form [(ta, i), (tb, j), (tc, k), ...]

        Returns:
            tidx, idxout:   index argument list for `numpy.einsum`
        """
        # generate continuous indices for all tensors
        tids = [t.tid for t in self.tensors]
        tidx = []
        maxidx = 0
        for t in self.tensors:
            assert t.deg >= 0
            tidx.append(list(range(maxidx, maxidx + t.deg)))
            maxidx += t.deg
        # now identify to-be contracted indices
        for bond in self.bonds:
            # unpack bond
            a = tids.index(bond.tids[0])
            b = tids.index(bond.tids[1])
            i, j = bond.axes
            if tidx[a][i] < tidx[b][j]:
                tidx[b][j] = tidx[a][i]
            else:
                tidx[a][i] = tidx[b][j]
        # condense indices
        idxmap = maxidx * [-1]
        c = 0
        for i in range(len(tidx)):
            for j in range(len(tidx[i])):
                if idxmap[tidx[i][j]] == -1:
                    # use next available index
                    idxmap[tidx[i][j]] = c
                    c += 1
                tidx[i][j] = idxmap[tidx[i][j]]
        # generate output indices
        idxout = [tidx[tids.index(ta)][i] for (ta, i) in outlist]
        return tidx, idxout
