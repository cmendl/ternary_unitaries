import numpy as np
import networkx as nx


class ContractionNode(object):
    """
    Node in a contraction tree,
    following numpy.einsum's index specification convention
    for describing the contraction.
    """
    def __init__(self, tid, nL, idxL, nR, idxR, idxout):
        self.tid = tid                  # index of new tensor after contraction
        # indices as used by numpy.einsum; `idxL` and `idxR` can be empty in case this is a leaf node
        self.idxL = list(idxL)          # contraction indices for left child node tensor
        self.idxR = list(idxR)          # contraction indices for right child node tensor
        self.idxout = list(idxout)      # contraction indices for new (output) tensor
        self.parent = None              # parent node
        self.children = [nL, nR]        # left and right child nodes
        if nL: nL.parent = self
        if nR: nR.parent = self

    @property
    def is_leaf(self):
        return not any(self.children)

    def permute_axes(self, sort_indices):
        """
        Permute the axes of the tensor represented by the node.
        """
        self.idxout = [self.idxout[i] for i in sort_indices]
        if self.parent:
            parent = self.parent
            if self == parent.children[0]:      # whether left child
                parent.idxL = [parent.idxL[i] for i in sort_indices]
            elif self == parent.children[1]:    # whether right child
                parent.idxR = [parent.idxR[i] for i in sort_indices]
            else:
                assert False, "node not found among children of its parent"

    def as_graph(self) -> nx.Graph:
        """
        Construct graph representation of contraction tree.
        """
        graph = nx.Graph()
        self._build_graph(graph)
        return graph

    def _build_graph(self, graph: nx.Graph):
        """
        Helper function for recursive graph construction.
        """
        if self.is_leaf:
            graph.add_node(self.tid)
        else:
            self.children[0]._build_graph(graph)
            self.children[1]._build_graph(graph)
            graph.add_edge(self.tid, self.children[0].tid)
            graph.add_edge(self.tid, self.children[1].tid)


def find_contraction_node(root: ContractionNode, tid: int):
    """
    Search for the contraction node with tensor ID `tid`
    within tree with root node `root`.
    """
    if root.tid == tid:
        return root
    for c in root.children:
        if c:
            node = find_contraction_node(c, tid)
            if node:
                return node
    # not found
    return None


def track_open_axis(node: ContractionNode, i: int) -> int:
    """
    Track the i-th axis of the current node upstream to the root node of the tree.
    """
    # check if we have already reached the root node
    if not node.parent:
        return i
    parent = node.parent
    if node == parent.children[0]:      # whether left child
        return track_open_axis(parent, parent.idxout.index(parent.idxL[i]))
    elif node == parent.children[1]:    # whether right child
        return track_open_axis(parent, parent.idxout.index(parent.idxR[i]))
    else:
        assert False, "node not found among children of its parent"


def perform_contraction(node: ContractionNode, tensor_dict) -> np.array:
    """
    Perform contraction as specified by contraction tree with root `node`.
    """
    if node.is_leaf:
        return tensor_dict[node.tid]
    assert node.children[0] and node.children[1], "both child nodes must be set"
    tL = perform_contraction(node.children[0], tensor_dict)
    tR = perform_contraction(node.children[1], tensor_dict)
    return np.einsum(tL, node.idxL, tR, node.idxR, node.idxout)
