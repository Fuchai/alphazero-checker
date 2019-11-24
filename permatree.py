# the permanent tree data structure
class PermaTree:
    def __init__(self, checker):
        self.root=PermaNode(checker.state)

    def move_root(self, node):
        # move from root to a immediate child
        # update parent to None
        self.root = node
        self.root.parent = None

    # def update(self, edge):
    #     pass
    #
    # def update(self, node):
    #     pass


class PermaEdge:
    """
    Guarantees that from_node and to_node are not None
    """

    def __init__(self, perma_tree, action, from_node):
        self.perma_tree = perma_tree
        # an Action object of the checker program
        self.action = action
        # a Node object for where the action comes from
        self.from_node = from_node
        # initialize node whenever an edge is created, guarantees the data structure property
        self.to_node = PermaNode(perma_tree, action, self.from_node, self)
        # self.to_node = None  # create new child node in expand() and update this

        # # these values are initialized at expand() and updated in backup()
        self.prior_probability = None
        self.visit_count = None
        self.total_action_value = None
        self.mean_action_value = None
        self.value_when_expanded = None
        # side effect, initialize the node that the edge points to


class PermaNode:
    """
    Guarantees that from_edge is not None. May not have self.edges
    """

    def __init__(self, perma_tree, checker_state, parent=None, from_edge=None):
        self.perma_tree = perma_tree
        # every element in self.edges is an Edge object
        self.checker_state = checker_state
        self.parent = parent  # parent is an edge, None when root
        self.from_edge = from_edge
        # adjacency list implementation
        self.edges = []

    def is_leaf(self):
        return len(self.edges) == 0

    def construct_edges(self):
        # call get_legal_actions from checker
        actions, _ = self.checker_state.get_legal_actions()

        # init and add edges into node
        for action in actions:
            new_edge = PermaEdge(self.perma_tree, action, self)  # prior_prob will be updated in expand()
            self.edges.append(new_edge)

    def expand_values_get_prior_prob(self, nn):
        values=[]
        for edge in self.edges:
            val=edge.init_value(nn)
            values.append(val)
        return values

    def get_board(self):
        return self.checker_state.board
