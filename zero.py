import checker
import math
import numpy as np
# the permanent tree data structure
import numpy as np
import math


class PermaTree:
    def __init__(self):
        self.root = None

    def set_root(self, root):
        self.root = root

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


class Edge:
    """
    Guarantees that from_node and to_node are not None
    """

    # update edge's value by just changing edge.value(?)

    def __init__(self, from_node, action, perma_tree, prior_probability=None):
        # stipuate the property that whenever there is an edge, there must be a
        # destination node
        # a Node object for where the action comes from
        self.from_node = from_node
        # initialize node
        self.to_node = Node(action, perma_tree, self.from_node, self)
        # self.to_node = None  # create new child node in expand() and update this
        # # a reference to the checker action, not MCTS specific, can be anything
        # # these values are updated in backup()
        self.action = action
        self.visit_count = None
        self.action_value = None
        self.mean_action_value = None
        self.prior_probability = prior_probability
        self.v = None

        self.perma_tree = perma_tree
        # side effect, initialize the node that the edge points to

    def get_stats(self):
        return self.visit_count, self.action_value, self.mean_action_value, self.prior_probability

    def update_parent(self, childnode):
        # called in expand()
        self.to_node = childnode

    def assign_value(self, nn):
        self.v = nn(self.to_node)


class Node:
    """
    Guarantees that from_edge is not None. May not have self.edges
    """

    def __init__(self, checker_state, perma_tree, parent=None, from_edge=None):
        # adjacency list implementation
        # every element in self.edges is an Edge object
        self.checker_state = checker_state
        self.parent = parent  # parent is an edge, None when root
        self.from_edge = from_edge
        # self.parent_node=parent_node
        self.edges = []
        self.board = self.get_board()
        # call get_legal_actions from checker
        # actions, _ = board.get_legal_actions()
        self.perma_tree = perma_tree

        # # init and add edges into node
        # for action in actions:
        #     edge = Edge(self, action, None)  # prior_prob will be updated in expand()
        #     self.edges.append(edge)
        #     # or self.edges+[pack_action_into_edge(action)]
        #     # another append syntax

    def is_leaf(self):
        return len(self.edges) == 0

    def construct_edges(self, nn):
        actions, _ = self.checker_state.get_legal_actions()

        # init and add edges into node
        for action in actions:
            new_edge = Edge(self, action, None)  # prior_prob will be updated in expand()
            new_edge.assign_value(nn)
            self.edges.append(new_edge)
            # or self.edges+[pack_action_into_edge(action)]
            # another append syntax

    def get_board(self):
        return self.checker_state.board

    def get_edges(self):
        return self.edges


# TODO deadend
# tree search does not check same node
# algorithm
# Input parameters: node (; node of a tree), tree structure, neural net
class MCTS:
    def __init__(self, nn):
        self.permaTree = PermaTree()
        self.nn = nn
        self.temperature = 123  # needs to be changed to correct val
        self.puct = 0.1

    def run(self):
        pass

    # run select, expand, backup L-times on a root node
    def simulation(self, root):
        total_epochs = 100
        for epoch in range(total_epochs):
            current_node = root
            L = 1600
            for l in range(L):
                selected_edge = self.select(current_node)
                selected_node = selected_edge.to_node
                if selected_node.is_leaf():
                    self.expand(selected_node)
                    self.backup(selected_node)
                    current_node = root
                else:
                    current_node = selected_node
                    # if dead end ?
            self.play()

    def play(self):
        root = self.permaTree.root
        scores = []
        for level_one_edge in root.edges:
            vc = level_one_edge.visit_count
            scores.append(self.temperature_adjust(vc))
        sum_socres = sum(scores)
        scores = [score / sum_socres for score in scores]
        best_action = np.argmax(scores)
        self.permaTree.move_root(best_action.to_node)

    def temperature_adjust(self, count):
        return count ** (1 / self.temperature)

    def select(self, node):

        # argmax_input_list = contains a list of [Q(s,a)+U(s,a)] vals of all outbound edges of a node
        argmax_input_list = []
        # every node/node should have a list of next possible actions
        outbound_edge_list = node.edges
        # loop through all the child edges of a node
        for edge in outbound_edge_list:
            Nsa = edge.visit_count
            Nsb = sum(edge.sibling_visit_count_list)  # does Nsb mean edge sibling visit count ? <verify>
            # u(s,a) = controls exploration
            Usa = (self.puct * edge.prior_probability * math.sqrt(Nsb)) / (1 + Nsa)
            # add to argmax_input_list
            argmax_input_list.append(edge.mean_action_value + Usa)

        # pick the edge that is returned by the argmax and return it
        # make it node
        selected_action = np.argmax(argmax_input_list)
        return selected_action

    # expand and evaluate the leaf node sl of the selected edge
    # input parameter: a node
    # return the leaf node
    def expand(self, node):
        # Let sl be the selected node
        # get p,v from neural network
        # p = NeuralNetwork.p
        # v = NeuralNetwork.v
        # add
        # will call assign value for each child
        node.construct_edges(self.nn)
        # values should not be None
        p = self.nn.children_values_to_probability(children_value_tensors)
        # assert that all edges must not be shuffled

        outbound_edge_list = node.edges
        for idx, edge in enumerate(outbound_edge_list):
            edge.visit_count = 0
            edge.action_value = 0
            edge.mean_action_value = 0
            edge.prior_probability = p[idx]
            # update perma tree with current edge
            # self.permaTree.update(edge)

    # trace back the whole path from given node till root node while updating edges on the path
    def backup(self, leaf_node):
        # parent of root node is null
        assert (leaf_node.is_leaf())
        current_node = leaf_node
        while current_node.parent is not None:
            edge = current_node.from_edge
            edge.visit_count = edge.visit_count + 1
            edge.action_value = edge.action_value + v
            edge.mean_action_value = edge.action_value / edge.visit_count
            current_node = edge.from_node


class NeuralNetwork:
    def __call__(self, node):
        """
        the (p,v)=f_theta(s) function
        f_theta=NeuralNetwork()
        p,v=f_theta(node)
        """
        p = 1
        v = 2
        print("Neural network is run")
        return p, v


if __name__ == "__main__":
    edges = [1, 2, 3, 4]
    for idx, edge in enumerate(edges):
        print(edge)
        print(edges[idx])

    f_theta = NeuralNetwork()
    node = None
    p, v = f_theta(node)
