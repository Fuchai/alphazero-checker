import checker
import math
import numpy as np
# the permanent tree data structure
import numpy as np
import math


class PermaTree:
    def __init__(self, root):
        self.root = root

    def go(self):
        # what is run?
        self.root.run()

    def choose(self, state):
        # move from root to a immediate child
        # update parent to None
        self.root = state
        self.root.parent = None


class Edge():
    # update edge's value by just changing edge.value(?)

    def __init__(self, from_state, action, prior_probability):
        # a State object for where the action comes from
        self.from_state = from_state
        self.to_state = None  # create new child state in expand() and update this
        # a reference to the checker action, not MCTS specific, can be anything
        # these values are updated in backup()
        self.action = action
        self.visit_count = 0
        self.action_value = 0
        self.mean_action_value = 0
        self.prior_probability = prior_probability

    def get_stats(self):
        return self.visit_count, self.action_value, self.mean_action_value, self.prior_probability

    def update_parent(self, childstate):
        # called in expand()
        self.to_state = childstate


class State():
    def __init__(self, checker_state, parent=None):
        # adjacency list implementation
        # every element in self.edges is an Edge object
        self.checker_state = checker_state
        self.parent = parent  # parent is an edge, None when root
        self.edges = []
        board = self.get_board()
        # call get_legal_actions from checker
        actions, _ = board.get_legal_actions()

        # init and add edges into state
        for action in actions:
            edge = Edge(self, action, None)  # prior_prob will be updated in expand()
            self.edges.append(edge)
            # or self.edges+[pack_action_into_edge(action)]
            # another append syntax

    def get_board(self):
        return self.checker_state.board

    def get_edges(self):
        return self.edges


# tree search does not check same state
# algorithm
# Input parameters: state (; node of a tree), tree structure, neural net
class MCTS():
    def __init__(self, permaTree, nn):
        self.permaTree = permaTree
        self.nn = nn
        self.temperature = 123  # needs to be changed to correct val
        self.puct = 0.1

    def simulation(self):
        pass

    def select(self, node):

        # argmax_input_list = contains a list of [Q(s,a)+U(s,a)] vals of all outbound edges of a node
        argmax_input_list = []
        # every node/state should have a list of next possible actions
        outbound_edge_list = node.edges
        # loop through all the child edges of a node
        for edge in outbound_edge_list:
            Nsa = edge.visit_count
            Nsb = sum(edge.sibling_visit_count_list)  # does Nsb mean edge sibling visit count ? <verify>
            # u(s,a) = controls exploration
            Usa = (self.puct*edge.prior_probability*math.sqrt(Nsb))/(1+Nsa)
            # add to argmax_input_list
            argmax_input_list.append(edge.mean_action_value + Usa)

        # pick the edge that is returned by the argmax and return it
        select_action = np.argmax(argmax_input_list)
        return select_action

    # expand and evaluate the leaf node sl of the selected edge
    # input parameter: a node
    def expand(self, node):
        # Let sl be the selected node
        # get p,v from neural network
        p = NeuralNetwork.p
        v = NeuralNetwork.v
        outbound_edge_list = node.edges
        for edges in outbound_edge_list:
            edges.Nsa = 0
            edges.action_value = 0
            edges.mean_action_value = 0
            edges.prior_probability = v
            # update perma tree with current edge
            PermaTree(edges)

    # trace back the whole path from given node till root node while updating edges on the path
    def backup(self, node):
        # parent of root node is null
        while parent.node != null:
            edge = parent.node
            edge.visit_count = edge.visit_count + 1
            edge.action_value = edge.action_value + v
            edge.mean_action_value = edge.action_value / edge.visit_count


class NeuralNetwork():
    def __call__(self, state):
        """
        the (p,v)=f_theta(s) function
        f_theta=NeuralNetwork()
        p,v=f_theta(state)
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
    state = None
    p, v = f_theta(state)
