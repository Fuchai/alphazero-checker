from checker import Checker
import numpy as np
import math
from permatree import PermaTree

# tree search does not check same node
# algorithm
# Input parameters: node (; node of a tree), tree structure, neural net
class MCTS:
    """
    Selects from root until a leaf node. A leaf node denotes a leaf node in the PermaTree, not the end game (ref Zero
    page 2 right column paragraph 1).
    """

    def __init__(self, nn):
        self.checker=Checker()
        self.permaTree = PermaTree(self.checker)
        self.nn = nn
        self.temperature = 1
        self.puct = 0.1

    def run(self):
        terminal=False
        while not terminal:
            simulations_per_play=1600
            for epoch in range(simulations_per_play):
                self.simulation()
            self.play()


    def simulation(self):
        """
        run select, expand, backup L-times on a root node
        :return: L: the length of simulation from root to leaf node
        """
        current_node = self.permaTree.root
        l = 1
        while not current_node.is_leaf():
            selected_edge = self.select(current_node)
            selected_node = selected_edge.to_node
            current_node = selected_node
            l+=1
        self.expand(selected_node)
        self.backup(selected_node)
        return l

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
        """
        select function is called from the root to the leaf node in perma_tree for each simulation of L simulations
        :param node:
        :return:
        """
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

    def expand(self, leaf_node):
        """
        expand and evaluate the leaf node sl of the selected edge
        value and prior probability are assigned
        :param leaf_node:
        :return:
        """

        # will call assign value for each child
        leaf_node.construct_edges(self.nn)

        # initialize basic statistics
        for idx, edge in enumerate(leaf_node.edges):
            edge.visit_count = 0
            edge.action_value = 0
            edge.mean_action_value = 0
            # update perma tree with current edge
            # self.permaTree.update(edge)

        # initialize value of all children
        children_values=leaf_node.expand_values_get_prior_prob()

        # initialize the prior probability of all children
        p = self.nn.children_values_to_probability(children_values)
        # assert that all edges must not be shuffled
        edge.prior_probability = p[idx]

    def backup(self, leaf_node):
        """
        trace back the whole path from given node till root node while updating edges on the path

        :param leaf_node:
        :return:
        """
        # parent of root node is null
        assert (leaf_node.is_leaf())
        current_node = leaf_node
        while current_node.parent is not None:
            edge = current_node.from_edge
            edge.visit_count = edge.visit_count + 1
            edge.total_action_value = edge.total_action_value + v
            edge.mean_action_value = edge.total_action_value / edge.visit_count
            current_node = edge.from_node

if __name__ == "__main__":
    edges = [1, 2, 3, 4]
    for idx, edge in enumerate(edges):
        print(edge)
        print(edges[idx])

    f_theta = NeuralNetwork()
    node = None
    p, v = f_theta(node)
