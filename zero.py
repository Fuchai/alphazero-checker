import checker
# the permanent tree data structure
class PermaTree:
    def __init__(self,root):
        self.root=root

    def go():
        self.root.run()

    def choose(state):
        # move from root to a immediate child
        pass


class Edge():
    def __init__(self, from_state, to_state, action, prior_probability):
        # a State object for where the action comes from
        self.from_state=from_state
        self.to_state=to_state
        # a refernce to the checker action, not MCTS specific, can be anything
        self.action=action
        self.visit_count=None
        self.action_value=None
        self.mean_action_value=None
        # does the initialization take the prior_probability?
        self.prior_probability=prior_probability

class State():
    def __init__(self,board):
        self.board=board
        # adjacency list implementation
        # every element in self.edges is a Edge object
        self.edges=[] # call get_legal_actions from checker
        actions, _ =self.board.get_legal_actions()
        for action in actions:

            edge=Edge(self,None,action,)
            self.edges.append(pack_action_into_edge(action))
            # or self.edges+[pack_action_into_edge(action)]
            # another append syntax

    def pack_action_into_edge(action):
        return Edge(self,action)


# tree search does not check same state
# algorithm
# Input parameters: state (; node of a tree), tree structure, neural net
class MCTS():
    def __init__(self, permaTree, nn):
        self.permaTree=permaTree
        self.nn=nn
        self.temperature=123 # needs to be changed to correct val
        self.puct=0.1

    def select():

        # argmax_input_list = contains a list of [Q(s,a)+U(s,a)] vals of all outbound edges of a node
        argmax_input_list = []
        # every node/state should have a list of next possible actions
        outbound_edge_list = self.edges
        # loop through all the child edges of a node
        for edge in outbound_edge_list:
            Nsa = edge.visit_count
            Nsb = sum(edge.sibling_visit_count_list) # does Nsb mean edge sibling visit count ? <verify>
            # u(s,a) = controls exploration
            Usa = (self.puct*edge.prior_probability*math.sqrt(Nsb))/(1+edge.visit_count)
            # add to argmax_input_list
            argmax_input_list.append(edge.mean_action_value + Usa)

        # pick the edge that is returned by the argmax and return it
        select_action = numpy.argmax(argmax_input_list)
        return select_action

    # expand and evaluate the leaf node sl of the selected edge
    # input parameter: a node
    def expand(node):
        # Let sl be the selected node
        # get p,v from neural network
        p = NeuralNetwork.p
        v = NeuralNetwork.v
        outbound_edge_list = sl.edges
        for edges in outbound_edge_list:
            edges.Nsa = 0
            edges.action_value = 0
            edges.mean_action_value = 0
            edges.prior_probability = v
            # update perma tree with current edge
            PermaTree(edges)


    # trace back the whole path from given node till root node while updating edges on the path
    def backup(node):

        # parent of root node is null
        while parent.node != null:
            edge = parent.node
            edge.visit_count = edge.visit_count + 1
            edge.action_value =  edge.action_value + v
            edge.mean_action_value = edge.action_value/edge.visit_count



class NeuralNetwork():
    def __call__(self,state):
        """
        the (p,v)=f_theta(s) function
        f_theta=NeuralNetwork()
        p,v=f_theta(state)
        """
        p=1
        v=2
        print("Neural network is run")
        return p,v



if __name__=="__main__":
    edges=[1,2,3,4]
    for idx,edge in enumerate(edges):
        print(edge)
        print(edges[idx])

    f_theta=NeuralNetwork()
    p,v=f_theta(state)
