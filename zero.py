import checker
# the permanent tree data structure
class PermaTree:
    def __init__(self,root):
        self.root=root

    def go(self):
        # what is run?
        self.root.run()

    def choose(state):
        # move from root to a immediate child
        # update parent to None
        self.root=state
        root.parent=None


class Edge():
# update edge's value by just changing edge.value(?)

    def __init__(self, from_state, action, prior_probability):
        # a State object for where the action comes from
        self.from_state = from_state
        self.to_state = None # create new child state in expand() and update this
        # a reference to the checker action, not MCTS specific, can be anything
        # these values are updated in backup()
        self.action =  action
        self.visit_count = 0
        self.action_value = 0
        self.mean_action_value = 0
        self.prior_probability = prior_probability

    def get_stats(self):
        return self.visit_count, self.action_value, self.mean_action_value, self.prior_probability

    def update_parent(self,childstate):
        # called in expand()
        self.to_state = childstate


class State():
    def __init__(self,checker_state,parent=None):
        # adjacency list implementation
        # every element in self.edges is an Edge object
        self.checker_state = checker_state
        self.parent = parent # parent is an edge, None when root
        self.edges=[]
        board = self.get_board()
        # call get_legal_actions from checker
        actions, _ = self.board.get_legal_actions()

        # init and add edges into state
        for action in actions:
            edge=Edge(self,action,None) # prior_prob will be updated in expand()
            self.edges.append(edge)
            # or self.edges+[pack_action_into_edge(action)]
            # another append syntax

    def get_board(self):
        return self.checker_state.board

    def get_edges(self):
        return self.edges


# tree search does not check same state
# algorithm
class MCTS():
    def __init__(self, permaTree, nn):
        self.permaTree=permaTree
        self.nn=nn
        self.temperature=123
        self.puct=0.1

    def select(self):
        # first calculate the Q(s,a) and U(s,a)
        # a MCTS call corresponds to a path?

        # for example, pick the (s,a) with the highest U, if that is the correct heurisitics
        highest=-float.inf()
        highestsa=None
        Usa = self.puct*
        for s,a in satuples:
            if highest < U(s,a):
                highest=U(s,a)
                highestsa=(s,a)
        s,a=highestsa

        return s,a

    def U(s,a):
        pass

    def expand(self):
        pass

    def backup(self):
        pass




if __name__=="__main__":
    edges=[1,2,3,4]
    for idx,edge in enumerate(edges):
        print(edge)
        print(edges[idx])

    f_theta=NeuralNetwork()
    state=None
    p,v=f_theta(state)
