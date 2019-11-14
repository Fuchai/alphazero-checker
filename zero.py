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
class MCTS():
    def __init__(self, permaTree, nn):
        self.permaTree=permaTree
        self.nn=nn
        self.temperature=123
        self.puct=0.1

    def select():
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

    def expand():
        pass

    def backup():
        pass


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
