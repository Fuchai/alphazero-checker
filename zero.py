
class Zero:
    def __init__(self):
        pass
    # first of all, I think we still need a policy network independently from value. Somehow I think this is the best
    # We run MCTS until the end
    # We then sample the MCTS timesteps to train the neural network (how many times?)
    # Then we repeat. (with new MCTS tree? right?)
    # We need to design the asynchronous behavior.

