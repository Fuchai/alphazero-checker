
# load the model, make decision
from alphabeta import AlphaBeta
from mcts import MCTS


class Agent:
    def respond(self, opponent_state):
        """
        Override this method

        :param opponent_state: the opponent resulting state
        :return: agent action
        """
        action=None
        return action

class MinimaxAgent(Agent, AlphaBeta):
    def __init__(self, first, *args, **kwargs):
        super(MinimaxAgent, self).__init__(human_first= not first, human_second=first, *args, **kwargs)
        current_round=0

    def respond(self, opponent_state):
        """
        takes a unflipped opponent state
        responds, do not flip back
        :param opponent_state:
        :return:
        """
        self.human_play(opponent_state)
        ret = self.auto_play()
        self.state.print_board()
        return self.state

    def versus(self):

        for i in range(self.max_rounds):
            ret=True
            if i%2==0:
                if self.human_first:
                    self.verbose_human_play()
                else:
                    ret=self.auto_play()
                    print("Alpha played")
                    self.state.get_flipped_state().print_board(player_view=False)
            else:
                if self.human_second:
                    self.verbose_human_play()
                else:
                    ret=self.auto_play()
                    print("Beta played")
                    self.state.get_flipped_state().print_board(player_view=False)
            if not ret:
                print("Game finished")
                break
        if ret:
            print("Game reached max rounds")
        self.print_winner()

class NeuralAgent(Agent):
    def __init__(self,mcts):
        self.mcts=mcts
        self.step=0
        self.mcts.simulations_per_play=2

    def respond(self, opponent_state):
        """
        similar to play_until_terminal except this only play once
        :param opponent_state: opponent's resulting state
        :return: agent's resulting state
        """
        # move permaTree root according to opponent's move
        self.run_simulation()
        found_child=self.mcts.permaTree.root.find_child(opponent_state)
        self.mcts.permaTree.move_root(found_child)
        # simulation
        # step as input?
        # if self.step == self.mcts.temperature_change_at:
        #     self.mcts.temperature = False
        # for simulation in range(self.mcts.simulations_per_play):
        #     self.mcts.simulation()
        self.run_simulation()
        terminal = self.mcts.play()
        if terminal:
            print("Terminated at step", self.step)
        # return root aka board state after agent's action
        self.step+=2
        return self.mcts.permaTree.root.checker_state, terminal

    def run_simulation(self):
        if self.step == self.mcts.temperature_change_at:
            self.mcts.temperature = False
        for simulation in range(self.mcts.simulations_per_play):
            self.mcts.simulation()