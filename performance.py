
# load the model, make decision
from alphabeta import AlphaBeta


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