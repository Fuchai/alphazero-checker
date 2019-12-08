from alphabeta import *
from checker import *
from performance import *
from zero import *


def human_first_play():
    depth=5
    ab = AlphaBeta(0, depth, human_first=True)
    ab.start_game()

def human_second_play():
    depth=5
    ab = AlphaBeta(depth, 0, human_second=True)
    ab.start_game()

def play_with_your_friend():
    ab = AlphaBeta(0, 0, human_first=True, human_second=True)
    ab.start_game()

def machine_versus_machine():
    # first player (alpha) probably wins
    alpha_depth=5
    beta_depth=3
    ab = AlphaBeta(alpha_depth, beta_depth, max_rounds=300)
    ab.start_game()

def alphazero_versus_alphazero():
    # first player is AlphaZero
    # second player is AlphaBeta

    # load in neural network
    az1=alphazero_factory("alternate",28)
    az2=alphazero_factory("alternate",35)

    # play game
    print("Current board")
    az1.mcts.print_root()
    action = az1.mcts.permaTree.root
    for i in range(az1.mcts.max_game_length):
        ret = True
        if i % 2 == 0:
            action, end = az1.respond(action)
            ret = not end
            print("Zero played")
            az1.print_root()
        else:
            response = az2.respond(action)
            print("Beta played")
            # flip it for alphago because it does not know how to flip it
            action = response.get_flipped_state()
            action.print_board(player_view=False)
        if not ret:
            print("Game finished")
            break
    if ret:
        print("Game reached max rounds")
    az1.print_winner()

def alphazero_factory(model_name="alternate",epoch=28):
    # load in neural network
    alphazero = AlphaZero(model_name, is_cuda=False)
    alphazero.scale = 1
    alphazero.starting_epoch = epoch
    alphazero.load_model()
    nn_thread_edge_queue = queue.Queue(maxsize=alphazero.max_queue_size)
    mcts = MCTS(nn_thread_edge_queue, nn=alphazero.nn, is_cuda=False,
                max_game_length=alphazero.max_game_length, peace=alphazero.peace,
                simulations_per_play=alphazero.simulations_per_play, debug=False)

    # construct agents
    agz = NeuralAgent(mcts)
    return agz

def alphazero_versus_alphabeta():
    # first player is AlphaZero
    # second player is AlphaBeta

    # load in neural network
    alphazero = AlphaZero("alternate", is_cuda=False)
    alphazero.scale=1

    nn_thread_edge_queue = queue.Queue(maxsize=alphazero.max_queue_size)
    mcts = MCTS(nn_thread_edge_queue, nn = alphazero.nn, is_cuda = False,
                max_game_length = alphazero.max_game_length, peace = alphazero.peace,
                simulations_per_play = alphazero.simulations_per_play, debug=False)

    # construct agents
    agz = NeuralAgent(mcts)
    ab = MinimaxAgent(first=False)

    # play game
    print("Current board")
    agz.mcts.print_root()
    action = agz.mcts.permaTree.root
    for i in range(agz.mcts.max_game_length):
        ret = True
        if i % 2 == 0:
            action, end = agz.respond(action)
            ret = not end
            print("Zero played")
            agz.print_root()
        else:
            response = ab.respond(action)
            print("Beta played")
            # flip it for alphago because it does not know how to flip it
            action=response.get_flipped_state()
            action.print_board(player_view=False)
        if not ret:
            print("Game finished")
            break
    if ret:
        print("Game reached max rounds")
    ab.print_winner()

if __name__ == '__main__':
    alphazero_versus_alphazero()