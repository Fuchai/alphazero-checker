# first of all, I think we still need a policy network independently from value. Somehow I think this is the best
# We run MCTS until the end
# We then sample the MCTS timesteps to train the neural network (how many times?)
# Then we repeat. (with new MCTS tree? right?)
# We need to design the asynchronous behavior.

from mcts import MCTS, TimeStep
from neuralnetwork import NoPolicy, PaperLoss
import random
import torch
import torch.optim
import torch.nn as nn
import os
from pathlib import Path
from os.path import abspath
import datetime
from neuralnetwork import states_to_batch_tensor
import numpy as np
import time
import queue
import threading


class AlphaZero:
    """
    Use MCTS to generate g number of games
    Train based on these games with repeatedly sampled time steps
    Refresh the g number of games, repeat
    """

    def __init__(self, model_name, is_cuda=True):
        self.model_name = model_name
        self.nn = NoPolicy()
        self.is_cuda = is_cuda
        if self.is_cuda:
            self.nn = self.nn.cuda()
        # TODO
        self.loss_fn = PaperLoss()
        self.optim = torch.optim.Adam(self.nn.parameters(), weight_decay=0.01)
        # control how many time steps each loss.backwards() is called for.
        # controls the GPU memory allocation
        self.training_time_step_batch_size = 16
        # controls how many boards is fed into a neural network at once
        # controls the speed of gpu computation.
        self.neural_network_batch_size = 128
        # time steps contain up to self.game_size different games.
        self.time_steps = []
        self.game_size = 2
        self.total_epochs = 20
        self.train_period = 2000
        self.validation_period = 100
        self.print_period = 10
        self.save_period = 1000
        self.log_file = "log/" + self.model_name + "_" + datetime_filename() + ".txt"
        self.refresh_period = 10
        self.simulations_per_play = 200
        # this is a tuned parameter, do not change
        self.eval_batch_size = 25600 // self.simulations_per_play
        self.debug = True
        self.max_queue_size = self.eval_batch_size * 2

    def mcts_refresh_game(self):
        with torch.no_grad():
            self.nn.eval()
            self.time_steps = []
            for i in range(self.game_size):
                nn_thread_edge_queue = queue.Queue(maxsize=self.max_queue_size)
                # def gpu_thread_worker(nn, queue, eval_batch_size, is_cuda):
                gpu_thread = threading.Thread(target=gpu_thread_worker,
                                              args=(self.nn, nn_thread_edge_queue, self.eval_batch_size, self.is_cuda))
                gpu_thread.start()
                mcts = MCTS(nn_thread_edge_queue, self.is_cuda, self.simulations_per_play, self.debug)
                mcts.play_until_terminal()
                nn_thread_edge_queue.put(None)
                print("Terminal sentinel is put on queue")
                nn_thread_edge_queue.join()
                if self.debug:
                    print("Queue has joined")
                gpu_thread.join()
                if self.debug:
                    print("Thread has joined")
                self.time_steps += mcts.time_steps
                print("Successful generation of one game")
                print("Queue empty:", nn_thread_edge_queue.empty())

    def train(self):
        for epoch in range(self.total_epochs):
            for ti in range(self.train_period):
                if ti % self.refresh_period == 0:
                    self.mcts_refresh_game()
                train_loss = self.train_one_round()
                if ti % self.print_period == 0:
                    self.log_print(
                        "%14s " % self.model_name +
                        "train epoch %4d, batch %4d. running loss: %.5f" %
                        (epoch, ti, train_loss))
                if ti % self.validation_period == 0:
                    self.validation()

    def run_one_round(self):
        pass

    def train_one_round(self):
        self.nn.train()
        # sample self.batch_size number of time steps, bundle them together
        batch_tss = random.choices(self.time_steps, k=self.training_time_step_batch_size)

        # compile value tensor
        value_batches = len(batch_tss) // self.neural_network_batch_size
        values_outputs = []
        for batch_idx in range(value_batches):
            value_inputs = batch_tss[
                batch_idx * self.neural_network_batch_size, (batch_idx + 1) * self.neural_network_batch_size]
            value_inputs = [ts.checker_state for ts in value_inputs]
            value_tensor = states_to_batch_tensor(value_inputs, is_cuda=self.is_cuda)
            value_output = self.nn(value_tensor)
            values_outputs.append(value_output)
        # TODO is this function call correct?
        value_output = torch.cat(values_outputs, dim=0)

        # compile policy tensor
        # queue up children_states
        # slice output tensor
        tss_policy_output = {}
        #
        policy_inputs_queue = []
        dim_ts = []
        for ts in batch_tss:
            tss_policy_output[ts] = []
            for child in ts.children_states:
                if len(policy_inputs_queue) != self.neural_network_batch_size:
                    # queue up
                    dim_ts.append(ts)
                    policy_inputs_queue.append(child)
                else:
                    ### process
                    self.policy_bonanza(policy_inputs_queue, dim_ts, tss_policy_output)
                    policy_inputs_queue = []
                    dim_ts = []
        # remnant in the queue
        self.policy_bonanza(policy_inputs_queue, dim_ts, tss_policy_output)

        # policy transpose
        loss=0
        for ts in batch_tss:
            logits=torch.cat(tss_policy_output[ts])
            p=self.nn.logits_to_probability(logits)
            loss += self.loss_fn(ts.z, target)
        loss.backward()
        self.optim.step()
        return loss.data

    def policy_bonanza(self, policy_inputs_queue, dim_ts, tss_policy_output):
        policy_tensor = states_to_batch_tensor(policy_inputs_queue)
        policy_output = self.nn.policy_logit(policy_tensor)
        # slice and append
        last_ts = None
        tsbegin = None
        for tsidx, ts in enumerate(dim_ts):
            if ts != last_ts:
                if last_ts is not None:
                    # slice the policy output
                    # TODO check the slicing operation
                    tss_policy_output[ts].append(policy_output[tsbegin: ts, :, :])
                last_ts = ts
                tsbegin = ts
        # take care of the last ones
        tss_policy_output[ts].append(policy_output[tsbegin: ts, :, :])

    def validation(self):
        with torch.no_grad():
            self.nn.eval()
            validation_time_steps = []
            for i in range(self.game_size):
                mcts = MCTS(self.nn)
                mcts.play_until_terminal()
                validation_time_steps += mcts.time_steps
            self.batch_tss = random.choices(validation_time_steps, k=self.training_time_step_batch_size)
            input, target = self.time_steps_to_tensor(self.batch_tss)
            output = self.nn(input)
            loss = self.loss_fn(output, target)
            return loss.data

    def time_steps_to_tensor(self, batch_tss):
        # what is the strategy for training time neural network call?
        # I suppose we should do one pass on all values, then we do passes over the probabilities
        # sometimes a batch represent different time steps
        # sometimes a batch represents the same time step but different children of the time step
        # cast and view bonanza.
        # keep track of the loss coefficient.
        # beware not to keep too many boards in the memory at once: when doing probability, do n boards at once

        # compatible with our next design that the probability head is independent from value head, but no other
        # design compatibility

        return None

    def log_print(self, message):
        string = str(message)
        if self.log_file is not None and self.log_file != False:
            with open(self.log_file, 'a') as handle:
                handle.write(string + '\n')
        print(string)

    def save_model(self, epoch, iteration):

        epoch = int(epoch)
        task_dir = os.path.dirname(abspath(__file__))
        if not os.path.isdir(Path(task_dir) / "saves"):
            os.mkdir(Path(task_dir) / "saves")

        pickle_file = Path(task_dir).joinpath(
            "saves/" + self.model_name + "_" + str(epoch) + "_" + str(iteration) + ".pkl")
        with pickle_file.open('wb') as fhand:
            torch.save((self.nn, self.optim, epoch, iteration), fhand)

        print("saved model", self.model_name, "at", pickle_file)

    def load_model(self, computer, optim, starting_epoch, starting_iteration, model_name):
        task_dir = os.path.dirname(abspath(__file__))
        save_dir = Path(task_dir) / "saves"
        highest_epoch = 0
        highest_iter = 0

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        for child in save_dir.iterdir():
            if child.name.split("_")[0] == model_name:
                epoch = child.name.split("_")[1]
                iteration = child.name.split("_")[2].split('.')[0]
                iteration = int(iteration)
                epoch = int(epoch)
                # some files are open but not written to yet.
                if child.stat().st_size > 20480:
                    if epoch > highest_epoch or (iteration > highest_iter and epoch == highest_epoch):
                        highest_epoch = epoch
                        highest_iter = iteration

        if highest_epoch == 0 and highest_iter == 0:
            print("nothing to load")
            return computer, optim, starting_epoch, starting_iteration

        if starting_epoch == 0 and starting_iteration == 0:
            pickle_file = Path(task_dir).joinpath(
                "saves/" + model_name + "_" + str(highest_epoch) + "_" + str(highest_iter) + ".pkl")
            print("loading model at", pickle_file)
            with pickle_file.open('rb') as pickle_file:
                computer, optim, epoch, iteration = torch.load(pickle_file)
            print('Loaded model at epoch ', highest_epoch, 'iteration', highest_iter)
        else:
            pickle_file = Path(task_dir).joinpath(
                "saves/" + model_name + "_" + str(starting_epoch) + "_" + str(starting_iteration) + ".pkl")
            print("loading model at", pickle_file)
            with pickle_file.open('rb') as pickle_file:
                computer, optim, epoch, iteration = torch.load(pickle_file)
            print('Loaded model at epoch ', starting_epoch, 'iteration', starting_iteration)

        return computer, optim, highest_epoch, highest_iter




def datetime_filename():
    return datetime.datetime.now().strftime("%m_%d_%X")


def gpu_thread_worker(nn, edge_queue, eval_batch_size, is_cuda):
    while True:
        with torch.no_grad():
            nn.eval()
            edges = []
            last_batch = False
            for i in range(eval_batch_size):
                try:
                    edge = edge_queue.get_nowait()
                    if edge is None:
                        last_batch = True
                        print("Sentinel received. GPU will process this batch and terminate afterwards")
                    else:
                        edges.append(edge)
                except queue.Empty:
                    pass

            if len(edges) != 0:
                # batch process
                states = [edge.to_node.checker_state for edge in edges]
                input_tensor = states_to_batch_tensor(states, is_cuda)
                # this line is the bottleneck
                value_tensor = nn(input_tensor)
                p = nn.children_values_to_probability(value_tensor)
                # GPU done, CPU begins
                # prior probability
                npp = p.cpu().numpy().tolist()
                # value
                value_array = value_tensor.cpu().numpy()
                value_array = np.squeeze(value_array, axis=1)
                value_list = value_array.tolist()
                # assignment and lock open
                for edx, edge in enumerate(edges):
                    edge.value = value_list[edx]
                    edge.prior_probability = npp[edx]
                    edge_queue.task_done()
                    edge.from_node.unassigned -= 1
                    if edge.from_node.unassigned == 0:
                        edge.from_node.lock.release()
            else:
                time.sleep(0.1)

            if last_batch:
                edge_queue.task_done()
                print("Queue task done signal sent. Queue will join. Thread may still be running.")
                return


if __name__ == '__main__':
    az = AlphaZero("first", is_cuda=True)
    az.train()
