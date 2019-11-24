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


class Zero:
    """
    Use MCTS to generate g number of games
    Train based on these games with repeatedly sampled time steps
    Refresh the g number of games, repeat
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.nn = NoPolicy()
        # TODO
        self.loss = PaperLoss()
        self.optim = torch.optim.Adam(self.nn.parameters(), weight_decay=0.01)
        self.batch_size = 16
        # time steps contain up to self.game_size different games.
        self.time_steps = []
        self.game_size = 10
        self.total_epochs = 20
        self.train_period = 2000
        self.validation_period = 100
        self.print_period = 10
        self.save_period = 1000
        self.log_file = "log/" + self.save_str + "_" + self.model_name + "_" + datetime_filename() + ".txt"

    def refresh_games(self):
        self.time_steps = []
        for i in range(self.game_size):
            mcts = MCTS(self.nn)
            mcts.play_until_terminal()
            self.time_steps += mcts.time_steps

    def train_one_round(self):
        # sample self.batch_size number of time steps, bundle them together
        self.batch_tss = random.choices(self.time_steps, k=self.batch_size)
        input, target = self.time_steps_to_tensor(self.batch_tss)
        output = self.nn(input)
        loss = self.loss(output, target)
        loss.backward()
        self.optim.step()
        return loss.data

    def validation(self):
        self.nn.eval()
        validation_time_steps = []
        for i in range(self.game_size):
            mcts = MCTS(self.nn)
            mcts.play_until_terminal()
            validation_time_steps += mcts.time_steps
        self.batch_tss = random.choices(validation_time_steps, k=self.batch_size)
        input, target = self.time_steps_to_tensor(self.batch_tss)
        output = self.nn(input)
        loss = self.loss(output, target)
        return loss.data

    def train(self):
        for epoch in range(self.total_epochs):
            for ti in range(self.train_period):
                train_loss = self.train_one_round()
                if ti % self.print_period == 0:
                    self.logprint(
                        "%14s " % self.model_name +
                        "train epoch %4d, batch %4d. running loss: %.5f" %
                        (epoch, ti, train_loss))
                if ti % self.validation_period == 0:
                    self.validation()

    def time_steps_to_tensor(self, batch_tss):
        # TODO
        return None

    def log_print(self, message):
        string = str(message)
        if self.log_file is not None and self.log_file != False:
            with open(self.log_file, 'a') as handle:
                handle.write(string + '\n')
        print(string)

    def save_models(self, epoch, iteration):

        epoch = int(epoch)
        task_dir = os.path.dirname(abspath(__file__))
        if not os.path.isdir(Path(task_dir) / "saves"):
            os.mkdir(Path(task_dir) / "saves")

        pickle_file = Path(task_dir).joinpath(
            "saves/" + self.save_str + "/" + self.model_name + "_" + str(epoch) + "_" + str(iteration) + ".pkl")
        with pickle_file.open('wb') as fhand:
            torch.save((self.nn, self.optim, epoch, iteration), fhand)

        print("saved model", self.model_name, "at", pickle_file)

    def load_models(self, epoch=0):
        """

        :param epoch: 0 is load the newest, or load the epoch epoch.
        :return:
        """
        models = []
        optims = []
        hes = []
        his = []

        if len(self.optims) == 0:
            raise ValueError("Please initialize models and optims first")

        for model, name, optim in zip(self.models, self.model_names, self.optims):
            model, optim, highest_epoch, highest_iter = self.load_model(model, optim, epoch, 0, self.save_str, name)
            models.append(model)
            optims.append(optim)
            hes.append(highest_epoch)
            his.append(highest_iter)
        try:
            he = hes[0]
            hi = his[0]
        except:
            raise ValueError("No model to load")
        for i in range(len(self.models)):
            assert (hes[i] == he)
            assert (his[i] == hi)

        self.models = models
        self.optims = optims
        print("All models loaded")
        self.he = he
        self.hi = hi

        return he, hi


def datetime_filename():
    return datetime.datetime.now().strftime("%m_%d_%X")
