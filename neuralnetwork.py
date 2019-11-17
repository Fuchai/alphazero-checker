import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, scale):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(scale, scale, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(scale)
        self.conv2 = nn.Conv2d(scale, scale, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(scale)

    def forward(self, input):
        out = self.conv1(input)
        out1 = self.bn1(out)
        out2 = F.relu(out1)
        out3 = self.conv2(out2)
        out4 = self.bn2(out3)
        out5 = out4 + input
        out6 = F.relu(out5)
        return out6


class NeuralNetwork(nn.Module):

    def __init__(self):
        scale = 64
        tower_len = 9
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, scale, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(scale)
        self.tower = nn.ModuleList([ResBlock(scale)] * tower_len)
        self.policy_linear = nn.Linear(2 * 8 * 8, 8 * 8 + 1)
        self.value_linear_1 = nn.Linear(1* 8 * 8, scale * 2)
        self.value_linear_2 = nn.Linear(scale * 2, 1)

        # how to turn this into a policy if without knowing the child
        # board state? Move in goal is easy and can be represented as a
        # logit over the whole board, but checker allows jumps
        self.policy_head = nn.Sequential(
            nn.Conv2d(scale, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(scale, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

    def forward(self, state):
        """
        the (p,v)=f_theta(s) function
        f_theta=NeuralNetwork()
        p,v=f_theta(state)
        """
        board_array = state.get_board()
        board_tensor = torch.Tensor(board_array)

        out = self.conv1(board_tensor)
        out1 = self.bn1(out)
        out2 = F.relu(out1)
        for res in self.tower:
            out3 = res(out2)

        p = self.policy_head(out3)
        p2 = p.reshape(p.shape[0],-1)
        p2 = self.policy_linear(p2)

        v = self.value_head(out3)
        v1 = v.reshape(v.shape[0],-1)
        v2 = self.value_linear_1(v1)
        v2 = F.relu(v2)
        v2 = self.value_linear_2(v2)
        v2 = F.tanh(v2)
        return p2, v2


class BoardWrapper():
    def __init__(self, nparray):
        self.nparray = nparray

    def get_board(self):
        return self.nparray


if __name__ == '__main__':
    i = BoardWrapper(np.zeros((16, 1, 8, 8)))
    nn = NeuralNetwork()
    print(nn(i))
