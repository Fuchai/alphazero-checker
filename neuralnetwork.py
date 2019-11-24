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
        out2 = torch.relu(out1)
        out3 = self.conv2(out2)
        out4 = self.bn2(out3)
        out5 = out4 + input
        out6 = torch.relu(out5)
        return out6


class NoPolicy(nn.Module):
    """
    Modified for checker.
    """

    def __init__(self):
        scale = 64
        tower_len = 9
        super(NoPolicy, self).__init__()
        self.conv1 = nn.Conv2d(4, scale, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(scale)
        self.tower = nn.ModuleList([ResBlock(scale)] * tower_len)
        # self.policy_linear = nn.Linear(2 * 8 * 8, 8 * 8 + 1)
        self.value_linear_1 = nn.Linear(1 * 8 * 8, scale * 2)
        self.value_linear_2 = nn.Linear(scale * 2, 1)

        # # how to turn this into a policy if without knowing the child
        # # board state? Move in Go is easy and can be represented as a
        # # logit over the whole board, but checker allows jumps
        # self.policy_head = nn.Sequential(
        #     nn.Conv2d(scale, 2, 1),
        #     nn.BatchNorm2d(2),
        #     nn.ReLU(),
        # )

        self.value_head = nn.Sequential(
            nn.Conv2d(scale, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

    def forward(self, checker_state):
        """
        the (p,v)=f_theta(s) function
        f_theta=NeuralNetwork()
        p,v=f_theta(state)

        Modified for checker, does not return p. Only v

        To get policy, use self.child_values_to_probability([...])

        This implementation might not work in the end.

        :param checker_state: a checker state
        :return: value of the state
        """

        out = self.conv1(input)
        out1 = self.bn1(out)
        out2 = torch.relu(out1)
        for res in self.tower:
            out3 = res(out2)

        # p = self.policy_head(out3)
        # p2 = p.reshape(p.shape[0],-1)
        # p2 = self.policy_linear(p2)

        v = self.value_head(out3)
        v1 = v.reshape(v.shape[0], -1)
        v2 = self.value_linear_1(v1)
        v2 = torch.relu(v2)
        v2 = self.value_linear_2(v2)
        v2 = torch.tanh(v2)
        return v2

    def children_values_to_probability(self, children_value_tensors):
        """

        :param children_value_tensors: Must be tensors, because it needs to receive gradient at backup.
        :return:
        """
        vals = torch.stack(children_value_tensors)
        policy = torch.softmax(vals, dim=0)
        return policy


class PaperLoss(nn.Module):
    def __init__(self):
        super(PaperLoss, self).__init__()
        self.l1=torch.nn.L1Loss()
        self.lsm=nn.LogSoftmax()
        self.c=0.01

    def forward(self,v,z,logit_p,pi, network):
        return self.l1(v,z) - pi*self.lsm(logit_p)


def binary_board(board):
    # a checker board is represented with 2,1,0,-1,-2
    # represent checker board as a four dimension board instead: whether 2, whether 1, whether -1, whether -2
    two = board == 2
    one = board == 1
    negone = board == -1
    negtwo = board == -2

    ret = np.stack((two, one, negone, negtwo))
    return ret

def batch_board_tensor(board_list):
    batch_tensor=[torch.Tensor(binary_board(board).astype(np.uint8)) for board in board_list]
    ret=torch.stack(batch_tensor)
    return ret

class BoardWrapper():
    def __init__(self, nparray):
        self.nparray = nparray

    def get_board(self):
        return self.nparray


def main_example_1():
    i = torch.Tensor(np.random.rand(16, 4, 8, 8))
    nn = NoPolicy()
    ret=nn(i)
    print(ret)
    print(ret.shape)


def main_example_2():
    fake_board = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, -1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, -1, 0, -1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [-1, 0, 0, 0, 0, 0, -1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 2]])
    fake_boards=[fake_board]*16
    board_array = batch_board_tensor(fake_boards)
    print(board_array)
    print(board_array.shape)




if __name__ == '__main__':
    main_example_2()