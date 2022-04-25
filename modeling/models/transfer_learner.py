import torch
import torch.nn as nn

class TransferLearner(nn.Module):
    """
    Just a linear-nonlinear model with an exponential
    nonlinearity and explicit spatial form of the weights.
    Presumably used for transfer learning on extracted
    CNN features
    """
    def __init__(self, in_shape, out_num):
        """
        in_shape is a tuple or Shape object with the shape of an input
        (in CxHxW form)

        out_num is the dimensionality of outputs (presumably
        the number of neurons)
        """
        super().__init__()
        self.bias = nn.Parameter(torch.empty((out_num,),
            dtype=torch.float32))

        self.c_in = in_shape[0]
        self.h = in_shape[1]
        self.w = in_shape[2]
        self.out_num = out_num
        self.weight = nn.Parameter(torch.empty((self.c_in, self.h, self.w, out_num),
            dtype=torch.float32))

        self.initialize_std()

    def initialize_std(self):
        nn.init.zeros_(self.bias)
        # not sure if this stddev is reasonable
        nn.init.normal_(self.weight, std=1 / (self.c_in * self.h * self.w))

    def forward(self, x):
        return torch.exp(nn.functional.linear(x.view(x.shape[0], -1), 
                self.weight.view(self.out_num, -1), self.bias))
