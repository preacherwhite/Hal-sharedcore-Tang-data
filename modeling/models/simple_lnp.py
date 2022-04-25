import torch
import torch.nn as nn

class LinearNonlinear(nn.Module):
    def __init__(self, input_shape, output_size):
        super().__init__()
        self.input_size = input_shape[0] * input_shape[1]
        self.output_size = output_size

        self.linear = nn.Linear(self.input_size, self.output_size,
                bias=True)
        self.nonlinear = nn.Softplus()

    def forward(self, x):
        return self.nonlinear(self.linear(x.view(x.shape[0], -1)))
