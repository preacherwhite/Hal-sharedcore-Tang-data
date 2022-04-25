import torch
import torch.nn as nn

from modeling.models.cnns.bethge import FactorizedLinear

class AltNearSurroundTransfer(nn.Module):
    """
    An extension of the near-surround transfer model,
    changing its recurrent processing. Now, the
    feedforward input is kept directly in the computation
    of every state, and its influence gated by a learned
    parameter at each timestep for each channel.
    """
    def __init__(self, base_transform, trans_out_size,
            trans_out_channels, conv_out_channels,
            out_neurons, iterations, downscale=1,
            conv_k=3, r_conv_k=3, conv_groups=1, dilation=1):
        """
        base_transform: the Module that is being
        transfer-learned from

        trans_out_size: the height/width of the
        (square) output of the base_transform

        trans_out_channels: the number of channels
        in the output of the base_transform

        conv_out_channels: the number of channels in
        the learned convolutional layer

        out_neurons: the number of neurons to be read
        out from the convolutional layer

        downscale: the level of (max-pooling) downsampling
        to perform between the convolutional layer and
        the readout

        iterations: the number of timesteps to run
        the recurrent processing for (can be updated
        at any point)

        conv_k: the size of the feedforward conovlutional
        kernels

        r_conv_k: the size of the recurrent convolutional
        kernels

        conv_groups: the number of groups in the recurrent
        convolutional kernels

        dilation: the dilation of the recurrent convolutional
        kernel
        """
        super().__init__()
        self.nonlin = nn.Softplus()
        self.out_neurons = out_neurons
        self.iterations = iterations

        self.base = base_transform

        self.ff_conv = nn.Conv2d(trans_out_channels,
                conv_out_channels, kernel_size=conv_k,
                padding=conv_k // 2)

        self.r_conv = nn.Conv2d(conv_out_channels,
                conv_out_channels, kernel_size=r_conv_k,
                padding=int(dilation * (r_conv_k // 2)), groups=conv_groups,
                dilation=dilation)

        self.t_kern = nn.Parameter(torch.ones(conv_out_channels, 
            self.iterations - 1))

        self.pool = nn.MaxPool2d(downscale, downscale)

        self.readout = FactorizedLinear(conv_out_channels,
                int(trans_out_size // downscale), out_neurons)

        # for hooks on the full output
        self.ident = nn.Identity()

    def forward(self, x):
        out = torch.empty(x.shape[0], self.out_neurons, self.iterations).to(x.device)

        conv_out = self.nonlin(self.ff_conv(self.base(x)))
        this_out = conv_out
        out[:, :, 0] = self.nonlin(self.readout(self.pool(conv_out)))

        for i in range(1, self.iterations):
            this_out = self.nonlin(conv_out * self.t_kern[:, i-1].unsqueeze(1).unsqueeze(2) + 
                    self.r_conv(this_out))
            out[:, :, i] = self.nonlin(self.readout(self.pool(this_out)))

        out = self.ident(out)

        return out
