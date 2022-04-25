import torch
import torch.nn as nn

from modeling.models.cnns.bethge import FactorizedLinear

class NearSurroundTransfer(nn.Module):
    """
    Transfer learning, but with a recurrent
    convolutional layer between the readout and
    the base task-driven CNN layer.
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

        self.pool = nn.MaxPool2d(downscale, downscale)

        self.readout = FactorizedLinear(conv_out_channels,
                int(trans_out_size // downscale), out_neurons)

        # for hooks on the intermediate layer and full output
        self.int_ident = nn.Identity()
        self.ident = nn.Identity()

    def forward(self, x):
        out = torch.empty(x.shape[0], self.out_neurons, self.iterations).to(x.device)

        conv_out = self.int_ident(self.nonlin(self.ff_conv(self.base(x))))
        out[:, :, 0] = self.nonlin(self.readout(self.pool(conv_out)))

        for i in range(1, self.iterations):
            conv_out = self.int_ident(self.nonlin(conv_out + self.r_conv(conv_out)))
            out[:, :, i] = self.nonlin(self.readout(self.pool(conv_out)))

        out = self.ident(out)

        return out
