import torch
import torch.nn as nn

from modeling.models.cnns.bethge import FactorizedLinear

class FarNearSurroundTransfer(nn.Module):
    """
    The near-surround model extended with
    later layers of the base CNN used to
    implement "far-surround" feedback.
    """
    def __init__(self, base_transform, trans_out_sizes,
            trans_out_channels, conv_out_channels,
            out_neurons, iterations, downscale=1,
            conv_k=3, fb_conv_k=3, r_conv_k=3, conv_groups=1,
            p_dropout=0.0):
        """
        base_transform: the Module that is being
        transfer-learned from, which will output
        a list of tensors, the first being the
        feedforward input, and the later ones being
        used as far-surround feedback

        trans_out_sizes: list of the height/width of the
        (square) output of each part of base_transform

        trans_out_channels: list of the number of channels
        in the output of each base_transform

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
        
        p_dropout: the dropout probability for the
        far-surround feedback layers
        """
        super().__init__()
        self.nonlin = nn.Softplus()
        self.out_neurons = out_neurons
        self.iterations = iterations

        self.base = base_transform

        self.ff_conv = nn.Conv2d(trans_out_channels[0],
                conv_out_channels, kernel_size=conv_k,
                padding=conv_k // 2)

        self.fb_convs = nn.ModuleList([nn.Conv2d(trans_out_channels[i],
            conv_out_channels, kernel_size=fb_conv_k, padding=fb_conv_k//2)
            for i in range(1,len(trans_out_channels))])
        self.upsample = nn.Upsample(trans_out_sizes[0], mode='bilinear')

        self.near_conv = nn.Conv2d(conv_out_channels,
                conv_out_channels, kernel_size=r_conv_k,
                padding=r_conv_k // 2, groups=conv_groups)

        self.pool = nn.MaxPool2d(downscale, downscale)

        self.readout = FactorizedLinear(conv_out_channels,
                int(trans_out_sizes[0] // downscale), out_neurons)

        if p_dropout > 0:
            self.dropout = nn.Dropout(p=p_dropout)
        else:
            self.dropout = nn.Identity()

        # for hooks on the full output
        self.ident = nn.Identity()

    def forward(self, x):
        out = torch.empty(x.shape[0], self.out_neurons, self.iterations).to(x.device)

        base = self.base(x)
        conv_out = self.nonlin(self.ff_conv(base[0]))
        out[:, :, 0] = self.nonlin(self.readout(self.pool(conv_out)))

        for i in range(1, self.iterations):
            conv_out = self.nonlin(conv_out + self.near_conv(conv_out) +
                    torch.sum(torch.stack([self.upsample(self.fb_convs[j](
                        self.dropout(base[j+1])))
                        for j in range(len(self.fb_convs))]), 0))
            out[:, :, i] = self.nonlin(self.readout(self.pool(conv_out)))

        out = self.ident(out)
        return out
