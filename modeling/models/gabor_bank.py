### Gabor filter bank
import torch
import torch.nn as nn
import numpy as np

from math import ceil
from itertools import product
from skimage.filters import gabor_kernel
from skimage.util import pad

class GaborBank(nn.Module):
    """
    Gabor filter bank. Convolves the input image with
    a bank of Gabor filters, and applies a linear transform
    and ELU (softplus) nonlinearity to the output.
    """

    def __init__(self, frequencies, orientations, sigmas,
            aspect_ratios, strides, input_size, output_num):
        """
        frequencies: a list of spatial frequencies for the filters
        orientations: a list of orientations for the filters
        sigmas: a list of standard deviations for the x-axis of 
            the filters
        aspect_ratios: a list of ratios between the standard
            deviation of the y-axis and that of the x-axis
        strides: a list of strides for the convolution of the
            filters
        input_size: the shape of the grayscale input images (HxW)
        output_num: the number of output neurons to model

        The filter bank will have a filter for each combination
        of frequency, orientation, sigma, aspect ratio, and stride.
        Watch out for combinatorial explosion.
        """
        super().__init__()

        h, w = input_size
        n_filters = len(frequencies) * len(orientations) * \
                len(sigmas) * len(aspect_ratios)
        n_convs = len(strides) * 2
        self.strides = strides

        # get list of convolutions
        # also keep track of their shapes and strides
        self.real_convs = nn.ParameterList()
        self.imag_convs = nn.ParameterList()
        for stride in strides:
            real_convs = []
            imag_convs = []
            for freq, orient, sigma, ratio in product(
                    frequencies, orientations, sigmas, aspect_ratios):

                kernel = gabor_kernel(freq, orient, sigma_x=sigma,
                        sigma_y = ratio * sigma)

                # add real and imaginary parts, with correct shape
                real_convs.append(torch.FloatTensor(
                    kernel.real).unsqueeze(0).unsqueeze(0))
                imag_convs.append(torch.FloatTensor(
                    kernel.imag).unsqueeze(0).unsqueeze(0))

            # compute the max size, pad them all out to that
            # loses some activations on the ends, but is much more efficicent
            # and the edges of images are aperture anyway (basically padded)
            heights = [c.shape[2] for c in real_convs]
            widths = [c.shape[3] for c in real_convs]
            self.shape = (max(heights), max(widths)) # same for each stride

            # do the padding
            # list comprehension would be faster but less clear
            for i in range(n_filters):
                height_diff = self.shape[0] - heights[i]
                width_diff = self.shape[1] - widths[i]

                # deals with even and odd differences correctly
                real_convs[i] = torch.tensor(pad(real_convs[i], ((0, 0), (0, 0),
                    (height_diff // 2, round(height_diff / 2)),
                    (width_diff // 2, round(width_diff / 2)))))

                imag_convs[i] = torch.tensor(pad(imag_convs[i], ((0, 0), (0, 0),
                    (height_diff // 2, round(height_diff / 2)),
                    (width_diff // 2, round(width_diff / 2)))))

            # now, with kernels all same size, can combine into
            # single convolutional weight tensor -- efficient!
            self.real_convs.append(nn.Parameter(torch.cat(real_convs)))
            self.imag_convs.append(nn.Parameter(torch.cat(imag_convs)))


        # now need to compute output sizes
        each_output_size = np.floor(1 + ((np.array(input_size) - np.array(self.shape)) /
                np.array(self.strides)[..., np.newaxis]))

        # real, imaginary, and energy for each
        total_output_size = int(3 * n_filters * each_output_size.prod(1).sum(0))

        # finally, setup the regression weight
        self.weight = nn.Parameter(torch.empty((
            output_num, total_output_size)))
        self.bias = nn.Parameter(torch.empty((output_num,)))
        self.elu = nn.Softplus()

        # and initialize the weights
        self.initialize_weights()

    def forward(self, image):
        """
        Convolve the real and imaginary parts of each Gabor filter
        over the image, then compute the energy at each point as
        well. Then, take a linear combination of all those values,
        run it through an exponential linear unit, and return it.
        """
        n = image.shape[0] # batch size
        # compute real and imaginary activations, then
        # the energy, from them
        real_outputs = [nn.functional.conv2d(image, kernel, stride=stride) for
                kernel, stride in zip(self.real_convs, self.strides)]
        imag_outputs = [nn.functional.conv2d(image, kernel, stride=stride) for
                kernel, stride in zip(self.imag_convs, self.strides)]
        energy_outputs = [torch.sqrt(real ** 2 + imag ** 2) for
                real, imag in zip(real_outputs, imag_outputs)]

        # just combine them all, spatial orientation doesn't matter
        bank_outputs = torch.cat([out.view(n, -1) for out in 
            real_outputs + imag_outputs + energy_outputs], dim=1)

        linear_comb = nn.functional.linear(bank_outputs, self.weight,
                bias=self.bias)

        return self.elu(linear_comb)

    def initialize_weights(self):
        """
        Initialize the regression weight to small normal values.
        """
        # don't need some fancy init scheme since it's just one 'layer'
        nn.init.normal_(self.weight, std=0.1)
