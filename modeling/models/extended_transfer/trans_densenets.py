import torch
import torch.nn as nn
import torchvision.models.densenet as densenet

NET_DICT = {
        121: densenet.densenet121,
        161: densenet.densenet161,
        169: densenet.densenet169,
        201: densenet.densenet201
        }

def intermediate_size(net_type, block, input_size, inpt=True):
    """
    Utitility function to determine the intermediate
    size of the representation at a ResNet layer.
    Just works with square images (input_size is
    an int).
    Returns the whole NxCxHxW tensor shape
    """
    net = NET_DICT[net_type](pretrained=False)
    blocks = [m for m in net.modules() if
            isinstance(m, densenet._DenseLayer)]
    sizes = []
    if inpt:
        hook = blocks[block].register_forward_hook(
                lambda m,i,o: sizes.append(i[0].shape))
    else:
        hook = blocks[block].register_forward_hook(
                lambda m,i,o: sizes.append(o.shape))
    img = torch.zeros((1, 3, input_size, input_size))
    _ = net(img)

    return sizes[0]

class TransferDensenet(nn.Module):
    """
    A wrapper around a DenseNet allowing extraction
    from arbitrary convolutional layers.
    Outputs a single tensor (if one layer is specified)
    or a list, if multiple are.
    Only works with the "_DenseLayer" components
    of the network.
    By default extracts the input to the layer (since
    DenseNet layers have very few output channels
    individually).
    """
    def __init__(self, net_type, blocks, inpt=True):
        super().__init__()
        assert net_type in NET_DICT.keys()

        self.network = NET_DICT[net_type](pretrained=True)
        self.network.fc = nn.Identity() # allows arbitrary image size
        self.blocks = blocks
        self.inpt = inpt

        network_blocks = [m for m in self.network.modules()
                if isinstance(m, densenet._DenseLayer)]

        if isinstance(blocks, int):
            # just a single layer
            self.singular = True
            self.acts = []
            if inpt:
                self.hook = network_blocks[blocks].register_forward_hook(
                        lambda m,i,o: self.acts.append(i[0]))
            else:
                self.hook = network_blocks[blocks].register_forward_hook(
                        lambda m,i,o: self.acts.append(o))

        else:
            self.singular = False
            self.acts = []
            if inpt:
                self.hook = [network_blocks[blocks[j]].register_forward_hook(
                    lambda m,i,o: self.acts.append(i)) for j
                    in range(len(blocks))]
            else:
                self.hook = [network_blocks[blocks[j]].register_forward_hook(
                    lambda m,i,o: self.acts.append(o)) for j
                    in range(len(blocks))]

    def forward(self, x):
        _ = self.network(x)
        if self.singular:
            act = self.acts[0]
            self.acts = []
            return act
        else:
            acts = [self.acts[i] for i in range(len(self.blocks))]
            self.acts = []
            return acts
