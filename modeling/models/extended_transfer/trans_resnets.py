import torch
import torch.nn as nn
import torchvision.models.resnet as resnet

NET_DICT = {
        18: resnet.resnet18,
        34: resnet.resnet34,
        50: resnet.resnet50,
        101: resnet.resnet101,
        152: resnet.resnet152,
        }

def intermediate_size(net_type, block, input_size):
    """
    Utitility function to determine the intermediate
    size of the representation at a ResNet layer.
    Just works with square images (input_size is
    an int).
    Returns the whole NxCxHxW tensor shape
    """
    net = NET_DICT[net_type](pretrained=False)
    blocks = [m for m in net.modules() if
            isinstance(m, resnet.BasicBlock) or
            isinstance(m, resnet.Bottleneck)]
    sizes = []
    hook = blocks[block].register_forward_hook(
            lambda m,i,o: sizes.append(o.shape))
    img = torch.zeros((1, 3, input_size, input_size))
    _ = net(img)

    return sizes[0]

class TransferResnet(nn.Module):
    """
    A wrapper around a ResNet allowing extraction
    from arbitrary convolutional layers.
    Outputs a single tensor (if one layer is specified)
    or a list, if multiple are.
    Only works with the "BasicBlock" 
    (or "Bottleneck" for larger ones) components of
    the network.
    """
    def __init__(self, net_type, blocks):
        super().__init__()
        assert net_type in [18, 34, 50, 101, 152]

        self.network = NET_DICT[net_type](pretrained=True)
        self.network.fc = nn.Identity() # allows arbitrary image size
        self.blocks = blocks

        network_blocks = [m for m in self.network.modules()
                if isinstance(m, resnet.BasicBlock) or
                isinstance(m, resnet.Bottleneck)]

        if isinstance(blocks, int):
            # just a single layer
            self.singular = True
            self.acts = []
            self.hook = network_blocks[blocks].register_forward_hook(
                    lambda m,i,o: self.acts.append(o))
        else:
            self.singular = False
            self.acts = []
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
