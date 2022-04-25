### 062620 -- training "extended transfer learning models"
### with near and far-surround recurrence
### (note these are more like templates, constantly changed)

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from functools import partial
from scipy.stats import pearsonr

from analysis.data_utils import get_neural_data, spike_counts, \
        trial_average
from modeling import LOG_DIR, SAVE_DIR
from modeling.losses import *
from modeling.models.extended_transfer import trans_resnets, \
        near_surround, far_surround, trans_densenets, alt_near_surround
from modeling.models.cnns.utils import num_params
from modeling.data_utils import get_images, train_val_test_split, \
        torchvision_normalize
from modeling.train_utils import array_to_dataloader, simple_train_loop

# general things
DATASET = 'tang'
CORR_THRESHOLD = 0.7
START = 530
END = 1130
WINDOW_SIZE = 100
SORT = 'batch'
LOG_FILE = sys.stdout # for testing

# data setup
DOWNSAMPLE = 2
images = get_images(DATASET, downsample=DOWNSAMPLE,
        torch_format=True, normalize=False)
images = torchvision_normalize(images)

data = get_neural_data(DATASET, CORR_THRESHOLD,
        elecs=False,sort=SORT)
data = trial_average(data)
data = np.array([spike_counts(data, start, start+WINDOW_SIZE)
    for start in range(START, END, WINDOW_SIZE)])
data = data.transpose((1, 2,0 )) # to CxNxT
iterations = data.shape[2] # time dimension

train_idx, val_idx, test_idx = train_val_test_split(
        total_size=2250, train_size=1250, val_size=500,
        deterministic=True)

train_x = images[train_idx]
val_x = images[val_idx]
test_x = images[test_idx]

train_y = data[train_idx]
val_y = data[val_idx]
test_y = data[test_idx]

# training params
lr = 3e-3
scale = 1e-2

# model params
base_net = 'densenet'
net_size = 121
#block = 3
channels = 32
ff_k = 1
r_k = 3
groups = 1
downscale = 1

for block in [8, 10]:
    for dilation in [1]:
        for r_k in [3]:
            for scale in [3e-4, 1e-3, 1e-2]:
                for sd in range(2):
                    # changed a lot
                    num_epochs = 300
                    print_every=30

                    key = f'near_surround_{SORT}_{DATASET}_it{iterations}_ds{DOWNSAMPLE}__{base_net}{net_size}_block{block}_chan{channels}_ff{ff_k}_r{r_k}_dil{dilation}_gr{groups}_ds{downscale}__lr{lr}_scale{scale}_e{num_epochs}_sd{sd}'

                    save_dir = SAVE_DIR + f'transfer_learning/{key}/'
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)
                    LOG_FILE = open(save_dir + 'train_log.txt', 'w')

                    base_network = trans_densenets.TransferDensenet(net_size, block)
                    out_shape = trans_densenets.intermediate_size(net_size, block, 252 // DOWNSAMPLE)
                    model = near_surround.NearSurroundTransfer(base_network, trans_out_size=out_shape[2],
                            trans_out_channels=out_shape[1], conv_out_channels=channels,
                            out_neurons=data.shape[1], iterations=iterations, downscale=downscale,
                            conv_k=ff_k, r_conv_k=r_k, conv_groups=groups, dilation=dilation)


                    def temp_poisson_loss(p, y, n):
                        return torch.mean(torch.stack([poisson_loss(p[:, :, i], y[:, :, i])
                            for i in range(iterations)]))

                    # don't actually use temporal response
                    # analagous to yimeng's models
                    #def temp_poisson_loss(p, y, n):
                        #return poisson_loss(p.sum(2), y.sum(2))

                    def full_loss(p, y, n, scale):
                        base = temp_poisson_loss(p, y, n)

                        spatial_map_flat = n.readout.weight_spatial.view(
                                train_y.shape[1], -1)
                        feature_map_flat = n.readout.weight_feature.view(
                                train_y.shape[1], -1)
                        readout_sparsity = scale * torch.mean(
                                torch.sum(torch.abs(spatial_map_flat), 1) *
                                torch.sum(torch.abs(feature_map_flat), 1))

                        return base + readout_sparsity

                    full_loss = partial(full_loss, scale=scale)

                    ### training
                    train_loader = array_to_dataloader(train_x, train_y, batch_size=60)
                    val_loader = array_to_dataloader(val_x, val_y, batch_size=125)
                    test_loader = array_to_dataloader(test_x, test_y, batch_size=125)

                    params = list(set(model.parameters()) - set(base_network.parameters()))
                    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-3)

                    trained, train_loss, val_loss = simple_train_loop(train_loader,
                            val_loader, model, optimizer, full_loss, temp_poisson_loss,
                            num_epochs, print_every, stop_criterion=None, device='cuda',
                            log_file=LOG_FILE, clipping=False)

                    for p_group in optimizer.param_groups:
                        p_group['lr'] = p_group['lr'] / 10.0

                    trained, train_loss, val_loss = simple_train_loop(train_loader,
                            val_loader, trained, optimizer, full_loss, temp_poisson_loss,
                            num_epochs, print_every, stop_criterion=None, device='cuda',
                            log_file=LOG_FILE, clipping=False)

                    ### evaluation

                    trained = trained.cpu().eval()
                    train_preds = []
                    test_preds = []
                    with torch.no_grad():
                        for x, y in train_loader:
                            train_preds.append(trained(x.float()))

                        for x, y in test_loader:
                            test_preds.append(trained(x.float()))

                    train_preds = torch.cat(train_preds, dim=0)
                    test_preds = torch.cat(test_preds, dim=0)

                    overall_train_corrs = np.zeros(data.shape[1])
                    overall_test_corrs = np.zeros(data.shape[1])
                    temp_train_corrs = np.zeros(data.shape[1:])
                    temp_test_corrs = np.zeros(data.shape[1:])
                    for neu in range(data.shape[1]):
                        overall_train_resp = train_y[:, neu, :].sum(1)
                        overall_train_preds = train_preds[:, neu, :].sum(1)
                        overall_train_corrs[neu] = pearsonr(overall_train_resp.flatten(),
                                overall_train_preds.flatten())[0]

                        overall_test_resp = test_y[:, neu, :].sum(1)
                        overall_test_preds = test_preds[:, neu, :].sum(1)
                        overall_test_corrs[neu] = pearsonr(overall_test_resp.flatten(),
                                overall_test_preds.flatten())[0]

                        for time in range(data.shape[2]):
                            temp_train_resp = train_y[:, neu, time]
                            temp_train_preds = train_preds[:, neu, time]
                            temp_train_corrs[neu, time] = pearsonr(temp_train_resp.flatten(),
                                    temp_train_preds.flatten())[0]

                            temp_test_resp = test_y[:, neu, time]
                            temp_test_preds = test_preds[:, neu, time]
                            temp_test_corrs[neu, time] = pearsonr(temp_test_resp.flatten(),
                                    temp_test_preds.flatten())[0]

                    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', file=LOG_FILE)
                    print(f'time-averaged train corrs: {overall_train_corrs}', file=LOG_FILE)
                    print(f'average time-averaged train corrs: {overall_train_corrs.mean()}', file=LOG_FILE)
                    print(f'time-averaged test corrs: {overall_test_corrs}', file=LOG_FILE)
                    print(f'average time-averaged test corrs: {overall_test_corrs.mean()}', file=LOG_FILE)
                    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', file=LOG_FILE)
                    print(f'train corrs: {np.round(temp_train_corrs, 2)}', file=LOG_FILE)
                    print(f'average train corrs: {temp_train_corrs.mean()}', file=LOG_FILE)
                    print(f'test corrs: {np.round(temp_test_corrs, 2)}', file=LOG_FILE)
                    print(f'average test corrs: {temp_test_corrs.mean()}', file=LOG_FILE)
                    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', file=LOG_FILE)
                    print(f'train corrs: {np.round(temp_train_corrs, 2)}')
                    print(f'average train corrs: {temp_train_corrs.mean()}')
                    print(f'test corrs: {np.round(temp_test_corrs, 2)}')
                    print(f'average test corrs: {temp_test_corrs.mean()}')

                    ### saving
                    plt.figure()
                    plt.imshow(temp_test_corrs, cmap='viridis', interpolation='nearest')
                    plt.colorbar()
                    plt.xlabel(f'{WINDOW_SIZE}ms time bin')
                    plt.ylabel('neuron')
                    plt.title(f'test correlation over neurons and times')
                    plt.savefig(save_dir + 'test_corrs.png')

                    torch.save(trained.state_dict(), save_dir + 'model.pt')
                    np.save(save_dir + 'test_corrs', temp_test_corrs)
                    LOG_FILE.close()
